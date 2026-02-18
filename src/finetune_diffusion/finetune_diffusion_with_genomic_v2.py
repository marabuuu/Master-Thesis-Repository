#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune Diffusion Model with Genomic Conditioning (v2 — MoPaDi-native)

This version subclasses MoPaDi's ``LitModel`` to inherit all of its
infrastructure:

 - TensorBoard scalar + image logging
 - EMA updates (full state_dict, configurable decay)
 - Periodic sample generation  (``log_sample``)
 - FID / LPIPS evaluation      (``evaluate_scores``)
 - Gradient clipping            (``on_before_optimizer_step``)
 - Multi-GPU DDP via PyTorch Lightning
 - Mixed-precision (``precision="16-mixed"``)
 - ModelCheckpoint callbacks

Only the genomic-specific parts are added on top:

 1. A frozen (or jointly trainable) **ProjectionHead** that maps
    genomic vectors → pseudo image features.
 2. A custom **GenomicTileDataset** that pairs genomic H5 files with
    tile images from zip archives.
 3. An overridden ``training_step`` that projects the genomic vector
    *before* handing it to the diffusion sampler.
 4. An overridden ``configure_optimizers`` that optionally includes
    the projection head parameters.

Architecture
------------
Genomic (512-dim) → ProjectionHead → pseudo image features (512-dim) → cDDIM → tile

Usage
-----
    python finetune_diffusion_with_genomic_v2.py \\
        --projection-head-ckpt ./projection_head_best.pt \\
        --diffusion-ckpt ./last.ckpt \\
        --genomic-h5-dir /path/to/genomic_features \\
        --tiles-zip-dir /path/to/tile_zips \\
        --out-dir ./finetuned_genomic \\
        --epochs 10 --batch-size 8 --lr 1e-5
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import re
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# MoPaDi imports
from mopadi.train_diff_autoenc import LitModel, ema, is_time
from mopadi.configs.config import TrainConfig
from mopadi.configs.templates import tcga_brca_autoenc
from mopadi.configs.choices import TrainMode


# ======================================================================
#  Projection Head
# ======================================================================

class ProjectionHead(nn.Module):
    """MLP / linear / residual projection  genomic → image-feature space."""

    def __init__(
        self,
        in_dim: int = 512,
        out_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        arch: str = "mlp",
        dropout: float = 0.1,
        normalize_output: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.arch = arch
        self.normalize_output = normalize_output

        if arch == "linear":
            self.net = nn.Linear(in_dim, out_dim)
        elif arch == "mlp":
            layers = []
            dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(dropout))
            self.net = nn.Sequential(*layers)
        elif arch == "residual":
            layers = []
            for i in range(num_layers):
                layers.append(
                    nn.Linear(hidden_dim if i > 0 else in_dim, hidden_dim)
                )
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.net = nn.Sequential(*layers)
            self.skip = (
                nn.Identity() if in_dim == out_dim
                else nn.Linear(in_dim, out_dim)
            )
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.arch == "residual":
            out = self.net(x) + self.skip(x)
        else:
            out = self.net(x)
        if self.normalize_output:
            out = F.normalize(out, p=2, dim=-1)
        return out


def load_projection_head(ckpt_path: str, device: str = "cpu") -> ProjectionHead:
    """Restore a trained projection head from its checkpoint."""
    # Fail early if the checkpoint path is invalid
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Projection head checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    head = ProjectionHead(
        in_dim=cfg.get("in_dim", 512),
        out_dim=cfg.get("out_dim", 512),
        hidden_dim=cfg.get("hidden_dim", 512),
        num_layers=cfg.get("num_layers", 2),
        arch=cfg.get("arch", "mlp"),
    )
    head.load_state_dict(ckpt["state_dict"])
    head.to(device)
    head.eval()
    print(f"[OK] Loaded projection head from {ckpt_path}  (config: {cfg})")

    # Basic sanity checks on loaded weights
    try:
        state_vals = list(head.state_dict().values())
        if state_vals:
            sample = state_vals[0]
            if torch.isnan(sample).any():
                print("[WARN] Projection head contains NaNs after load")
    except Exception:
        pass
    return head


# ======================================================================
#  Dataset
# ======================================================================

def canonical_patient_id(name: str) -> str:
    """Extract TCGA-XX-XXXX from various filename formats."""
    name = Path(name).stem.upper()
    for sep in ("_", "."):
        name = name.replace(sep, "-")
    while "--" in name:
        name = name.replace("--", "-")
    parts = name.split("-")
    if len(parts) >= 3 and parts[0].startswith("TCGA"):
        return "-".join(parts[:3])
    return name


class GenomicTileDataset(Dataset):
    """
    Pairs genomic H5 features with tile images from zip archives.

    Each ``__getitem__`` returns::

        {"img": tensor(3, H, W),   # [-1, 1] normalised
         "feat": tensor(D,),       # genomic feature (matches MoPaDi key)
         "patient_id": str}

    The key is called ``"feat"`` (not ``"genomic"``) so it is directly
    compatible with MoPaDi's ``training_step`` expectations after the
    projection head transforms it.
    """

    def __init__(
        self,
        genomic_h5_dir: str,
        tiles_zip_dir: str,
        genomic_key: str = "feats",
        transform=None,
        tiles_per_patient: int = 1,
        split: str = "train",
    ):
        super().__init__()
        self.genomic_h5_dir = Path(genomic_h5_dir).expanduser()
        self.tiles_zip_dir = Path(tiles_zip_dir).expanduser()
        self.genomic_key = genomic_key
        self.transform = transform
        self.tiles_per_patient = tiles_per_patient

        # Discover genomic files (support train/test subdirs)
        genomic_files = self._find_genomic_files(split)

        zip_files = {
            canonical_patient_id(f.name): f
            for f in self.tiles_zip_dir.glob("*.zip")
        }

        common_ids = sorted(set(genomic_files) & set(zip_files))
        if not common_ids:
            raise RuntimeError(
                f"No matching patients between\n"
                f"  Genomic: {self.genomic_h5_dir}\n"
                f"  Tiles:   {self.tiles_zip_dir}"
            )
        print(f"[GenomicTileDataset] {len(common_ids)} matched patients (split={split})")

        self.pairs = [
            (genomic_files[pid], zip_files[pid], pid) for pid in common_ids
        ]

        # Cache genomic features (small — one vector per patient)
        self.genomic_cache: dict[str, torch.Tensor] = {}
        for gpath, _, pid in tqdm(self.pairs, desc="Caching genomic"):
            with h5py.File(gpath, "r") as f:
                arr = np.array(f[self.genomic_key])
                if arr.ndim == 2:
                    arr = arr.mean(axis=0)
                # Validate shape and values
                if arr.ndim != 1:
                    raise ValueError(f"Genomic vector for {pid} has unexpected shape {arr.shape}")
                if np.isnan(arr).any():
                    raise ValueError(f"Genomic vector for {pid} contains NaNs")
                self.genomic_cache[pid] = torch.from_numpy(arr.astype(np.float32))

        # Index tile names inside each zip
        self.tile_lists: dict[str, list[str]] = {}
        bad_pids: set[str] = set()
        for _, zpath, pid in tqdm(self.pairs, desc="Indexing zips"):
            try:
                with zipfile.ZipFile(zpath, "r") as zf:
                    names = [
                        n for n in zf.namelist()
                        if n.lower().endswith((".png", ".jpg", ".jpeg"))
                    ]
                    if not names:
                        print(f"[WARN] No image tiles found in zip for patient {pid}: {zpath}")
                        bad_pids.add(pid)
                    else:
                        self.tile_lists[pid] = names
            except zipfile.BadZipFile:
                print(f"[WARN] Bad zip skipped: {zpath}")
                bad_pids.add(pid)

        if bad_pids:
            self.pairs = [
                (g, z, p) for g, z, p in self.pairs if p not in bad_pids
            ]

        # Expand to tiles_per_patient samples per patient
        self.samples: list[tuple[str, Path]] = []
        for _, zpath, pid in self.pairs:
            n = min(self.tiles_per_patient, len(self.tile_lists.get(pid, [])))
            self.samples.extend([(pid, zpath)] * n)

        print(f"[GenomicTileDataset] {len(self.samples)} total samples")

    # ------------------------------------------------------------------
    def _find_genomic_files(self, split: str) -> dict[str, Path]:
        train_dir = self.genomic_h5_dir / "train"
        test_dir = self.genomic_h5_dir / "test"
        if train_dir.is_dir() or test_dir.is_dir():
            files: dict[str, Path] = {}
            if split in ("train", "all") and train_dir.is_dir():
                files.update({
                    canonical_patient_id(f.name): f
                    for f in train_dir.glob("*.h5")
                })
            if split in ("test", "all") and test_dir.is_dir():
                files.update({
                    canonical_patient_id(f.name): f
                    for f in test_dir.glob("*.h5")
                })
            return files
        return {
            canonical_patient_id(f.name): f
            for f in self.genomic_h5_dir.glob("*.h5")
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        pid, zpath = self.samples[idx]
        genomic = self.genomic_cache[pid]  # raw genomic vector

        # Random tile from zip
        tile_name = random.choice(self.tile_lists[pid])
        with zipfile.ZipFile(zpath, "r") as zf:
            with zf.open(tile_name) as fh:
                img = Image.open(BytesIO(fh.read())).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        # Validate image tensor
        if not torch.is_tensor(img):
            raise TypeError(f"Transformed image is not a tensor for pid={pid}")
        if img.ndim != 3 or img.shape[0] != 3:
            raise ValueError(f"Image tensor has invalid shape {img.shape} for pid={pid}")

        return {
            "img": img,
            "feat": genomic,        # NOTE: raw genomic — projection happens in training_step
            "patient_id": pid,
        }


# ======================================================================
#  LitModel subclass for genomic fine-tuning
# ======================================================================

class LitModelGenomicFinetune(LitModel):
    """
    Subclass of MoPaDi's ``LitModel`` that fine-tunes a pretrained
    diffusion autoencoder with genomic conditioning via a projection head.

    Inherited from ``LitModel`` (no reimplementation needed):
      - TensorBoard scalar & image logging
      - EMA updates (full ``state_dict``, configurable ``ema_decay``)
      - ``log_sample``  — periodic sample generation → TensorBoard + disk
      - ``evaluate_scores`` — FID / LPIPS during training
      - ``on_before_optimizer_step`` — gradient clipping
      - Multi-GPU / DDP support (via PyTorch Lightning)
      - Mixed precision (``precision='16-mixed'``)

    Overridden / added:
      - ``__init__``  : creates the projection head
      - ``setup``     : creates the ``GenomicTileDataset``
      - ``training_step`` : projects genomic → image-feature space first
      - ``on_train_batch_end`` : same as parent but passes projected cond
      - ``configure_optimizers`` : optionally adds projection head params
      - ``train_dataloader`` : returns the genomic tile dataloader
    """

    def __init__(
        self,
        conf: TrainConfig,
        projection_head: ProjectionHead,
        freeze_projection_head: bool = True,
        genomic_h5_dir: str = "",
        tiles_zip_dir: str = "",
        tiles_per_patient: int = 10,
        split: str = "train",
        img_size: int = 512,
    ):
        # LitModel.__init__ creates self.model, self.ema_model, samplers, etc.
        super().__init__(conf)

        self.projection_head = projection_head
        self.freeze_projection_head = freeze_projection_head
        self.genomic_h5_dir = genomic_h5_dir
        self.tiles_zip_dir = tiles_zip_dir
        self.tiles_per_patient = tiles_per_patient
        self.split = split
        self._img_size = img_size

        if self.freeze_projection_head:
            self.projection_head.eval()
            for p in self.projection_head.parameters():
                p.requires_grad = False

        # Basic projection head checks
        if not hasattr(self.projection_head, "in_dim") or not hasattr(self.projection_head, "out_dim"):
            raise AttributeError("projection_head must expose in_dim and out_dim attributes")

    # ------------------------------------------------------------------
    # Startup hooks — override parent hooks that assume WDS data pipeline
    # ------------------------------------------------------------------

    def on_fit_start(self):
        """
        Override to skip MoPaDi's WebDataset sanity check.

        The parent's ``on_fit_start`` calls ``expand_shards(self.conf.data_dirs)``
        and ``sanity_check_precomputed_feats()`` — both assume a WebDataset
        (WebDataset + precomputed image features).  We use a plain map-style
        ``GenomicTileDataset`` with no shards, so we skip that block entirely
        and just make sure the projection head is on the right device.

        Risks fixed:
          #1  ValueError from expand_shards([]) when data_dirs is empty.
          #2  AttributeError from feat_extractor=None inside sanity check.
        """
        # Move projection head to training device
        self.projection_head = self.projection_head.to(self.device)
        if self.freeze_projection_head:
            self.projection_head.eval()

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------

    def setup(self, stage=None):
        """Create the genomic-tile dataset (replaces MoPaDi's WDS setup)."""
        if self.conf.seed is not None:
            from mopadi.utils.dist_utils import get_world_size
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        xform = transforms.Compose([
            transforms.Resize(
                self._img_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(self._img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.train_data = GenomicTileDataset(
            genomic_h5_dir=self.genomic_h5_dir,
            tiles_zip_dir=self.tiles_zip_dir,
            transform=xform,
            tiles_per_patient=self.tiles_per_patient,
            split=self.split,
        )
        self.val_data = self.train_data

        # Fail fast if dataset is empty
        if len(self.train_data) == 0:
            raise RuntimeError("Train dataset is empty — check genomic and tile paths")

        # We do NOT need a feature extractor — the projection head
        # replaces the image encoder.
        self.feat_extractor = None
        self.model.feat_extractor = None
        self.ema_model.feat_extractor = None

    def train_dataloader(self):
        """Standard DataLoader for a map-style Dataset."""
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.conf.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """
        Same as MoPaDi's training_step, but projects genomic features
        through the projection head first.

        Key differences from the v1 script:
         - Uses ``T_sampler.sample()`` (importance-weighted) instead of
           ``torch.randint``.
         - Does NOT normalise projected features with ``conds_mean/std``
           — MoPaDi's original training doesn't normalise either.
         - Logs all loss components to TensorBoard.
        """
        from torch.amp.autocast_mode import autocast

        with autocast(device_type="cuda", enabled=False):
            imgs = batch["img"].to(self.device)
            genomic_raw = batch["feat"].to(self.device, dtype=torch.float32)

            # ---- projection head ----
            with torch.set_grad_enabled(not self.freeze_projection_head):
                feats = self.projection_head(genomic_raw)   # (B, D)

            # Validate projected features
            if not torch.is_tensor(feats):
                raise TypeError("Projection head did not return a tensor")
            if feats.dim() != 2 or feats.shape[0] != imgs.shape[0]:
                raise ValueError(f"Projected feats shape mismatch: {feats.shape} vs imgs {imgs.shape}")

            # NOTE: we do *not* z-normalise here.  MoPaDi's original
            # training passes raw features to the UNet; the projection
            # head should already output vectors in the correct range
            # (ensured by the distribution-matching pre-training).

            model_kwargs = {"cond": feats}

            if self.conf.train_mode == TrainMode.diffusion:
                t, _ = self.T_sampler.sample(len(imgs), imgs.device)
                losses = self.sampler.training_losses(
                    model=self.model,
                    x_start=imgs,
                    cond=feats,
                    t=t,
                    model_kwargs=model_kwargs,
                )
            else:
                raise NotImplementedError(self.conf.train_mode)

            loss = losses["loss"].mean()

            # Log — reuse MoPaDi's pattern (all_gather for multi-GPU)
            for key in ["loss", "vae", "mmd", "chamfer", "arg_cnt"]:
                if key in losses:
                    gathered = self.all_gather(losses[key])

                    def _mean_obj(o):
                        # Return the mean for tensors; recurse into
                        # containers (dict, list, tuple). For lists/tuples
                        # of tensors, return the scalar mean; otherwise
                        # reconstruct the sequence with means.
                        if torch.is_tensor(o):
                            return o.mean()
                        if isinstance(o, dict):
                            return {k: _mean_obj(v) for k, v in o.items()}
                        if isinstance(o, (list, tuple)):
                            seq = [_mean_obj(v) for v in o]
                            # if all elements are tensors (scalars), stack
                            if all(torch.is_tensor(x) for x in seq):
                                return torch.stack(seq).mean()
                            return type(o)(seq)
                        try:
                            return torch.as_tensor(o).mean()
                        except Exception:
                            return o

                    losses[key] = _mean_obj(gathered)

            if self.global_rank == 0:
                # Safely access TensorBoard SummaryWriter via logger.experiment
                # Some Lightning loggers may not expose `.experiment` or may
                # have a different API — fall back to Lightning's logger
                # methods when necessary.
                exp = getattr(self.logger, "experiment", None)
                step = int(self.num_samples)
                if exp is not None and hasattr(exp, "add_scalar"):
                    try:
                        exp.add_scalar("loss", losses["loss"], step)
                        for key in ["vae", "mmd", "chamfer", "arg_cnt"]:
                            if key in losses:
                                exp.add_scalar(f"loss/{key}", losses[key], step)
                    except Exception:
                        # If the SummaryWriter call fails, fall back.
                        exp = None

                if exp is None:
                    # Build a plain metrics dict and try the Lightning logger
                    metrics = {}
                    # Convert tensors to Python scalars where appropriate
                    def to_scalar(x):
                        try:
                            return float(x)
                        except Exception:
                            return x

                    metrics["loss"] = to_scalar(losses.get("loss"))
                    for key in ["vae", "mmd", "chamfer", "arg_cnt"]:
                        if key in losses:
                            metrics[f"loss/{key}"] = to_scalar(losses[key])

                    # Try logger.log_metrics (Lightning logger API)
                    try:
                        _logger = getattr(self, "logger", None)
                        _log_fn = getattr(_logger, "log_metrics", None)
                        if callable(_log_fn):
                            _log_fn(metrics, step=step)
                        else:
                            # Last resort: use self.log so Lightning still records
                            for k, v in metrics.items():
                                self.log(k, v, prog_bar=False, logger=True)
                    except Exception:
                        # Silently ignore logging failures to avoid crashing training
                        pass

        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """EMA update + sample logging — identical to LitModel, but
        we project the genomic features before passing them as ``cond``
        to ``log_sample``."""
        if self.is_last_accum(batch_idx):
            ema(self.model, self.ema_model, self.conf.ema_decay)

            imgs = batch["img"]
            genomic_raw = batch["feat"].to(self.device, dtype=torch.float32)
            with torch.no_grad():
                conds = self.projection_head(genomic_raw)

            self.log_sample(x_start=imgs, cond=conds)
            # NOTE: evaluate_scores() is intentionally NOT called here.
            # The parent's FID/LPIPS evaluation routes images through the
            # diffusion sampler with raw (unprojected) features as cond,
            # which bypasses the projection head → meaningless metrics.
            # See evaluate_scores() override below.

    def evaluate_scores(self):
        """
        Override to disable FID/LPIPS evaluation.

        Risk #3: The parent's ``evaluate_scores`` calls ``evaluate_fid`` which
        generates images conditioned on raw features (not projected genomic
        vectors), producing completely wrong FID scores.

        During genomic fine-tuning the conditioning vector is only valid
        after passing through the projection head; the evaluation sampler
        inside ``evaluate_fid`` does not know about this.

        Solution: skip FID/LPIPS here.  Qualitative quality can be judged
        from the periodic sample images that ``log_sample`` saves to
        TensorBoard and disk (those do go through the projection head).
        """
        pass  # intentionally empty

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """
        Use the parent's optimizer & scheduler logic, but add the
        projection head parameters when it is not frozen.
        """
        out = super().configure_optimizers()

        if not self.freeze_projection_head:
            optim = out["optimizer"]
            # Risk #4: optim.defaults["lr"] is not guaranteed for all optimisers
            # (e.g. Lion stores it differently).  Use conf.lr instead.
            optim.add_param_group({
                "params": list(self.projection_head.parameters()),
                "lr": self.conf.lr,
            })
            print("[INFO] Projection head parameters added to optimizer")

        return out

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint):
        """Persist projection head state alongside the diffusion model."""
        checkpoint["projection_head_state_dict"] = (
            self.projection_head.state_dict()
        )
        checkpoint["projection_head_config"] = {
            "in_dim": self.projection_head.in_dim,
            "out_dim": self.projection_head.out_dim,
            "arch": self.projection_head.arch,
        }
        checkpoint["freeze_projection_head"] = self.freeze_projection_head

    def on_load_checkpoint(self, checkpoint):
        """Restore projection head from a genomic fine-tuning checkpoint."""
        if "projection_head_state_dict" in checkpoint:
            self.projection_head.load_state_dict(
                checkpoint["projection_head_state_dict"]
            )
            print("[OK] Projection head restored from checkpoint")


# ======================================================================
#  Training entry point  (mirrors ``mopadi.train_diff_autoenc.train``)
# ======================================================================

def train_genomic_finetune(
    conf: TrainConfig,
    projection_head: ProjectionHead,
    diffusion_ckpt: str,
    genomic_h5_dir: str,
    tiles_zip_dir: str,
    out_dir: str,
    gpus: list[int],
    *,
    freeze_projection_head: bool = True,
    tiles_per_patient: int = 10,
    split: str = "train",
    img_size: int = 512,
    epochs: int = 10,
    nodes: int = 1,
):
    """
    High-level training function that mirrors ``mopadi.train_diff_autoenc.train``.

    It creates a ``LitModelGenomicFinetune``, sets up Lightning callbacks
    (TensorBoard logger, ModelCheckpoint, LearningRateMonitor), and calls
    ``trainer.fit``.
    """
    # Point logdir into out_dir
    conf.base_dir = out_dir
    conf.name = "genomic_finetune"

    model = LitModelGenomicFinetune(
        conf=conf,
        projection_head=projection_head,
        freeze_projection_head=freeze_projection_head,
        genomic_h5_dir=genomic_h5_dir,
        tiles_zip_dir=tiles_zip_dir,
        tiles_per_patient=tiles_per_patient,
        split=split,
        img_size=img_size,
    )

    # ------------------------------------------------------------------
    # Load pretrained diffusion weights
    # ------------------------------------------------------------------
    # Ensure checkpoint exists and provide diagnostic info
    if not Path(diffusion_ckpt).exists():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {diffusion_ckpt}")
    print(f"[INFO] Loading diffusion checkpoint: {diffusion_ckpt}")
    ckpt = torch.load(diffusion_ckpt, map_location="cpu")
    try:
        print(f"[INFO] Checkpoint keys: {list(ckpt.keys())}")
    except Exception:
        pass

    if "state_dict" in ckpt:
        # Full LitModel checkpoint — load directly (keeps EMA, buffers, …)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"[OK] Loaded LitModel state_dict from {diffusion_ckpt}")
    else:
        # Bare model state dict
        model.model.load_state_dict(ckpt, strict=False)
        model.ema_model = copy.deepcopy(model.model)
        model.ema_model.requires_grad_(False)
        model.ema_model.eval()
        print(f"[OK] Loaded bare model weights from {diffusion_ckpt}")

    # Retrieve conds_mean / conds_std from checkpoint (informational)
    for key in ("conds_mean", "conds_std"):
        val = ckpt.get(key, None)
        if val is None and "state_dict" in ckpt:
            val = ckpt["state_dict"].get(key, None)
        if val is not None:
            print(f"  {key}: shape={tuple(val.shape)}")

    # ------------------------------------------------------------------
    # Lightning boilerplate (same as mopadi.train_diff_autoenc.train)
    # ------------------------------------------------------------------
    os.makedirs(conf.logdir, exist_ok=True)

    ckpt_callback = ModelCheckpoint(
        dirpath=conf.logdir,
        save_last=True,
        save_top_k=1,
        every_n_train_steps=max(
            1,
            conf.save_every_samples // conf.batch_size_effective,
        ),
    )

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=conf.logdir, name=None, version=""
    )

    # Accelerator / strategy
    if len(gpus) == 1 and nodes == 1:
        accelerator, strategy = "gpu", "auto"
    elif len(gpus) > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=True)
        accelerator = "gpu"
    else:
        accelerator, strategy = "cpu", "auto"

    trainer = pl.Trainer(
        max_epochs=epochs,
        devices=gpus,
        strategy=strategy,
        num_nodes=nodes,
        accelerator=accelerator,
        precision="16-mixed" if conf.fp16 else 32,
        callbacks=[ckpt_callback, LearningRateMonitor()],
        logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
    )

    # Check for an existing checkpoint in out_dir to resume from
    last_ckpt = os.path.join(conf.logdir, "last.ckpt")
    if os.path.exists(last_ckpt):
        print(f"[INFO] Resuming from {last_ckpt}")
        trainer.fit(model, ckpt_path=last_ckpt)
    else:
        trainer.fit(model)

    print("\n" + "=" * 60)
    print("GENOMIC FINE-TUNING COMPLETE")
    print(f"  Output directory : {conf.logdir}")
    print(f"  TensorBoard logs : {conf.logdir}")
    print("=" * 60 + "\n")


# ======================================================================
#  CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a pretrained MoPaDi diffusion model with genomic "
            "conditioning (v2 — subclasses LitModel for full MoPaDi "
            "infrastructure)."
        ),
    )

    # Required paths
    g = parser.add_argument_group("paths")
    g.add_argument("--projection-head-ckpt", type=str, required=True)
    g.add_argument("--diffusion-ckpt", type=str, required=True)
    g.add_argument("--genomic-h5-dir", type=str, required=True)
    g.add_argument("--tiles-zip-dir", type=str, required=True)
    g.add_argument("--out-dir", type=str, required=True)

    # Training
    g = parser.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=10)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--accum-batches", type=int, default=4,
                   help="Gradient accumulation steps")
    g.add_argument("--lr", type=float, default=1e-5)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adam", "adamw", "lion"])
    g.add_argument("--warmup", type=int, default=0,
                   help="Warmup steps (not epochs)")
    g.add_argument("--fp16", action="store_true")
    g.add_argument("--grad-clip", type=float, default=1.0)

    # Data
    g = parser.add_argument_group("data")
    g.add_argument("--tiles-per-patient", type=int, default=10)
    g.add_argument("--split", type=str, default="train",
                   choices=["train", "test", "all"])
    g.add_argument("--img-size", type=int, default=512)
    g.add_argument("--num-workers", type=int, default=4)

    # Model
    g = parser.add_argument_group("model")
    g.add_argument("--freeze-projection-head", action="store_true")
    g.add_argument("--ema-decay", type=float, default=0.9999)

    # Logging / eval intervals (in number of samples seen)
    g = parser.add_argument_group("logging")
    g.add_argument("--reconstruct-every-samples", type=int, default=20_000,
                   help="Nr of samples between periodic image generation to TensorBoard")
    g.add_argument("--eval-every-samples", type=int, default=200_000,
                   help="FID evaluation interval (disabled in genomic mode; kept large)")
    # Risk #5: fine-tuning datasets are much smaller than pretraining ones.
    # Default 5_000 → checkpoint every ~156 optimizer steps at batch_size=32.
    g.add_argument("--save-every-samples", type=int, default=5_000,
                   help="Nr of samples between checkpoint saves (default lowered for fine-tuning)")
    g.add_argument("--sample-size", type=int, default=16,
                   help="Nr of images for periodic sample generation")

    # Hardware
    g = parser.add_argument_group("hardware")
    g.add_argument("--gpus", type=int, nargs="+", default=[0])

    args = parser.parse_args()

    # ---- Build MoPaDi TrainConfig ----
    conf = tcga_brca_autoenc()

    # Override with CLI values
    conf.batch_size = args.batch_size
    conf.lr = args.lr
    conf.weight_decay = args.weight_decay
    conf.fp16 = args.fp16
    conf.grad_clip = args.grad_clip
    conf.accum_batches = args.accum_batches
    conf.warmup = args.warmup
    conf.ema_decay = args.ema_decay
    conf.num_workers = args.num_workers
    conf.img_size = args.img_size
    conf.sample_size = args.sample_size
    conf.reconstruct_every_samples = args.reconstruct_every_samples
    # Risk #5: eval_every_samples controls FID — now disabled via evaluate_scores
    # override, but keep large so accidental triggering is also avoided.
    conf.eval_every_samples = args.eval_every_samples
    conf.eval_ema_every_samples = args.eval_every_samples
    # Risk #5: default save_every_samples=100_000 is calibrated for large-scale
    # pretraining (millions of samples).  Fine-tuning is much shorter, so we
    # use a smaller default (5_000) to ensure at least one checkpoint per epoch.
    conf.save_every_samples = args.save_every_samples

    # Map optimizer string → enum
    from mopadi.configs.choices import OptimizerType
    _opt_map = {
        "adam": OptimizerType.adam,
        "adamw": OptimizerType.adamw,
        "lion": OptimizerType.lion,
    }
    conf.optimizer = _opt_map[args.optimizer]

    # We do NOT use MoPaDi's WDS data pipeline — the genomic dataset
    # is created inside setup().  Set dummy values so conf doesn't
    # complain. Use setattr to avoid static type-checker errors if
    # TrainConfig doesn't declare these attributes.
    setattr(conf, "data_dirs", [])
    setattr(conf, "feature_dirs", [])
    setattr(conf, "feat_extractor", None)

    # ---- Load projection head ----
    device = f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu"
    proj_head = load_projection_head(args.projection_head_ckpt, device="cpu")

    # ---- Print banner ----
    print("\n" + "=" * 60)
    print("DIFFUSION FINE-TUNING WITH GENOMIC CONDITIONING (v2)")
    print("=" * 60)
    print(f"  PyTorch        : {torch.__version__}")
    # Resolve PyTorch Lightning version robustly to avoid static-analysis
    # warnings from Pylance when importing __version__ directly.
    pl_version = "unknown"
    try:
        # Prefer importlib.metadata (Py3.8+). Try both PyPI package names.
        _pkg_version = None
        try:
            from importlib.metadata import version as _pkg_version
        except ImportError:
            pass

        if _pkg_version is not None:
            for _name in ("pytorch-lightning", "pytorch_lightning"):
                try:
                    pl_version = _pkg_version(_name)
                    break
                except Exception:
                    continue
        
        if pl_version == "unknown":
            import pytorch_lightning as _pl
            pl_version = getattr(_pl, "__version__", "unknown")
    except Exception:
        pl_version = "unknown"

    print(f"  Lightning      : {pl_version}")
    print(f"  GPUs           : {args.gpus}")
    print(f"  Batch size     : {args.batch_size} × {args.accum_batches} accum "
          f"= {args.batch_size * args.accum_batches} effective")
    print(f"  Learning rate  : {args.lr}")
    print(f"  Optimizer      : {args.optimizer}")
    print(f"  Epochs         : {args.epochs}")
    print(f"  Freeze proj    : {args.freeze_projection_head}")
    print(f"  Image size     : {args.img_size}")
    print(f"  EMA decay      : {args.ema_decay}")
    print("=" * 60 + "\n")

    # ---- Train ----
    train_genomic_finetune(
        conf=conf,
        projection_head=proj_head,
        diffusion_ckpt=args.diffusion_ckpt,
        genomic_h5_dir=args.genomic_h5_dir,
        tiles_zip_dir=args.tiles_zip_dir,
        out_dir=args.out_dir,
        gpus=args.gpus,
        freeze_projection_head=args.freeze_projection_head,
        tiles_per_patient=args.tiles_per_patient,
        split=args.split,
        img_size=args.img_size,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
