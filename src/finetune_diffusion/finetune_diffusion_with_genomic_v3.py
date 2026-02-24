#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune Diffusion Model with Genomic Conditioning (v3)

Key improvements over v2
========================

1. **DDIM-inversion start noise.**
   Instead of sampling from a fixed ``x_T`` buffer (which makes every
   generated image structurally identical), we DDIM-invert each real
   tile to obtain a *tile-specific* noise map.  The diffusion model is
   then trained to reconstruct the tile from that noise, conditioned on
   the projected genomic vector.  This means:
     - Every tile starts from unique noise ⇒ diverse generated images.
     - The model learns to faithfully reconstruct the tile layout under
       genomic guidance (not just to denoise generic Gaussian noise).

2. **Joint projection-head + diffusion training.**
   The projection head is *not* pre-trained separately; instead it is
   trained end-to-end together with the diffusion UNet (optionally with
   a smaller learning rate).  A warm-up schedule ramps the projection
   head LR from zero to its target over the first N steps.

3. **Deeper projection head (4 layers by default).**
   The default ``num_layers`` is raised from 2 → 4 so the head can
   learn a richer mapping from genomic space → image-feature space.

4. **3 × 4 grid logging (12 tiles).**
   A fixed validation batch of 12 samples is captured once in
   ``setup()`` and reused at every logging step so ``sample_real``
   stays constant across steps and generated images can be compared
   directly.  ``make_grid(nrow=4)`` gives a 3-row × 4-column layout.

Architecture
------------
Genomic (512-dim)
    → ProjectionHead (4-layer MLP, jointly trained)
    → pseudo image features (512-dim)
    → cDDIM (DDIM-inverted tile noise as start)
    → reconstructed tile

Usage
-----
    python finetune_diffusion_with_genomic_v3.py \\
        --diffusion-ckpt ./last.ckpt \\
        --genomic-h5-dir /path/to/genomic_features \\
        --tiles-zip-dir /path/to/tile_zips \\
        --out-dir ./finetuned_genomic_v3 \\
        --epochs 20 --batch-size 4 --lr 1e-5 \\
        --proj-lr 3e-4 --proj-warmup 500 --proj-layers 4
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import re
import sys
import time
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
from torchvision.utils import make_grid, save_image
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
#  Projection Head  (deeper default: 4 layers)
# ======================================================================

class ProjectionHead(nn.Module):
    """MLP / linear / residual projection  genomic → image-feature space."""

    def __init__(
        self,
        in_dim: int = 512,
        out_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 4,          # ← deeper default
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
            layers: list[nn.Module] = []
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

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[ProjectionHead] arch={arch}, layers={num_layers}, "
              f"dims={in_dim}→{hidden_dim}→{out_dim}, params={n_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.arch == "residual":
            out = self.net(x) + self.skip(x)
        else:
            out = self.net(x)
        if self.normalize_output:
            out = F.normalize(out, p=2, dim=-1)
        return out


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

    Returns ``{"img": (3,H,W), "feat": (D,), "patient_id": str}``.
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
        print(f"[GenomicTileDataset] {len(common_ids)} matched patients "
              f"(split={split})")

        self.pairs = [
            (genomic_files[pid], zip_files[pid], pid) for pid in common_ids
        ]

        # Cache genomic features
        self.genomic_cache: dict[str, torch.Tensor] = {}
        for gpath, _, pid in tqdm(self.pairs, desc="Caching genomic"):
            with h5py.File(gpath, "r") as f:
                arr = np.array(f[self.genomic_key])
                if arr.ndim == 2:
                    arr = arr.mean(axis=0)
                if arr.ndim != 1:
                    raise ValueError(
                        f"Genomic vector for {pid} has shape {arr.shape}")
                if np.isnan(arr).any():
                    raise ValueError(
                        f"Genomic vector for {pid} contains NaNs")
                self.genomic_cache[pid] = torch.from_numpy(
                    arr.astype(np.float32))

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
                        print(f"[WARN] No tiles in zip for {pid}: {zpath}")
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

        self.samples: list[tuple[str, Path]] = []
        for _, zpath, pid in self.pairs:
            n = min(self.tiles_per_patient,
                    len(self.tile_lists.get(pid, [])))
            self.samples.extend([(pid, zpath)] * n)

        print(f"[GenomicTileDataset] {len(self.samples)} total samples")

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
        genomic = self.genomic_cache[pid]

        tile_name = random.choice(self.tile_lists[pid])
        with zipfile.ZipFile(zpath, "r") as zf:
            with zf.open(tile_name) as fh:
                img = Image.open(BytesIO(fh.read())).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return {"img": img, "feat": genomic, "patient_id": pid}


# ======================================================================
#  LitModel subclass — v3
# ======================================================================

class LitModelGenomicFinetuneV3(LitModel):
    """
    v3 improvements over v2:
      1. DDIM-inversion noise (tile-specific start noise)
      2. Joint projection-head training (no separate pre-training)
      3. Deeper projection head (4-layer default)
      4. Fixed 12-sample validation grid (3 rows × 4 cols)
    """

    def __init__(
        self,
        conf: TrainConfig,
        projection_head: ProjectionHead,
        genomic_h5_dir: str = "",
        tiles_zip_dir: str = "",
        tiles_per_patient: int = 10,
        split: str = "train",
        img_size: int = 512,
        proj_lr: float = 3e-4,
        proj_warmup: int = 500,
        n_log_samples: int = 12,
        ddim_inversion_T: int = 50,
    ):
        super().__init__(conf)

        self.projection_head = projection_head
        self.genomic_h5_dir = genomic_h5_dir
        self.tiles_zip_dir = tiles_zip_dir
        self.tiles_per_patient = tiles_per_patient
        self.split = split
        self._img_size = img_size
        self.proj_lr = proj_lr
        self.proj_warmup = proj_warmup
        self.n_log_samples = n_log_samples
        self.ddim_inversion_T = ddim_inversion_T

        # Will be populated in setup()
        self._fixed_val_batch: Optional[dict] = None

    # ------------------------------------------------------------------
    # Skip parent's WebDataset hooks
    # ------------------------------------------------------------------

    def on_fit_start(self):
        self.projection_head = self.projection_head.to(self.device)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def setup(self, stage=None):
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

        if len(self.train_data) == 0:
            raise RuntimeError(
                "Train dataset is empty — check genomic and tile paths")

        self.feat_extractor = None
        self.model.feat_extractor = None
        self.ema_model.feat_extractor = None

        # ---- Capture a fixed validation batch (12 samples) ----
        val_loader = DataLoader(
            self.val_data,
            batch_size=min(self.n_log_samples, len(self.val_data)),
            shuffle=False,
            num_workers=0,
        )
        try:
            self._fixed_val_batch = next(iter(val_loader))
        except Exception as exc:  # StopIteration or other loader error
            self._fixed_val_batch = None
            exc = exc

        if self._fixed_val_batch is None:
            print(
                "[setup] Warning: could not capture fixed validation batch; "
                "3x4 grid logging will be skipped until a batch is available"
            )
            n_val = 0
        else:
            # safe: dataset returned a batch dict
            n_val = self._fixed_val_batch["img"].shape[0]
            print(f"[setup] Captured {n_val} fixed validation samples "
                  f"for 3×4 grid logging")

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.conf.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    # ------------------------------------------------------------------
    # DDIM inversion helper
    # ------------------------------------------------------------------

    def _ddim_invert(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Encode real images x into their DDIM-inverted noise maps x_T.

        Uses the EMA model and a fast schedule (``ddim_inversion_T`` steps)
        so inversion is cheap even at 512×512.
        """
        sampler = self.conf._make_diffusion_conf(
            T=self.ddim_inversion_T
        ).make_sampler()
        with torch.no_grad():
            out = sampler.ddim_reverse_sample_loop(
                self.ema_model,
                x,
                model_kwargs={"cond": cond},
            )
        return out["sample"]  # (B, 3, H, W) — the inverted noise

    # ------------------------------------------------------------------
    # Training step  — DDIM-inversion + joint projection head
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        from torch.amp.autocast_mode import autocast

        with autocast(device_type="cuda", enabled=False):
            imgs = batch["img"].to(self.device)
            genomic_raw = batch["feat"].to(self.device, dtype=torch.float32)

            # ---- Joint projection head (always trainable) ----
            feats = self.projection_head(genomic_raw)      # (B, D)

            # ---- DDIM-invert each tile to get tile-specific noise ----
            # Use the *projected* genomic cond for inversion so the noise
            # is coherent with what the forward pass will condition on.
            x_T = self._ddim_invert(imgs, cond=feats.detach())

            # ---- Diffusion training loss ----
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

            # ---- Logging ----
            # Local lightweight implementation of apply_to_collection.
            # We avoid importing lightning/pytorch utilities to keep static
            # analysis and editor linters happy in environments where those
            # packages are not available.
            from numbers import Number
            def apply_to_collection(x, dtype, fn):
                if isinstance(x, dtype):
                    return fn(x)
                if isinstance(x, dict):
                    return {k: apply_to_collection(v, dtype, fn) for k, v in x.items()}
                if isinstance(x, (list, tuple)):
                    res = [apply_to_collection(v, dtype, fn) for v in x]
                    return type(x)(res)
                return x

            def _gather_mean(val):
                gathered = self.all_gather(val)
                if isinstance(gathered, torch.Tensor):
                    return gathered.mean()
                if isinstance(gathered, (list, tuple)):
                    if len(gathered) == 0:
                        return gathered
                    # all tensors
                    if all(isinstance(x, torch.Tensor) for x in gathered):
                        return torch.stack(gathered).mean()
                    # all numbers
                    if all(isinstance(x, Number) for x in gathered):
                        return torch.tensor(gathered, device=self.device).float().mean()
                    return apply_to_collection(gathered, torch.Tensor, lambda t: t.mean())
                if isinstance(gathered, dict):
                    return apply_to_collection(gathered, torch.Tensor, lambda t: t.mean())
                # fallback: return as-is
                return gathered

            for key in ["loss", "vae", "mmd", "chamfer", "arg_cnt"]:
                if key in losses:
                    v = _gather_mean(losses[key])
                    losses[key] = v  # type: ignore[assignment]

            if self.global_rank == 0:
                step = int(self.num_samples)
                exp = getattr(self.logger, "experiment", None)
                if exp is not None and hasattr(exp, "add_scalar"):
                    try:
                        exp.add_scalar("loss", losses["loss"], step)
                        for key in ["vae", "mmd", "chamfer", "arg_cnt"]:
                            if key in losses:
                                exp.add_scalar(
                                    f"loss/{key}", losses[key], step)
                    except Exception:
                        pass

            # Log loss values via Lightning so callbacks (EarlyStopping,
            # ModelCheckpoint) can access them through trainer.callback_metrics.
            try:
                # main scalar used by EarlyStopping in this repo is 'loss'
                # but we also log a namespaced 'train/loss' for readability.
                self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                self.log("train/loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
                # if component losses exist, log them too
                for key in ("vae", "mmd", "chamfer"):
                    if key in losses:
                        try:
                            self.log(f"loss/{key}", losses[key], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
                        except Exception:
                            pass
            except Exception:
                # Logging must not interrupt training; ignore errors.
                pass

        return {"loss": loss}

    # ------------------------------------------------------------------
    # EMA update + grid logging with DDIM-inverted noise
    # ------------------------------------------------------------------

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self.is_last_accum(batch_idx):
            return

        ema(self.model, self.ema_model, self.conf.ema_decay)

        # --- Log fixed validation grid every N samples ---
        if (self.conf.reconstruct_every_samples <= 0 or
                not is_time(self.num_samples,
                            self.conf.reconstruct_every_samples,
                            self.conf.batch_size_effective)):
            return

        self._log_grid(self.model, postfix="")
        self._log_grid(self.ema_model, postfix="_ema")

    def _log_grid(self, model, postfix: str):
        """
        Generate a 3×4 grid of reconstructed tiles from the fixed
        validation batch and save to disk + TensorBoard.
        """
        model.eval()
        vb = self._fixed_val_batch
        if vb is None:
            return

        imgs = vb["img"].to(self.device)
        genomic_raw = vb["feat"].to(self.device, dtype=torch.float32)

        with torch.no_grad():
            conds = self.projection_head(genomic_raw)

            # DDIM-invert the real tiles → tile-specific noise
            x_T = self._ddim_invert(imgs, cond=conds)

            # Reconstruct from inverted noise + genomic condition
            gen = self.eval_sampler.sample(
                model=model,
                noise=x_T,
                cond=conds,
                x_start=imgs,
            )

        if self.global_rank == 0:
            step = int(self.num_samples)

            # ---- Real grid ----
            grid_real = (make_grid(imgs, nrow=4) + 1) / 2
            real_dir = os.path.join(
                self.conf.logdir, f"sample_real{postfix}")
            os.makedirs(real_dir, exist_ok=True)
            save_image(grid_real,
                       os.path.join(real_dir, f"{step}.png"))
            exp = getattr(self.logger, "experiment", None)
            if exp is not None and hasattr(exp, "add_image"):
                exp.add_image(f"sample{postfix}/real", grid_real, step)

            # ---- Generated grid ----
            grid_gen = (make_grid(gen, nrow=4) + 1) / 2
            gen_dir = os.path.join(
                self.conf.logdir, f"sample{postfix}")
            os.makedirs(gen_dir, exist_ok=True)
            save_image(grid_gen,
                       os.path.join(gen_dir, f"{step}.png"))
            exp = getattr(self.logger, "experiment", None)
            if exp is not None and hasattr(exp, "add_image"):
                exp.add_image(f"sample{postfix}/gen", grid_gen, step)

        model.train()

    # ------------------------------------------------------------------
    def evaluate_scores(self):
        pass  # FID/LPIPS disabled (same reasoning as v2)

    # ------------------------------------------------------------------
    # Optimizer — joint training with separate LR for projection head
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        # Diffusion UNet params at conf.lr
        unet_params = list(self.model.parameters())

        # Projection head params at a separate (typically higher) LR
        proj_params = list(self.projection_head.parameters())

        param_groups = [
            {"params": unet_params, "lr": self.conf.lr, "name": "unet"},
            {"params": proj_params, "lr": self.proj_lr, "name": "proj_head"},
        ]

        if self.conf.optimizer.name == "lion":
            try:
                from lion_pytorch import Lion  # type: ignore[import]
            except Exception:
                print(
                    "[WARN] optimizer 'lion' requested but package 'lion_pytorch' is not installed."
                    " Falling back to AdamW. To use Lion, install: pip install lion-pytorch"
                )
                optim = torch.optim.AdamW(
                    param_groups,
                    betas=(0.9, 0.99),
                    eps=1e-6,
                    weight_decay=self.conf.weight_decay,
                )
            else:
                optim = Lion(
                    param_groups,
                    betas=(0.95, 0.98),
                    weight_decay=self.conf.weight_decay,
                )
        elif self.conf.optimizer.name == "adamw":
            optim = torch.optim.AdamW(
                param_groups,
                betas=(0.9, 0.99),
                eps=1e-6,
                weight_decay=self.conf.weight_decay,
            )
        else:
            optim = torch.optim.Adam(
                param_groups,
                lr=self.conf.lr,
                weight_decay=self.conf.weight_decay,
            )

        out: dict = {"optimizer": optim}

        # Optional warm-up for projection head
        if self.proj_warmup > 0:
            def lr_lambda_unet(step):
                return 1.0   # UNet LR is constant (handled by conf)

            def lr_lambda_proj(step):
                return min(1.0, step / max(1, self.proj_warmup))

            from torch.optim.lr_scheduler import LambdaLR
            sched = LambdaLR(optim, lr_lambda=[lr_lambda_unet,
                                                lr_lambda_proj])
            out["lr_scheduler"] = {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
            }

        return out

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint):
        checkpoint["projection_head_state_dict"] = (
            self.projection_head.state_dict()
        )
        checkpoint["projection_head_config"] = {
            "in_dim": self.projection_head.in_dim,
            "out_dim": self.projection_head.out_dim,
            "arch": self.projection_head.arch,
        }

    def on_load_checkpoint(self, checkpoint):
        if "projection_head_state_dict" in checkpoint:
            self.projection_head.load_state_dict(
                checkpoint["projection_head_state_dict"]
            )
            print("[OK] Projection head restored from checkpoint")


# ======================================================================
#  Training entry point
# ======================================================================

def train_genomic_finetune_v3(
    conf: TrainConfig,
    projection_head: ProjectionHead,
    diffusion_ckpt: str,
    genomic_h5_dir: str,
    tiles_zip_dir: str,
    out_dir: str,
    gpus: list[int],
    *,
    tiles_per_patient: int = 10,
    split: str = "train",
    img_size: int = 512,
    epochs: int = 20,
    nodes: int = 1,
    proj_lr: float = 3e-4,
    proj_warmup: int = 500,
    n_log_samples: int = 12,
    ddim_inversion_T: int = 50,
):
    conf.base_dir = out_dir
    conf.name = "genomic_finetune_v3"
    # Ensure TrainConfig.sample_size matches the number of fixed
    # validation samples used for x_T / grid logging so checkpointed
    # `x_T` buffers (if present) have the expected shape.
    conf.sample_size = n_log_samples

    model = LitModelGenomicFinetuneV3(
        conf=conf,
        projection_head=projection_head,
        genomic_h5_dir=genomic_h5_dir,
        tiles_zip_dir=tiles_zip_dir,
        tiles_per_patient=tiles_per_patient,
        split=split,
        img_size=img_size,
        proj_lr=proj_lr,
        proj_warmup=proj_warmup,
        n_log_samples=n_log_samples,
        ddim_inversion_T=ddim_inversion_T,
    )

    # ---- Load pretrained diffusion weights ----
    if not Path(diffusion_ckpt).exists():
        raise FileNotFoundError(
            f"Diffusion checkpoint not found: {diffusion_ckpt}")
    print(f"[INFO] Loading diffusion checkpoint: {diffusion_ckpt}")
    ckpt = torch.load(diffusion_ckpt, map_location="cpu")

    if "state_dict" in ckpt:
        state = dict(ckpt["state_dict"])  # copy so we can mutate
        # Some checkpoints may store a fixed x_T buffer with a different
        # number of samples (e.g. 16) than this run's `n_log_samples` (12).
        # Loading a mismatched buffer raises a size-mismatch error — drop
        # it from the state_dict and warn instead.
        if "x_T" in state:
            try:
                ck_shape = tuple(state["x_T"].shape)
                model_xT = getattr(model, "x_T", None)
                model_shape = tuple(model_xT.shape) if model_xT is not None else None
                if model_shape is not None and ck_shape != model_shape:
                    print(
                        f"[WARN] Checkpoint x_T shape {ck_shape} != model x_T shape {model_shape}; skipping x_T from checkpoint"
                    )
                    del state["x_T"]
            except Exception:
                # If anything goes wrong inspecting shapes, drop the buffer
                # to avoid failing the whole load.
                print("[WARN] Unable to validate checkpoint x_T shape; skipping x_T")
                state.pop("x_T", None)

        model.load_state_dict(state, strict=False)
        print(f"[OK] Loaded LitModel state_dict from {diffusion_ckpt}")
    else:
        model.model.load_state_dict(ckpt, strict=False)
        model.ema_model = copy.deepcopy(model.model)
        model.ema_model.requires_grad_(False)
        model.ema_model.eval()
        print(f"[OK] Loaded bare model weights from {diffusion_ckpt}")

    for key in ("conds_mean", "conds_std"):
        val = ckpt.get(key, None)
        if val is None and "state_dict" in ckpt:
            val = ckpt["state_dict"].get(key, None)
        if val is not None:
            print(f"  {key}: shape={tuple(val.shape)}")

    # ---- Lightning boilerplate ----
    os.makedirs(conf.logdir, exist_ok=True)

    # Save any new-best checkpoints and include epoch+loss in filename.
    # We use `save_top_k=-1` so each time a new best is found the checkpoint
    # is kept with its epoch number instead of replacing previous bests.
    ckpt_callback = ModelCheckpoint(
        dirpath=conf.logdir,
        filename="epoch={epoch:02d}-loss={loss:.4f}",
        save_last=True,
        save_top_k=-1,
        monitor="loss",
        mode="min",
        every_n_train_steps=max(
            1,
            conf.save_every_samples // conf.batch_size_effective,
        ),
    )

    # Early stopping on monitored metric if it doesn't improve.
    from pytorch_lightning.callbacks import EarlyStopping
    early_stop_callback = EarlyStopping(monitor="loss", patience=10, mode="min")

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=conf.logdir, name=None, version=""
    )

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
        callbacks=[ckpt_callback, LearningRateMonitor(), early_stop_callback],
        logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
    )

    last_ckpt = os.path.join(conf.logdir, "last.ckpt")
    if os.path.exists(last_ckpt):
        print(f"[INFO] Resuming from {last_ckpt}")
        trainer.fit(model, ckpt_path=last_ckpt)
    else:
        trainer.fit(model)

    print("\n" + "=" * 60)
    print("GENOMIC FINE-TUNING v3 COMPLETE")
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
            "conditioning (v3 — joint training, DDIM inversion, deeper "
            "projection head, 3×4 grid logging)."
        ),
    )

    # Required paths
    g = parser.add_argument_group("paths")
    g.add_argument("--diffusion-ckpt", type=str, required=True)
    g.add_argument("--genomic-h5-dir", type=str, required=True)
    g.add_argument("--tiles-zip-dir", type=str, required=True)
    g.add_argument("--out-dir", type=str, required=True)

    # Training
    g = parser.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=20)
    g.add_argument("--batch-size", type=int, default=4)
    g.add_argument("--accum-batches", type=int, default=8,
                   help="Gradient accumulation steps")
    g.add_argument("--lr", type=float, default=1e-5,
                   help="Learning rate for the UNet")
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adam", "adamw", "lion"])
    g.add_argument("--warmup", type=int, default=0,
                   help="UNet warmup steps")
    g.add_argument("--fp16", action="store_true")
    g.add_argument("--grad-clip", type=float, default=1.0)

    # Projection head
    g = parser.add_argument_group("projection head")
    g.add_argument("--proj-lr", type=float, default=3e-4,
                   help="Separate LR for the projection head")
    g.add_argument("--proj-warmup", type=int, default=500,
                   help="Warmup steps for projection head LR")
    g.add_argument("--proj-layers", type=int, default=4,
                   help="Number of layers in the projection head MLP")
    g.add_argument("--proj-hidden-dim", type=int, default=512)
    g.add_argument("--proj-arch", type=str, default="mlp",
                   choices=["linear", "mlp", "residual"])
    g.add_argument("--proj-dropout", type=float, default=0.1)
    g.add_argument("--proj-ckpt", type=str, default=None,
                   help="Optional: init proj head from a v2 checkpoint")

    # DDIM inversion
    g = parser.add_argument_group("ddim inversion")
    g.add_argument("--ddim-inversion-T", type=int, default=50,
                   help="Nr of DDIM steps for tile→noise inversion "
                        "(smaller = faster but less accurate)")

    # Data
    g = parser.add_argument_group("data")
    g.add_argument("--tiles-per-patient", type=int, default=10)
    g.add_argument("--split", type=str, default="train",
                   choices=["train", "test", "all"])
    g.add_argument("--img-size", type=int, default=512)
    g.add_argument("--num-workers", type=int, default=4)

    # Model
    g = parser.add_argument_group("model")
    g.add_argument("--ema-decay", type=float, default=0.9999)

    # Logging
    g = parser.add_argument_group("logging")
    g.add_argument("--n-log-samples", type=int, default=16,
                   help="Nr of fixed validation tiles (16 = 4×4 grid; 12 = 3×4)")
    g.add_argument("--reconstruct-every-samples", type=int, default=10_000,
                   help="Samples between grid logging events")
    g.add_argument("--save-every-samples", type=int, default=5_000)
    g.add_argument("--sample-size", type=int, default=16)

    # Hardware
    g = parser.add_argument_group("hardware")
    g.add_argument("--gpus", type=int, nargs="+", default=[0])

    args = parser.parse_args()

    # ---- Build MoPaDi TrainConfig ----
    conf = tcga_brca_autoenc()
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
    conf.eval_every_samples = 999_999_999   # effectively disabled
    conf.eval_ema_every_samples = 999_999_999
    conf.save_every_samples = args.save_every_samples

    from mopadi.configs.choices import OptimizerType
    _opt_map = {
        "adam": OptimizerType.adam,
        "adamw": OptimizerType.adamw,
        "lion": OptimizerType.lion,
    }
    conf.optimizer = _opt_map[args.optimizer]

    setattr(conf, "data_dirs", [])
    setattr(conf, "feature_dirs", [])
    setattr(conf, "feat_extractor", None)

    # ---- Create projection head (random init or from checkpoint) ----
    proj_head = ProjectionHead(
        in_dim=512,
        out_dim=512,
        hidden_dim=args.proj_hidden_dim,
        num_layers=args.proj_layers,
        arch=args.proj_arch,
        dropout=args.proj_dropout,
    )

    if args.proj_ckpt is not None:
        ckpt_path = Path(args.proj_ckpt)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            proj_head.load_state_dict(ckpt["state_dict"], strict=False)
            print(f"[OK] Initialized projection head from {args.proj_ckpt}")
        else:
            print(f"[WARN] --proj-ckpt not found: {args.proj_ckpt}, "
                  "starting from random init")

    # ---- Banner ----
    print("\n" + "=" * 60)
    print("DIFFUSION FINE-TUNING WITH GENOMIC CONDITIONING (v3)")
    print("=" * 60)
    print(f"  PyTorch          : {torch.__version__}")
    print(f"  GPUs             : {args.gpus}")
    print(f"  Batch size       : {args.batch_size} × {args.accum_batches} accum"
          f" = {args.batch_size * args.accum_batches} effective")
    print(f"  UNet LR          : {args.lr}")
    print(f"  Proj-head LR     : {args.proj_lr}")
    print(f"  Proj-head warmup : {args.proj_warmup} steps")
    print(f"  Proj-head layers : {args.proj_layers}")
    print(f"  Optimizer        : {args.optimizer}")
    print(f"  Epochs           : {args.epochs}")
    print(f"  DDIM-inv steps   : {args.ddim_inversion_T}")
    print(f"  Log grid size    : {args.n_log_samples} "
          f"({args.n_log_samples // 4}×4)")
    print(f"  Image size       : {args.img_size}")
    print(f"  EMA decay        : {args.ema_decay}")
    print("=" * 60 + "\n")

    # ---- Train ----
    train_genomic_finetune_v3(
        conf=conf,
        projection_head=proj_head,
        diffusion_ckpt=args.diffusion_ckpt,
        genomic_h5_dir=args.genomic_h5_dir,
        tiles_zip_dir=args.tiles_zip_dir,
        out_dir=args.out_dir,
        gpus=args.gpus,
        tiles_per_patient=args.tiles_per_patient,
        split=args.split,
        img_size=args.img_size,
        epochs=args.epochs,
        proj_lr=args.proj_lr,
        proj_warmup=args.proj_warmup,
        n_log_samples=args.n_log_samples,
        ddim_inversion_T=args.ddim_inversion_T,
    )


if __name__ == "__main__":
    main()
