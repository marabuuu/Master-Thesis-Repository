#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune Diffusion Model with Genomic Conditioning 
=========================================================

Joint single-stage training
----------------------------

    Genomic (512-d) → ProjectionHead (trainable) → cond (512-d)
    Tile (x_start) + Gaussian noise + cond → UNet → ε̂

The projection head and the UNet are trained jointly via the
standard diffusion loss.  By pairing each tile with its patient's
genomic vector, the gradient flows back through the UNet and the
projection head — teaching the head which genomic dimensions matter
for reconstructing histology tiles. The tiles are shown alongside 
the genomic vectors during every training step.

Usage
-----
    python finetune_diffusion_with_genomic.py \
        --diffusion-ckpt ./last.ckpt \
        --genomic-h5-dir /path/to/genomic_features \
        --tiles-zip-dir /path/to/tile_zips \
        --out-dir ./finetuned_genomic_v4 \
        --epochs 20 --batch-size 4 --lr 1e-5 \
        --proj-lr 3e-4 --proj-warmup 500 --proj-layers 4
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

# optional YAML config
import yaml

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
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

# MoPaDi imports
from mopadi.train_diff_autoenc import LitModel, ema, is_time
from mopadi.configs.config import TrainConfig
from mopadi.configs.templates import tcga_brca_autoenc
from mopadi.configs.choices import TrainMode


# ======================================================================
#  Projection Head
# ======================================================================

class ProjectionHead(nn.Module):
    """MLP projection: genomic space (in_dim) → UNet cond space (out_dim).

    Trained jointly with the diffusion UNet — the diffusion loss gradient
    flows through this head, teaching it which genomic dimensions are
    relevant for reconstructing histo-pathology tiles.
    """

    def __init__(
        self,
        in_dim: int = 512,
        out_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        layers: list[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:                # skip after last linear
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[ProjectionHead] layers={num_layers}, "
              f"dims={in_dim}→{hidden_dim}→{out_dim}, params={n_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================================
#  Patient-ID helper
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


# ======================================================================
#  Patient discovery (lightweight — no data loaded)
# ======================================================================

def discover_common_patients(
    genomic_h5_dir: str,
    tiles_zip_dir: str,
    split: str = "train",
) -> list[str]:
    """Return sorted patient IDs that have both genomic H5 and tile zip."""
    g_dir = Path(genomic_h5_dir).expanduser()
    t_dir = Path(tiles_zip_dir).expanduser()

    # Genomic patient IDs
    train_dir = g_dir / "train"
    test_dir = g_dir / "test"
    g_ids: set[str] = set()
    if train_dir.is_dir() or test_dir.is_dir():
        if split in ("train", "all") and train_dir.is_dir():
            g_ids.update(
                canonical_patient_id(f.name) for f in train_dir.glob("*.h5"))
        if split in ("test", "all") and test_dir.is_dir():
            g_ids.update(
                canonical_patient_id(f.name) for f in test_dir.glob("*.h5"))
    else:
        g_ids.update(
            canonical_patient_id(f.name) for f in g_dir.glob("*.h5"))

    # Tile patient IDs
    z_ids = {canonical_patient_id(f.name) for f in t_dir.glob("*.zip")}

    common = sorted(g_ids & z_ids)
    print(f"[discover_common_patients] {len(g_ids)} genomic, "
          f"{len(z_ids)} tile zips, {len(common)} in common (split={split})")
    return common


# ======================================================================
#  Dataset
# ======================================================================

class GenomicTileDataset(Dataset):
    """
    Pairs genomic H5 feature vectors with tile images from zip archives.

    Each item is {"img": (3,H,W), "feat": (D,), "patient_id": str}.
    The tile and genomic vector come from the same patient so the
    diffusion model sees them together during training.
    """

    def __init__(
        self,
        genomic_h5_dir: str,
        tiles_zip_dir: str,
        genomic_key: str = "feats",
        transform=None,
        tiles_per_patient: int = 1,
        split: str = "train",
        patient_ids: Optional[list[str]] = None,
    ):
        super().__init__()
        self.genomic_h5_dir = Path(genomic_h5_dir).expanduser()
        self.tiles_zip_dir = Path(tiles_zip_dir).expanduser()
        self.genomic_key = genomic_key
        self.transform = transform
        self.tiles_per_patient = tiles_per_patient
        
        # Corruption tracking
        self.bad_tiles: dict[str, list[str]] = {}  # patient_id -> list of bad tile names
        self.corruption_retries: int = 5  # max retries per sample

        genomic_files = self._find_genomic_files(split)
        zip_files = {
            canonical_patient_id(f.name): f
            for f in self.tiles_zip_dir.glob("*.zip")
        }
        common_ids = sorted(set(genomic_files) & set(zip_files))
        if patient_ids is not None:
            allowed = set(patient_ids)
            common_ids = [pid for pid in common_ids if pid in allowed]
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

        # Retry loop for handling corrupted tiles
        available_tiles = [
            t for t in self.tile_lists[pid]
            if t not in self.bad_tiles.get(pid, [])
        ]
        
        if not available_tiles:
            raise RuntimeError(
                f"[GenomicTileDataset] All tiles corrupted for patient {pid}")

        for attempt in range(self.corruption_retries):
            tile_name = random.choice(available_tiles)
            try:
                with zipfile.ZipFile(zpath, "r") as zf:
                    with zf.open(tile_name) as fh:
                        img = Image.open(BytesIO(fh.read())).convert("RGB")

                if self.transform is not None:
                    img = self.transform(img)
                else:
                    img = transforms.ToTensor()(img)

                return {"img": img, "feat": genomic, "patient_id": pid}

            except (zipfile.BadZipFile, EOFError, Exception) as e:
                # Log the bad tile
                if pid not in self.bad_tiles:
                    self.bad_tiles[pid] = []
                if tile_name not in self.bad_tiles[pid]:
                    self.bad_tiles[pid].append(tile_name)
                    print(f"[WARN] Corrupted tile skipped ({attempt+1}/{self.corruption_retries}): "
                          f"{pid} / {tile_name} ({type(e).__name__})")
                
                # Remove from available list and retry
                available_tiles = [t for t in available_tiles if t != tile_name]
                
                if not available_tiles:
                    # No more tiles to try for this patient
                    raise RuntimeError(
                        f"[GenomicTileDataset] Patient {pid}: all {len(self.bad_tiles[pid])} "
                        f"tiles corrupted after {attempt+1} attempts")
        
        # Should not reach here, but just in case
        raise RuntimeError(
            f"[GenomicTileDataset] Failed to load tile for patient {pid} "
            f"after {self.corruption_retries} retries")

    def log_corruption_summary(self):
        """Log statistics about corrupted tiles encountered during training."""
        if not self.bad_tiles:
            print("\n[GenomicTileDataset] No corrupted tiles detected ✓")
            return
        
        total_bad = sum(len(v) for v in self.bad_tiles.values())
        affected_patients = len(self.bad_tiles)
        
        print(f"\n{'='*70}")
        print(f"[GenomicTileDataset] CORRUPTION SUMMARY")
        print(f"{'='*70}")
        print(f"  Total corrupted tiles : {total_bad}")
        print(f"  Affected patients    : {affected_patients}")
        print(f"  Patients in dataset  : {len(self.pairs)}")
        print(f"  Success rate         : {(1 - affected_patients/len(self.pairs)) * 100:.1f}%")
        print(f"\nCorrupted tiles by patient:")
        for pid in sorted(self.bad_tiles.keys()):
            tiles = self.bad_tiles[pid]
            print(f"  {pid}: {len(tiles)} tile(s)")
            for tile in tiles[:3]:  # Show first 3 tiles
                print(f"    - {tile[:60]}...")
            if len(tiles) > 3:
                print(f"    ... and {len(tiles) - 3} more")
        print(f"{'='*70}\n")


# ======================================================================
#  Image transforms
# ======================================================================

def make_tile_transform(img_size: int = 512) -> transforms.Compose:
    """Standard tile transform: resize, crop, normalise to [-1, 1]."""
    return transforms.Compose([
        transforms.Resize(
            img_size,
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


# ======================================================================
#  Joint diffusion fine-tuning with genomic conditioning
# ======================================================================

class LitDiffusionGenomicV4(LitModel):
    """
    Fine-tune the diffusion model with genomic conditioning.

    The projection head and UNet train **jointly**: paired (tile, genomic)
    samples are fed every step so the diffusion loss gradient flows
    through ``cond = proj_head(genomic)`` into the projection head,
    teaching it which genomic dimensions are useful for each tile.

    Standard Gaussian noise — no DDIM inversion.
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
        n_log_samples: int = 16,
        val_fraction: float = 0.1,
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
        self.val_fraction = val_fraction

        print(f"Projection head JOINTLY trained (lr={proj_lr}, "
              f"warmup={proj_warmup} steps)")

        self._fixed_val_batch: Optional[dict] = None

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------

    def on_fit_start(self):
        self.projection_head = self.projection_head.to(self.device)

    def setup(self, stage=None):
        if self.conf.seed is not None:
            seed = self.conf.seed + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        xform = make_tile_transform(self._img_size)

        if self.val_fraction > 0:
            # Discover all matched patients, then split for train / val
            all_pids = discover_common_patients(
                self.genomic_h5_dir, self.tiles_zip_dir, self.split)
            n_val = max(1, int(len(all_pids) * self.val_fraction))
            rng = np.random.RandomState(42)  # deterministic split
            shuffled = list(all_pids)
            rng.shuffle(shuffled)
            val_pids = sorted(shuffled[:n_val])
            train_pids = sorted(shuffled[n_val:])
            print(f"[setup] Patient split: {len(train_pids)} train, "
                  f"{len(val_pids)} val ({self.val_fraction:.0%})")

            self.train_data = GenomicTileDataset(
                genomic_h5_dir=self.genomic_h5_dir,
                tiles_zip_dir=self.tiles_zip_dir,
                transform=xform,
                tiles_per_patient=self.tiles_per_patient,
                split=self.split,
                patient_ids=train_pids,
            )
            self.val_data = GenomicTileDataset(
                genomic_h5_dir=self.genomic_h5_dir,
                tiles_zip_dir=self.tiles_zip_dir,
                transform=xform,
                tiles_per_patient=self.tiles_per_patient,
                split=self.split,
                patient_ids=val_pids,
            )
        else:
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
                "Dataset is empty — check genomic and tile paths")

        # Disable parent's feat_extractor (we use projection head instead)
        self.feat_extractor = None
        self.model.feat_extractor = None
        self.ema_model.feat_extractor = None

        # Capture fixed validation batch for grid logging
        val_loader = DataLoader(
            self.val_data,
            batch_size=min(self.n_log_samples, len(self.val_data)),
            shuffle=False,
            num_workers=0,
        )
        try:
            batch = next(iter(val_loader))
            # Validate batch structure
            if batch is None or not hasattr(batch, "__getitem__") or "img" not in batch:
                self._fixed_val_batch = None
                print("[setup] Warning: validation loader returned empty or unexpected batch")
            else:
                # store batch as-is (tensors will be moved to device later)
                self._fixed_val_batch = batch
                # Compute number of samples robustly (guard against None, lists, tensors)
                fixed = self._fixed_val_batch
                if fixed is None or not hasattr(fixed, "__getitem__") or "img" not in fixed:
                    n_val = None
                else:
                    try:
                        img = fixed["img"]
                        if hasattr(img, "shape"):
                            n_val = int(img.shape[0])
                        else:
                            n_val = int(len(img))
                    except Exception:
                        n_val = None
                print(f"[setup] Captured {n_val if n_val is not None else 'unknown'} fixed validation samples "
                      f"for grid logging")
        except Exception as e:
            self._fixed_val_batch = None
            print(f"[setup] Warning: could not capture validation batch: {e}")

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.conf.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_fraction <= 0:
            return []  # skip validation when no val set
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.conf.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    # ------------------------------------------------------------------
    #  Validation step
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        imgs = batch["img"].to(self.device)
        genomic_raw = batch["feat"].to(self.device, dtype=torch.float32)

        cond = self.projection_head(genomic_raw)

        t, _ = self.T_sampler.sample(len(imgs), imgs.device)
        losses = self.sampler.training_losses(
            model=self.model,
            x_start=imgs,
            cond=cond,
            t=t,
            model_kwargs={"cond": cond},
        )
        loss = losses["loss"].mean()

        self.log("val_loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    #  Training step — joint UNet + projection head
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        imgs = batch["img"].to(self.device)
        genomic_raw = batch["feat"].to(self.device, dtype=torch.float32)

        # Project genomic → cond (gradient flows into proj head)
        cond = self.projection_head(genomic_raw)

        # Standard diffusion training loss
        t, _ = self.T_sampler.sample(len(imgs), imgs.device)
        losses = self.sampler.training_losses(
            model=self.model,
            x_start=imgs,
            cond=cond,
            t=t,
            model_kwargs={"cond": cond},
        )
        loss = losses["loss"].mean()

        # Lightning logging (accessible by ModelCheckpoint / callbacks)
        self.log("loss", loss, prog_bar=True,
                 on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss", loss, prog_bar=False,
                 on_step=False, on_epoch=True, sync_dist=True)

        # Also attempt to log to TensorBoard at self.num_samples (matches MoPaDi style).
        # Guard against `self.logger.experiment` being None (some loggers defer creation).
        if self.global_rank == 0:
            try:
                tb = getattr(self.logger, "experiment", None)
                step = int(self.num_samples) if hasattr(self, "num_samples") else None
                if tb is not None:
                    tb.add_scalar("loss", loss.item(), step)
                else:
                    # Fallback to Lightning logger so metrics are still captured
                    self.log("loss_tensorboard_fallback", loss, on_step=False, on_epoch=True)
            except Exception:
                # Ensure logging never crashes training
                self.log("loss_logging_error", loss, on_step=False, on_epoch=True)

        return {"loss": loss}

    # ------------------------------------------------------------------
    #  EMA + grid logging  (override parent's on_train_batch_end)
    # ------------------------------------------------------------------

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """EMA update + periodic 4×4 grid of real vs generated tiles."""
        if not self.is_last_accum(batch_idx):
            return

        # EMA update
        ema(self.model, self.ema_model, self.conf.ema_decay)

        # Grid logging
        if (self.conf.reconstruct_every_samples > 0
                and is_time(self.num_samples,
                            self.conf.reconstruct_every_samples,
                            self.conf.batch_size_effective)):
            self._log_grid(self.model, postfix="")
            self._log_grid(self.ema_model, postfix="_ema")

    def _log_grid(self, model, postfix: str = ""):
        """Generate a 4×4 grid from the fixed validation batch."""
        vb = self._fixed_val_batch
        if vb is None:
            return

        model.eval()
        imgs = vb["img"].to(self.device)
        genomic = vb["feat"].to(self.device, dtype=torch.float32)

        with torch.no_grad():
            cond = self.projection_head(genomic)

            # Sample from Gaussian noise conditioned on genomic
            noise = torch.randn_like(imgs)
            gen = self.eval_sampler.sample(
                model=model,
                noise=noise,
                cond=cond,
                x_start=imgs,
            )

        if self.global_rank == 0:
            step = int(self.num_samples)
            nrow = 4

            # Real tiles
            grid_real = (make_grid(imgs, nrow=nrow) + 1) / 2
            real_dir = os.path.join(
                self.conf.logdir, f"sample_real{postfix}")
            os.makedirs(real_dir, exist_ok=True)
            save_image(grid_real, os.path.join(real_dir, f"{step}.png"))

            # Generated tiles
            grid_gen = (make_grid(gen, nrow=nrow) + 1) / 2
            gen_dir = os.path.join(
                self.conf.logdir, f"sample{postfix}")
            os.makedirs(gen_dir, exist_ok=True)
            save_image(grid_gen, os.path.join(gen_dir, f"{step}.png"))

            # TensorBoard images (guarded)
            try:
                tb = getattr(self.logger, "experiment", None)
                if tb is not None:
                    tb.add_image(f"sample{postfix}/real", grid_real, step)
                    tb.add_image(f"sample{postfix}", grid_gen, step)
                else:
                    # Log a simple scalar as a marker that images were saved to disk
                    self.log(f"sample{postfix}/grid_saved", 1, on_step=False, on_epoch=True)
            except Exception as e:
                print(f"[log_grid] Warning: TensorBoard logging failed: {e}")
                self.log(f"sample{postfix}/grid_logging_error", 1, on_step=False, on_epoch=True)

        model.train()

    # ------------------------------------------------------------------
    #  Optimizer  (separate param groups for UNet + projection head)
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        param_groups = [
            {"params": list(self.model.parameters()),
             "lr": self.conf.lr, "name": "unet"},
            {"params": list(self.projection_head.parameters()),
             "lr": self.proj_lr, "name": "proj_head"},
        ]

        optim = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.99),
            eps=1e-6,
            weight_decay=self.conf.weight_decay,
        )

        # Optional linear warm-up for the projection head
        if self.proj_warmup > 0:
            from torch.optim.lr_scheduler import LambdaLR

            def warmup_fn(step):
                """Ramp from 0 → 1 over proj_warmup steps."""
                if step < self.proj_warmup:
                    return float(step) / float(max(1, self.proj_warmup))
                return 1.0

            # Apply warm-up to ALL param groups (UNet also benefits from
            # a short ramp since we're loading pre-trained weights).
            # Must pass a list of lambda fns (one per param group).
            scheduler = LambdaLR(optim, lr_lambda=[warmup_fn, warmup_fn])
            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return {"optimizer": optim}

    # ------------------------------------------------------------------
    #  Evaluation placeholder
    # ------------------------------------------------------------------

    def evaluate_scores(self):
        pass

    # ------------------------------------------------------------------
    #  Checkpoint helpers
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint):
        checkpoint["projection_head_state_dict"] = (
            self.projection_head.state_dict()
        )

    def on_load_checkpoint(self, checkpoint):
        if "projection_head_state_dict" in checkpoint:
            self.projection_head.load_state_dict(
                checkpoint["projection_head_state_dict"]
            )
            print("[OK] Projection head restored from checkpoint")


# ======================================================================
#  Entry point
# ======================================================================

def run(args):
    """Build model, load checkpoint, train."""

    # ---- Projection head (trains from scratch jointly with UNet) ----
    proj_head = ProjectionHead(
        in_dim=512,
        out_dim=512,
        hidden_dim=args.proj_hidden_dim,
        num_layers=args.proj_layers,
        dropout=args.proj_dropout,
    )

    # Optionally warm-start from a previous checkpoint
    if args.proj_ckpt:
        ckpt_path = Path(args.proj_ckpt)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            proj_head.load_state_dict(
                ckpt.get("state_dict", ckpt), strict=False)
            print(f"[OK] Warm-started projection head from {args.proj_ckpt}")
        else:
            print(f"[WARN] --proj-ckpt not found: {args.proj_ckpt}")

    # ---- MoPaDi config ----
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
    conf.sample_size = args.n_log_samples
    conf.reconstruct_every_samples = args.reconstruct_every_samples
    conf.eval_every_samples = 999_999_999
    conf.eval_ema_every_samples = 999_999_999
    conf.save_every_samples = args.save_every_samples
    conf.base_dir = args.out_dir
    conf.name = "genomic_finetune_v4"

    setattr(conf, "data_dirs", [])
    setattr(conf, "feature_dirs", [])
    setattr(conf, "feat_extractor", None)

    # ---- Build model ----
    model = LitDiffusionGenomicV4(
        conf=conf,
        projection_head=proj_head,
        genomic_h5_dir=args.genomic_h5_dir,
        tiles_zip_dir=args.tiles_zip_dir,
        tiles_per_patient=args.tiles_per_patient,
        split=args.split,
        img_size=args.img_size,
        proj_lr=args.proj_lr,
        proj_warmup=args.proj_warmup,
        n_log_samples=args.n_log_samples,
        val_fraction=args.val_fraction,
    )

    # ---- Load pretrained diffusion weights ----
    if not Path(args.diffusion_ckpt).exists():
        raise FileNotFoundError(
            f"Diffusion checkpoint not found: {args.diffusion_ckpt}")
    print(f"[INFO] Loading diffusion checkpoint: {args.diffusion_ckpt}")
    ckpt = torch.load(args.diffusion_ckpt, map_location="cpu")

    if "state_dict" in ckpt:
        state = dict(ckpt["state_dict"])
        # Drop x_T if shape doesn't match
        if "x_T" in state:
            model_xT = getattr(model, "x_T", None)
            if model_xT is not None:
                if tuple(state["x_T"].shape) != tuple(model_xT.shape):
                    print(f"[WARN] Dropping x_T from checkpoint "
                          f"(shape {tuple(state['x_T'].shape)} "
                          f"!= {tuple(model_xT.shape)})")
                    del state["x_T"]
        model.load_state_dict(state, strict=False)
        print(f"[OK] Loaded state_dict from {args.diffusion_ckpt}")
    else:
        model.model.load_state_dict(ckpt, strict=False)
        model.ema_model = copy.deepcopy(model.model)
        model.ema_model.requires_grad_(False)
        model.ema_model.eval()
        print(f"[OK] Loaded bare model weights from {args.diffusion_ckpt}")

    # ---- Trainer ----
    os.makedirs(conf.logdir, exist_ok=True)

    if args.val_fraction > 0:
        ckpt_callback = ModelCheckpoint(
            dirpath=conf.logdir,
            filename="epoch={epoch:02d}-val_loss={val_loss:.4f}",
            save_last=True,
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        )
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping_patience,
            min_delta=1e-4,
            mode="min",
            verbose=True,
        )
        callbacks = [ckpt_callback, early_stop_callback,
                     LearningRateMonitor()]
    else:
        ckpt_callback = ModelCheckpoint(
            dirpath=conf.logdir,
            filename="epoch={epoch:02d}-loss={loss:.4f}",
            save_last=True,
            save_top_k=3,
            monitor="loss",
            mode="min",
            every_n_train_steps=max(
                1,
                conf.save_every_samples // conf.batch_size_effective,
            ),
        )
        callbacks = [ckpt_callback, LearningRateMonitor()]

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=conf.logdir, name=None, version="",
    )

    gpus = args.gpus
    if len(gpus) == 1:
        accelerator, strategy = "gpu", "auto"
    elif len(gpus) > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=True)
        accelerator = "gpu"
    else:
        accelerator, strategy = "cpu", "auto"

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=gpus,
        strategy=strategy,
        accelerator=accelerator,
        precision="16-mixed" if conf.fp16 else 32,
        callbacks=callbacks,
        logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
    )

    last_ckpt = os.path.join(conf.logdir, "last.ckpt")
    if os.path.exists(last_ckpt):
        print(f"[INFO] Resuming from {last_ckpt}")
        trainer.fit(model, ckpt_path=last_ckpt)
    else:
        trainer.fit(model)

    # Log corruption statistics after training
    if hasattr(model, 'train_data') and model.train_data is not None:
        model.train_data.log_corruption_summary()

    print(f"\n{'='*60}")
    print("GENOMIC DIFFUSION FINE-TUNING COMPLETE")
    print(f"  Output directory : {conf.logdir}")
    print(f"  TensorBoard logs : {conf.logdir}")
    print(f"{'='*60}\n")


# ======================================================================
#  CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune diffusion model with genomic conditioning (v4)"
                    " — joint projection-head + UNet training",
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file with training settings")
    parser.add_argument("--diffusion-ckpt", type=str, required=False,
                        help="Path to pretrained diffusion checkpoint")
    parser.add_argument("--genomic-h5-dir", type=str, required=False,
                        help="Dir with per-patient genomic .h5 files")
    parser.add_argument("--tiles-zip-dir", type=str, required=False,
                        help="Dir with per-patient tile .zip archives")
    parser.add_argument("--out-dir", type=str, required=False)

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum-batches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="UNet learning rate")
    parser.add_argument("--proj-lr", type=float, default=3e-4,
                        help="Projection head learning rate")
    parser.add_argument("--proj-warmup", type=int, default=500,
                        help="Linear warm-up steps for LR scheduler")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=0,
                        help="Global warm-up (MoPaDi config)")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)

    # Projection head architecture
    parser.add_argument("--proj-ckpt", type=str, default=None,
                        help="Optional: warm-start proj head from checkpoint")
    parser.add_argument("--proj-layers", type=int, default=4)
    parser.add_argument("--proj-hidden-dim", type=int, default=512)
    parser.add_argument("--proj-dropout", type=float, default=0.1)

    # Data
    parser.add_argument("--tiles-per-patient", type=int, default=10)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)

    # Validation & early stopping
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of patients held out for validation "
                             "(0 to disable)")
    parser.add_argument("--early-stopping-patience", type=int, default=5,
                        help="Stop if val_loss does not improve for this "
                             "many epochs")

    # Logging
    parser.add_argument("--n-log-samples", type=int, default=16,
                        help="Fixed validation tiles for 4×4 grid logging")
    parser.add_argument("--reconstruct-every-samples", type=int,
                        default=10_000)
    parser.add_argument("--save-every-samples", type=int, default=5_000)

    # Hardware
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])

    args = parser.parse_args()

    # if config file is supplied, override corresponding args
    # the YAML may contain a top‑level "training" dict or specify options
    # directly (e.g. diffusion_ckpt, tiles_zip_dir) – any key matching an
    # argparse attribute will be applied.
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        applied = {}
        # first process top‑level keys
        for key, value in cfg.items():
            if key == 'training':
                continue
            if hasattr(args, key):
                setattr(args, key, value)
                applied[key] = value
        # then process nested training block if present
        train_cfg = cfg.get('training', {})
        for key, value in train_cfg.items():
            if hasattr(args, key):
                setattr(args, key, value)
                applied[key] = value
        if applied:
            print(f"[config] Applied settings from {args.config}: {applied}")

    # Ensure numeric types are correct (YAML may parse as strings in edge cases)
    args.lr = float(args.lr)
    args.proj_lr = float(args.proj_lr)
    args.batch_size = int(args.batch_size)
    args.epochs = int(args.epochs)
    args.val_fraction = float(args.val_fraction)
    args.early_stopping_patience = int(args.early_stopping_patience)

    # validate required arguments now that config merges have happened
    missing = []
    for req in ["diffusion_ckpt", "genomic_h5_dir", "tiles_zip_dir", "out_dir"]:
        if getattr(args, req, None) is None:
            missing.append(req)
    if missing:
        parser.error(f"the following arguments are required (either via CLI or config): {', '.join(missing)}")

    # ---- Banner ----
    print(f"\n{'='*60}")
    print("DIFFUSION FINE-TUNING WITH GENOMIC CONDITIONING (v4)")
    print(f"  Joint UNet + ProjectionHead training")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  UNet LR : {args.lr}  |  Proj LR : {args.proj_lr}")
    print(f"{'='*60}\n")

    run(args)


if __name__ == "__main__":
    main()
