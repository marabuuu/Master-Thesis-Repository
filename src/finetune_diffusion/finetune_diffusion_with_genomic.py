#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune Diffusion Model with Genomic Conditioning

This script fine-tunes a pretrained MoPaDi diffusion model to generate tiles
conditioned on genomic feature vectors (projected through a trained projection head).

Architecture:
    Genomic (512-dim) → Projection Head → Pseudo Image Features (512-dim) → cDDIM → Tile

The projection head was trained separately to match the distribution of image features.
Now we fine-tune the diffusion model so it learns the semantic relationship between
projected genomic features and tile content.

Usage:
    python finetune_diffusion_with_genomic.py \\
        --projection-head-ckpt ./projection_head_best.pt \\
        --diffusion-ckpt ./diffusion_without_encoder.ckpt \\
        --genomic-h5-dir /path/to/genomic_features \\
        --tiles-zip-dir /path/to/tile_zips \\
        --out-dir ./finetuned_diffusion \\
        --epochs 10
"""

import argparse
import copy
import math
import os
import random
import re
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# ----------------------------------------------------------------------
#   Projection Head (copied from projection_head_genomic.py)
# ----------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """Projection head for genomic → image feature space mapping."""
    
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
                layers.append(nn.Linear(hidden_dim if i > 0 else in_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.net = nn.Sequential(*layers)
            self.skip = nn.Identity() if in_dim == hidden_dim else nn.Linear(in_dim, out_dim)
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
    """Load a trained projection head from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    
    projection_head = ProjectionHead(
        in_dim=config.get("in_dim", 512),
        out_dim=config.get("out_dim", 512),
        hidden_dim=config.get("hidden_dim", 512),
        num_layers=config.get("num_layers", 2),
        arch=config.get("arch", "mlp"),
    )
    projection_head.load_state_dict(ckpt["state_dict"])
    projection_head.to(device)
    projection_head.eval()
    
    print(f"[OK] Loaded projection head from {ckpt_path}")
    print(f"     Config: {config}")
    
    return projection_head


# ----------------------------------------------------------------------
#   Helper Functions
# ----------------------------------------------------------------------

def canonical_patient_id(name: str) -> str:
    """Extract TCGA-XX-XXXX patient ID from various filename formats."""
    name = Path(name).stem.upper()
    for sep in ("_", "."):
        name = name.replace(sep, "-")
    while "--" in name:
        name = name.replace("--", "-")
    parts = name.split("-")
    if len(parts) >= 3 and parts[0].startswith("TCGA"):
        return "-".join(parts[:3])
    return name


# ----------------------------------------------------------------------
#   Dataset
# ----------------------------------------------------------------------

class GenomicTileDataset(Dataset):
    """
    Dataset that pairs genomic H5 features with tile images from zip files.
    
    For each patient:
    - genomic: single 512-dim vector from H5 file
    - tile: random tile image from corresponding zip file
    """
    
    def __init__(
        self,
        genomic_h5_dir: str,
        tiles_zip_dir: str,
        genomic_key: str = "feats",
        transform=None,
        tiles_per_patient: int = 1,  # How many tiles to sample per patient per epoch
        split: str = "train",
    ):
        super().__init__()
        self.genomic_h5_dir = Path(genomic_h5_dir).expanduser()
        self.tiles_zip_dir = Path(tiles_zip_dir).expanduser()
        self.genomic_key = genomic_key
        self.transform = transform
        self.tiles_per_patient = tiles_per_patient
        
        # Find matching files
        # Check for train/test subdirs
        train_dir = self.genomic_h5_dir / "train"
        test_dir = self.genomic_h5_dir / "test"
        
        if train_dir.is_dir() or test_dir.is_dir():
            if split == "train" and train_dir.is_dir():
                genomic_files = {canonical_patient_id(f.name): f 
                                 for f in train_dir.glob("*.h5")}
            elif split == "test" and test_dir.is_dir():
                genomic_files = {canonical_patient_id(f.name): f 
                                 for f in test_dir.glob("*.h5")}
            else:
                genomic_files = {}
                if train_dir.is_dir():
                    genomic_files.update({canonical_patient_id(f.name): f 
                                          for f in train_dir.glob("*.h5")})
                if test_dir.is_dir():
                    genomic_files.update({canonical_patient_id(f.name): f 
                                          for f in test_dir.glob("*.h5")})
        else:
            genomic_files = {canonical_patient_id(f.name): f 
                             for f in self.genomic_h5_dir.glob("*.h5")}
        
        zip_files = {canonical_patient_id(f.name): f 
                     for f in self.tiles_zip_dir.glob("*.zip")}
        
        common_ids = sorted(set(genomic_files) & set(zip_files))
        if not common_ids:
            raise RuntimeError(
                f"No matching patient IDs between\n"
                f"  Genomic: {self.genomic_h5_dir}\n"
                f"  Tiles: {self.tiles_zip_dir}"
            )
        
        print(f"[GenomicTileDataset] Found {len(common_ids)} matched patients (split={split})")
        self.pairs = [(genomic_files[pid], zip_files[pid], pid) for pid in common_ids]
        
        # Cache genomic features (they're small, one per patient)
        self.genomic_cache = {}
        print("[INFO] Caching genomic features...")
        for genomic_path, _, pid in tqdm(self.pairs, desc="Loading genomic"):
            with h5py.File(genomic_path, "r") as f:
                genomic = np.array(f[self.genomic_key])
                if genomic.ndim == 2:
                    genomic = genomic.mean(axis=0)
                self.genomic_cache[pid] = torch.from_numpy(genomic.astype(np.float32))
        
        # Pre-list tiles in each zip (for faster random sampling)
        print("[INFO] Indexing tile files...")
        self.tile_lists = {}
        bad_zips = []
        for _, zip_path, pid in tqdm(self.pairs, desc="Indexing zips"):
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    tiles = [n for n in zf.namelist() 
                             if n.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    self.tile_lists[pid] = tiles
            except zipfile.BadZipFile:
                print(f"[WARNING] Skipping bad/corrupt zip: {zip_path}")
                bad_zips.append((pid, zip_path))
        
        # Remove bad zips from pairs
        if bad_zips:
            bad_pids = {pid for pid, _ in bad_zips}
            self.pairs = [(g, z, p) for g, z, p in self.pairs if p not in bad_pids]
            print(f"[WARNING] Skipped {len(bad_zips)} corrupt zip files")
        
        # Create expanded index for tiles_per_patient > 1
        self.samples = []
        for genomic_path, zip_path, pid in self.pairs:
            n_tiles = min(self.tiles_per_patient, len(self.tile_lists[pid]))
            for _ in range(n_tiles):
                self.samples.append((pid, zip_path))
        
        print(f"[GenomicTileDataset] Total samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_random_tile(self, zip_path: Path, pid: str) -> Image.Image:
        tile_name = random.choice(self.tile_lists[pid])
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open(tile_name) as f:
                return Image.open(BytesIO(f.read())).convert("RGB")
    
    def __getitem__(self, idx):
        pid, zip_path = self.samples[idx]
        
        # Get cached genomic features
        genomic = self.genomic_cache[pid]
        
        # Load random tile
        img = self._load_random_tile(zip_path, pid)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        return {
            "img": img,
            "genomic": genomic,
            "patient_id": pid,
        }


# ----------------------------------------------------------------------
#   Early Stopping
# ----------------------------------------------------------------------

class EarlyStopping:
    """Stop training when the monitored loss stops improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
                  Set to 0 to disable.
        min_delta: Minimum absolute decrease in loss to count as an
                   improvement.  Helps ignore noise.
        restore_best: If True the caller should reload the best checkpoint
                      after training ends (signalled by `self.should_stop`).
    """

    def __init__(self, patience: int = 0, min_delta: float = 0.0,
                 restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best_loss: Optional[float] = None
        self.epochs_without_improvement: int = 0
        self.should_stop: bool = False
        self.best_epoch: int = 0

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def step(self, loss: float, epoch: int) -> bool:
        """Call once per epoch.  Returns True when training should stop."""
        if not self.enabled:
            return False

        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.epochs_without_improvement = 0
            self.best_epoch = epoch
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            return True
        return False

    def status_message(self) -> str:
        if not self.enabled:
            return "early stopping disabled"
        return (f"patience {self.epochs_without_improvement}/{self.patience} "
                f"(best={self.best_loss:.6f} @ epoch {self.best_epoch})")


# ----------------------------------------------------------------------
#   LR Scheduler helpers
# ----------------------------------------------------------------------

class _LinearWarmupCosineDecay(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup for `warmup_epochs`, then cosine decay to `eta_min`."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 eta_min_factor: float = 0.01, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min_factor = eta_min_factor
        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            # linear warmup from 0 → 1
            return max(epoch / max(self.warmup_epochs, 1), 1e-6)
        # cosine decay from 1 → eta_min_factor
        progress = (epoch - self.warmup_epochs) / max(
            self.total_epochs - self.warmup_epochs, 1)
        import math
        return self.eta_min_factor + 0.5 * (1.0 - self.eta_min_factor) * (
            1.0 + math.cos(math.pi * progress))


def build_scheduler(optimizer, args):
    """Build a learning-rate scheduler from CLI arguments.

    Supported values for ``args.scheduler``:
    * ``cosine``        – CosineAnnealingLR (default, current behaviour)
    * ``cosine_warmup`` – Linear warmup + cosine decay
    * ``plateau``       – ReduceLROnPlateau (reacts to loss stagnation)
    """
    name = getattr(args, "scheduler", "cosine")
    warmup = getattr(args, "warmup_epochs", 0)

    if name == "cosine_warmup" or (name == "cosine" and warmup > 0):
        print(f"[INFO] LR scheduler: cosine with {warmup}-epoch linear warmup")
        return _LinearWarmupCosineDecay(
            optimizer,
            warmup_epochs=warmup,
            total_epochs=args.epochs,
            eta_min_factor=0.01,
        )
    elif name == "plateau":
        print("[INFO] LR scheduler: ReduceLROnPlateau (factor=0.5, patience=3)")
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3,
        )
    else:  # default: plain cosine
        print("[INFO] LR scheduler: CosineAnnealingLR")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
        )


# ----------------------------------------------------------------------
#   Diffusion Fine-tuning
# ----------------------------------------------------------------------

def finetune_diffusion(
    projection_head: ProjectionHead,
    diffusion_ckpt_path: str,
    train_loader: DataLoader,
    args,
    device: str,
):
    """
    Fine-tune the diffusion model with genomic conditioning.
    
    The projection head can be:
    - Frozen: Only the diffusion model learns
    - Trainable: Both adapt together (joint fine-tuning)
    """
    
    # Import MoPaDi
    try:
        from mopadi.configs.templates import tcga_brca_autoenc
        from mopadi.model.unet import BeatGANsUNetModel
        from mopadi.model.nn import timestep_embedding
    except ImportError as e:
        raise RuntimeError(f"Failed to import MoPaDi: {e}")
    
    print("\n" + "=" * 60)
    print("LOADING DIFFUSION MODEL")
    print("=" * 60)
    
    # Load config and create model
    conf = tcga_brca_autoenc()
    
    # Load the checkpoint
    ckpt = torch.load(diffusion_ckpt_path, map_location=device)
    
    # The checkpoint might be from LitModel or just the model state
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        # Filter to only model weights (not ema_model or other things)
        model_state = {}
        ema_state = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                model_state[k[6:]] = v  # Remove "model." prefix
            elif k.startswith("ema_model."):
                ema_state[k[10:]] = v  # Remove "ema_model." prefix
        
        # Prefer EMA weights if available
        if ema_state:
            print("[INFO] Using EMA model weights")
            state_dict = ema_state
        elif model_state:
            state_dict = model_state
    else:
        state_dict = ckpt
    
    # Create model
    model = conf.make_model_conf().make_model()
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    
    # Create EMA model
    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)
    ema_model.eval()
    
    print(f"[OK] Loaded diffusion model")
    print(f"     Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get conditioning statistics from checkpoint or use defaults
    conds_mean = ckpt.get("conds_mean", None)
    conds_std = ckpt.get("conds_std", None)
    
    if conds_mean is None and "state_dict" in ckpt:
        # Try to find in state_dict
        if "conds_mean" in ckpt["state_dict"]:
            conds_mean = ckpt["state_dict"]["conds_mean"]
        if "conds_std" in ckpt["state_dict"]:
            conds_std = ckpt["state_dict"]["conds_std"]
    
    if conds_mean is not None:
        if not isinstance(conds_mean, torch.Tensor):
            conds_mean = torch.tensor(conds_mean, dtype=torch.float32)
        conds_mean = conds_mean.to(device)
        if conds_mean.dim() == 2:
            conds_mean = conds_mean.squeeze(0)
        print(f"[OK] conds_mean loaded, shape: {conds_mean.shape}")
    else:
        conds_mean = torch.zeros(512, device=device)
        print("[WARN] No conds_mean found, using zeros")
    
    if conds_std is not None:
        if not isinstance(conds_std, torch.Tensor):
            conds_std = torch.tensor(conds_std, dtype=torch.float32)
        conds_std = conds_std.to(device)
        if conds_std.dim() == 2:
            conds_std = conds_std.squeeze(0)
        print(f"[OK] conds_std loaded, shape: {conds_std.shape}")
    else:
        conds_std = torch.ones(512, device=device)
        print("[WARN] No conds_std found, using ones")
    
    # Create diffusion sampler
    sampler = conf.make_diffusion_conf().make_sampler()
    
    # Setup optimizer
    if args.freeze_projection_head:
        projection_head.eval()
        for p in projection_head.parameters():
            p.requires_grad = False
        trainable_params = list(model.parameters())
        print("[INFO] Projection head FROZEN")
    else:
        trainable_params = list(model.parameters()) + list(projection_head.parameters())
        print("[INFO] Projection head TRAINABLE (joint fine-tuning)")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    scheduler = build_scheduler(optimizer, args)
    is_plateau_scheduler = isinstance(
        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=getattr(args, "early_stopping_patience", 0),
        min_delta=getattr(args, "early_stopping_min_delta", 0.0),
    )
    
    print(f"\n[INFO] Training config:")
    print(f"       Epochs: {args.epochs}")
    print(f"       Batch size: {args.batch_size}")
    print(f"       Grad accum steps: {args.grad_accum_steps}")
    print(f"       Effective batch size: {args.batch_size * args.grad_accum_steps}")
    print(f"       Learning rate: {args.lr}")
    print(f"       Scheduler: {getattr(args, 'scheduler', 'cosine')}")
    print(f"       Warmup epochs: {getattr(args, 'warmup_epochs', 0)}")
    print(f"       Freeze projection: {args.freeze_projection_head}")
    print(f"       Mixed precision (fp16): {args.fp16}")
    if early_stopping.enabled:
        print(f"       Early stopping: patience={early_stopping.patience}, "
              f"min_delta={early_stopping.min_delta}")
    else:
        print(f"       Early stopping: disabled")
    
    # Setup mixed precision
    use_amp = args.fp16 and device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("[INFO] Using mixed precision training (fp16)")
    
    # Training loop
    print("\n" + "=" * 60)
    print("STARTING FINE-TUNING")
    print("=" * 60 + "\n")
    
    best_loss = float("inf")
    training_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        if not args.freeze_projection_head:
            projection_head.train()
        
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        
        optimizer.zero_grad()
        accum_loss = 0.0
        
        for batch_idx, batch in enumerate(pbar):
            imgs = batch["img"].to(device)
            genomic = batch["genomic"].to(device)
            B = imgs.shape[0]
            
            # Mixed precision forward pass
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                # Project genomic features
                with torch.set_grad_enabled(not args.freeze_projection_head):
                    projected = projection_head(genomic)  # (B, 512)
                
                # Normalize using diffusion model's statistics
                cond = (projected - conds_mean) / (conds_std + 1e-6)
                
                # Sample timesteps
                t = torch.randint(0, conf.T, (B,), device=device).long()
                
                # Compute diffusion loss
                model_kwargs = {"cond": cond}
                losses = sampler.training_losses(
                    model=model,
                    x_start=imgs,
                    cond=cond,
                    t=t,
                    model_kwargs=model_kwargs,
                )
                
                # Scale loss for gradient accumulation
                loss = losses["loss"].mean() / args.grad_accum_steps
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            accum_loss += loss.item() * args.grad_accum_steps
            
            # Step optimizer every grad_accum_steps
            if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update EMA after each optimizer step
                with torch.no_grad():
                    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                        p_ema.data.mul_(0.9999).add_(p.data, alpha=0.0001)
                
                epoch_loss += accum_loss
                pbar.set_postfix({"loss": accum_loss})
                accum_loss = 0.0
        
        # Step the LR scheduler
        # Average loss: we accumulate per optimizer step, divide by number of steps
        num_optim_steps = (len(train_loader) + args.grad_accum_steps - 1) // args.grad_accum_steps
        epoch_loss /= max(num_optim_steps, 1)

        if is_plateau_scheduler:
            scheduler.step(epoch_loss)
        else:
            scheduler.step()

        # Current LR (works for all scheduler types)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch}/{args.epochs} | Time: {epoch_time:.1f}s | "
              f"Loss: {epoch_loss:.6f} | LR: {current_lr:.2e}")
        
        if device.startswith("cuda"):
            mem = torch.cuda.memory_allocated(device) / 1024**3
            print(f"  GPU Memory: {mem:.2f} GB")
        
        # Save best
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(
                args.out_dir, "diffusion_genomic_best.pt",
                model, ema_model, projection_head, 
                conds_mean, conds_std, epoch, epoch_loss, args
            )
            print(f"  ★ NEW BEST! Saved checkpoint")
        
        # Periodic save
        if epoch % 5 == 0 or epoch == args.epochs:
            save_checkpoint(
                args.out_dir, f"diffusion_genomic_epoch{epoch:03d}.pt",
                model, ema_model, projection_head,
                conds_mean, conds_std, epoch, epoch_loss, args
            )
            print(f"  Checkpoint saved: epoch {epoch}")
        
        # Early stopping check
        if early_stopping.step(epoch_loss, epoch):
            print(f"\n[EARLY STOPPING] No improvement for {early_stopping.patience} "
                  f"epochs. Best loss {early_stopping.best_loss:.6f} at epoch "
                  f"{early_stopping.best_epoch}. Stopping.")
            break
        elif early_stopping.enabled:
            print(f"  Early stopping: {early_stopping.status_message()}")
    
    total_time = time.time() - training_start
    stopped_early = early_stopping.should_stop
    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE" + (" (early stopped)" if stopped_early else ""))
    print(f"  Epochs run: {epoch}/{args.epochs}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Best loss: {best_loss:.6f}")
    if early_stopping.enabled:
        print(f"  Best epoch: {early_stopping.best_epoch}")
    print(f"  Output: {args.out_dir}")
    print("=" * 60 + "\n")


def save_checkpoint(
    out_dir: str,
    filename: str,
    model: nn.Module,
    ema_model: nn.Module,
    projection_head: ProjectionHead,
    conds_mean: torch.Tensor,
    conds_std: torch.Tensor,
    epoch: int,
    loss: float,
    args,
):
    """Save a combined checkpoint with diffusion model + projection head."""
    os.makedirs(out_dir, exist_ok=True)
    save_path = Path(out_dir) / filename
    
    torch.save({
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "ema_model_state_dict": ema_model.state_dict(),
        "projection_head_state_dict": projection_head.state_dict(),
        "projection_head_config": {
            "in_dim": projection_head.in_dim,
            "out_dim": projection_head.out_dim,
            "arch": projection_head.arch,
        },
        "conds_mean": conds_mean.cpu(),
        "conds_std": conds_std.cpu(),
        "args": vars(args),
    }, save_path)


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune diffusion model with genomic conditioning"
    )
    
    # Required paths
    parser.add_argument("--projection-head-ckpt", type=str, required=True,
                        help="Path to trained projection head checkpoint")
    parser.add_argument("--diffusion-ckpt", type=str, required=True,
                        help="Path to pretrained diffusion model checkpoint")
    parser.add_argument("--genomic-h5-dir", type=str, required=True,
                        help="Directory with genomic feature H5 files")
    parser.add_argument("--tiles-zip-dir", type=str, required=True,
                        help="Directory with tile zip files")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for checkpoints")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (reduce if OOM)")
    parser.add_argument("--grad-accum-steps", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (lower than pretraining)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    
    # Scheduler & early stopping
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "cosine_warmup", "plateau"],
                        help="LR scheduler type (default: cosine)")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Number of linear-warmup epochs before cosine decay "
                             "(only used with cosine/cosine_warmup scheduler)")
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop after N epochs without improvement (0 = disabled)")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0,
                        help="Minimum loss decrease to count as improvement")
    
    parser.add_argument("--tiles-per-patient", type=int, default=10,
                        help="Tiles to sample per patient per epoch")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test", "all"])
    
    # Model settings
    parser.add_argument("--freeze-projection-head", action="store_true",
                        help="Keep projection head frozen during fine-tuning")
    parser.add_argument("--img-size", type=int, default=512,
                        help="Image size for training")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training (fp16) to reduce memory")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    
    args = parser.parse_args()
    
    # Startup banner
    print("\n" + "=" * 60)
    print("DIFFUSION MODEL FINE-TUNING WITH GENOMIC CONDITIONING")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("=" * 60 + "\n")
    
    # Device setup
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = args.device
        props = torch.cuda.get_device_properties(0)
        print(f"[OK] Using GPU: {props.name} ({props.total_memory/1024**3:.1f} GB)")
    else:
        device = "cpu"
        print("[WARN] Using CPU (this will be slow)")
    
    # Validate paths
    assert Path(args.projection_head_ckpt).exists(), \
        f"Projection head checkpoint not found: {args.projection_head_ckpt}"
    assert Path(args.diffusion_ckpt).exists(), \
        f"Diffusion checkpoint not found: {args.diffusion_ckpt}"
    assert Path(args.genomic_h5_dir).exists(), \
        f"Genomic H5 directory not found: {args.genomic_h5_dir}"
    assert Path(args.tiles_zip_dir).exists(), \
        f"Tiles zip directory not found: {args.tiles_zip_dir}"
    
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[OK] Output directory: {args.out_dir}")
    
    # Load projection head
    print("\n" + "=" * 60)
    print("LOADING PROJECTION HEAD")
    print("=" * 60)
    projection_head = load_projection_head(args.projection_head_ckpt, device)
    
    # Create dataset
    print("\n" + "=" * 60)
    print("CREATING DATASET")
    print("=" * 60)
    
    transform = transforms.Compose([
        transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    
    dataset = GenomicTileDataset(
        genomic_h5_dir=args.genomic_h5_dir,
        tiles_zip_dir=args.tiles_zip_dir,
        transform=transform,
        tiles_per_patient=args.tiles_per_patient,
        split=args.split,
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    print(f"[OK] DataLoader: {len(train_loader)} batches")
    
    # Fine-tune
    finetune_diffusion(
        projection_head=projection_head,
        diffusion_ckpt_path=args.diffusion_ckpt,
        train_loader=train_loader,
        args=args,
        device=device,
    )
    
    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
