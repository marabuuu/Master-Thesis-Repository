#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Projection Head Training for Genomic-to-Image Feature Alignment

This script trains a learnable projection head to transform genomic feature vectors
(one per patient, 512-dim) into the conditioning space expected by a pretrained
conditional denoising diffusion model (MoPaDi).

Architecture Overview
---------------------
The MoPaDi diffusion model expects:
- Input noise x_T of shape (B, 3, H, W)
- Conditioning vector cond of shape (B, 512) ← this is what the projection head learns

Training Modes
--------------
1. DISTRIBUTION_MATCHING (recommended, genomic-only):
   Train using ONLY genomic H5 files. The projection head learns to map genomic
   features into the conditioning distribution the diffusion model expects.
   Uses the diffusion model's learned conds_mean/conds_std as targets.
   
2. FEATURE_ALIGNMENT: Match projected genomic features to real image features
   (requires both genomic h5 and image feature h5 files for the same patients)
   
3. RECONSTRUCTION: Decode x_T using projected genomic conditioning and minimize
   reconstruction loss (requires genomic h5 + tile images, more expensive)

Usage
-----
# Recommended: Train with only genomic features (distribution matching)
python projection_head_genomic.py \\
    --mode distribution_matching \\
    --genomic-h5-dir /path/to/genomic_features \\
    --diffusion-ckpt ./split_ckpts/diffusion_without_encoder.ckpt \\
    --out-dir ./projection_head_output \\
    --epochs 50

# Alternative: Train with image features (if available)
python projection_head_genomic.py \\
    --mode feature_alignment \\
    --genomic-h5-dir /path/to/genomic_features \\
    --image-h5-dir /path/to/image_features \\
    --out-dir ./projection_head_output \\
    --epochs 50

After training, convert genomic features to pseudo-image-feature format:
python projection_head_genomic.py --convert \\
    --checkpoint ./projection_head_output/projection_head_best.pt \\
    --genomic-h5-dir /path/to/genomic_features \\
    --tiles-zip-dir /path/to/tile_zips \\
    --output-h5-dir /path/to/output_features
"""

import argparse
import logging
import os
import random
import re
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ----------------------------------------------------------------------
#   Logging Setup
# ----------------------------------------------------------------------

def setup_logging(out_dir: str, verbose: bool = True) -> logging.Logger:
    """Configure logging to both console and file."""
    logger = logging.getLogger("projection_head")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_fmt = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)
    
    # File handler
    os.makedirs(out_dir, exist_ok=True)
    file_handler = logging.FileHandler(Path(out_dir) / "training.log")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)
    
    return logger


def check_cuda_availability(requested_device: str) -> str:
    """Check CUDA availability and return the appropriate device."""
    print("=" * 60)
    print("CUDA / GPU CHECK")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print(f"[WARNING] CUDA is not available. Falling back to CPU.")
        print(f"  - torch.cuda.is_available(): {torch.cuda.is_available()}")
        return "cpu"
    
    # CUDA is available
    num_gpus = torch.cuda.device_count()
    print(f"[OK] CUDA is available with {num_gpus} GPU(s)")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    - Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"    - Compute Capability: {props.major}.{props.minor}")
    
    # Parse requested device
    if requested_device.startswith("cuda"):
        if ":" in requested_device:
            device_idx = int(requested_device.split(":")[1])
            if device_idx >= num_gpus:
                print(f"[ERROR] Requested GPU {device_idx} but only {num_gpus} available!")
                raise RuntimeError(f"GPU {device_idx} not available. Only {num_gpus} GPUs found.")
            print(f"[OK] Using requested device: {requested_device}")
        else:
            print(f"[OK] Using default CUDA device: cuda:0")
            requested_device = "cuda:0"
    else:
        print(f"[INFO] Using CPU as requested")
    
    # Set device and do a quick test
    device = torch.device(requested_device)
    try:
        test_tensor = torch.zeros(1, device=device)
        del test_tensor
        print(f"[OK] Device test passed: {device}")
    except Exception as e:
        print(f"[ERROR] Failed to use device {device}: {e}")
        raise
    
    print("=" * 60)
    return requested_device

# ----------------------------------------------------------------------
#   Projection Head Architecture
# ----------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    Learnable projection head that transforms genomic features into
    image-feature-like conditioning vectors for the diffusion model.
    
    Architecture options:
    - linear: Single linear layer (baseline)
    - mlp: Multi-layer perceptron with ReLU activations
    - residual: MLP with residual connection (good when dims match)
    - attention: Self-attention based projection
    """
    
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
        
        # Validate inputs
        assert in_dim > 0, f"in_dim must be positive, got {in_dim}"
        assert out_dim > 0, f"out_dim must be positive, got {out_dim}"
        assert hidden_dim > 0, f"hidden_dim must be positive, got {hidden_dim}"
        assert num_layers >= 1, f"num_layers must be >= 1, got {num_layers}"
        assert 0.0 <= dropout < 1.0, f"dropout must be in [0, 1), got {dropout}"
        assert arch in ["linear", "mlp", "residual", "attention"], \
            f"Unknown architecture: {arch}. Must be one of: linear, mlp, residual, attention"
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.arch = arch
        self.normalize_output = normalize_output
        
        print(f"[ProjectionHead] Initializing {arch} architecture:")
        print(f"  - Input dim: {in_dim}")
        print(f"  - Hidden dim: {hidden_dim}")
        print(f"  - Output dim: {out_dim}")
        print(f"  - Num layers: {num_layers}")
        print(f"  - Dropout: {dropout}")
        print(f"  - Normalize output: {normalize_output}")
        
        if arch == "linear":
            self.net = nn.Linear(in_dim, out_dim)
            
        elif arch == "mlp":
            layers = []
            dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:  # No activation after last layer
                    layers.append(nn.LayerNorm(dims[i + 1]))
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(dropout))
            self.net = nn.Sequential(*layers)
            
        elif arch == "residual":
            assert in_dim == out_dim, f"Residual requires in_dim == out_dim, got {in_dim} != {out_dim}"
            layers = []
            for i in range(num_layers):
                layers.append(nn.Linear(hidden_dim if i > 0 else in_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.net = nn.Sequential(*layers)
            self.skip = nn.Identity() if in_dim == hidden_dim else nn.Linear(in_dim, out_dim)
            
        elif arch == "attention":
            self.input_proj = nn.Linear(in_dim, hidden_dim)
            self.attention = nn.MultiheadAttention(
                hidden_dim, num_heads=8, dropout=dropout, batch_first=True
            )
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, out_dim)
            self.net = nn.Identity()  # Placeholder, not used in attention arch
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        # Print parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Genomic features of shape (B, in_dim)
        Returns:
            Projected features of shape (B, out_dim)
        """
        # Input validation
        assert x.dim() == 2, f"Expected 2D input (B, D), got shape {x.shape}"
        assert x.shape[1] == self.in_dim, \
            f"Input dim mismatch: expected {self.in_dim}, got {x.shape[1]}"
        
        if self.arch == "attention":
            # Treat each feature as a sequence of length 1
            h = self.input_proj(x).unsqueeze(1)  # (B, 1, hidden_dim)
            attn_out, _ = self.attention(h, h, h)
            h = self.norm1(h + attn_out)
            h = self.norm2(h + self.ffn(h))
            out = self.output_proj(h.squeeze(1))
        elif self.arch == "residual":
            out = self.net(x) + self.skip(x)
        else:
            out = self.net(x)
        
        if self.normalize_output:
            out = F.normalize(out, p=2, dim=-1)
        
        return out


# ----------------------------------------------------------------------
#   Helper functions
# ----------------------------------------------------------------------

def canonical_patient_id(name: str) -> str:
    """Extract TCGA-XX-XXXX patient ID from various filename formats."""
    name = Path(name).stem.upper()
    # Normalize separators
    for sep in ("_", "."):
        name = name.replace(sep, "-")
    while "--" in name:
        name = name.replace("--", "-")
    parts = name.split("-")
    if len(parts) >= 3 and parts[0].startswith("TCGA"):
        return "-".join(parts[:3])
    return name


def parse_tile_coords_from_filename(filename: str) -> Tuple[float, float]:
    """
    Parse tile coordinates from filenames like:
    - 'tile_(1234.5, 6789.0).png'
    - 'tile_1234_6789.png'
    """
    base = Path(filename).stem
    # Try pattern: tile_(x, y)
    match = re.search(r'\((-?[\d.]+),\s*(-?[\d.]+)\)', base)
    if match:
        return float(match.group(1)), float(match.group(2))
    # Try pattern: tile_x_y
    match = re.search(r'tile_(-?[\d.]+)_(-?[\d.]+)', base)
    if match:
        return float(match.group(1)), float(match.group(2))
    return 0.0, 0.0


def get_tile_coords_from_zip(zip_path: Path) -> np.ndarray:
    """Extract all tile coordinates from a zip file."""
    coords = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                x, y = parse_tile_coords_from_filename(name)
                coords.append([x, y])
    return np.array(coords, dtype=np.float32) if coords else np.zeros((0, 2), dtype=np.float32)


# ----------------------------------------------------------------------
#   Datasets
# ----------------------------------------------------------------------

class GenomicOnlyDataset(Dataset):
    """
    Dataset for DISTRIBUTION_MATCHING mode.
    Uses only genomic h5 files - no image features needed.
    
    Supports directory structures:
    1. Flat: genomic_h5_dir/*.h5
    2. Split subdirs: genomic_h5_dir/{train,test}/*.h5
    3. CSV-based: uses clinical_table.csv with 'PATIENT' and 'split' columns
    """
    
    def __init__(
        self,
        genomic_h5_dir: str,
        genomic_key: str = "feats",
        split: str = "train",  # "train", "test", or "all"
        clinical_csv: Optional[str] = None,  # Path to clinical_table.csv
    ):
        super().__init__()
        
        print("=" * 60)
        print("LOADING GENOMIC DATASET")
        print("=" * 60)
        
        self.genomic_h5_dir = Path(genomic_h5_dir).expanduser()
        self.genomic_key = genomic_key
        self.split = split
        
        # Validate directory exists
        assert self.genomic_h5_dir.exists(), \
            f"Genomic H5 directory does not exist: {self.genomic_h5_dir}"
        assert self.genomic_h5_dir.is_dir(), \
            f"Path is not a directory: {self.genomic_h5_dir}"
        
        print(f"[INFO] Genomic H5 directory: {self.genomic_h5_dir}")
        print(f"[INFO] Split: {split}")
        print(f"[INFO] Feature key: {genomic_key}")
        
        # Try to find H5 files based on directory structure
        self.h5_files = self._find_h5_files(clinical_csv)
        
        if not self.h5_files:
            raise RuntimeError(
                f"No H5 files found!\n"
                f"  Directory: {self.genomic_h5_dir}\n"
                f"  Split: {split}\n"
                f"  Clinical CSV: {clinical_csv}\n"
                f"Please check:\n"
                f"  1. The directory contains .h5 files\n"
                f"  2. If using splits, ensure 'train/' or 'test/' subdirs exist\n"
                f"  3. If using CSV, ensure 'PATIENT' and 'split' columns exist"
            )
        
        # Validate first file to check key exists
        first_file = self.h5_files[0]
        with h5py.File(first_file, "r") as f:
            available_keys = list(f.keys())
            assert self.genomic_key in available_keys, \
                f"Key '{self.genomic_key}' not found in {first_file}. Available keys: {available_keys}"
            
            # Get feature dimension
            sample = np.array(f[self.genomic_key])
            if sample.ndim == 2:
                feat_dim = sample.shape[1]
            else:
                feat_dim = sample.shape[0]
            print(f"[INFO] Feature dimension: {feat_dim}")
        
        print(f"[OK] Found {len(self.h5_files)} genomic H5 files (split={split})")
        print("=" * 60)
        
        print(f"[GenomicOnlyDataset] Found {len(self.h5_files)} genomic H5 files (split={split})")
    
    def _find_h5_files(self, clinical_csv: Optional[str]) -> List[Path]:
        """Find H5 files based on directory structure or CSV."""
        
        # Option 1: Check for train/test subdirectories
        train_dir = self.genomic_h5_dir / "train"
        test_dir = self.genomic_h5_dir / "test"
        
        if train_dir.is_dir() or test_dir.is_dir():
            print(f"[INFO] Found train/test subdirectory structure")
            # Subdirectory structure exists
            if self.split == "train" and train_dir.is_dir():
                files = sorted(train_dir.glob("*.h5"))
                print(f"[INFO] Using train subdir: {train_dir} ({len(files)} files)")
                return files
            elif self.split == "test" and test_dir.is_dir():
                files = sorted(test_dir.glob("*.h5"))
                print(f"[INFO] Using test subdir: {test_dir} ({len(files)} files)")
                return files
            elif self.split == "all":
                files = []
                if train_dir.is_dir():
                    files.extend(train_dir.glob("*.h5"))
                if test_dir.is_dir():
                    files.extend(test_dir.glob("*.h5"))
                print(f"[INFO] Using all files from train+test subdirs ({len(files)} files)")
                return sorted(files)
            else:
                # Fallback to the specified split dir
                split_dir = self.genomic_h5_dir / self.split
                if split_dir.is_dir():
                    files = sorted(split_dir.glob("*.h5"))
                    print(f"[INFO] Using custom split subdir: {split_dir} ({len(files)} files)")
                    return files
        
        # Option 2: Use clinical CSV if provided
        if clinical_csv:
            csv_path = Path(clinical_csv).expanduser()
            if not csv_path.exists():
                # Try looking in the genomic_h5_dir
                csv_path = self.genomic_h5_dir / "clinical_table.csv"
            
            if csv_path.exists():
                print(f"[INFO] Using clinical CSV: {csv_path}")
                import pandas as pd
                df = pd.read_csv(csv_path)
                
                # Normalize column names (handle variations)
                df.columns = df.columns.str.strip().str.upper()
                
                if "PATIENT" in df.columns and "SPLIT" in df.columns:
                    if self.split != "all":
                        df = df[df["SPLIT"].str.lower() == self.split.lower()]
                    
                    patient_ids = set(df["PATIENT"].astype(str).str.upper())
                    print(f"[INFO] Found {len(patient_ids)} patients in CSV for split='{self.split}'")
                    
                    # Find all H5 files and filter by patient ID
                    all_h5 = list(self.genomic_h5_dir.glob("*.h5"))
                    # Also check subdirs
                    all_h5.extend(self.genomic_h5_dir.glob("*/*.h5"))
                    
                    filtered = []
                    for h5_path in all_h5:
                        pid = canonical_patient_id(h5_path.name)
                        if pid in patient_ids:
                            filtered.append(h5_path)
                    
                    print(f"[INFO] Matched {len(filtered)} H5 files with CSV patient IDs")
                    return sorted(filtered)
                else:
                    print(f"[WARNING] CSV missing 'PATIENT' or 'split' columns. Found: {list(df.columns)}")
        
        # Option 3: Check for clinical_table.csv in the directory automatically
        auto_csv = self.genomic_h5_dir / "clinical_table.csv"
        if auto_csv.exists() and clinical_csv is None:
            import pandas as pd
            df = pd.read_csv(auto_csv)
            df.columns = df.columns.str.strip().str.upper()
            
            if "PATIENT" in df.columns and "SPLIT" in df.columns:
                if self.split != "all":
                    df = df[df["SPLIT"].str.lower() == self.split.lower()]
                
                patient_ids = set(df["PATIENT"].astype(str).str.upper())
                
                all_h5 = list(self.genomic_h5_dir.glob("*.h5"))
                all_h5.extend(self.genomic_h5_dir.glob("*/*.h5"))
                
                filtered = []
                for h5_path in all_h5:
                    pid = canonical_patient_id(h5_path.name)
                    if pid in patient_ids:
                        filtered.append(h5_path)
                
                if filtered:
                    return sorted(filtered)
        
        # Option 4: Flat directory (no splits) - use all files
        return sorted(self.genomic_h5_dir.glob("*.h5"))
    
    def __len__(self):
        return len(self.h5_files)
    
    def __getitem__(self, idx):
        h5_path = self.h5_files[idx]
        
        with h5py.File(h5_path, "r") as f:
            genomic = np.array(f[self.genomic_key])
            if genomic.ndim == 2:
                genomic = genomic.mean(axis=0)
        
        return (
            torch.from_numpy(genomic.astype(np.float32)),
            str(h5_path.stem),
        )


class GenomicImageFeatureDataset(Dataset):
    """
    Dataset for FEATURE_ALIGNMENT mode.
    Matches genomic h5 files with image feature h5 files by patient ID.
    """
    
    def __init__(
        self,
        genomic_h5_dir: str,
        image_h5_dir: str,
        genomic_key: str = "feats",
        image_key: str = "feats",
        random_tile: bool = True,
    ):
        super().__init__()
        self.genomic_h5_dir = Path(genomic_h5_dir).expanduser()
        self.image_h5_dir = Path(image_h5_dir).expanduser()
        self.genomic_key = genomic_key
        self.image_key = image_key
        self.random_tile = random_tile
        
        # Find matching files
        genomic_files = {canonical_patient_id(f.name): f 
                         for f in self.genomic_h5_dir.glob("*.h5")}
        image_files = {canonical_patient_id(f.name): f 
                       for f in self.image_h5_dir.glob("*.h5")}
        
        common_ids = sorted(set(genomic_files) & set(image_files))
        if not common_ids:
            raise RuntimeError(
                f"No matching patient IDs between\n"
                f"  Genomic: {self.genomic_h5_dir}\n"
                f"  Image: {self.image_h5_dir}"
            )
        
        print(f"[GenomicImageFeatureDataset] Found {len(common_ids)} matched patients")
        self.pairs = [(genomic_files[pid], image_files[pid]) for pid in common_ids]
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        genomic_path, image_path = self.pairs[idx]
        
        # Load genomic features (one per patient)
        with h5py.File(genomic_path, "r") as f:
            genomic = np.array(f[self.genomic_key])
            if genomic.ndim == 2:
                genomic = genomic.mean(axis=0)  # Average if multiple
        
        # Load image features (one per tile)
        with h5py.File(image_path, "r") as f:
            image_feats = np.array(f[self.image_key])
            if image_feats.ndim == 1:
                image_feat = image_feats
            else:
                # Random tile or average
                if self.random_tile:
                    idx_tile = random.randint(0, len(image_feats) - 1)
                    image_feat = image_feats[idx_tile]
                else:
                    image_feat = image_feats.mean(axis=0)
        
        return (
            torch.from_numpy(genomic.astype(np.float32)),
            torch.from_numpy(image_feat.astype(np.float32)),
            str(genomic_path.stem),
        )


class GenomicReconstructionDataset(Dataset):
    """
    Dataset for RECONSTRUCTION mode.
    Uses genomic h5 files and tile images from zip files.
    """
    
    def __init__(
        self,
        genomic_h5_dir: str,
        tiles_zip_dir: str,
        genomic_key: str = "feats",
        transform=None,
    ):
        super().__init__()
        self.genomic_h5_dir = Path(genomic_h5_dir).expanduser()
        self.tiles_zip_dir = Path(tiles_zip_dir).expanduser()
        self.genomic_key = genomic_key
        self.transform = transform
        
        # Find matching files
        genomic_files = {canonical_patient_id(f.name): f 
                         for f in self.genomic_h5_dir.glob("*.h5")}
        zip_files = {canonical_patient_id(f.name): f 
                     for f in self.tiles_zip_dir.glob("*.zip")}
        
        common_ids = sorted(set(genomic_files) & set(zip_files))
        if not common_ids:
            raise RuntimeError("No matching patient IDs found")
        
        print(f"[GenomicReconstructionDataset] Found {len(common_ids)} matched patients")
        self.pairs = [(genomic_files[pid], zip_files[pid]) for pid in common_ids]
    
    def __len__(self):
        return len(self.pairs)
    
    def _load_random_tile(self, zip_path: Path) -> Image.Image:
        with zipfile.ZipFile(zip_path, "r") as zf:
            candidates = [n for n in zf.namelist() 
                          if n.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not candidates:
                raise RuntimeError(f"No images in {zip_path}")
            inner = random.choice(candidates)
            with zf.open(inner) as f:
                return Image.open(BytesIO(f.read())).convert("RGB")
    
    def __getitem__(self, idx):
        genomic_path, zip_path = self.pairs[idx]
        
        # Load genomic features
        with h5py.File(genomic_path, "r") as f:
            genomic = np.array(f[self.genomic_key])
            if genomic.ndim == 2:
                genomic = genomic.mean(axis=0)
        
        # Load random tile
        img = self._load_random_tile(zip_path)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        return (
            torch.from_numpy(genomic.astype(np.float32)),
            img,
            str(genomic_path.stem),
        )


# ----------------------------------------------------------------------
#   Loss Functions
# ----------------------------------------------------------------------

class FeatureAlignmentLoss(nn.Module):
    """
    Loss for aligning projected genomic features with image features.
    """
    
    def __init__(
        self,
        mode: str = "mse",  # "mse", "cosine", "infonce"
        temperature: float = 0.07,
    ):
        super().__init__()
        self.mode = mode
        self.temperature = temperature
    
    def forward(
        self,
        projected: torch.Tensor,  # (B, D) - projected genomic
        target: torch.Tensor,     # (B, D) - image features
    ) -> torch.Tensor:
        
        if self.mode == "mse":
            return F.mse_loss(projected, target)
        
        elif self.mode == "cosine":
            # Cosine similarity loss
            projected_norm = F.normalize(projected, p=2, dim=-1)
            target_norm = F.normalize(target, p=2, dim=-1)
            return 1 - (projected_norm * target_norm).sum(dim=-1).mean()
        
        elif self.mode == "infonce":
            # InfoNCE / contrastive loss
            projected_norm = F.normalize(projected, p=2, dim=-1)
            target_norm = F.normalize(target, p=2, dim=-1)
            
            # Positive pairs are same-patient genomic-image
            logits = projected_norm @ target_norm.T / self.temperature
            labels = torch.arange(len(projected), device=projected.device)
            
            loss_g2i = F.cross_entropy(logits, labels)
            loss_i2g = F.cross_entropy(logits.T, labels)
            
            return (loss_g2i + loss_i2g) / 2
        
        else:
            raise ValueError(f"Unknown loss mode: {self.mode}")


# ----------------------------------------------------------------------
#   Training Functions
# ----------------------------------------------------------------------

def train_feature_alignment(
    projection_head: ProjectionHead,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    args,
    device: str,
):
    """Train projection head using feature alignment loss."""
    
    optimizer = torch.optim.AdamW(
        projection_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    loss_fn = FeatureAlignmentLoss(mode=args.loss_mode, temperature=args.temperature)
    
    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        # Training
        projection_head.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for genomic, image_feat, _ in pbar:
            genomic = genomic.to(device)
            image_feat = image_feat.to(device)
            
            # Forward
            projected = projection_head(genomic)
            loss = loss_fn(projected, image_feat)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projection_head.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Validation
        val_loss = 0.0
        if val_loader:
            projection_head.eval()
            with torch.no_grad():
                for genomic, image_feat, _ in val_loader:
                    genomic = genomic.to(device)
                    image_feat = image_feat.to(device)
                    projected = projection_head(genomic)
                    val_loss += loss_fn(projected, image_feat).item()
            val_loss /= len(val_loader)
        
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={scheduler.get_last_lr()[0]:.2e}")
        
        # Save best
        if val_loss < best_val_loss or not val_loader:
            best_val_loss = val_loss if val_loader else train_loss
            save_path = Path(args.out_dir) / "projection_head_best.pt"
            torch.save({
                "epoch": epoch,
                "state_dict": projection_head.state_dict(),
                "val_loss": best_val_loss,
                "config": {
                    "in_dim": projection_head.in_dim,
                    "out_dim": projection_head.out_dim,
                    "arch": projection_head.arch,
                },
            }, save_path)
            print(f"  → Saved best model to {save_path}")
    
    return projection_head


def train_reconstruction(
    projection_head: ProjectionHead,
    train_loader: DataLoader,
    args,
    device: str,
):
    """Train projection head using reconstruction loss with diffusion model."""
    
    # Load MoPaDi diffusion model
    try:
        from mopadi.configs.templates import tcga_brca_autoenc
        from mopadi.utils.encode import ImageEncoder
    except ImportError as e:
        raise RuntimeError("Failed to import MoPaDi. Ensure it's installed.") from e
    
    diffusion = ImageEncoder(
        tcga_brca_autoenc(),
        autoenc_path=args.diffusion_ckpt,
        device=device,
        feat_extractor=None,
    )
    diffusion.model.ema_model.eval()
    for p in diffusion.model.parameters():
        p.requires_grad = False
    
    optimizer = torch.optim.AdamW(
        projection_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    cond_dim = getattr(diffusion.model.conf, "feat_dim", 512)
    
    for epoch in range(1, args.epochs + 1):
        projection_head.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for genomic, img, _ in pbar:
            genomic = genomic.to(device)
            img = img.to(device)
            B = img.shape[0]
            
            # Project genomic features
            cond_pred = projection_head(genomic)
            
            # Encode image to noise (using neutral conditioning)
            neutral = torch.zeros(B, cond_dim, device=device)
            with torch.no_grad():
                x_T = diffusion.encode_to_noise(img, neutral, T=args.encode_steps)
            
            # Decode using projected conditioning
            recon = diffusion.decode_image(x_T, cond_pred, T=args.decode_steps)
            
            # Reconstruction loss
            loss = F.mse_loss(recon, (img + 1) / 2)  # decode_image returns [0,1]
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projection_head.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        
        print(f"Epoch {epoch}: loss={epoch_loss/len(train_loader):.6f}")
        
        # Save checkpoint
        save_path = Path(args.out_dir) / f"projection_head_epoch{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "state_dict": projection_head.state_dict(),
        }, save_path)
    
    return projection_head


def train_distribution_matching(
    projection_head: ProjectionHead,
    train_loader: DataLoader,
    args,
    device: str,
):
    """
    Train projection head using ONLY genomic features.
    
    The idea: project genomic features into a distribution that matches the
    conditioning space the diffusion model expects. We use several losses:
    
    1. MEAN MATCHING: Match the mean of projected features to the diffusion
       model's learned conditioning mean (conds_mean)
    
    2. VARIANCE REGULARIZATION: Ensure projected features have reasonable
       variance (not collapsed to a point)
    
    3. DIVERSITY LOSS: Encourage different patients to have different projections
       (prevents mode collapse)
    
    4. SMOOTHNESS: Regularize the projection to be smooth
    """
    import time
    
    print("\n" + "-" * 50)
    print("DISTRIBUTION MATCHING TRAINING")
    print("-" * 50)
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Num batches: {len(train_loader)}")
    print(f"  Num samples: {len(train_loader.dataset)}")
    print("-" * 50 + "\n")
    
    # Load MoPaDi diffusion model to get conditioning statistics
    try:
        from mopadi.configs.templates import tcga_brca_autoenc
        from mopadi.utils.encode import ImageEncoder
    except ImportError as e:
        raise RuntimeError(
            "Failed to import MoPaDi. Ensure mopadi is installed:\n"
            "  pip install -e /path/to/mopadi\n"
            f"Original error: {e}"
        ) from e
    
    print("[STEP 1/3] Loading diffusion model for conditioning statistics...")
    start_time = time.time()
    
    diffusion = ImageEncoder(
        tcga_brca_autoenc(),
        autoenc_path=args.diffusion_ckpt,
        device=device,
        feat_extractor=None,
    )
    diffusion.model.ema_model.eval()
    for p in diffusion.model.parameters():
        p.requires_grad = False
    
    print(f"  Diffusion model loaded in {time.time() - start_time:.2f}s")
    
    # Extract target statistics from the diffusion model
    print("[STEP 2/3] Extracting target conditioning statistics...")
    
    conds_mean = getattr(diffusion.model, "conds_mean", None)
    conds_std = getattr(diffusion.model, "conds_std", None)
    
    if conds_mean is not None:
        if not isinstance(conds_mean, torch.Tensor):
            conds_mean = torch.tensor(conds_mean, dtype=torch.float32)
        conds_mean = conds_mean.to(device)
        if conds_mean.dim() == 2:
            conds_mean = conds_mean.mean(dim=0)  # Average over patches if needed
        print(f"  [OK] conds_mean shape: {conds_mean.shape}")
        print(f"       conds_mean range: [{conds_mean.min().item():.4f}, {conds_mean.max().item():.4f}]")
    else:
        # Fallback: use zero mean (the model was trained on centered features)
        conds_mean = torch.zeros(args.out_dim, device=device)
        print("  [WARN] No conds_mean found, using zero mean as target")
    
    if conds_std is not None:
        if not isinstance(conds_std, torch.Tensor):
            conds_std = torch.tensor(conds_std, dtype=torch.float32)
        conds_std = conds_std.to(device)
        if conds_std.dim() == 2:
            conds_std = conds_std.mean(dim=0)
        print(f"  [OK] conds_std shape: {conds_std.shape}")
        print(f"       conds_std range: [{conds_std.min().item():.4f}, {conds_std.max().item():.4f}]")
    else:
        # Fallback: assume unit variance
        conds_std = torch.ones(args.out_dim, device=device)
        print("  [WARN] No conds_std found, using unit std as target")
    
    print("[STEP 3/3] Setting up optimizer and starting training...")
    print(f"  Loss weights: mean={args.weight_mean}, var={args.weight_var}, diversity={args.weight_diversity}")
    print(f"  Target diversity: {args.target_diversity}")
    print()
    
    optimizer = torch.optim.AdamW(
        projection_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    best_loss = float("inf")
    training_start_time = time.time()
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING LOOP")
    print("=" * 60 + "\n")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        projection_head.train()
        epoch_losses = {"total": 0.0, "mean": 0.0, "var": 0.0, "diversity": 0.0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            genomic = batch[0].to(device)  # (B, D)
            B = genomic.shape[0]
            
            # Verify input on first batch of first epoch
            if epoch == 1 and batch_idx == 0:
                print(f"\n[DEBUG] First batch info:")
                print(f"  Batch shape: {genomic.shape}")
                print(f"  Batch dtype: {genomic.dtype}")
                print(f"  Batch device: {genomic.device}")
                print(f"  Value range: [{genomic.min().item():.4f}, {genomic.max().item():.4f}]")
            
            # Project genomic features
            projected = projection_head(genomic)  # (B, out_dim)
            
            # ============ LOSS 1: Mean Matching ============
            # Push the mean of projected features toward the target mean
            batch_mean = projected.mean(dim=0)
            loss_mean = F.mse_loss(batch_mean, conds_mean)
            
            # ============ LOSS 2: Variance Regularization ============
            # Encourage the projected features to have similar variance as target
            batch_std = projected.std(dim=0) + 1e-6
            loss_var = F.mse_loss(batch_std, conds_std)
            
            # ============ LOSS 3: Diversity Loss ============
            # Prevent all projections from collapsing to the same point
            # Use pairwise distances within the batch
            if B > 1:
                # Compute pairwise L2 distances
                dists = torch.cdist(projected, projected, p=2)
                # Mask diagonal (self-distances)
                mask = ~torch.eye(B, dtype=torch.bool, device=device)
                mean_dist = dists[mask].mean()
                # Encourage distances to be non-zero (hinge loss)
                target_dist = args.target_diversity  # hyperparameter
                loss_diversity = F.relu(target_dist - mean_dist)
            else:
                loss_diversity = torch.tensor(0.0, device=device)
            
            # ============ Total Loss ============
            loss = (
                args.weight_mean * loss_mean +
                args.weight_var * loss_var +
                args.weight_diversity * loss_diversity
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projection_head.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses["total"] += loss.item()
            epoch_losses["mean"] += loss_mean.item()
            epoch_losses["var"] += loss_var.item()
            epoch_losses["diversity"] += loss_diversity.item()
            
            pbar.set_postfix({
                "loss": loss.item(),
                "mean": loss_mean.item(),
                "var": loss_var.item(),
            })
        
        scheduler.step()
        epoch_time = time.time() - epoch_start_time
        n_batches = len(train_loader)
        
        # Print epoch summary with timing
        print(f"\nEpoch {epoch}/{args.epochs} | Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"  Losses: total={epoch_losses['total']/n_batches:.6f}, "
              f"mean={epoch_losses['mean']/n_batches:.6f}, "
              f"var={epoch_losses['var']/n_batches:.6f}, "
              f"diversity={epoch_losses['diversity']/n_batches:.6f}")
        
        # Print GPU memory usage if on CUDA
        if device.startswith("cuda") and torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f"  GPU Memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
        
        # Save best
        epoch_total = epoch_losses["total"] / n_batches
        if epoch_total < best_loss:
            best_loss = epoch_total
            save_path = Path(args.out_dir) / "projection_head_best.pt"
            torch.save({
                "epoch": epoch,
                "state_dict": projection_head.state_dict(),
                "loss": best_loss,
                "config": {
                    "in_dim": projection_head.in_dim,
                    "out_dim": projection_head.out_dim,
                    "arch": projection_head.arch,
                },
                "target_mean": conds_mean.cpu(),
                "target_std": conds_std.cpu(),
            }, save_path)
            print(f"  ★ NEW BEST! Saved to {save_path}")
        
        # Save periodic checkpoint
        if epoch % 10 == 0 or epoch == args.epochs:
            save_path = Path(args.out_dir) / f"projection_head_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "state_dict": projection_head.state_dict(),
                "loss": epoch_total,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "config": {
                    "in_dim": projection_head.in_dim,
                    "out_dim": projection_head.out_dim,
                    "arch": projection_head.arch,
                },
            }, save_path)
            print(f"  Checkpoint saved: {save_path}")
    
    total_time = time.time() - training_start_time
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Output dir: {args.out_dir}")
    print("=" * 60 + "\n")
    
    return projection_head


# ----------------------------------------------------------------------
#   Conversion: Genomic → Pseudo Image-Feature H5
# ----------------------------------------------------------------------

def convert_genomic_to_image_format(
    projection_head: ProjectionHead,
    genomic_h5_dir: Path,
    tiles_zip_dir: Path,
    output_h5_dir: Path,
    genomic_key: str = "feats",
    device: str = "cpu",
):
    """
    Convert genomic feature H5 files to image-feature-like H5 format.
    
    For each patient:
    1. Load genomic vector and pass through projection head
    2. Get tile coordinates from the corresponding zip file
    3. Replicate projected vector for all tile positions
    4. Save as H5 with 'coords' and 'feats' keys
    """
    
    output_h5_dir = Path(output_h5_dir)
    output_h5_dir.mkdir(parents=True, exist_ok=True)
    
    # Match files
    genomic_files = {canonical_patient_id(f.name): f 
                     for f in Path(genomic_h5_dir).glob("*.h5")}
    zip_files = {canonical_patient_id(f.name): f 
                 for f in Path(tiles_zip_dir).glob("*.zip")}
    
    common_ids = sorted(set(genomic_files) & set(zip_files))
    print(f"Converting {len(common_ids)} patients...")
    
    projection_head.eval()
    
    for pid in tqdm(common_ids):
        genomic_path = genomic_files[pid]
        zip_path = zip_files[pid]
        
        # Load and project genomic features
        with h5py.File(genomic_path, "r") as f:
            genomic = np.array(f[genomic_key])
            if genomic.ndim == 2:
                genomic = genomic.mean(axis=0)
        
        with torch.no_grad():
            genomic_t = torch.from_numpy(genomic.astype(np.float32)).unsqueeze(0).to(device)
            projected = projection_head(genomic_t).cpu().numpy().squeeze(0)
        
        # Get tile coordinates
        coords = get_tile_coords_from_zip(zip_path)
        n_tiles = len(coords)
        
        if n_tiles == 0:
            print(f"Warning: No tiles found in {zip_path}, skipping")
            continue
        
        # Replicate projected features for all tiles
        feats = np.tile(projected, (n_tiles, 1))
        
        # Save
        out_path = output_h5_dir / f"{pid}.h5"
        with h5py.File(out_path, "w") as f:
            f.create_dataset("coords", data=coords, dtype=np.float32)
            f.create_dataset("feats", data=feats, dtype=np.float32)
        
    print(f"Saved {len(common_ids)} H5 files to {output_h5_dir}")


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train projection head for genomic-to-image feature alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="distribution_matching",
                        choices=["feature_alignment", "reconstruction", "distribution_matching", "convert"],
                        help="Training mode or conversion")
    parser.add_argument("--convert", action="store_true",
                        help="Convert mode (shortcut for --mode convert)")
    
    # Data paths
    parser.add_argument("--genomic-h5-dir", type=str, required=True,
                        help="Directory with genomic feature H5 files (can have train/test subdirs)")
    parser.add_argument("--image-h5-dir", type=str, default=None,
                        help="Directory with image feature H5 files (for feature_alignment)")
    parser.add_argument("--tiles-zip-dir", type=str, default=None,
                        help="Directory with tile zip files (for reconstruction/convert)")
    parser.add_argument("--output-h5-dir", type=str, default=None,
                        help="Output directory for converted H5 files")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test", "all"],
                        help="Which split to use for training (default: train)")
    parser.add_argument("--clinical-csv", type=str, default=None,
                        help="Path to clinical_table.csv with PATIENT and split columns")
    
    # Model architecture
    parser.add_argument("--in-dim", type=int, default=512,
                        help="Genomic feature dimension")
    parser.add_argument("--out-dim", type=int, default=512,
                        help="Output (image feature) dimension")
    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="Hidden layer dimension")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of layers")
    parser.add_argument("--arch", type=str, default="mlp",
                        choices=["linear", "mlp", "residual", "attention"],
                        help="Projection head architecture")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--normalize-output", action="store_true",
                        help="L2-normalize output features")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--loss-mode", type=str, default="mse",
                        choices=["mse", "cosine", "infonce"],
                        help="Loss function for feature alignment")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Temperature for InfoNCE loss")
    parser.add_argument("--genomic-key", type=str, default="feats")
    parser.add_argument("--image-key", type=str, default="feats")
    
    # Reconstruction mode
    parser.add_argument("--diffusion-ckpt", type=str, default=None,
                        help="Diffusion model checkpoint (for reconstruction/distribution_matching mode)")
    parser.add_argument("--encode-steps", type=int, default=250)
    parser.add_argument("--decode-steps", type=int, default=20)
    
    # Distribution matching mode (genomic-only training)
    parser.add_argument("--weight-mean", type=float, default=1.0,
                        help="Weight for mean matching loss")
    parser.add_argument("--weight-var", type=float, default=0.5,
                        help="Weight for variance matching loss")
    parser.add_argument("--weight-diversity", type=float, default=0.1,
                        help="Weight for diversity loss (prevents collapse)")
    parser.add_argument("--target-diversity", type=float, default=1.0,
                        help="Target pairwise distance for diversity loss")
    
    # Output
    parser.add_argument("--out-dir", type=str, default="./projection_head_output")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Load checkpoint (for convert mode)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # =============================================
    # STARTUP BANNER
    # =============================================
    print("\n" + "=" * 60)
    print("PROJECTION HEAD TRAINING FOR GENOMIC-TO-IMAGE ALIGNMENT")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    print("=" * 60 + "\n")
    
    if args.convert:
        args.mode = "convert"
    
    # Check CUDA availability and set device
    device = check_cuda_availability(args.device)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[INFO] Output directory: {args.out_dir}")
    
    # Setup logging
    logger = setup_logging(args.out_dir, verbose=args.verbose)
    logger.info(f"Starting {args.mode} mode")
    
    # =============================================
    # VALIDATE ARGUMENTS
    # =============================================
    print("\n" + "=" * 60)
    print("VALIDATING ARGUMENTS")
    print("=" * 60)
    
    # Validate genomic-h5-dir exists
    genomic_dir = Path(args.genomic_h5_dir).expanduser()
    assert genomic_dir.exists(), f"Genomic H5 directory does not exist: {genomic_dir}"
    print(f"[OK] Genomic H5 dir exists: {genomic_dir}")
    
    # Validate mode-specific arguments
    if args.mode == "distribution_matching":
        assert args.diffusion_ckpt is not None, \
            "--diffusion-ckpt is required for distribution_matching mode"
        ckpt_path = Path(args.diffusion_ckpt).expanduser()
        assert ckpt_path.exists(), f"Diffusion checkpoint not found: {ckpt_path}"
        print(f"[OK] Diffusion checkpoint exists: {ckpt_path}")
        
    elif args.mode == "reconstruction":
        assert args.diffusion_ckpt is not None, \
            "--diffusion-ckpt is required for reconstruction mode"
        assert args.tiles_zip_dir is not None, \
            "--tiles-zip-dir is required for reconstruction mode"
        ckpt_path = Path(args.diffusion_ckpt).expanduser()
        tiles_dir = Path(args.tiles_zip_dir).expanduser()
        assert ckpt_path.exists(), f"Diffusion checkpoint not found: {ckpt_path}"
        assert tiles_dir.exists(), f"Tiles zip directory not found: {tiles_dir}"
        print(f"[OK] Diffusion checkpoint exists: {ckpt_path}")
        print(f"[OK] Tiles zip dir exists: {tiles_dir}")
        
    elif args.mode == "feature_alignment":
        assert args.image_h5_dir is not None, \
            "--image-h5-dir is required for feature_alignment mode"
        image_dir = Path(args.image_h5_dir).expanduser()
        assert image_dir.exists(), f"Image H5 directory not found: {image_dir}"
        print(f"[OK] Image H5 dir exists: {image_dir}")
        
    elif args.mode == "convert":
        assert args.checkpoint is not None, \
            "--checkpoint is required for convert mode"
        assert args.tiles_zip_dir is not None, \
            "--tiles-zip-dir is required for convert mode"
        assert args.output_h5_dir is not None, \
            "--output-h5-dir is required for convert mode"
        ckpt_path = Path(args.checkpoint).expanduser()
        tiles_dir = Path(args.tiles_zip_dir).expanduser()
        assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
        assert tiles_dir.exists(), f"Tiles zip directory not found: {tiles_dir}"
        print(f"[OK] Checkpoint exists: {ckpt_path}")
        print(f"[OK] Tiles zip dir exists: {tiles_dir}")
    
    # Validate training hyperparameters
    assert args.epochs > 0, f"epochs must be > 0, got {args.epochs}"
    assert args.batch_size > 0, f"batch_size must be > 0, got {args.batch_size}"
    assert args.lr > 0, f"lr must be > 0, got {args.lr}"
    assert 0.0 <= args.val_split < 1.0, f"val_split must be in [0, 1), got {args.val_split}"
    
    print(f"[OK] Training hyperparameters validated")
    print("=" * 60 + "\n")
    
    # =============================================
    # BUILD/LOAD PROJECTION HEAD
    # =============================================
    print("=" * 60)
    print("BUILDING PROJECTION HEAD")
    print("=" * 60)
    
    projection_head = ProjectionHead(
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        arch=args.arch,
        dropout=args.dropout,
        normalize_output=args.normalize_output,
    ).to(device)
    
    print(f"[OK] Projection head moved to device: {device}")
    
    if args.checkpoint:
        print(f"[INFO] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        projection_head.load_state_dict(ckpt["state_dict"])
        print(f"[OK] Checkpoint loaded successfully")
        if "config" in ckpt:
            print(f"[INFO] Checkpoint config: {ckpt['config']}")
    
    print("=" * 60 + "\n")
    
    # =============================================
    # EXECUTE MODE
    # =============================================
    print("=" * 60)
    print(f"EXECUTING MODE: {args.mode.upper()}")
    print("=" * 60 + "\n")
    
    if args.mode == "convert":
        print("[INFO] Convert mode: Transform genomic features to image-feature format")
        print(f"  - Input genomic H5 dir: {args.genomic_h5_dir}")
        print(f"  - Input tiles zip dir: {args.tiles_zip_dir}")
        print(f"  - Output H5 dir: {args.output_h5_dir}")
        
        convert_genomic_to_image_format(
            projection_head=projection_head,
            genomic_h5_dir=Path(args.genomic_h5_dir),
            tiles_zip_dir=Path(args.tiles_zip_dir),
            output_h5_dir=Path(args.output_h5_dir),
            genomic_key=args.genomic_key,
            device=device,
        )
        
    elif args.mode == "feature_alignment":
        print("[INFO] Feature alignment mode: Learn to map genomic -> image features")
        print(f"  - Genomic H5 dir: {args.genomic_h5_dir}")
        print(f"  - Image H5 dir: {args.image_h5_dir}")
        
        # Create dataset
        full_dataset = GenomicImageFeatureDataset(
            genomic_h5_dir=args.genomic_h5_dir,
            image_h5_dir=args.image_h5_dir,
            genomic_key=args.genomic_key,
            image_key=args.image_key,
            random_tile=True,
        )
        
        print(f"[INFO] Total samples: {len(full_dataset)}")
        
        # Split
        n_val = int(len(full_dataset) * args.val_split)
        n_train = len(full_dataset) - n_val
        print(f"[INFO] Train samples: {n_train}, Val samples: {n_val}")
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val]
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        ) if n_val > 0 else None
        
        print(f"[INFO] Train batches: {len(train_loader)}")
        if val_loader:
            print(f"[INFO] Val batches: {len(val_loader)}")
        
        train_feature_alignment(projection_head, train_loader, val_loader, args, device)
        
    elif args.mode == "reconstruction":
        print("[INFO] Reconstruction mode: Learn via image reconstruction loss")
        print(f"  - Genomic H5 dir: {args.genomic_h5_dir}")
        print(f"  - Tiles zip dir: {args.tiles_zip_dir}")
        print(f"  - Diffusion checkpoint: {args.diffusion_ckpt}")
        
        transform = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        
        dataset = GenomicReconstructionDataset(
            genomic_h5_dir=args.genomic_h5_dir,
            tiles_zip_dir=args.tiles_zip_dir,
            genomic_key=args.genomic_key,
            transform=transform,
        )
        
        print(f"[INFO] Total samples: {len(dataset)}")
        
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )
        
        print(f"[INFO] Train batches: {len(loader)}")
        
        train_reconstruction(projection_head, loader, args, device)
    
    elif args.mode == "distribution_matching":
        print("[INFO] Distribution matching mode: Match target image feature distribution")
        print(f"  - Genomic H5 dir: {args.genomic_h5_dir}")
        print(f"  - Diffusion checkpoint: {args.diffusion_ckpt}")
        print(f"  - Split: {args.split}")
        if args.clinical_csv:
            print(f"  - Clinical CSV: {args.clinical_csv}")
        
        dataset = GenomicOnlyDataset(
            genomic_h5_dir=args.genomic_h5_dir,
            genomic_key=args.genomic_key,
            split=args.split,
            clinical_csv=args.clinical_csv,
        )
        
        print(f"[INFO] Total samples: {len(dataset)}")
        
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            drop_last=True,  # Important for batch statistics
        )
        
        print(f"[INFO] Train batches: {len(loader)}")
        print(f"[INFO] Note: drop_last=True for stable batch statistics")
        
        train_distribution_matching(projection_head, loader, args, device)
    
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    print("\n" + "=" * 60)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
