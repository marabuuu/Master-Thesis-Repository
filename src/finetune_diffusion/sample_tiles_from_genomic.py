#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sample Tiles from Genomic Features using Fine-tuned Diffusion Model

This script generates synthetic tile images conditioned on genomic feature vectors.
It loads:
1. The fine-tuned diffusion model (with projection head integrated)
2. Genomic features from H5 files
And generates tile images for each patient.

Usage:
    python sample_tiles_from_genomic.py \\
        --checkpoint ./diffusion_genomic_best.pt \\
        --genomic-h5-dir /path/to/genomic_features \\
        --output-dir ./generated_tiles \\
        --num-samples-per-patient 4

Or with the original diffusion + projection head (without fine-tuning):
    python sample_tiles_from_genomic.py \\
        --diffusion-ckpt ./diffusion_without_encoder.ckpt \\
        --projection-head-ckpt ./projection_head_best.pt \\
        --genomic-h5-dir /path/to/genomic_features \\
        --output-dir ./generated_tiles
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image, make_grid
from tqdm import tqdm


# ----------------------------------------------------------------------
#   Projection Head
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


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor [-1, 1] to PIL Image."""
    tensor = tensor.detach().cpu()
    tensor = (tensor + 1) / 2  # [-1, 1] -> [0, 1]
    tensor = tensor.clamp(0, 1)
    tensor = tensor.mul(255).byte()
    tensor = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(tensor)


# ----------------------------------------------------------------------
#   Genomic Conditioned Sampler
# ----------------------------------------------------------------------

class GenomicConditionedSampler:
    """
    Wrapper that handles sampling tiles conditioned on genomic features.
    """
    
    def __init__(
        self,
        diffusion_model: nn.Module,
        projection_head: nn.Module,
        sampler,
        conds_mean: torch.Tensor,
        conds_std: torch.Tensor,
        device: str = "cuda:0",
        img_size: int = 512,
    ):
        self.diffusion_model = diffusion_model.to(device)
        self.projection_head = projection_head.to(device)
        self.sampler = sampler
        self.conds_mean = conds_mean.to(device)
        self.conds_std = conds_std.to(device)
        self.device = device
        self.img_size = img_size
        
        self.diffusion_model.eval()
        self.projection_head.eval()
    
    @torch.no_grad()
    def sample(
        self,
        genomic: torch.Tensor,
        num_samples: int = 1,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate tile images from genomic features.
        
        Args:
            genomic: (D,) or (B, D) genomic feature vector
            num_samples: number of tiles to generate per genomic vector
            noise: optional starting noise, shape (B*num_samples, 3, H, W)
        
        Returns:
            Generated images, shape (B*num_samples, 3, H, W)
        """
        if genomic.dim() == 1:
            genomic = genomic.unsqueeze(0)
        
        B = genomic.shape[0]
        genomic = genomic.to(self.device)
        
        # Project genomic features
        projected = self.projection_head(genomic)  # (B, 512)
        
        # Normalize
        cond = (projected - self.conds_mean) / (self.conds_std + 1e-6)
        
        # Expand for multiple samples per genomic vector
        if num_samples > 1:
            cond = cond.repeat_interleave(num_samples, dim=0)  # (B*num_samples, 512)
        
        total_samples = B * num_samples
        
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn(
                total_samples, 3, self.img_size, self.img_size,
                device=self.device
            )
        
        # Sample using DDIM
        model_kwargs = {"cond": cond}
        samples = self.sampler.sample(
            model=self.diffusion_model,
            noise=noise,
            cond=cond,
            model_kwargs=model_kwargs,
            progress=True,
        )
        
        return samples


# ----------------------------------------------------------------------
#   Main Functions
# ----------------------------------------------------------------------

def load_model_from_combined_checkpoint(
    ckpt_path: str,
    device: str = "cpu",
):
    """Load diffusion model and projection head from a fine-tuned combined checkpoint."""
    
    print(f"[INFO] Loading combined checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Load projection head
    proj_config = ckpt.get("projection_head_config", {})
    projection_head = ProjectionHead(
        in_dim=proj_config.get("in_dim", 512),
        out_dim=proj_config.get("out_dim", 512),
        arch=proj_config.get("arch", "mlp"),
    )
    projection_head.load_state_dict(ckpt["projection_head_state_dict"])
    print(f"[OK] Loaded projection head")
    
    # Load diffusion model
    from mopadi.configs.templates import tcga_brca_autoenc
    conf = tcga_brca_autoenc()
    model = conf.make_model_conf().make_model()
    
    # Prefer EMA weights
    if "ema_model_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_model_state_dict"])
        print("[OK] Loaded EMA model weights")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        print("[OK] Loaded model weights")
    
    # Get conditioning stats
    conds_mean = ckpt.get("conds_mean", torch.zeros(512))
    conds_std = ckpt.get("conds_std", torch.ones(512))
    
    # Create sampler
    sampler = conf.make_eval_diffusion_conf().make_sampler()
    
    return model, projection_head, sampler, conds_mean, conds_std, conf


def load_model_from_separate_checkpoints(
    diffusion_ckpt_path: str,
    projection_head_ckpt_path: str,
    device: str = "cpu",
):
    """Load from separate diffusion and projection head checkpoints."""
    
    print(f"[INFO] Loading diffusion checkpoint: {diffusion_ckpt_path}")
    print(f"[INFO] Loading projection head checkpoint: {projection_head_ckpt_path}")
    
    # Load projection head
    proj_ckpt = torch.load(projection_head_ckpt_path, map_location=device)
    proj_config = proj_ckpt.get("config", {})
    projection_head = ProjectionHead(
        in_dim=proj_config.get("in_dim", 512),
        out_dim=proj_config.get("out_dim", 512),
        arch=proj_config.get("arch", "mlp"),
    )
    projection_head.load_state_dict(proj_ckpt["state_dict"])
    print(f"[OK] Loaded projection head")
    
    # Get target mean/std from projection head checkpoint if available
    conds_mean = proj_ckpt.get("target_mean", torch.zeros(512))
    conds_std = proj_ckpt.get("target_std", torch.ones(512))
    
    # Load diffusion model
    from mopadi.configs.templates import tcga_brca_autoenc
    conf = tcga_brca_autoenc()
    model = conf.make_model_conf().make_model()
    
    diff_ckpt = torch.load(diffusion_ckpt_path, map_location=device)
    if "state_dict" in diff_ckpt:
        state_dict = diff_ckpt["state_dict"]
        # Extract model or ema_model
        model_state = {}
        ema_state = {}
        for k, v in state_dict.items():
            if k.startswith("ema_model."):
                ema_state[k[10:]] = v
            elif k.startswith("model."):
                model_state[k[6:]] = v
        
        if ema_state:
            model.load_state_dict(ema_state, strict=False)
            print("[OK] Loaded EMA model weights")
        elif model_state:
            model.load_state_dict(model_state, strict=False)
            print("[OK] Loaded model weights")
        else:
            model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(diff_ckpt, strict=False)
    
    # Try to get conds_mean/std from diffusion checkpoint
    if "conds_mean" in diff_ckpt:
        conds_mean = diff_ckpt["conds_mean"]
    elif "state_dict" in diff_ckpt and "conds_mean" in diff_ckpt["state_dict"]:
        conds_mean = diff_ckpt["state_dict"]["conds_mean"]
    
    if "conds_std" in diff_ckpt:
        conds_std = diff_ckpt["conds_std"]
    elif "state_dict" in diff_ckpt and "conds_std" in diff_ckpt["state_dict"]:
        conds_std = diff_ckpt["state_dict"]["conds_std"]
    
    # Ensure tensor format
    if not isinstance(conds_mean, torch.Tensor):
        conds_mean = torch.tensor(conds_mean, dtype=torch.float32)
    if not isinstance(conds_std, torch.Tensor):
        conds_std = torch.tensor(conds_std, dtype=torch.float32)
    
    # Create sampler
    sampler = conf.make_eval_diffusion_conf().make_sampler()
    
    return model, projection_head, sampler, conds_mean, conds_std, conf


def main():
    parser = argparse.ArgumentParser(
        description="Sample tiles from genomic features using diffusion model"
    )
    
    # Model loading options
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Combined checkpoint (fine-tuned diffusion + projection head)")
    parser.add_argument("--diffusion-ckpt", type=str, default=None,
                        help="Separate diffusion checkpoint")
    parser.add_argument("--projection-head-ckpt", type=str, default=None,
                        help="Separate projection head checkpoint")
    
    # Data
    parser.add_argument("--genomic-h5-dir", type=str, required=True,
                        help="Directory with genomic H5 files")
    parser.add_argument("--patient-ids", type=str, nargs="+", default=None,
                        help="Specific patient IDs to sample (default: all)")
    parser.add_argument("--max-patients", type=int, default=None,
                        help="Maximum number of patients to sample")
    
    # Sampling settings
    parser.add_argument("--num-samples-per-patient", type=int, default=4,
                        help="Number of tiles to generate per patient")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--save-grid", action="store_true",
                        help="Save samples as grid images")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Output
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for generated tiles")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.checkpoint is None and (args.diffusion_ckpt is None or args.projection_head_ckpt is None):
        raise ValueError(
            "Must provide either --checkpoint (combined) or "
            "both --diffusion-ckpt and --projection-head-ckpt"
        )
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Banner
    print("\n" + "=" * 60)
    print("SAMPLE TILES FROM GENOMIC FEATURES")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("=" * 60 + "\n")
    
    # Device
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = args.device
        props = torch.cuda.get_device_properties(0)
        print(f"[OK] Using GPU: {props.name}")
    else:
        device = "cpu"
        print("[WARN] Using CPU")
    
    # Load models
    print("\n" + "=" * 60)
    print("LOADING MODELS")
    print("=" * 60)
    
    if args.checkpoint:
        model, projection_head, sampler, conds_mean, conds_std, conf = \
            load_model_from_combined_checkpoint(args.checkpoint, device)
    else:
        model, projection_head, sampler, conds_mean, conds_std, conf = \
            load_model_from_separate_checkpoints(
                args.diffusion_ckpt, args.projection_head_ckpt, device
            )
    
    # Ensure conds_mean/std are 1D
    if conds_mean.dim() == 2:
        conds_mean = conds_mean.squeeze(0)
    if conds_std.dim() == 2:
        conds_std = conds_std.squeeze(0)
    
    print(f"[OK] conds_mean shape: {conds_mean.shape}")
    print(f"[OK] conds_std shape: {conds_std.shape}")
    
    # Create sampler wrapper
    genomic_sampler = GenomicConditionedSampler(
        diffusion_model=model,
        projection_head=projection_head,
        sampler=sampler,
        conds_mean=conds_mean,
        conds_std=conds_std,
        device=device,
        img_size=args.img_size,
    )
    
    # Find genomic H5 files
    print("\n" + "=" * 60)
    print("FINDING GENOMIC FILES")
    print("=" * 60)
    
    genomic_dir = Path(args.genomic_h5_dir).expanduser()
    
    # Check for train/test subdirs
    train_dir = genomic_dir / "train"
    test_dir = genomic_dir / "test"
    
    h5_files = []
    if train_dir.is_dir():
        h5_files.extend(train_dir.glob("*.h5"))
    if test_dir.is_dir():
        h5_files.extend(test_dir.glob("*.h5"))
    if not h5_files:
        h5_files = list(genomic_dir.glob("*.h5"))
    
    if args.patient_ids:
        # Filter to specific patients
        target_ids = set(p.upper() for p in args.patient_ids)
        h5_files = [f for f in h5_files if canonical_patient_id(f.name) in target_ids]
    
    if args.max_patients:
        h5_files = h5_files[:args.max_patients]
    
    print(f"[OK] Found {len(h5_files)} genomic H5 files")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate samples
    print("\n" + "=" * 60)
    print("GENERATING SAMPLES")
    print("=" * 60 + "\n")
    
    for h5_path in tqdm(h5_files, desc="Sampling"):
        pid = canonical_patient_id(h5_path.name)
        
        # Load genomic features
        with h5py.File(h5_path, "r") as f:
            genomic = np.array(f["feats"])
            if genomic.ndim == 2:
                genomic = genomic.mean(axis=0)
        
        genomic_tensor = torch.from_numpy(genomic.astype(np.float32))
        
        # Generate samples
        samples = genomic_sampler.sample(
            genomic=genomic_tensor,
            num_samples=args.num_samples_per_patient,
        )
        
        # Save
        patient_dir = output_dir / pid
        patient_dir.mkdir(exist_ok=True)
        
        if args.save_grid:
            # Save as a single grid image
            grid = make_grid(samples, nrow=int(np.sqrt(args.num_samples_per_patient)))
            grid = (grid + 1) / 2  # [-1, 1] -> [0, 1]
            save_image(grid, patient_dir / "grid.png")
        else:
            # Save individual images
            for i, sample in enumerate(samples):
                img = tensor_to_pil(sample)
                img.save(patient_dir / f"sample_{i:02d}.png")
    
    print("\n" + "=" * 60)
    print("SAMPLING COMPLETE")
    print(f"  Output: {output_dir}")
    print(f"  Patients: {len(h5_files)}")
    print(f"  Samples per patient: {args.num_samples_per_patient}")
    print(f"  Total samples: {len(h5_files) * args.num_samples_per_patient}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
