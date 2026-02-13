#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sample Tiles from Genomic Features using Fine-tuned Diffusion Model

This script generates synthetic tile images conditioned on genomic feature vectors.
It supports two sampling modes:

1. RANDOM NOISE (default): Generate tiles from pure random noise
   - Fully synthetic generation
   - High diversity, different noise = different tile
   
2. ENCODE-DECODE: Use real tiles as starting points
   - Encodes real tile → stochastic noise x_T
   - Decodes with genomic conditioning
   - Preserves structure while applying genomic-specific features

Usage:
    # Mode 1: Random noise generation (default)
    python sample_tiles_from_genomic.py \\
        --checkpoint ./diffusion_genomic_best.pt \\
        --genomic-h5-dir /path/to/genomic_features \\
        --output-dir ./generated_tiles \\
        --num-samples-per-patient 4

    # Mode 2: Encode-decode with real tiles
    python sample_tiles_from_genomic.py \\
        --checkpoint ./diffusion_genomic_best.pt \\
        --genomic-h5-dir /path/to/genomic_features \\
        --tiles-zip-dir /path/to/tile_zips \\
        --output-dir ./generated_tiles \\
        --mode encode-decode \\
        --num-samples-per-patient 4

    # With separate checkpoints (no fine-tuning)
    python sample_tiles_from_genomic.py \\
        --diffusion-ckpt ./diffusion_without_encoder.ckpt \\
        --projection-head-ckpt ./projection_head_best.pt \\
        --genomic-h5-dir /path/to/genomic_features \\
        --output-dir ./generated_tiles
"""

import argparse
import os
import random
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# Import mopadi conditionally so static analyzers don't require it at analysis time.
if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from mopadi.configs.templates import tcga_brca_autoenc  # type: ignore
else:
    try:
        from mopadi.configs.templates import tcga_brca_autoenc
    except Exception as e:
        raise RuntimeError(
            "mopadi is required for sampling but could not be imported. "
            "Install mopadi in your environment (e.g. `pip install -e /path/to/mopadi`) "
            "or activate the interpreter that has mopadi installed. "
            f"Original error: {e}"
        ) from e


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
    t = tensor.detach().cpu()
    t = (t + 1) / 2  # [-1, 1] -> [0, 1]
    t = t.clamp(0, 1)
    # Convert to [0,255] uint8 numpy image with shape (H, W, C)
    np_img = (t.mul(255)).to(torch.uint8).permute(1, 2, 0).numpy()

    # Ensure C-contiguous for PIL and type-checkers
    if not np_img.flags.c_contiguous:
        np_img = np.ascontiguousarray(np_img)

    # PIL expects HxW or HxWxC uint8/float; ensure uint8
    if np_img.dtype != np.uint8:
        np_img = np_img.astype(np.uint8)

    return Image.fromarray(np_img)


def load_tiles_from_zip(
    zip_path: Path,
    num_tiles: int,
    img_size: int = 512,
    random_sample: bool = True,
) -> Tuple[List[torch.Tensor], List[str]]:
    """
    Load tile images from a zip file.
    
    Args:
        zip_path: Path to zip file
        num_tiles: Number of tiles to load
        img_size: Size to resize tiles to
        random_sample: If True, randomly sample tiles; else take first N
    
    Returns:
        Tuple of:
            - List of tensors, each (3, H, W) in range [-1, 1]
            - List of original tile names (basenames without path)
    """
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    
    tiles = []
    tile_names = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        candidates = [n for n in zf.namelist() 
                      if n.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not candidates:
            return tiles, tile_names
        
        # Select tiles
        if random_sample and len(candidates) > num_tiles:
            selected = random.sample(candidates, num_tiles)
        else:
            selected = candidates[:num_tiles]
        
        for tile_name in selected:
            with zf.open(tile_name) as f:
                img = Image.open(BytesIO(f.read())).convert("RGB")
                tiles.append(transform(img))
                # Store just the filename (basename) for matching
                tile_names.append(Path(tile_name).name)
    
    return tiles, tile_names


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
    
    def _get_conditioning(self, genomic: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Project genomic features and normalize for conditioning."""
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
        
        return cond
    
    @torch.no_grad()
    def sample(
        self,
        genomic: torch.Tensor,
        num_samples: int = 1,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate tile images from random noise + genomic conditioning.
        
        Args:
            genomic: (D,) or (B, D) genomic feature vector
            num_samples: number of tiles to generate per genomic vector
            noise: optional starting noise, shape (B*num_samples, 3, H, W)
        
        Returns:
            Generated images, shape (B*num_samples, 3, H, W)
        """
        cond = self._get_conditioning(genomic, num_samples)
        total_samples = cond.shape[0]
        
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
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor, encode_steps: Optional[int] = None) -> torch.Tensor:
        """
        Encode images to stochastic noise representation.
        
        Uses the forward diffusion process to map x_0 → x_T.
        
        Args:
            x: Images (B, 3, H, W) in range [-1, 1]
            encode_steps: Number of encoding steps (T)
        
        Returns:
            Encoded noise x_T, shape (B, 3, H, W)
        """
        x = x.to(self.device)
        B = x.shape[0]

        # Default to full schedule if not specified
        alphas_cumprod = getattr(self.sampler, "alphas_cumprod", None)
        if alphas_cumprod is None:
            # Fallback: try to read from sampler.conf or sampler.betas
            raise RuntimeError("Sampler does not expose 'alphas_cumprod'; cannot encode")

        # Ensure alphas_cumprod is a torch tensor on the correct device
        if not isinstance(alphas_cumprod, torch.Tensor):
            alphas_cumprod = torch.tensor(np.array(alphas_cumprod), dtype=torch.float32, device=self.device)
        else:
            alphas_cumprod = alphas_cumprod.to(self.device)

        # Determine timestep to encode to (use last index of requested steps)
        max_T = int(alphas_cumprod.shape[0])
        if encode_steps is None:
            T = max_T
        else:
            T = min(int(encode_steps), max_T)
            if int(encode_steps) > max_T:
                print(f"[WARN] encode_steps={encode_steps} exceeds schedule length {max_T}; clamping to {max_T}")

        t = torch.full((B,), T - 1, device=self.device, dtype=torch.long)

        # Simple stochastic encoding (x_t = sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * noise)
        noise = torch.randn_like(x)

        # Index alphas_cumprod at t and reshape for broadcasting
        alpha_bar = alphas_cumprod[t]  # shape (B,)
        # reshape to (B,1,1,1) for broadcasting over image dims
        alpha_bar = alpha_bar.view(B, 1, 1, 1)

        x_t = torch.sqrt(alpha_bar) * x + torch.sqrt(1.0 - alpha_bar) * noise

        return x_t
    
    @torch.no_grad()
    def encode_decode(
        self,
        x: torch.Tensor,
        genomic: torch.Tensor,
        encode_steps: int = 250,
    ) -> torch.Tensor:
        """
        Encode real tiles then decode with genomic conditioning.
        
        This preserves structure from real tiles while applying
        genomic-specific features.
        
        Args:
            x: Real tile images (B, 3, H, W) in range [-1, 1]
            genomic: (D,) or (B, D) genomic feature vector
            encode_steps: Number of encoding steps
        
        Returns:
            Reconstructed images with genomic conditioning, shape (B, 3, H, W)
        """
        B = x.shape[0]
        
        # Get conditioning from genomic features
        cond = self._get_conditioning(genomic, num_samples=1)
        
        # If single genomic for multiple tiles, expand
        if cond.shape[0] == 1 and B > 1:
            cond = cond.expand(B, -1)
        
        # Encode images to noise
        x_T = self.encode(x, encode_steps)
        
        # Decode with genomic conditioning
        model_kwargs = {"cond": cond}
        samples = self.sampler.sample(
            model=self.diffusion_model,
            noise=x_T,
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
    
    # Load diffusion model (tcga_brca_autoenc imported conditionally at module level)
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
    
    # Load diffusion model (tcga_brca_autoenc imported conditionally at module level)
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
    parser.add_argument("--mode", type=str, default="random",
                        choices=["random", "encode-decode"],
                        help="Sampling mode: 'random' generates from random noise, "
                             "'encode-decode' encodes real tiles and reconstructs")
    parser.add_argument("--num-samples-per-patient", type=int, default=4,
                        help="Number of tiles to generate per patient")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Encode-decode mode settings
    parser.add_argument("--tiles-zip-dir", type=str, default=None,
                        help="Directory with tile zip files (required for encode-decode mode)")
    parser.add_argument("--encode-steps", type=int, default=None,
                        help="Number of diffusion steps for encoding (default: full T steps). "
                             "Use smaller values (e.g., 250) for less noise/more preservation")
    
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
    
    if args.mode == "encode-decode" and args.tiles_zip_dir is None:
        raise ValueError(
            "--tiles-zip-dir is required for encode-decode mode"
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
    print(f"Mode: {args.mode}")
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
    
    # Load tile zips if encode-decode mode
    tiles_by_patient = {}
    if args.mode == "encode-decode":
        print("\n" + "=" * 60)
        print("LOADING TILE ZIPS FOR ENCODE-DECODE")
        print("=" * 60)
        
        tiles_dir = Path(args.tiles_zip_dir).expanduser()
        for h5_path in h5_files:
            pid = canonical_patient_id(h5_path.name)
            # Find matching zip file
            matching_zips = list(tiles_dir.glob(f"*{pid}*.zip")) + \
                           list(tiles_dir.glob(f"*{pid.replace('-', '')}*.zip"))
            
            if matching_zips:
                zip_path = matching_zips[0]
                tiles, tile_names = load_tiles_from_zip(
                    zip_path, 
                    num_tiles=args.num_samples_per_patient,
                    img_size=args.img_size
                )
                if tiles:
                    tiles_by_patient[pid] = (tiles, tile_names)
                    print(f"  [OK] Loaded {len(tiles)} tiles for {pid}")
                else:
                    print(f"  [WARN] No valid tiles in zip for {pid}")
            else:
                print(f"  [WARN] No zip file found for {pid}")
        
        print(f"[OK] Loaded tiles for {len(tiles_by_patient)} patients")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate samples
    print("\n" + "=" * 60)
    print(f"GENERATING SAMPLES (mode={args.mode})")
    print("=" * 60 + "\n")
    
    for h5_path in tqdm(h5_files, desc="Sampling"):
        pid = canonical_patient_id(h5_path.name)
        # Ensure `tile_names` is always defined (encode-decode branch will overwrite)
        tile_names: List[str] = []
        
        # Load genomic features
        with h5py.File(h5_path, "r") as f:
            genomic = np.array(f["feats"])
            if genomic.ndim == 2:
                genomic = genomic.mean(axis=0)
        
        genomic_tensor = torch.from_numpy(genomic.astype(np.float32))
        
        # Generate samples based on mode
        if args.mode == "random":
            # Random noise sampling
            samples = genomic_sampler.sample(
                genomic=genomic_tensor,
                num_samples=args.num_samples_per_patient,
            )
        else:
            # Encode-decode mode
            if pid not in tiles_by_patient:
                print(f"  [SKIP] No tiles for {pid}")
                continue
            
            real_tiles, tile_names = tiles_by_patient[pid]
            # Stack list of tensors into batch tensor
            real_tiles_batch = torch.stack(real_tiles, dim=0)
            samples = genomic_sampler.encode_decode(
                x=real_tiles_batch,
                genomic=genomic_tensor,
                encode_steps=args.encode_steps or 250,
            )
        
        # Save as zip file with individual PNGs
        # Use original tile names for encode-decode mode, indexed names for random mode
        zip_path = output_dir / f"{pid}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, sample in enumerate(samples):
                img = tensor_to_pil(sample)
                # Save to bytes buffer
                img_buffer = BytesIO()
                img.save(img_buffer, format="PNG")
                img_buffer.seek(0)
                # Use original tile name if available (encode-decode mode), else indexed
                if args.mode == "encode-decode" and i < len(tile_names):
                    out_name = tile_names[i]
                else:
                    out_name = f"sample_{i:02d}.png"
                # Write to zip
                zf.writestr(out_name, img_buffer.getvalue())
    
    print("\n" + "=" * 60)
    print("SAMPLING COMPLETE")
    print(f"  Output: {output_dir}")
    print(f"  Mode: {args.mode}")
    print(f"  Patients: {len(h5_files)}")
    print(f"  Samples per patient: {args.num_samples_per_patient}")
    if args.mode == "encode-decode":
        print(f"  Encode steps: {args.encode_steps or 'full T'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
