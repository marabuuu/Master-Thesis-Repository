#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Investigation utility for noising/denoising trajectory visualization.

Provides standalone functionality to visualize intermediate steps of the
diffusion process to ensure proper encoding/decoding behavior.

Can be called independently or integrated into reconstruction pipeline.

Usage:
  python -m src.reconstruction.investigate_noising \\
    --checkpoint experiments/.../last.ckpt \\
    --gene-expression dataframes/brca_gene_expression.csv \\
    --patient TCGA-XX-XX \\
    --output investigations/patient_XX_XX \\
    --steps 10
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def extract_patient_id(name: str) -> str:
    """Extract TCGA-XX-XXXX from filename."""
    stem = Path(name).stem.upper()
    for sep in ("_", "."):
        stem = stem.replace(sep, "-")
    while "--" in stem:
        stem = stem.replace("--", "-")
    parts = stem.split("-")
    if len(parts) >= 3 and parts[0].startswith("TCGA"):
        return "-".join(parts[:3]).lower()
    return stem.lower()


def tensor_to_image(x: torch.Tensor) -> np.ndarray:
    """Convert tensor (C, H, W) in [-1, 1] to uint8 RGB image."""
    if x.ndim == 4:
        x = x[0]
    x = x.cpu().detach()
    x = ((x + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
    return x.permute(1, 2, 0).numpy()


def load_checkpoint_simple(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple:
    """
    Load checkpoint (simple version for investigation script).
    
    Returns:
        (model, config, n_genes)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "hyper_parameters" not in ckpt:
        raise ValueError("Checkpoint missing hyperparameters")
    
    hp = ckpt["hyper_parameters"]
    conf = hp.get("conf")
    joint_cfg = hp.get("joint_cfg")
    n_genes = hp.get("n_genes")

    if conf is None or joint_cfg is None or n_genes is None:
        raise ValueError(
            "Checkpoint missing one of required hyperparameters: conf, joint_cfg, n_genes"
        )

    variant = hp.get("joint_variant")
    if not isinstance(variant, str) or not variant:
        has_cross = "cross_cfg" in hp or (isinstance(joint_cfg, dict) and "cross_attention" in joint_cfg)
        has_gene_token = "gene_token_transformer" in hp
        if has_cross and has_gene_token:
            variant = "gene_token_cross_attention_joint_training"
        elif has_gene_token:
            variant = "gene_token_transformer_joint_training"
        elif has_cross:
            variant = "cross_attention_joint_training"
        else:
            variant = "joint_training"

    if variant == "joint_training":
        try:
            from src.joint_training.model import JointLitModel as ModelCls  # type: ignore[import-not-found]
        except ImportError:
            from ..joint_training.model import JointLitModel as ModelCls  # type: ignore[import-not-found]
    elif variant == "cross_attention_joint_training":
        try:
            from src.cross_attention_joint_training.model import CrossAttentionJointLitModel as ModelCls  # type: ignore[import-not-found]
        except ImportError:
            from ..cross_attention_joint_training.model import CrossAttentionJointLitModel as ModelCls  # type: ignore[import-not-found]
    elif variant == "gene_token_transformer_joint_training":
        try:
            from src.gene_token_transformer_joint_training.model import GeneTokenTransformerJointLitModel as ModelCls  # type: ignore[import-not-found]
        except ImportError:
            from ..gene_token_transformer_joint_training.model import GeneTokenTransformerJointLitModel as ModelCls  # type: ignore[import-not-found]
    elif variant == "gene_token_cross_attention_joint_training":
        try:
            from src.gene_token_cross_attention_joint_training.model import GeneTokenCrossAttentionJointLitModel as ModelCls  # type: ignore[import-not-found]
        except ImportError:
            from ..gene_token_cross_attention_joint_training.model import GeneTokenCrossAttentionJointLitModel as ModelCls  # type: ignore[import-not-found]
    else:
        raise ValueError(f"Unsupported joint variant in checkpoint: {variant}")
    
    model = ModelCls(conf, joint_cfg, n_genes)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully (variant={variant})")
    return model, conf, n_genes


def load_gene_for_patient(
    csv_path: str,
    patient_id: str,
    patient_col: str = "Patient_ID",
) -> np.ndarray:
    """
    Load gene expression for a single patient from CSV.
    
    Parameters
    ----------
    csv_path : str
        Path to gene expression CSV
    patient_id : str
        Patient ID to look up
    patient_col : str
        Column name for patient IDs
    
    Returns
    -------
    np.ndarray
        Gene expression vector (n_genes,)
    """
    logger.info(f"Loading gene expression for {patient_id}")
    
    df = pd.read_csv(csv_path)
    
    # Find patient
    metadata_cols = {patient_col, "label", "Label", "SubType", "subtype"}
    gene_cols = [c for c in df.columns if c not in metadata_cols]
    
    # Match patient (case-insensitive, with canonical ID extraction)
    for _, row in df.iterrows():
        pid = extract_patient_id(str(row[patient_col]))
        if pid == extract_patient_id(patient_id):
            genes = row[gene_cols].values.astype(np.float32)
            logger.info(f"Found {patient_id}: {len(genes)} genes")
            return genes  # type: ignore[return-value]
    
    raise ValueError(f"Patient {patient_id} not found in {csv_path}")


def investigate_denoise_trajectory(
    model: torch.nn.Module,
    genomic: torch.Tensor,
    output_dir: Path,
    n_steps: int = 10,
    device: torch.device = torch.device("cuda"),
    seed: Optional[int] = None,
) -> None:
    """Generate and save denoising trajectory visualization.

    Creates a series of images showing the progression from pure noise
    to final reconstruction.

    Parameters
    ----------
    model : JointLitModel
        Loaded joint training model
    genomic : torch.Tensor
        Gene expression vector (n_genes,)
    output_dir : Path
        Directory to save intermediate frames
    n_steps : int
        Number of intermediate frames to save
    device : torch.device
        Device to run inference on
    seed : int, optional
        Optional seed for deterministic sampling.
    """
    logger.info(f"Generating denoise trajectory with {n_steps} frames")
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # Encode genomics to conditioning space
        cond = model.encode(genomic.unsqueeze(0))  # type: ignore[attr-defined]

        # Ensure deterministic behavior if requested
        if seed is not None:
            torch.manual_seed(seed)

        # Initialize pure noise
        sampler = getattr(model, "sampler", None)
        if sampler is None:
            raise RuntimeError("Model does not expose `sampler` for diffusion operations")

        T = getattr(sampler, "num_timesteps", None)
        if T is None:
            betas = getattr(sampler, "betas", None)
            T = len(betas) if betas is not None else 0
        T = int(T)
        if T <= 0:
            raise RuntimeError("Could not determine diffusion timesteps from sampler")

        img_size = getattr(model.conf, "img_size", None)
        if img_size is None:
            raise RuntimeError("Could not determine image size from model configuration")

        noise = torch.randn(
            cond.size(0), 3, int(img_size), int(img_size),
            device=device,
        )

        logger.info(f"Sampling trajectory with {T} diffusion steps")
        logger.info(f"Saving {n_steps} intermediate frames")

        # Determine which timesteps to save
        save_timesteps = set(np.linspace(0, T - 1, n_steps, dtype=int))

        # Save pure noise
        noise_path = output_dir / "noise.png"
        Image.fromarray(tensor_to_image(noise.squeeze(0))).save(noise_path)
        logger.info(f"  Saved {noise_path.name}")

        # Run reverse diffusion and save intermediate steps
        unet_model = getattr(model, "model", model)
        prog = sampler.p_sample_loop_progressive(
            model=unet_model,
            shape=noise.shape,
            noise=noise,
            model_kwargs={"cond": cond},
            device=device,
            progress=False,
        )

        final_sample = None
        for idx, out in enumerate(prog):
            t = T - 1 - idx
            if t in save_timesteps:
                out_img = out["sample"].squeeze(0)
                save_path = output_dir / f"t{int(t):04d}.png"
                Image.fromarray(tensor_to_image(out_img)).save(save_path)
                logger.info(f"  Saved {save_path.name}")
            final_sample = out["sample"]

        if final_sample is not None:
            final_path = output_dir / "final_reconstruction.png"
            Image.fromarray(tensor_to_image(final_sample.squeeze(0))).save(final_path)
            logger.info(f"  Saved {final_path.name}")
def investigate_encode_decode(
    model: torch.nn.Module,
    genomic: torch.Tensor,
    output_dir: Path,
    device: torch.device = torch.device("cuda"),
) -> None:
    """
    Investigate VAE encoding/decoding of gene expression.
    
    Visualizes:
      1. Input gene expression (reconstructed as image-like visualization)
      2. VAE latent representation statistics
      3. Decoded gene expression (should match input)
    
    Parameters
    ----------
    model : JointLitModel
        Loaded joint training model
    genomic : torch.Tensor
        Gene expression vector (n_genes,)
    output_dir : Path
        Directory to save analysis files
    device : torch.device
        Device to run inference on
    """
    logger.info("Investigating VAE encoding/decoding")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        genomic_batched = genomic.unsqueeze(0).to(device)

        # Encode to deterministic latent (mean)
        mean, log_var = model.vae.encoder(genomic_batched)  # type: ignore[attr-defined]
        z = mean

        # Decode and project
        recon = model.vae.decoder(z)  # type: ignore[attr-defined]
        cond = model.projection(z)  # type: ignore[attr-defined]

        # Save statistics
        stats = {
            "input_mean": genomic.mean().item(),
            "input_std": genomic.std().item(),
            "input_min": genomic.min().item(),
            "input_max": genomic.max().item(),
            "latent_mean": z.mean().item(),
            "latent_std": z.std().item(),
            "latent_min": z.min().item(),
            "latent_max": z.max().item(),
            "recon_mean": recon.mean().item(),
            "recon_std": recon.std().item(),
            "reconstruction_error": (genomic_batched - recon).abs().mean().item(),
            "conditioning_mean": cond.mean().item(),
            "conditioning_std": cond.std().item(),
        }

        # Save JSON
        import json
        stats_file = output_dir / "vae_statistics.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"  Saved {stats_file.name}")

        # Log statistics
        logger.info("VAE Encoding/Decoding Statistics:")
        for key, val in stats.items():
            logger.info(f"  {key}: {val:.6f}")


# ───────────────────────────────────────────────────────────────────────
#  CLI
# ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(
        description="Investigate noising/denoising trajectory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Visualize denoising trajectory
  python -m src.reconstruction.investigate_noising \\
    --checkpoint experiments/joint_training/checkpoints/last.ckpt \\
    --gene-expression dataframes/brca_gene_expression.csv \\
    --patient TCGA-XX-XX \\
    --output investigations/patient_XX \\
    --trajectory

  # Investigate VAE encoding/decoding
  python -m src.reconstruction.investigate_noising \\
    --checkpoint experiments/joint_training/checkpoints/last.ckpt \\
    --gene-expression dataframes/brca_gene_expression.csv \\
    --patient TCGA-XX-XX \\
    --output investigations/patient_XX \\
    --encode-decode
        """,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to joint training checkpoint",
    )
    parser.add_argument(
        "--gene-expression", type=str, required=True,
        help="Path to gene expression CSV",
    )
    parser.add_argument(
        "--patient", type=str, required=True,
        help="Patient ID to investigate (e.g., TCGA-XX-XX)",
    )
    parser.add_argument(
        "--output", type=str, default="investigations",
        help="Output directory for investigation results",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config (for auto-lookup if needed)",
    )
    parser.add_argument(
        "--steps", type=int, default=10,
        help="Number of frames to save in trajectory",
    )
    parser.add_argument(
        "--trajectory", action="store_true",
        help="Visualize denoising trajectory",
    )
    parser.add_argument(
        "--encode-decode", action="store_true",
        help="Investigate VAE encoding/decoding",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (e.g., cuda:0, cpu)",
    )
    
    args = parser.parse_args()
    
    if not args.trajectory and not args.encode_decode:
        parser.error("Please specify --trajectory or --encode-decode (or both)")
    
    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Load model
    model, conf, n_genes = load_checkpoint_simple(args.checkpoint, device)
    
    # Load gene expression
    gene_vec = load_gene_for_patient(
        args.gene_expression,
        args.patient,
    )
    genomic = torch.from_numpy(gene_vec).to(device, dtype=torch.float32)
    
    # Create output directory
    output_dir = Path(args.output)
    
    # Run investigations
    if args.trajectory:
        traj_dir = output_dir / "trajectory"
        investigate_denoise_trajectory(
            model, genomic, traj_dir,
            n_steps=args.steps,
            device=device,
        )
    
    if args.encode_decode:
        ed_dir = output_dir / "encode_decode"
        investigate_encode_decode(
            model, genomic, ed_dir,
            device=device,
        )
    
    logger.info(f"✅ Investigation complete. Results saved to {output_dir}")
