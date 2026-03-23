#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reconstruct tiles from genomic features using joint VAE-Diffusion model.

Three reconstruction modes:
  1. Image-guided: encode tile to noise + genomic conditioning → denoise
  2. Random noise: random noise + genomic conditioning → denoise (separate script)
  3. Investigation: save intermediate diffusion steps to inspect encoding/decoding

Pipeline:
  1. Load joint training checkpoint (VAE + Diffusion + Projection)
  2. Load bulk RNA-seq for test patients
  3. Encode genomics to VAE latent → projection space
  4. Generate/reconstruct tiles conditioned on genomics
  5. Compute metrics (SSIM, MSE, PSNR)
  6. Save results and reconstructed images

Usage:
  python -m src.reconstruction.reconstruct_tiles \\
    --checkpoint experiments/.../last.ckpt \\
    --config src/config.yaml \\
    --patients TCGA-XX-XX TCGA-YY-YY \\
    --save-dir experiments/reconstructed_tiles \\
    --n-tiles-per-patient 20
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import zipfile
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from skimage.metrics import mean_squared_error, structural_similarity
from torchvision.transforms import ToPILImage

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────
#  Utility Functions
# ───────────────────────────────────────────────────────────────────────

def extract_patient_id(name: str) -> str:
    """Extract TCGA-XX-XXXX from filename."""
    stem = Path(name).stem.upper()
    for sep in ("_", "."):
        stem = stem.replace(sep, "-")
    while "--" in stem:
        stem = stem.replace("--", "-")
    parts = stem.split("-")
    if len(parts) >= 3 and parts[0].startswith("TCGA"):
        return "-".join(parts[:3])
    return stem


def tensor_to_image(x: torch.Tensor) -> np.ndarray:
    """Convert tensor (C, H, W) in [-1, 1] to uint8 RGB image."""
    if x.ndim == 4:
        x = x[0]  # Remove batch dim
    x = x.cpu().detach()
    x = ((x + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
    return x.permute(1, 2, 0).numpy()


def image_to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert uint8 RGB image to tensor in [-1, 1]."""
    x = torch.from_numpy(img).to(device).to(torch.float32)
    x = x.permute(2, 0, 1)  # (H, W, C) → (C, H, W)
    x = x / 127.5 - 1.0  # [0, 255] → [-1, 1]
    return x


def compute_metrics(img_original: np.ndarray, img_reconstructed: np.ndarray) -> Dict[str, float]:
    """Compute SSIM, MSE, PSNR between original and reconstructed tiles."""
    if img_original.shape != img_reconstructed.shape:
        logger.warning(f"Shape mismatch: {img_original.shape} vs {img_reconstructed.shape}")
        return {"ssim": np.nan, "mse": np.nan, "psnr": np.nan}
    
    # Convert to grayscale for SSIM
    if img_original.ndim == 3:
        from skimage.color import rgb2gray
        img_orig_gray = rgb2gray(img_original.astype(np.float32) / 255.0)
        img_recon_gray = rgb2gray(img_reconstructed.astype(np.float32) / 255.0)
    else:
        img_orig_gray = img_original.astype(np.float32) / 255.0
        img_recon_gray = img_reconstructed.astype(np.float32) / 255.0
    
    ssim_val = structural_similarity(img_orig_gray, img_recon_gray, data_range=1.0)
    if isinstance(ssim_val, tuple):
        ssim_val = ssim_val[0]
    mse = mean_squared_error(img_original.astype(np.float32), 
                             img_reconstructed.astype(np.float32))
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else np.inf
    
    return {"ssim": float(ssim_val), "mse": float(mse), "psnr": float(psnr)}


def create_investigation_strip(
    investigate_dir: Path,
    n_steps: int = 5,
) -> None:
    """Create a horizontal strip showing original | forward | noise | denoise | final.
    
    Loads saved intermediate frames and assembles them into a composite visualization.
    """
    from PIL import Image as PILImage
    
    try:
        # Load images
        original = PILImage.open(investigate_dir / "original.png").convert("RGB")
        final_recon = PILImage.open(investigate_dir / "final_reconstruction.png").convert("RGB")
        encoded_noise = PILImage.open(investigate_dir / "encoded_noise.png").convert("RGB")
        
        # Collect forward and denoise frames
        forward_frames = sorted(investigate_dir.glob("forward_t*.png"))
        denoise_frames = sorted(investigate_dir.glob("denoise_t*.png"), reverse=True)
        
        # Resize all to a consistent height for strip
        target_height = 256
        aspect_ratio = original.width / original.height
        target_width = int(target_height * aspect_ratio)
        
        original_resized = original.resize((target_width, target_height), PILImage.Resampling.LANCZOS)
        final_resized = final_recon.resize((target_width, target_height), PILImage.Resampling.LANCZOS)
        noise_resized = encoded_noise.resize((target_width, target_height), PILImage.Resampling.LANCZOS)
        
        # Build strip: original | forward steps | noise | denoise steps | final
        strip_frames = [original_resized]
        
        # Add forward frames (sample of them)
        if forward_frames:
            forward_subset = forward_frames[::max(1, len(forward_frames) // 3)]
            for fp in forward_subset[:3]:
                img = PILImage.open(fp).convert("RGB").resize((target_width, target_height), PILImage.Resampling.LANCZOS)
                strip_frames.append(img)
        
        strip_frames.append(noise_resized)
        
        # Add denoise frames (sample of them)
        if denoise_frames:
            denoise_subset = denoise_frames[::max(1, len(denoise_frames) // 3)]
            for dp in denoise_subset[:3]:
                img = PILImage.open(dp).convert("RGB").resize((target_width, target_height), PILImage.Resampling.LANCZOS)
                strip_frames.append(img)
        
        strip_frames.append(final_resized)
        
        # Create horizontal strip
        total_width = sum(img.width for img in strip_frames) + (len(strip_frames) - 1) * 5
        strip = PILImage.new("RGB", (total_width, target_height), color="white")
        
        x_offset = 0
        for img in strip_frames:
            strip.paste(img, (x_offset, 0))
            x_offset += img.width + 5
        
        # Save strip
        strip_path = investigate_dir / "investigation_strip.png"
        strip.save(strip_path)
        logger.info(f"  Saved investigation strip -> {strip_path.name}")
        
    except FileNotFoundError as e:
        logger.warning(f"Could not create investigation strip: {e}")


# ───────────────────────────────────────────────────────────────────────
#  Data Loading
# ───────────────────────────────────────────────────────────────────────

def load_gene_expression(
    csv_path: str,
    patient_ids: Optional[List[str]] = None,
    patient_col: str = "Patient_ID",
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Load gene expression from CSV.
    
    Returns:
        gene_data: {patient_id: (n_genes,)}
        gene_names: list of gene column names
    """
    logger.info(f"Loading gene expression from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Extract gene columns (everything except metadata columns)
    metadata_cols = {patient_col, "label", "Label", "SubType", "subtype"}
    gene_cols = [c for c in df.columns if c not in metadata_cols]
    logger.info(f"Found {len(gene_cols)} gene columns")
    
    # Filter patients
    if patient_ids is not None:
        patient_ids_canonical = {pid.upper() for pid in patient_ids}
        df_filtered = []
        for _, row in df.iterrows():
            pid = str(row[patient_col])
            pid_canonical = extract_patient_id(pid)
            if pid_canonical in patient_ids_canonical:
                df_filtered.append(row)
        df = pd.DataFrame(df_filtered)
        logger.info(f"Filtered to {len(df)} patients")
    
    # Create dict
    gene_data = {}
    for _, row in df.iterrows():
        pid = extract_patient_id(str(row[patient_col]))
        genes = row[gene_cols].values.astype(np.float32)
        gene_data[pid] = genes
    
    logger.info(f"Loaded gene expression for {len(gene_data)} patients")
    return gene_data, gene_cols


def load_tiles_for_patients(
    tiles_dir: Path,
    patient_ids: List[str],
    n_tiles_per_patient: int = 20,
) -> Dict[str, List[Tuple[Union[Path, str], str]]]:
    """
    Load tile paths for specified patients.
    
    Returns:
        {patient_id: [(tile_path, tile_basename), ...]}
        where tile_path is either a Path (for directories) or a str "zip_path::member_name" (for zip files)
    """
    logger.info(f"Discovering tiles for {len(patient_ids)} patients from {tiles_dir}")
    patient_tiles = {}
    
    for patient_id in patient_ids:
        # Check for non-zip directories
        patient_folders = [
            f for f in tiles_dir.iterdir()
            if f.is_dir() and extract_patient_id(f.name).upper() == patient_id.upper()
        ]
        
        # Check for zip archives
        patient_zips = [
            f for f in tiles_dir.iterdir()
            if f.is_file() and f.suffix == ".zip" and extract_patient_id(f.name).upper() == patient_id.upper()
        ]
        
        tile_files = []
        if patient_folders:
            patient_folder = patient_folders[0]
            discovered_files = sorted(patient_folder.glob("*.jpg")) + sorted(patient_folder.glob("*.png"))
            tile_files = [(f, f.name) for f in discovered_files]
        elif patient_zips:
            patient_zip = patient_zips[0]
            try:
                with zipfile.ZipFile(patient_zip, 'r') as zf:
                    members = [m.filename for m in zf.infolist() if not m.is_dir() and (m.filename.endswith(".jpg") or m.filename.endswith(".png"))]
                    members = sorted(members)
                    tile_files = [(f"{patient_zip}::{m}", Path(m).name) for m in members]
            except Exception as e:
                logger.error(f"Failed to read zip file {patient_zip}: {e}")
                continue
                
        if not patient_folders and not patient_zips:
            logger.warning(f"No folder or zip found for patient {patient_id}")
            continue
            
        if len(tile_files) == 0:
            logger.warning(f"No tiles found for patient {patient_id}")
            continue
        
        # Sample tiles
        if len(tile_files) > n_tiles_per_patient:
            indices = np.random.choice(  # type: ignore[call-overload]
                len(tile_files), n_tiles_per_patient, replace=False
            )
            tile_files = [tile_files[i] for i in indices]
        
        patient_tiles[patient_id] = tile_files
        logger.info(f"  {patient_id}: {len(tile_files)} tiles")
    
    return patient_tiles


# ───────────────────────────────────────────────────────────────────────
#  Model Loading
# ───────────────────────────────────────────────────────────────────────

def load_checkpoint(
    checkpoint_path: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple:
    """
    Load JOINT TRAINING checkpoint (strict - no fallback to base mopadi).
    
    The checkpoint MUST have been saved from src.joint_training with:
      - conf: TrainConfig for the diffusion model
      - joint_cfg: Configuration dict for the joint training
      - n_genes: Number of genes in the genomic VAE
    
    Returns:
        (model, conf, joint_cfg)
    
    Raises:
        ValueError: If checkpoint is not from joint training
    """
    logger.info(f"Loading JOINT TRAINING checkpoint from {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load state dict
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint file: {e}")
    
    logger.info(f"Checkpoint keys: {list(ckpt.keys())}")
    
    # Extract config from checkpoint
    if "hyper_parameters" not in ckpt:
        raise ValueError(
            f"❌ Checkpoint is missing 'hyper_parameters' key.\n"
            f"   This does not appear to be a joint training checkpoint.\n"
            f"   Available keys: {list(ckpt.keys())}\n"
            f"   Expected: checkpoint from src.joint_training.train (JointLitModel)"
        )
    
    hp = ckpt["hyper_parameters"]
    logger.info(f"Checkpoint has {len(hp)} hyperparameters")
    
    # Strictly require joint training format
    conf = hp.get("conf")
    joint_cfg = hp.get("joint_cfg")
    n_genes = hp.get("n_genes")
    
    # Check each required parameter
    if conf is None:
        raise ValueError(
            f"❌ Checkpoint missing 'conf' (TrainConfig for diffusion).\n"
            f"   This is not a joint training checkpoint.\n"
            f"   Available hyperparameters: {sorted(hp.keys())}\n"
            f"   Expected parameters: ['conf', 'joint_cfg', 'n_genes']"
        )
    
    if joint_cfg is None:
        raise ValueError(
            f"❌ Checkpoint missing 'joint_cfg' (joint training configuration).\n"
            f"   This is not a joint training checkpoint.\n"
            f"   Available hyperparameters: {sorted(hp.keys())}\n"
            f"   Expected parameters: ['conf', 'joint_cfg', 'n_genes']"
        )
    
    if n_genes is None:
        raise ValueError(
            f"❌ Checkpoint missing 'n_genes' (number of genes).\n"
            f"   This is not a joint training checkpoint.\n"
            f"   Available hyperparameters: {sorted(hp.keys())}\n"
            f"   Expected parameters: ['conf', 'joint_cfg', 'n_genes']"
        )
    
    logger.info("✅ Detected JOINT TRAINING checkpoint format")
    logger.info(f"   conf: {type(conf).__name__}")
    logger.info(f"   joint_cfg: {type(joint_cfg).__name__} with keys: {list(joint_cfg.keys()) if isinstance(joint_cfg, dict) else 'N/A'}")
    logger.info(f"   n_genes: {n_genes}")
    
    # Load JointLitModel
    try:
        from src.joint_training.model import JointLitModel  # type: ignore[import-not-found]
    except ImportError:
        try:
            from ..joint_training.model import JointLitModel  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(f"Could not import JointLitModel: {e}")
    
    try:
        logger.info("Creating JointLitModel...")
        model = JointLitModel(conf, joint_cfg, n_genes)
        logger.info("✅ JointLitModel created")
    except Exception as e:
        raise ValueError(f"Failed to create JointLitModel: {e}")
    
    try:
        logger.info("Loading state dict...")
        model.load_state_dict(ckpt["state_dict"])
        logger.info("✅ State dict loaded")
    except Exception as e:
        raise ValueError(f"Failed to load state dict into JointLitModel: {e}")
    
    model = model.to(device)
    model.eval()
    
    logger.info("=" * 80)
    logger.info("✅ JOINT TRAINING CHECKPOINT LOADED SUCCESSFULLY")
    logger.info("=" * 80)
    return model, conf, joint_cfg


# ───────────────────────────────────────────────────────────────────────
#  Reconstruction
# ───────────────────────────────────────────────────────────────────────

def _reconstruct_from_image_with_cond(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    cond: torch.Tensor,
    device: torch.device,
    *,
    seed: Optional[int] = None,
    save_dir: Optional[Path] = None,
    n_steps: int = 5,
) -> torch.Tensor:
    """Encode a tile into noise and then decode it using the diffusion sampler.

    If ``save_dir`` is provided, this will also save intermediate forward and
    reverse denoising steps (useful for investigation).

    Parameters
    ----------
    model:
        JointLitModel containing `.sampler` and `.model` (UNet) attributes.
    img_tensor:
        Image tensor in [-1, 1], shape (1, 3, H, W).
    cond:
        Conditioning vector from `model.encode()` (shape (1, C)).
    seed:
        Optional integer seed for deterministic behavior.
    save_dir:
        Directory where intermediate frames should be saved.
    n_steps:
        Number of intermediate timesteps to save for forward and reverse passes.

    Returns
    -------
    torch.Tensor
        Reconstructed image tensor in [-1, 1].
    """
    # Prefer eval sampler (matches mopadi encode_stochastic); fallback to training sampler
    sampler = getattr(model, "eval_sampler", None) or getattr(model, "sampler", None)
    if sampler is None:
        raise RuntimeError("Model does not expose `sampler` - cannot run diffusion denoising")

    if seed is not None:
        torch.manual_seed(seed)

    # Get the full training sampler for encoding (T=1000 for thorough diffusion)
    # This ensures the image is truly destroyed into noise, not just partially corrupted
    full_sampler = getattr(model, "sampler", sampler)
    
    T_full = getattr(full_sampler, "num_timesteps", None)
    if T_full is None:
        T_full = len(getattr(full_sampler, "betas", []))
    if not isinstance(T_full, int) or T_full <= 0:
        raise RuntimeError("Could not determine diffusion timesteps from full sampler")

    # Get eval sampler timesteps for decoding (T_eval=20 for fast sampling)
    T_eval = getattr(sampler, "num_timesteps", None)
    if T_eval is None:
        T_eval = len(getattr(sampler, "betas", []))
    if not isinstance(T_eval, int) or T_eval <= 0:
        raise RuntimeError("Could not determine diffusion timesteps from eval sampler")

    # Use EMA UNet if available
    unet_model = getattr(model, "ema_model", getattr(model, "model", model))
    unet_model.eval()

    # Forward diffusion (encode) using the eval sampler schedule to match the decode schedule (T_eval)
    # - We reuse a single noise tensor `eps` across all timesteps so saved frames are comparable
    # - x_T is sampled at the final eval diffusion timestep (T_eval - 1)
    eps = torch.randn_like(img_tensor)
    decode_eta = 0.0  # deterministic decode
    logger.info(f"  Encoding tile to noise using forward q_sample (T_eval={T_eval}) to keep schedules symmetric...")
    with torch.no_grad():
        t_final = torch.full((img_tensor.shape[0],), T_eval - 1, device=device, dtype=torch.long)
        x_T = sampler.q_sample(img_tensor, t_final, noise=eps)

    # Save forward trajectory snapshots if requested
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        def _to_uint8(img: torch.Tensor) -> np.ndarray:
            # Visualization helper: min-max normalize to avoid channel bias / clipping artifacts
            arr = img.detach().cpu()
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            return (arr * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()

        target_steps = set(np.linspace(0, T_eval - 1, n_steps, dtype=int))
        logger.info(f"  [DEBUG] Target timesteps for saving (forward q_sample, eval schedule): {sorted(target_steps)}")
        for t_int in sorted(target_steps):
            t_tensor = torch.full((img_tensor.shape[0],), int(t_int), device=device, dtype=torch.long)
            sample_t = sampler.q_sample(img_tensor, t_tensor, noise=eps)
            save_path = save_dir / f"forward_t{t_int:04d}.png"
            Image.fromarray(_to_uint8(sample_t.squeeze(0))).save(save_path)
            logger.info(f"  Saved forward step t={t_int} -> {save_path.name}")

        # Save final encoded noise (already computed as x_T)
        encoded_noise_path = save_dir / "encoded_noise.png"
        Image.fromarray(_to_uint8(x_T.squeeze(0))).save(encoded_noise_path)
        logger.info(f"  Saved final encoded noise -> {encoded_noise_path.name}")

    # Denoising (reverse diffusion) with deterministic sampling (fast with eval sampler)
    prog = sampler.ddim_sample_loop_progressive(
        model=unet_model,
        shape=img_tensor.shape,
        noise=x_T,
        model_kwargs={"cond": cond},
        device=device,
        progress=False,
        eta=decode_eta,
    )

    final_reconstruction: torch.Tensor | None = None
    save_denoise_timesteps = set(np.linspace(0, T_eval - 1, n_steps, dtype=int)) if save_dir is not None else set()

    logger.info(f"  [DEBUG] Save denoise timesteps (from T_eval={T_eval}): {sorted(save_denoise_timesteps)}")
    
    denoise_frame_count = 0
    for idx, out in enumerate(prog):
        t = T_eval - 1 - idx
        if t in save_denoise_timesteps:
            out_img = out["sample"].squeeze(0)
            save_path = cast(Path, save_dir) / f"denoise_t{int(t):04d}.png"
            Image.fromarray(tensor_to_image(out_img)).save(save_path)
            logger.info(f"  Saved denoise step t={t} -> {save_path.name}")
            denoise_frame_count += 1
        final_reconstruction = out["sample"]
    
    logger.info(f"  [DEBUG] Total denoise frames saved: {denoise_frame_count} out of {idx + 1} iterations")

    if final_reconstruction is not None and save_dir is not None:
        final_path = cast(Path, save_dir) / "final_reconstruction.png"
        Image.fromarray(tensor_to_image(final_reconstruction.squeeze(0))).save(final_path)
        logger.info(f"  Saved final reconstruction -> {final_path.name}")

    if final_reconstruction is None:
        raise RuntimeError("No reconstruction produced from p_sample_loop_progressive")
    return final_reconstruction


def reconstruct_tile_image_guided(
    model: torch.nn.Module,
    tile_path: Union[Path, str],
    genomic: torch.Tensor,
    device: torch.device,
    seed: Optional[int] = None,
    investigate: bool = False,
    investigate_dir: Optional[Path] = None,
    n_steps: int = 5,
) -> Tuple[torch.Tensor, np.ndarray, Dict]:
    """Reconstruct a tile using diffusion encoding/decoding + genomic conditioning.

    When ``investigate=True``, intermediate forward/reverse diffusion steps are
    saved into ``investigate_dir`` and the final reconstruction is guaranteed to
    match the tile output.
    """

    # Load original image
    if isinstance(tile_path, str) and "::" in tile_path:
        zip_path, member_name = tile_path.split("::", 1)
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open(member_name) as f:
                img_original = Image.open(f).convert("RGB")
        tile_name = Path(member_name).name
    else:
        img_original = Image.open(tile_path).convert("RGB")
        tile_name = Path(tile_path).name

    img_array = np.array(img_original)

    logger.info(f"Reconstructing {tile_name} with genomic conditioning (encode-decode)")

    # Convert to tensor in [-1, 1]
    img_tensor = image_to_tensor(img_array, device).unsqueeze(0)

    if investigate:
        if investigate_dir is None:
            raise ValueError("investigate_dir must be provided when investigate=True")
        investigate_dir.mkdir(parents=True, exist_ok=True)

        # Save original tile (for comparison)
        orig_save_path = investigate_dir / "original.png"
        img_original.save(orig_save_path)
        logger.info(f"  Saved original tile -> {orig_save_path.name}")

    with torch.no_grad():
        cond = model.encode(genomic.unsqueeze(0))  # type: ignore[attr-defined]
        # Debug conditioning statistics to ensure variability across tiles/patients
        logger.info(
            "  [DEBUG] cond stats: mean=%.4f std=%.4f min=%.4f max=%.4f first5=%s"
            % (
                cond.mean().item(),
                cond.std().item(),
                cond.min().item(),
                cond.max().item(),
                [float(x) for x in cond[0, :5].tolist()],
            )
        )
        if os.getenv("ZERO_COND", "0") == "1":
            cond = torch.zeros_like(cond)
            logger.info("  [DEBUG] ZERO_COND=1 -> conditioning zeroed for ablation")
        recon_tensor = _reconstruct_from_image_with_cond(
            model,
            img_tensor,
            cond,
            device=device,
            seed=seed,
            save_dir=investigate_dir if investigate else None,
            n_steps=n_steps,
        )

    # Create investigation strip if frames were saved
    if investigate and investigate_dir is not None:
        create_investigation_strip(investigate_dir, n_steps=n_steps)

    recon_image = tensor_to_image(recon_tensor)
    metrics = compute_metrics(img_array, recon_image)

    return recon_tensor, recon_image, metrics


def _reconstruct_from_noise_with_cond(
    model: torch.nn.Module,
    cond: torch.Tensor,
    noise: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Decode from noise using the diffusion sampler (same as training)."""
    sampler = getattr(model, "sampler", None)
    if sampler is None:
        raise RuntimeError("Model does not expose `sampler` for diffusion decoding")

    unet_model = getattr(model, "model", model)
    out = sampler.p_sample_loop(
        model=unet_model,
        shape=noise.shape,
        noise=noise,
        model_kwargs={"cond": cond},
        device=device,
        progress=False,
    )
    return out


def reconstruct_tile_random_noise(
    model: torch.nn.Module,
    genomic: torch.Tensor,
    device: torch.device,
    seed: Optional[int] = None,
    investigate: bool = False,
    investigate_dir: Optional[Path] = None,
    n_steps: int = 5,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Pure random reconstruction: sample from random noise + genomic conditioning.

    If ``investigate=True``, intermediate noise + denoising steps are saved to
    ``investigate_dir``.
    """
    if investigate and investigate_dir is None:
        raise ValueError("investigate_dir must be provided when investigate=True")

    with torch.no_grad():
        cond = model.encode(genomic.unsqueeze(0))  # type: ignore[attr-defined]

        if seed is not None:
            torch.manual_seed(seed)

        img_size = cast(int, getattr(model.conf, "img_size", 512))  # type: ignore[attr-defined]
        noise = torch.randn(1, 3, int(img_size), int(img_size), device=device)  # type: ignore[call-arg]

        if investigate:
            # Save noise and intermediate denoising steps
            inv_dir = cast(Path, investigate_dir)
            inv_dir.mkdir(parents=True, exist_ok=True)
            noise_path = inv_dir / "noise.png"
            Image.fromarray(tensor_to_image(noise.squeeze(0))).save(noise_path)
            logger.info(f"  Saved noise -> {noise_path.name}")

            # Run sampler and save intermediate steps
            sampler = getattr(model, "sampler", None)
            if sampler is None:
                raise RuntimeError("Model does not expose `sampler` for diffusion decoding")

            T = getattr(sampler, "num_timesteps", None)
            if T is None:
                betas = getattr(sampler, "betas", None)
                T = len(betas) if betas is not None else 0
            T = int(T)
            if T <= 0:
                raise RuntimeError("Could not determine diffusion timesteps from sampler")

            # Determine timesteps to save
            save_timesteps = set(np.linspace(0, T - 1, n_steps, dtype=int))

            unet_model = getattr(model, "model", model)
            prog = sampler.p_sample_loop_progressive(
                model=unet_model,
                shape=noise.shape,
                noise=noise,
                model_kwargs={"cond": cond},
                device=device,
                progress=False,
            )

            final_reconstruction: torch.Tensor | None = None
            for idx, out in enumerate(prog):
                t = T - 1 - idx
                if t in save_timesteps:
                    out_img = out["sample"].squeeze(0)
                    save_path = inv_dir / f"denoise_t{int(t):04d}.png"
                    Image.fromarray(tensor_to_image(out_img)).save(save_path)
                    logger.info(f"  Saved denoise step t={t} -> {save_path.name}")
                final_reconstruction = out["sample"]

            if final_reconstruction is not None:
                final_path = inv_dir / "final_reconstruction.png"
                Image.fromarray(tensor_to_image(final_reconstruction.squeeze(0))).save(final_path)
                logger.info(f"  Saved final reconstruction -> {final_path.name}")

            recon_tensor: torch.Tensor = final_reconstruction if final_reconstruction is not None else torch.empty(0)
        else:
            recon_tensor = _reconstruct_from_noise_with_cond(model, cond, noise, device)

    recon_image = tensor_to_image(recon_tensor)
    return recon_tensor, recon_image


# ───────────────────────────────────────────────────────────────────────
#  Investigate Noising Steps
# ───────────────────────────────────────────────────────────────────────

def investigate_noising_steps(
    model: torch.nn.Module,
    genomic: torch.Tensor,
    output_dir: Path,
    n_steps: int = 5,
    device: torch.device = torch.device("cuda"),
    mode: str = "image_guided",
    tile_path: Union[Path, str] | None = None,
) -> None:
    """Investigate the diffusion noising + denoising process.

    Depending on mode:
      - image_guided: encode a tile into noise (forward) and decode it back.
      - random_noise: start from pure random noise and decode.

    This function saves:
      - original tile (only for image_guided)
      - noise tensor
      - forward noising steps (image -> noise) when image_guided
      - reverse denoising steps (noise -> image)
      - final reconstructed tile

    Parameters
    ----------
    model : JointLitModel
        Loaded joint training model
    genomic : torch.Tensor
        Gene expression vector (n_genes,)
    output_dir : Path
        Directory to save intermediate steps
    n_steps : int
        Number of snapshots to save
    device : torch.device
        Device to run inference on
    mode : str
        Either "image_guided" or "random_noise"
    tile_path : Path | str | None
        Path to the tile (required when mode=="image_guided")
    """
    logger.info(f"Investigating noising steps, saving {n_steps} snapshots")
    output_dir.mkdir(parents=True, exist_ok=True)

    img_tensor: torch.Tensor
    if mode == "image_guided":
        if tile_path is None:
            raise ValueError("tile_path must be provided when mode='image_guided'")

        # --- Load and save original tile -------------------------------------------------
        if isinstance(tile_path, str) and "::" in tile_path:
            zip_path, member_name = tile_path.split("::", 1)
            with zipfile.ZipFile(zip_path, "r") as zf:
                with zf.open(member_name) as f:
                    img_original = Image.open(f).convert("RGB")
            tile_name = Path(member_name).name
        else:
            img_original = Image.open(tile_path).convert("RGB")
            tile_name = Path(tile_path).name

        orig_save_path = output_dir / "original.png"
        img_original.save(orig_save_path)
        logger.info(f"  Saved original tile -> {orig_save_path.name}")

        # Normalize image to [-1, 1] for diffusion
        img_arr = np.array(img_original).astype(np.float32) / 255.0
        img_arr = img_arr * 2.0 - 1.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)
    else:
        # Random noise mode: start from pure noise, no tile needed.
        img_tensor = None  # type: ignore[assignment]
        tile_name = "random_noise"

    with torch.no_grad():
        # Encode genomics (conditioning vector)
        cond = model.encode(genomic.unsqueeze(0))  # type: ignore[attr-defined]

        # Forward (noising) process
        sampler = getattr(model, "sampler", None)
        if sampler is None:
            raise RuntimeError("Model does not expose a `sampler` for diffusion operations")

        # Determine total timesteps from sampler
        T = getattr(sampler, "num_timesteps", None)
        if T is None:
            betas = getattr(sampler, "betas", None)
            T = len(betas) if betas is not None else 0
        T = int(T)
        if T <= 0:
            raise RuntimeError("Could not determine diffusion timesteps from sampler")

        # Fixed noise used for forward noising (keep consistent across steps)
        img_size = cast(int, getattr(model.conf, "img_size", 512))  # type: ignore[attr-defined]
        fixed_noise = torch.randn(1, 3, int(img_size), int(img_size), device=device)  # type: ignore[call-arg]
        noise_save_path = output_dir / "noise.png"
        Image.fromarray(tensor_to_image(fixed_noise.squeeze(0))).save(noise_save_path)
        logger.info(f"  Saved fixed noise -> {noise_save_path.name}")

        # Choose timesteps to capture (0..T-1)
        forward_timesteps = np.linspace(0, T - 1, n_steps, dtype=int)
        forward_timesteps = sorted(set(forward_timesteps))

        x_T: torch.Tensor
        if mode == "image_guided":
            # Save forward noising steps from the original tile
            for i, t in enumerate(forward_timesteps):
                t_tensor = torch.tensor([int(t)], device=device)
                x_t = sampler.q_sample(img_tensor, t_tensor, noise=fixed_noise)
                save_path = output_dir / f"forward_t{int(t):04d}.png"
                Image.fromarray(tensor_to_image(x_t.squeeze(0))).save(save_path)
                logger.info(f"  Saved forward step t={t} -> {save_path.name}")

            # Use x_T from the tile (encode to full noise level)
            t_final = T - 1
            t_final_tensor = torch.tensor([int(t_final)], device=device)
            x_T = sampler.q_sample(img_tensor, t_final_tensor, noise=fixed_noise)

            denoise_timesteps = set(forward_timesteps)
            denoise_timesteps = {int(t) for t in denoise_timesteps}
            denoise_timesteps.add(0)
        else:
            # Random noise mode: start from pure noise, no tile encoding.
            x_T = fixed_noise
            denoise_timesteps = set()  # we will sample regularly

        logger.info("\nStarting reverse denoising (sampling) ...")
        unet_model = getattr(model, "model", model)
        prog = sampler.p_sample_loop_progressive(
            model=unet_model,
            shape=x_T.shape,
            noise=x_T,
            model_kwargs={"cond": cond},
            device=device,
            progress=False,
        )

        final_reconstruction = None
        for idx, out in enumerate(prog):
            t = T - 1 - idx
            if mode == "random_noise":
                # Save evenly spaced frames in random-noise mode
                if idx % max(1, T // max(1, n_steps)) == 0:
                    out_img = out["sample"].squeeze(0)
                    save_path = output_dir / f"denoise_t{int(t):04d}.png"
                    Image.fromarray(tensor_to_image(out_img)).save(save_path)
                    logger.info(f"  Saved denoise step t={t} -> {save_path.name}")
            else:
                if t in denoise_timesteps:
                    out_img = out["sample"].squeeze(0)
                    save_path = output_dir / f"denoise_t{int(t):04d}.png"
                    Image.fromarray(tensor_to_image(out_img)).save(save_path)
                    logger.info(f"  Saved denoise step t={t} -> {save_path.name}")
            final_reconstruction = out["sample"]

        if final_reconstruction is not None:
            final_path = output_dir / "final_reconstruction.png"
            Image.fromarray(tensor_to_image(final_reconstruction.squeeze(0))).save(final_path)
            logger.info(f"  Saved final reconstruction -> {final_path.name}")


# ───────────────────────────────────────────────────────────────────────
#  Main Orchestration
# ───────────────────────────────────────────────────────────────────────

def main(
    checkpoint_path: str,
    config_path: str,
    gene_csv_path: str,
    tiles_dir: str | Path,
    save_dir: str,
    patients: Optional[List[str] | str] = None,
    patient_splits_path: Optional[str] = None,
    n_tiles_per_patient: int = 20,
    investigate: bool = False,
    mode: str = "image_guided",
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> None:
    """
    Main reconstruction pipeline.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to joint training checkpoint
    config_path : str
        Path to YAML config
    gene_csv_path : str
        Path to bulk RNA-seq CSV
    tiles_dir : str
        Directory containing patient tile folders
    save_dir : str
        Output directory for reconstructions
    patients : list of str, str, or None, optional
        Specific patient IDs to process, a split name ("train", "val", "test"),
        or None to use all from CSV.
    patient_splits_path : str, optional
        Path to patient_splits.json (required if patients is a split name).
    n_tiles_per_patient : int
        Number of tiles per patient to reconstruct
    investigate : bool
        Save intermediate noising steps
    mode : str
        "image_guided" or "random_noise"
    seed : int, optional
        Base random seed for deterministic runs (per-tile variations are derived)
    device : str, optional
        Device (e.g., "cuda:0", "cpu")
    """
    
    # Setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    
    orig_dir = save_dir_path / "original_samples"
    recon_dir = save_dir_path / "reconstructed_samples"
    orig_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("RECONSTRUCTION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Checkpoint:  {checkpoint_path}")
    logger.info(f"Gene CSV:    {gene_csv_path}")
    logger.info(f"Tiles dir:   {tiles_dir}")
    logger.info(f"Save dir:    {save_dir}")
    logger.info(f"Mode:        {mode}")
    logger.info(f"Investigate: {investigate}")
    logger.info("")
    
    # Load config (optional if called from pipeline)
    rec_cfg = {}
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        joint_cfg = config.get("joint_training", {})
        rec_cfg = config.get("reconstruction", {})
    else:
        joint_cfg = {}
    
    # Load checkpoint (strict - requires joint training format)
    model, conf, joint_cfg_ckpt = load_checkpoint(checkpoint_path, device_obj)
    
    # Joint training checkpoint successfully loaded - genomic conditioning is available
    logger.info("✅ Using genomic conditioning from joint training model")
    assert joint_cfg_ckpt is not None, "Joint training checkpoint should always have joint_cfg"
    
    # Determine patients (CLI takes priority over config)
    if patients is None:
        patients = rec_cfg.get("patient_ids")  # Try config first
    
    if patient_splits_path is None:
        patient_splits_path = rec_cfg.get("patient_splits_path")  # Try config also
    
    # Handle split name ("train", "val", "test")
    patient_ids_list: List[str]
    if isinstance(patients, str) and patients in ("train", "val", "test"):
        split_name = patients
        if patient_splits_path is None:
            raise ValueError(
                f"patient_ids='{split_name}' requires 'patient_splits_path' in config or as argument"
            )
        if not Path(patient_splits_path).exists():
            raise FileNotFoundError(f"patient_splits.json not found: {patient_splits_path}")
        
        with open(patient_splits_path) as f:
            splits = json.load(f)
        if split_name not in splits:
            raise ValueError(
                f"Split '{split_name}' not found in {patient_splits_path}. "
                f"Available: {list(splits.keys())}"
            )
        split_data = splits[split_name]
        # Handle both nested format {"patients": [...], "n_patients": N} and flat list format
        split_patients = split_data.get("patients", split_data) if isinstance(split_data, dict) else split_data
        patient_ids_list = cast(List[str], split_patients if isinstance(split_patients, list) else [])
        logger.info(f"Using {len(patient_ids_list)} patients from '{split_name}' split")
    elif patients is None:
        # Use all patients in CSV
        df = pd.read_csv(gene_csv_path)
        patient_ids_list = [extract_patient_id(str(p)) for p in df.get("Patient_ID", [])]
        patient_ids_list = list(set(patient_ids_list))  # Unique
        logger.info(f"Using all {len(patient_ids_list)} patients from CSV")
    else:
        patient_ids_list = [extract_patient_id(p) for p in patients]
        logger.info(f"Using {len(patient_ids_list)} specified patients")
    
    # Load gene expressions
    gene_data, gene_names = load_gene_expression(
        gene_csv_path,
        patient_ids_list,
        patient_col=joint_cfg.get("patient_col", "Patient_ID"),
    )
    
    # Load tiles
    tiles_dir_path = Path(tiles_dir)
    patient_tiles = load_tiles_for_patients(
        tiles_dir_path,
        list(gene_data.keys()),
        n_tiles_per_patient,
    )
    
    # Reconstruction loop
    results = []
    
    for patient_id, tiles_list in patient_tiles.items():
        if patient_id not in gene_data:
            logger.warning(f"No gene data for {patient_id}, skipping")
            continue
        
        logger.info(f"\nProcessing {patient_id} ({len(tiles_list)} tiles)")
        
        # Convert gene expression to tensor
        genes_np = gene_data[patient_id]
        genomic_tensor = torch.from_numpy(genes_np).to(device_obj, dtype=torch.float32)
        
        orig_zip_path = orig_dir / f"{patient_id}.zip"
        recon_zip_path = recon_dir / f"{patient_id}.zip"
        
        with zipfile.ZipFile(orig_zip_path, 'w') as zf_orig, zipfile.ZipFile(recon_zip_path, 'w') as zf_recon:
            # Per-tile reconstruction
            for tile_path, tile_name in tiles_list:
                # Deterministic seed for reproducible runs (and consistent
                # investigation output). This is stable across Python processes.
                if seed is not None:
                    tile_seed = int(
                        hashlib.sha256(f"{seed}_{patient_id}_{tile_name}".encode()).hexdigest(),
                        16,
                    ) % (2**32)
                else:
                    tile_seed = None

                # Prepare investigation directory (if requested)
                inv_dir = None
                if investigate:
                    inv_dir = save_dir_path / "investigation" / f"{patient_id}_{Path(tile_name).stem}"

                try:
                    if mode == "image_guided":
                        recon_tensor, recon_image, metrics = reconstruct_tile_image_guided(
                            model,
                            tile_path,
                            genomic_tensor,
                            device_obj,
                            seed=tile_seed,
                            investigate=investigate,
                            investigate_dir=inv_dir,
                            n_steps=5,
                        )
                    elif mode == "random_noise":
                        recon_tensor, recon_image = reconstruct_tile_random_noise(
                            model,
                            genomic_tensor,
                            device_obj,
                            seed=tile_seed,
                            investigate=investigate,
                            investigate_dir=inv_dir,
                            n_steps=5,
                        )
                        metrics = {}  # No ground truth for comparison
                    else:
                        raise ValueError(f"Unknown mode: {mode}")

                    # Save reconstructed image to ZIP
                    recon_pil = Image.fromarray(recon_image)
                    recon_bytes = io.BytesIO()
                    recon_pil.save(recon_bytes, format="PNG")
                    zf_recon.writestr(f"{patient_id}_{tile_name}", recon_bytes.getvalue())

                    # Save original image to ZIP
                    if isinstance(tile_path, str) and "::" in tile_path:
                        zip_path, member_name = tile_path.split("::", 1)
                        with zipfile.ZipFile(zip_path, "r") as zf_inner:
                            with zf_inner.open(member_name) as fInner:
                                orig_pil = Image.open(fInner).convert("RGB")
                    else:
                        orig_pil = Image.open(tile_path).convert("RGB")
                        
                    orig_bytes = io.BytesIO()
                    orig_pil.save(orig_bytes, format="PNG")
                    zf_orig.writestr(f"{patient_id}_{tile_name}", orig_bytes.getvalue())

                    # Record result
                    result_row = {
                        "patient_id": patient_id,
                        "tile_name": tile_name,
                        "status": "success",
                        **metrics,
                    }
                    results.append(result_row)
                    logger.debug(f"  ✓ {tile_name}")
                    
                except Exception as e:
                    logger.error(f"  ✗ {tile_name}: {e}")
                    results.append({
                        "patient_id": patient_id,
                        "tile_name": tile_name,
                        "status": f"error: {str(e)}",
                    })
    
    # Save results CSV
    results_df = pd.DataFrame(results)
    results_csv = save_dir_path / "reconstruction_results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"\n✅ Results saved to {results_csv}")
    
    logger.info("=" * 80)
    logger.info("RECONSTRUCTION COMPLETE")
    logger.info("=" * 80)


# ───────────────────────────────────────────────────────────────────────
#  CLI
# ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Reconstruct tiles using joint VAE-Diffusion model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Reconstruct with genomic conditioning (image-guided mode)
  python -m src.reconstruction.reconstruct_tiles \\
    --checkpoint experiments/joint_training/checkpoints/last.ckpt \\
    --config src/config.yaml \\
    --patients TCGA-5L-AAT0 TCGA-5T-A9QA \\
    --n-tiles-per-patient 50

  # Random noise mode (generate from scratch)
  python -m src.reconstruction.reconstruct_tiles \\
    --checkpoint experiments/joint_training/checkpoints/last.ckpt \\
    --config src/config.yaml \\
    --mode random_noise \\
    --investigate
        """,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to joint training checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--patients", type=str, nargs="+", default=None,
        help="Specific patient IDs to process (e.g., TCGA-XX-XX TCGA-YY-YY)",
    )
    parser.add_argument(
        "--gene-csv", type=str, default=None,
        help="Path to gene expression CSV (auto-inferred from config if not provided)",
    )
    parser.add_argument(
        "--tiles-dir", type=str, default=None,
        help="Path to tiles directory (auto-inferred from config if not provided)",
    )
    parser.add_argument(
        "--save-dir", type=str, default="experiments/reconstructed_tiles",
        help="Output directory for reconstructions",
    )
    parser.add_argument(
        "--n-tiles-per-patient", type=int, default=20,
        help="Number of tiles per patient to reconstruct",
    )
    parser.add_argument(
        "--mode", type=str, choices=["image_guided", "random_noise"], default="image_guided",
        help="Reconstruction mode",
    )
    parser.add_argument(
        "--investigate", action="store_true",
        help="Save intermediate noising steps for inspection",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Optional base seed for deterministic (reproducible) sampling.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (e.g., cuda:0, cpu)",
    )
    
    args = parser.parse_args()
    
    # Auto-infer paths from config if needed
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    joint_cfg = config.get("joint_training", {})
    gene_csv = args.gene_csv or joint_cfg.get("csv_path")
    tiles_dir = args.tiles_dir or joint_cfg.get("tiles_zip_dir")
    
    if not gene_csv or not os.path.exists(gene_csv):
        parser.error(f"Gene CSV not found: {gene_csv}")
    if not tiles_dir or not os.path.exists(tiles_dir):
        parser.error(f"Tiles directory not found: {tiles_dir}")
    
    main(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        gene_csv_path=gene_csv,
        tiles_dir=tiles_dir,
        save_dir=args.save_dir,
        patients=args.patients,
        n_tiles_per_patient=args.n_tiles_per_patient,
        investigate=args.investigate,
        mode=args.mode,
        seed=args.seed,
        device=args.device,
    )
