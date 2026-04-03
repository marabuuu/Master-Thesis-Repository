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
import sys
import time
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
    label_col: Optional[str] = None,
    gene_list_path: Optional[str] = None,
    norm_means: Optional[np.ndarray] = None,
    norm_stds: Optional[np.ndarray] = None,
    apply_log1p: Optional[bool] = None,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Load gene expression from CSV.
    
    Returns:
        gene_data: {patient_id: (n_genes,)}
        gene_names: list of gene column names
    """
    logger.info(f"Loading gene expression from {csv_path}")
    df = pd.read_csv(csv_path)
    
    if patient_col not in df.columns:
        raise KeyError(f"Patient column '{patient_col}' not found in {csv_path}")

    # Match joint_training dataset preprocessing exactly:
    # 1) drop label col (if configured)
    # 2) numeric coercion + NaN handling
    # 3) optional gene subset
    # 4) conditional log1p
    # 5) z-score normalization
    work = df.copy()
    if label_col and label_col in work.columns:
        work = work.drop(columns=[label_col])

    metadata_cols = {patient_col, "label", "Label", "SubType", "subtype"}
    gene_cols = [c for c in work.columns if c not in metadata_cols]
    gene_df = work[gene_cols].apply(pd.to_numeric, errors="coerce")
    gene_df = gene_df.dropna(axis=1, how="all")
    gene_df = gene_df.fillna(0.0)

    if gene_list_path and Path(gene_list_path).exists():
        with open(gene_list_path) as f:
            selected_genes = [line.strip() for line in f if line.strip()]
        available = [g for g in selected_genes if g in gene_df.columns]
        if available:
            gene_df = gene_df[available]
            logger.info(f"Using {len(available)} genes from gene list: {gene_list_path}")

    gene_cols = list(gene_df.columns)
    logger.info(f"Found {len(gene_cols)} gene columns after preprocessing")

    values = gene_df.values.astype(np.float64)

    # Apply log1p using the training decision when provided, otherwise infer.
    if apply_log1p is not None:
        if apply_log1p:
            values = np.log1p(values)
    else:
        positive = values[values > 0]
        if positive.size > 0 and float(np.median(positive)) > 2.0:
            values = np.log1p(values)

    # Use training-fitted normalization stats when available to avoid distribution
    # shift between the training set and the inference set.
    if norm_means is not None and norm_stds is not None:
        means = np.asarray(norm_means, dtype=np.float64)
        stds = np.asarray(norm_stds, dtype=np.float64).copy()
        stds[stds < 1e-8] = 1.0
        logger.info("Using training-fitted normalization statistics for gene expression")
    else:
        means = values.mean(axis=0)
        stds = values.std(axis=0)
        stds[stds < 1e-8] = 1.0
        logger.warning(
            "No training normalization stats provided — computing from inference set. "
            "This may introduce a distribution shift vs. training. "
            "Pass norm_means/norm_stds from getattr(model, '_norm_stats', None)."
        )
    values = (values - means) / stds

    # Filter patients after normalization stats are computed on full table.
    patient_ids_canonical = {pid.upper() for pid in patient_ids} if patient_ids is not None else None

    gene_data: Dict[str, np.ndarray] = {}
    for row_idx, (_, row) in enumerate(df.iterrows()):
        pid = extract_patient_id(str(row[patient_col]))
        if patient_ids_canonical is not None and pid not in patient_ids_canonical:
            continue
        gene_data[pid] = values[row_idx].astype(np.float32)

    if patient_ids_canonical is not None:
        logger.info(f"Filtered to {len(gene_data)} patients")
    
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

def _ensure_mopadi_import_path() -> None:
    """Ensure local mopadi source is importable for checkpoint unpickling.

    Some checkpoints store objects under ``mopadi.configs`` in hyperparameters.
    When loading outside SLURM, ``PYTHONPATH`` may miss the local mopadi source.
    """
    env_path = os.getenv("MOPADI_SRC")
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path(env_path) if env_path else None,
        repo_root / "mopadi" / "src",
        repo_root.parent / "mopadi" / "src",
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        candidate = candidate.resolve()
        if (candidate / "mopadi").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
                logger.info(f"Added mopadi import path: {candidate_str}")
            break

def _detect_joint_variant(hp: dict, joint_cfg: dict) -> str:
    """Infer which joint-training variant produced a checkpoint."""
    variant = hp.get("joint_variant")
    if isinstance(variant, str) and variant:
        return variant

    has_cross = "cross_cfg" in hp or "cross_attention" in joint_cfg
    has_gene_token = "gene_token_transformer" in hp or any(
        k in joint_cfg
        for k in (
            "gene_token_transformer",
            "gene_token_transformer_joint_training",
            "gene_token_d_model",
            "gene_token_n_heads",
            "gene_token_n_layers",
        )
    )

    if has_cross and has_gene_token:
        return "gene_token_cross_attention_joint_training"
    if has_gene_token:
        return "gene_token_transformer_joint_training"
    if has_cross:
        return "cross_attention_joint_training"
    return "joint_training"


def _resolve_joint_model_class(variant: str):
    """Resolve Lightning model class for a given joint-training variant."""
    if variant == "joint_training":
        try:
            from src.joint_training.model import JointLitModel  # type: ignore[import-not-found]
        except ImportError:
            from ..joint_training.model import JointLitModel  # type: ignore[import-not-found]
        return JointLitModel

    if variant == "cross_attention_joint_training":
        try:
            from src.cross_attention_joint_training.model import CrossAttentionJointLitModel  # type: ignore[import-not-found]
        except ImportError:
            from ..cross_attention_joint_training.model import CrossAttentionJointLitModel  # type: ignore[import-not-found]
        return CrossAttentionJointLitModel

    if variant == "gene_token_transformer_joint_training":
        try:
            from src.gene_token_transformer_joint_training.model import GeneTokenTransformerJointLitModel  # type: ignore[import-not-found]
        except ImportError:
            from ..gene_token_transformer_joint_training.model import GeneTokenTransformerJointLitModel  # type: ignore[import-not-found]
        return GeneTokenTransformerJointLitModel

    if variant == "gene_token_cross_attention_joint_training":
        try:
            from src.gene_token_cross_attention_joint_training.model import GeneTokenCrossAttentionJointLitModel  # type: ignore[import-not-found]
        except ImportError:
            from ..gene_token_cross_attention_joint_training.model import GeneTokenCrossAttentionJointLitModel  # type: ignore[import-not-found]
        return GeneTokenCrossAttentionJointLitModel

    raise ValueError(
        f"Unsupported joint training variant '{variant}'. "
        "Expected one of: joint_training, cross_attention_joint_training, "
        "gene_token_transformer_joint_training, gene_token_cross_attention_joint_training"
    )


def _sanitize_joint_cfg_for_inference(joint_cfg: dict) -> dict:
    """Return a copy of joint_cfg with constructor-only preload ckpts disabled.

    During reconstruction we load the full state dict from the target checkpoint,
    so constructor-side optional preload checkpoints are unnecessary and can
    stall if paths point to slow/unavailable network mounts.
    """
    cfg = dict(joint_cfg)
    cfg["diffusion_ckpt"] = None
    cfg["encoder_ckpt"] = None
    return cfg

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
        _ensure_mopadi_import_path()
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
    
    variant = _detect_joint_variant(hp, joint_cfg if isinstance(joint_cfg, dict) else {})

    logger.info("✅ Detected JOINT TRAINING checkpoint format")
    logger.info(f"   conf: {type(conf).__name__}")
    logger.info(f"   joint_cfg: {type(joint_cfg).__name__} with keys: {list(joint_cfg.keys()) if isinstance(joint_cfg, dict) else 'N/A'}")
    logger.info(f"   n_genes: {n_genes}")
    logger.info(f"   variant: {variant}")
    
    # Load the exact model class variant used during training
    try:
        model_cls = _resolve_joint_model_class(variant)
    except ImportError as e:
        raise ImportError(f"Could not import model class for variant '{variant}': {e}")

    safe_joint_cfg = _sanitize_joint_cfg_for_inference(joint_cfg)
    
    try:
        logger.info(f"Creating {model_cls.__name__}...")
        t0 = time.time()
        model = model_cls(conf, safe_joint_cfg, n_genes)
        logger.info(f"Model init finished in {time.time() - t0:.1f}s")
        logger.info(f"✅ {model_cls.__name__} created")
    except Exception as e:
        raise ValueError(f"Failed to create {model_cls.__name__}: {e}")
    
    try:
        logger.info("Loading state dict...")
        t1 = time.time()
        model.load_state_dict(ckpt["state_dict"])
        logger.info(f"State dict load finished in {time.time() - t1:.1f}s")
        logger.info("✅ State dict loaded")
    except Exception as e:
        raise ValueError(f"Failed to load state dict into JointLitModel: {e}")
    
    model = model.to(device)
    model.eval()

    # Load training normalization stats so callers can pass them to
    # load_gene_expression() and avoid distribution shift at inference time.
    # Stats are saved by JointLitModel.setup() into <out_dir>/normalization_stats.json.
    model._norm_stats = None  # type: ignore[attr-defined]
    if isinstance(joint_cfg, dict):
        out_dir = joint_cfg.get("out_dir", "")
        if out_dir:
            norm_stats_path = Path(out_dir) / "normalization_stats.json"
            if norm_stats_path.exists():
                with open(norm_stats_path) as _nf:
                    _ns = json.load(_nf)
                model._norm_stats = {  # type: ignore[attr-defined]
                    "means": np.array(_ns["means"], dtype=np.float64),
                    "stds": np.array(_ns["stds"], dtype=np.float64),
                    "apply_log1p": bool(_ns["apply_log1p"]),
                }
                logger.info(f"Loaded normalization stats from {norm_stats_path}")
            else:
                logger.warning(
                    f"normalization_stats.json not found in {out_dir}. "
                    "Gene expression will be normalised from the inference CSV — "
                    "potential distribution shift vs. training. "
                    "Run at least one training epoch to generate the stats file."
                )

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
    cond_invert: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
    seed: Optional[int] = None,
    save_dir: Optional[Path] = None,
    n_steps: int = 5,
    inversion_steps: int = 250,
    decode_steps: Optional[int] = None,
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
        Conditioning vector used for **decoding** (shape (1, C)).  In
        cross-patient conditioning this is the target patient's genomic vector.
    cond_invert:
        Conditioning vector used for **inversion** (DDIM reverse pass).  When
        provided this should be the tile patient's own conditioning so that the
        inversion is self-consistent and the noise ``x_T`` encodes pure content.
        If ``None``, ``cond`` is reused for inversion (self-reconstruction).
    guidance_scale:
        Classifier-free guidance scale applied during **decoding only**.
        1.0 = no guidance (standard sampling).  Values > 1 amplify the
        conditioning signal: ``ε_guided = ε_uncond + scale*(ε_cond−ε_uncond)``.
        Requires the model to have been trained with ``cond_dropout_prob > 0``.
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
    def _make_sampler(steps: Optional[int], fallback_to_eval: bool = True):
        if steps is None:
            sampler_obj = getattr(model, "eval_sampler", None) if fallback_to_eval else None
            sampler_obj = sampler_obj or getattr(model, "sampler", None)
            if sampler_obj is None:
                raise RuntimeError("Model does not expose a diffusion sampler")
            return sampler_obj

        conf_obj = getattr(model, "conf", None)
        if conf_obj is not None and hasattr(conf_obj, "_make_diffusion_conf"):
            return conf_obj._make_diffusion_conf(T=int(steps)).make_sampler()  # type: ignore[attr-defined]

        sampler_obj = getattr(model, "sampler", None)
        if sampler_obj is None:
            raise RuntimeError("Could not construct sampler for custom steps")
        return sampler_obj

    if seed is not None:
        torch.manual_seed(seed)

    inversion_sampler = _make_sampler(inversion_steps, fallback_to_eval=False)
    decode_sampler = _make_sampler(decode_steps if decode_steps is not None else inversion_steps)

    T_inv = getattr(inversion_sampler, "num_timesteps", None)
    if T_inv is None:
        T_inv = len(getattr(inversion_sampler, "betas", []))
    if not isinstance(T_inv, int) or T_inv <= 0:
        raise RuntimeError("Could not determine inversion timesteps from sampler")

    T_dec = getattr(decode_sampler, "num_timesteps", None)
    if T_dec is None:
        T_dec = len(getattr(decode_sampler, "betas", []))
    if not isinstance(T_dec, int) or T_dec <= 0:
        raise RuntimeError("Could not determine decode timesteps from sampler")

    # Use EMA UNet if available
    unet_model = getattr(model, "ema_model", getattr(model, "model", model))
    unet_model.eval()

    # Wrap UNet with classifier-free guidance for the decode pass.
    # The inversion pass always uses the bare UNet (no CFG) so that the
    # encoded noise faithfully represents the tile content.
    if guidance_scale > 1.0:
        decode_unet = _CFGWrapper(unet_model, scale=guidance_scale)
        logger.info(f"  [CFG] guidance_scale={guidance_scale:.1f} applied to decode pass")
    else:
        decode_unet = unet_model

    # For cross-patient conditioning: invert with the tile patient's own cond so
    # that x_T captures pure content under the tile patient's genomic prior.
    # Decoding then uses the target patient's cond to apply their genomic style.
    # If cond_invert is None we fall back to cond (self-reconstruction path).
    _cond_for_inversion = cond_invert if cond_invert is not None else cond

    decode_eta = 0.0  # deterministic decode
    logger.info(
        f"  Encoding tile to noise using DDIM reverse inversion "
        f"(T_inv={T_inv}, T_dec={T_dec}, eta={decode_eta}, "
        f"cross_cond={'yes' if cond_invert is not None else 'no'})..."
    )
    with torch.no_grad():
        inversion_out = inversion_sampler.ddim_reverse_sample_loop(
            model=unet_model,
            x=img_tensor,
            clip_denoised=True,
            model_kwargs={"cond": _cond_for_inversion},
            eta=0.0,
            device=device,
        )
        x_T = inversion_out["sample"]

    # Save forward trajectory snapshots if requested
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        def _to_uint8(img: torch.Tensor) -> np.ndarray:
            # Keep fixed scaling in diffusion image range [-1, 1] to avoid amplifying
            # weak residual structure through per-image min-max normalization.
            arr = img.detach().cpu().clamp(-1, 1)
            arr = ((arr + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            return arr.permute(1, 2, 0).numpy()

        sample_t = inversion_out.get("sample_t", [])
        target_indices = sorted(set(np.linspace(0, max(0, len(sample_t) - 1), n_steps, dtype=int)))
        logger.info(
            f"  [DEBUG] Target inversion indices for saving (DDIM reverse): {target_indices}"
        )
        for idx in target_indices:
            xt = sample_t[idx]
            save_path = save_dir / f"forward_t{idx:04d}.png"
            Image.fromarray(_to_uint8(xt.squeeze(0))).save(save_path)
            logger.info(f"  Saved forward inversion step idx={idx} -> {save_path.name}")

        # Save final encoded noise (already computed as x_T)
        encoded_noise_path = save_dir / "encoded_noise.png"
        Image.fromarray(_to_uint8(x_T.squeeze(0))).save(encoded_noise_path)
        logger.info(f"  Saved final encoded noise -> {encoded_noise_path.name}")

    # Denoising (reverse diffusion) with deterministic sampling.
    # decode_unet is either the bare EMA model (guidance_scale=1.0) or a
    # _CFGWrapper that amplifies the conditioning signal.
    prog = decode_sampler.ddim_sample_loop_progressive(
        model=decode_unet,
        shape=img_tensor.shape,
        noise=x_T,
        model_kwargs={"cond": cond},
        device=device,
        progress=False,
        eta=decode_eta,
    )

    final_reconstruction: torch.Tensor | None = None
    save_denoise_timesteps = set(np.linspace(0, T_dec - 1, n_steps, dtype=int)) if save_dir is not None else set()

    logger.info(f"  [DEBUG] Save denoise timesteps (from T_dec={T_dec}): {sorted(save_denoise_timesteps)}")
    
    denoise_frame_count = 0
    total_iter = 0
    for idx, out in enumerate(prog):
        total_iter = idx + 1
        t = T_dec - 1 - idx
        if t in save_denoise_timesteps:
            out_img = out["sample"].squeeze(0)
            save_path = cast(Path, save_dir) / f"denoise_t{int(t):04d}.png"
            Image.fromarray(tensor_to_image(out_img)).save(save_path)
            logger.info(f"  Saved denoise step t={t} -> {save_path.name}")
            denoise_frame_count += 1
        final_reconstruction = out["sample"]
    
    logger.info(f"  [DEBUG] Total denoise frames saved: {denoise_frame_count} out of {total_iter} iterations")

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
    genomic_tile: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
    seed: Optional[int] = None,
    investigate: bool = False,
    investigate_dir: Optional[Path] = None,
    n_steps: int = 5,
    inversion_steps: int = 250,
    decode_steps: Optional[int] = None,
) -> Tuple[torch.Tensor, np.ndarray, Dict]:
    """Reconstruct a tile using diffusion encoding/decoding + genomic conditioning.

    Parameters
    ----------
    genomic:
        Genomic conditioning vector for the **target** patient (used during
        the DDIM decoding / reverse-diffusion pass).
    genomic_tile:
        Genomic vector of the patient whose **tile** is being encoded.  When
        provided it is used for the DDIM inversion pass so that the content
        noise ``x_T`` is computed under the tile patient's own genomic prior.
        For cross-patient conditioning this should always be supplied; omitting
        it causes inversion and decoding to use the same (target) conditioning,
        which mathematically guarantees reconstruction ≈ original tile.

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

        # Encode tile patient's own genomic vector for the inversion pass so that
        # x_T is computed under the tile patient's prior (cross-conditioning fix).
        cond_invert: Optional[torch.Tensor] = None
        if genomic_tile is not None:
            cond_invert = model.encode(genomic_tile.unsqueeze(0))  # type: ignore[attr-defined]
            logger.info(
                "  [DEBUG] cond_invert stats (tile patient): mean=%.4f std=%.4f first5=%s"
                % (
                    cond_invert.mean().item(),
                    cond_invert.std().item(),
                    [float(x) for x in cond_invert[0, :5].tolist()],
                )
            )

        recon_tensor = _reconstruct_from_image_with_cond(
            model,
            img_tensor,
            cond,
            device=device,
            cond_invert=cond_invert,
            guidance_scale=guidance_scale,
            seed=seed,
            save_dir=investigate_dir if investigate else None,
            n_steps=n_steps,
            inversion_steps=inversion_steps,
            decode_steps=decode_steps,
        )

    # Create investigation strip if frames were saved
    if investigate and investigate_dir is not None:
        create_investigation_strip(investigate_dir, n_steps=n_steps)

    recon_image = tensor_to_image(recon_tensor)
    metrics = compute_metrics(img_array, recon_image)

    return recon_tensor, recon_image, metrics


class _CFGWrapper(torch.nn.Module):
    """Classifier-free guidance wrapper for DDIM decoding.

    At each denoising step computes:
        ε_guided = ε_uncond + scale * (ε_cond − ε_uncond)

    This amplifies the conditioning signal at inference time.  The model must
    have been trained with a non-zero ``cond_dropout_prob`` so that it has
    learnt a meaningful unconditioned (zero-cond) behaviour.

    Only used for the **decoding** (reverse diffusion) pass.  The inversion pass
    should use the regular model to ensure a faithful content encoding.
    """

    def __init__(self, model: torch.nn.Module, scale: float) -> None:
        super().__init__()
        self.model = model
        self.scale = scale

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None, **kwargs):
        from types import SimpleNamespace
        out_cond = self.model(x, t, cond=cond, **kwargs)
        cond_zeros = torch.zeros_like(cond) if cond is not None else None
        out_uncond = self.model(x, t, cond=cond_zeros, **kwargs)
        pred_guided = out_uncond.pred + self.scale * (out_cond.pred - out_uncond.pred)
        return SimpleNamespace(pred=pred_guided)


def _reconstruct_from_noise_with_cond(
    model: torch.nn.Module,
    cond: torch.Tensor,
    noise: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Decode from noise using the diffusion sampler (same as training)."""
    sampler = getattr(model, "eval_sampler", None) or getattr(model, "sampler", None)
    if sampler is None:
        raise RuntimeError("Model does not expose `sampler` for diffusion decoding")

    unet_model = getattr(model, "ema_model", getattr(model, "model", model))
    unet_model.eval()

    if hasattr(sampler, "ddim_sample_loop_progressive"):
        prog = sampler.ddim_sample_loop_progressive(
            model=unet_model,
            shape=noise.shape,
            noise=noise,
            model_kwargs={"cond": cond},
            device=device,
            progress=False,
            eta=0.0,
        )
        final_reconstruction: torch.Tensor | None = None
        for out in prog:
            final_reconstruction = out["sample"]
        if final_reconstruction is None:
            raise RuntimeError("No reconstruction produced from DDIM sampling loop")
        return final_reconstruction

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
    guidance_scale: float = 1.0,
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

            unet_model = getattr(model, "ema_model", getattr(model, "model", model))
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
            # For consistency with the investigate=True path, use the EMA model
            # and prefer DDIM sampling when available.
            sampler = getattr(model, "eval_sampler", None) or getattr(model, "sampler", None)
            unet_model = getattr(model, "ema_model", getattr(model, "model", model))
            unet_model.eval()
            if guidance_scale > 1.0:
                unet_model = _CFGWrapper(unet_model, scale=guidance_scale)
                logger.info(f"  [CFG] guidance_scale={guidance_scale:.1f} applied to random-noise sampling")
            logger.debug(f"Sampling using sampler={type(sampler).__name__ if sampler is not None else None} unet={type(unet_model).__name__}")

            if sampler is not None and hasattr(sampler, "ddim_sample_loop_progressive"):
                prog = sampler.ddim_sample_loop_progressive(
                    model=unet_model,
                    shape=noise.shape,
                    noise=noise,
                    model_kwargs={"cond": cond},
                    device=device,
                    progress=False,
                    eta=0.0,
                )
                final_reconstruction = None
                for out in prog:
                    final_reconstruction = out["sample"]
                if final_reconstruction is None:
                    raise RuntimeError("No reconstruction produced from DDIM progressive sampler")
                recon_tensor = final_reconstruction
            else:
                # Fallback: use p_sample_loop_progressive (DDPM) if DDIM not available
                recon_tensor = _reconstruct_from_noise_with_cond(model, cond, noise, device)

    # Debug: log tensor stats before conversion
    if isinstance(recon_tensor, torch.Tensor):
        arr = recon_tensor.detach().cpu().numpy()
        logger.info(f"[DEBUG] recon_tensor shape: {arr.shape} dtype: {arr.dtype} min: {arr.min():.4f} max: {arr.max():.4f} mean: {arr.mean():.4f}")
        if arr.ndim == 3 and arr.shape[0] == 3:
            ch_means = arr.mean(axis=(1,2))
            logger.info(f"[DEBUG] recon_tensor per-channel means: R={ch_means[0]:.4f} G={ch_means[1]:.4f} B={ch_means[2]:.4f}")
        elif arr.ndim == 4 and arr.shape[1] == 3:
            ch_means = arr.mean(axis=(0,2,3))
            logger.info(f"[DEBUG] recon_tensor per-channel means: R={ch_means[0]:.4f} G={ch_means[1]:.4f} B={ch_means[2]:.4f}")
        else:
            logger.info(f"[DEBUG] recon_tensor not 3-channel, shape: {arr.shape}")
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

        sampler = getattr(model, "eval_sampler", None) or getattr(model, "sampler", None)
        if sampler is None:
            raise RuntimeError("Model does not expose a diffusion sampler")

        T = getattr(sampler, "num_timesteps", None)
        if T is None:
            betas = getattr(sampler, "betas", None)
            T = len(betas) if betas is not None else 0
        T = int(T)
        if T <= 0:
            raise RuntimeError("Could not determine diffusion timesteps from sampler")

        unet_model = getattr(model, "ema_model", getattr(model, "model", model))
        unet_model.eval()

        x_T: torch.Tensor
        denoise_timesteps: set[int]

        if mode == "image_guided":
            logger.info("  Running DDIM reverse inversion for investigation")
            inversion_out = sampler.ddim_reverse_sample_loop(
                model=unet_model,
                x=img_tensor,
                clip_denoised=True,
                model_kwargs={"cond": cond},
                eta=0.0,
                device=device,
            )
            x_T = inversion_out["sample"]

            sample_t = inversion_out.get("sample_t", [])
            forward_indices = sorted(set(np.linspace(0, max(0, len(sample_t) - 1), n_steps, dtype=int)))

            for idx in forward_indices:
                xt = sample_t[idx]
                save_path = output_dir / f"forward_t{idx:04d}.png"
                Image.fromarray(tensor_to_image(xt.squeeze(0))).save(save_path)
                logger.info(f"  Saved forward inversion idx={idx} -> {save_path.name}")

            noise_save_path = output_dir / "noise.png"
            Image.fromarray(tensor_to_image(x_T.squeeze(0))).save(noise_save_path)
            logger.info(f"  Saved encoded noise -> {noise_save_path.name}")

            denoise_timesteps = {int(t) for t in np.linspace(0, T - 1, n_steps, dtype=int)}
            denoise_timesteps.add(0)
        else:
            img_size = cast(int, getattr(model.conf, "img_size", 512))  # type: ignore[attr-defined]
            x_T = torch.randn(1, 3, int(img_size), int(img_size), device=device)  # type: ignore[call-arg]
            noise_save_path = output_dir / "noise.png"
            Image.fromarray(tensor_to_image(x_T.squeeze(0))).save(noise_save_path)
            logger.info(f"  Saved random noise -> {noise_save_path.name}")
            denoise_timesteps = set()

        logger.info("\nStarting DDIM reverse denoising (sampling) ...")
        prog = sampler.ddim_sample_loop_progressive(
            model=unet_model,
            shape=x_T.shape,
            noise=x_T,
            model_kwargs={"cond": cond},
            device=device,
            progress=False,
            eta=0.0,
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
    split: Optional[str] = None,
    subtypes: Optional[List[str]] = None,
    subtype_col: Optional[str] = None,
    conditioning_patients: Optional[List[str] | str] = None,
    patient_splits_path: Optional[str] = None,
    n_tiles_per_patient: int = 20,
    investigate: bool = False,
    mode: str = "image_guided",
    seed: Optional[int] = None,
    device: Optional[str] = None,
    inversion_steps: int = 250,
    decode_steps: Optional[int] = None,
    guidance_scale: float = 1.0,
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
    conditioning_patients : list of str, str, or None, optional
        Patient IDs (or split name) to use for genomic conditioning.
        If None, each tile patient is conditioned on its own RNA profile (default).
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
    
    # Handle split name ("train", "val", "test") for tile source patients
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

    # If explicit split argument provided via CLI, it takes precedence
    if split is not None:
        # normalize to expected keywords
        if split not in ("train", "val", "test"):
            raise ValueError("--split must be one of: train, val, test")
        patients = split
        # reuse existing split handling by re-entering the split branch
        if patient_splits_path is None:
            patient_splits_path = rec_cfg.get("patient_splits_path")
        if patient_splits_path is None:
            raise ValueError(
                f"patient_splits_path is required when using --split {split}"
            )
        if not Path(patient_splits_path).exists():
            raise FileNotFoundError(f"patient_splits.json not found: {patient_splits_path}")
        with open(patient_splits_path) as f:
            splits = json.load(f)
        split_data = splits.get(split)
        split_patients = split_data.get("patients", split_data) if isinstance(split_data, dict) else split_data
        patient_ids_list = cast(List[str], split_patients if isinstance(split_patients, list) else [])
        logger.info(f"Using {len(patient_ids_list)} patients from '--split {split}'")

    # Apply subtype filtering if requested
    if subtypes:
        # Resolve subtype column name from arguments or config
        subtype_col_resolved = (
            subtype_col
            or rec_cfg.get("subtype_col")
            or joint_cfg.get("label_col")
            or (joint_cfg_ckpt.get("label_col") if joint_cfg_ckpt else None)
            or "Majority_Subtype_mRNA"
        )
        logger.info(f"Filtering patients by subtypes {subtypes} using column '{subtype_col_resolved}'")
        df_raw = pd.read_csv(gene_csv_path)
        if subtype_col_resolved not in df_raw.columns:
            raise KeyError(f"Subtype column '{subtype_col_resolved}' not found in {gene_csv_path}")

        # Build set of patient IDs matching requested subtypes (case-insensitive)
        wanted = {s.upper() for s in subtypes}
        matching = set()
        pid_col = joint_cfg_ckpt.get("patient_col") if joint_cfg_ckpt else "Patient_ID"
        for _, row in df_raw.iterrows():
            raw_pid = row.get(pid_col, row.get("Patient_ID", ""))
            pid = extract_patient_id(str(raw_pid))
            val = row.get(subtype_col_resolved)
            if pd.isna(val):
                continue
            if str(val).upper() in wanted:
                matching.add(pid)

        before = len(patient_ids_list)
        patient_ids_list = [p for p in patient_ids_list if p in matching]
        logger.info(f"Filtered patients by subtype: {before} -> {len(patient_ids_list)}")

    # Optional cross-patient conditioning IDs
    conditioning_ids_list: Optional[List[str]]
    if conditioning_patients is None:
        conditioning_ids_list = None
    elif isinstance(conditioning_patients, str) and conditioning_patients in ("train", "val", "test"):
        split_name = conditioning_patients
        if patient_splits_path is None:
            raise ValueError(
                f"conditioning_patients='{split_name}' requires 'patient_splits_path' in config or as argument"
            )
        if not Path(patient_splits_path).exists():
            raise FileNotFoundError(f"patient_splits.json not found: {patient_splits_path}")

        with open(patient_splits_path) as f:
            splits = json.load(f)
        if split_name not in splits:
            raise ValueError(
                f"Conditioning split '{split_name}' not found in {patient_splits_path}. "
                f"Available: {list(splits.keys())}"
            )
        split_data = splits[split_name]
        split_patients = split_data.get("patients", split_data) if isinstance(split_data, dict) else split_data
        conditioning_ids_list = cast(List[str], split_patients if isinstance(split_patients, list) else [])
        conditioning_ids_list = [extract_patient_id(p) for p in conditioning_ids_list]
        logger.info(f"Using {len(conditioning_ids_list)} conditioning patients from '{split_name}' split")
    elif isinstance(conditioning_patients, list):
        conditioning_ids_list = [extract_patient_id(p) for p in conditioning_patients]
        logger.info(f"Using {len(conditioning_ids_list)} specified conditioning patients")
    else:
        raise ValueError(
            "conditioning_patients must be None, a patient ID list, or one of: train/val/test"
        )

    if conditioning_ids_list is not None and len(conditioning_ids_list) == 0:
        raise ValueError("conditioning_patients resolved to an empty list")
    
    # Load gene expressions
    gene_patient_ids = set(patient_ids_list)
    if conditioning_ids_list is not None:
        gene_patient_ids.update(conditioning_ids_list)

    _norm_stats = getattr(model, "_norm_stats", None)
    gene_data, gene_names = load_gene_expression(
        gene_csv_path,
        sorted(gene_patient_ids),
        patient_col=joint_cfg_ckpt.get("patient_col", "Patient_ID"),
        label_col=joint_cfg_ckpt.get("label_col"),
        gene_list_path=joint_cfg_ckpt.get("gene_list_path"),
        norm_means=_norm_stats["means"] if _norm_stats else None,
        norm_stds=_norm_stats["stds"] if _norm_stats else None,
        apply_log1p=_norm_stats["apply_log1p"] if _norm_stats else None,
    )
    
    # Load tiles
    tiles_dir_path = Path(tiles_dir)
    patient_tiles = load_tiles_for_patients(
        tiles_dir_path,
        patient_ids_list,
        n_tiles_per_patient,
    )
    
    # Reconstruction loop
    results = []
    
    for patient_id, tiles_list in patient_tiles.items():
        logger.info(f"\nProcessing tile patient {patient_id} ({len(tiles_list)} tiles)")

        cond_ids_for_tiles = conditioning_ids_list if conditioning_ids_list is not None else [patient_id]
        cond_ids_for_tiles = [pid for pid in cond_ids_for_tiles if pid in gene_data]
        if len(cond_ids_for_tiles) == 0:
            logger.warning(f"No matching gene data for conditioning patients (tile patient={patient_id}), skipping")
            continue
        if conditioning_ids_list is not None:
            logger.info(f"Conditioning each tile with {len(cond_ids_for_tiles)} RNA profile(s)")
        
        orig_zip_path = orig_dir / f"{patient_id}.zip"

        with zipfile.ZipFile(orig_zip_path, 'w') as zf_orig:
            for cond_idx, cond_patient_id in enumerate(cond_ids_for_tiles):
                genes_np = gene_data[cond_patient_id]
                genomic_tensor = torch.from_numpy(genes_np).to(device_obj, dtype=torch.float32)

                if conditioning_ids_list is None:
                    recon_zip_path = recon_dir / f"{patient_id}.zip"
                else:
                    recon_zip_path = recon_dir / f"{patient_id}__cond_{cond_patient_id}.zip"

                logger.info(f"  Conditioning on RNA patient {cond_patient_id} -> {recon_zip_path.name}")

                with zipfile.ZipFile(recon_zip_path, 'w') as zf_recon:
                    for tile_path, tile_name in tiles_list:
                        # Deterministic seed for reproducible runs (and consistent
                        # investigation output). This is stable across Python processes.
                        if seed is not None:
                            tile_seed = int(
                                hashlib.sha256(
                                    f"{seed}_{patient_id}_{cond_patient_id}_{tile_name}".encode()
                                ).hexdigest(),
                                16,
                            ) % (2**32)
                        else:
                            tile_seed = None

                        inv_dir = None
                        if investigate:
                            inv_dir = (
                                save_dir_path
                                / "investigation"
                                / f"{patient_id}__cond_{cond_patient_id}_{Path(tile_name).stem}"
                            )

                        try:
                            if mode == "image_guided":
                                # Provide the tile patient's own genomic vector for
                                # inversion so that x_T is self-consistent with their
                                # genomic prior.  The decoding step uses genomic_tensor
                                # (the conditioning/target patient) to apply their style.
                                tile_genes_np = gene_data.get(patient_id)
                                genomic_tile_tensor = (
                                    torch.from_numpy(tile_genes_np).to(device_obj, dtype=torch.float32)
                                    if tile_genes_np is not None
                                    else None
                                )
                                recon_tensor, recon_image, metrics = reconstruct_tile_image_guided(
                                    model,
                                    tile_path,
                                    genomic_tensor,
                                    device_obj,
                                    genomic_tile=genomic_tile_tensor,
                                    guidance_scale=guidance_scale,
                                    seed=tile_seed,
                                    investigate=investigate,
                                    investigate_dir=inv_dir,
                                    n_steps=5,
                                    inversion_steps=inversion_steps,
                                    decode_steps=decode_steps,
                                )
                            elif mode == "random_noise":
                                recon_tensor, recon_image = reconstruct_tile_random_noise(
                                    model,
                                    genomic_tensor,
                                    device_obj,
                                    guidance_scale=guidance_scale,
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

                            # Save original image once (first conditioning pass only)
                            if cond_idx == 0:
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
                                "conditioning_patient_id": cond_patient_id,
                                "tile_name": tile_name,
                                "status": "success",
                                **metrics,
                            }
                            results.append(result_row)
                            logger.debug(f"  ✓ {tile_name} (cond={cond_patient_id})")

                        except Exception as e:
                            logger.error(f"  ✗ {tile_name} (cond={cond_patient_id}): {e}")
                            results.append({
                                "patient_id": patient_id,
                                "conditioning_patient_id": cond_patient_id,
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
        "--split", type=str, choices=["train", "val", "test"], default=None,
        help="Optional split name to select patients from (train/val/test)",
    )
    parser.add_argument(
        "--subtypes", type=str, nargs="+", default=None,
        help=(
            "Optional list of subtype group names to filter patients by (matches column in gene CSV), "
            "e.g. --subtypes LumA Basal"
        ),
    )
    parser.add_argument(
        "--subtype-col", type=str, default=None,
        help="Column name in gene CSV that contains subtype labels (default inferred from config)",
    )
    parser.add_argument(
        "--conditioning-patients", type=str, nargs="+", default=None,
        help=(
            "Optional patient IDs used for genomic conditioning. "
            "If omitted, each tile patient uses its own RNA profile."
        ),
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
    parser.add_argument(
        "--inversion-steps", type=int, default=250,
        help="Number of DDIM reverse inversion steps for image-guided mode (MoPaDi-style).",
    )
    parser.add_argument(
        "--decode-steps", type=int, default=None,
        help="Optional number of DDIM decode steps. Defaults to --inversion-steps.",
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
        split=args.split,
        subtypes=args.subtypes,
        subtype_col=args.subtype_col,
        conditioning_patients=args.conditioning_patients,
        n_tiles_per_patient=args.n_tiles_per_patient,
        investigate=args.investigate,
        mode=args.mode,
        seed=args.seed,
        device=args.device,
        inversion_steps=args.inversion_steps,
        decode_steps=args.decode_steps,
    )
