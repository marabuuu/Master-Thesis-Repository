#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FID-based subtype separability evaluation.

Generates tiles for LumA and Basal test patients via random-noise diffusion
conditioned on each patient's genomic vector, then computes a 2×2 Fréchet
Distance matrix using InceptionV3 pool3 features (pytorch-fid style):

    rows = {real LumA,  real Basal}
    cols = {gen  LumA,  gen  Basal}

    FD(real_A, gen_A)   FD(real_A, gen_B)
    FD(real_B, gen_A)   FD(real_B, gen_B)

Diagonal = within-class FD (fidelity; lower = better).
Off-diagonal = cross-class FD (separation; higher = better conditioning signal).

Class imbalance note
--------------------
LumA has more test patients than Basal (51 vs 14). Because we target a fixed
*total* tile count per subtype (``--n-tiles``), both distributions always have
the same number of samples entering the Gaussian fit — we simply generate more
tiles per Basal patient. This keeps all four FD values directly comparable.

Usage
-----
# Quick smoke-test (5 tiles per patient, ~5-10 min on GPU):
python -m src.quality_assurance.fid_subtype_evaluation \\
    --checkpoint experiments/20260407.../joint/last.ckpt \\
    --experiment-dir experiments/20260407... \\
    --gene-csv dataframes/brca_gene_expression_with_subtypes.csv \\
    --tiles-dir ../data/BRCA-tumor-tiles-corrected \\
    --output-dir experiments/20260407.../fid_evaluation \\
    --gene-list experiments/20260424.../gene_list.txt \\
    --test

# Full run (10 000 tiles per subtype, run via SLURM):
python -m src.quality_assurance.fid_subtype_evaluation \\
    --checkpoint experiments/20260407.../joint/last.ckpt \\
    --experiment-dir experiments/20260407... \\
    --gene-csv dataframes/brca_gene_expression_with_subtypes.csv \\
    --tiles-dir ../data/BRCA-tumor-tiles-corrected \\
    --output-dir experiments/20260407.../fid_evaluation \\
    --gene-list experiments/20260424.../gene_list.txt \\
    --n-tiles 10000
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Path helpers
# ─────────────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]          # Master-Thesis-Repository/
_WORKSPACE_ROOT = _REPO_ROOT.parent                        # genhist/ (experiments/ lives here)


def _resolve_path(p: str, must_exist: bool = True) -> Path:
    """Resolve a path that may be relative to the workspace root or the CWD.

    Resolution order:
      1. Absolute paths → used as-is.
      2. Workspace-root candidate (genhist/<p>) → preferred when it exists,
         so that ``experiments/...`` and ``dataframes/...`` work without ``../``.
      3. CWD-relative fallback.

    This avoids false positives from stray directories created inside the repo.
    """
    path = Path(p)
    if path.is_absolute():
        resolved = path
    else:
        workspace_candidate = (_WORKSPACE_ROOT / p).resolve()
        cwd_candidate = path.resolve()
        if workspace_candidate.exists():
            resolved = workspace_candidate
        elif cwd_candidate.exists():
            resolved = cwd_candidate
        else:
            resolved = workspace_candidate  # will trigger FileNotFoundError below

    if must_exist and not resolved.exists():
        raise FileNotFoundError(
            f"Path not found: {p}\n"
            f"  tried: {(_WORKSPACE_ROOT / p).resolve()}\n"
            f"  tried: {Path(p).resolve()}\n"
            "  Hint: use an absolute path, or pass the path relative to "
            "Master-Thesis-Repository/ with ../ prefix (e.g. ../data/...)"
        )
    return resolved


def _ensure_imports() -> None:
    """Add mopadi and src/drafts to sys.path so checkpoints can be unpickled."""
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root.parent / "mopadi" / "src",
        repo_root / "mopadi" / "src",
    ]
    for c in candidates:
        if (c / "mopadi").exists():
            s = str(c)
            if s not in sys.path:
                sys.path.insert(0, s)
            break
    drafts = str(repo_root / "src" / "drafts")
    if drafts not in sys.path:
        sys.path.insert(0, drafts)


_ensure_imports()


# ─────────────────────────────────────────────────────────────────────────────
#  InceptionV3 feature extractor  (pytorch-fid style)
# ─────────────────────────────────────────────────────────────────────────────

class _InceptionV3Extractor:
    """InceptionV3 pool3 features (2048-dim), matching pytorch-fid.

    Images are resized to 299×299, converted to [0, 1], then scaled to [-1, 1]
    (the normalization expected by the original TF InceptionV3 weights that
    pytorch-fid uses).
    """

    def __init__(self, device: str = "cpu") -> None:
        import torchvision.transforms as T
        from torchvision.models import Inception_V3_Weights, inception_v3

        logger.info("Loading InceptionV3 (pretrained)…")
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.aux_logits = False
        model.AuxLogits = None  # type: ignore[assignment]
        model.fc = torch.nn.Identity()  # pool3 → 2048-d features
        model.eval().to(device)
        self.model = model
        self.device = device
        # [-1, 1] normalization used by pytorch-fid / original TF inception
        self.transform = T.Compose([
            T.Resize(299),
            T.CenterCrop(299),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        logger.info(f"InceptionV3 ready on '{device}'")

    @torch.inference_mode()
    def encode_batch(self, pil_images: List[Image.Image]) -> np.ndarray:
        """Return pool3 features for a list of PIL images.

        Returns:
            float32 array of shape (N, 2048).
        """
        batch = torch.stack([self.transform(img) for img in pil_images]).to(self.device)
        out = self.model(batch)
        if isinstance(out, tuple):
            out = out[0]
        return out.float().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
#  Fréchet Distance
# ─────────────────────────────────────────────────────────────────────────────

def compute_statistics(features: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a multivariate Gaussian to a feature array of shape (N, D).

    A small diagonal regularisation (eps·I) is added to keep the covariance
    positive-definite when N < D (important in test/debug mode).
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    sigma += eps * np.eye(sigma.shape[0])
    return mu, sigma


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    """FD = ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2√(Σ1Σ2))."""
    from scipy.linalg import sqrtm

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            logger.warning("sqrtm produced large imaginary component; using real part")
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fd = float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)
    return fd


# ─────────────────────────────────────────────────────────────────────────────
#  Tile generation (batched random-noise diffusion)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_tiles_for_patient(
    model: torch.nn.Module,
    genomic_tensor: torch.Tensor,
    n_tiles: int,
    device: torch.device,
    guidance_scale: float,
    gen_batch_size: int,
    base_seed: Optional[int],
    patient_idx: int,
) -> List[np.ndarray]:
    """Generate ``n_tiles`` uint8 HWC numpy images for one patient."""
    import torchvision.transforms.functional as TF

    from src.reconstruction.reconstruct_tiles import reconstruct_tile_random_noise

    images: List[np.ndarray] = []
    for tile_idx in range(n_tiles):
        seed = None
        if base_seed is not None:
            seed = (base_seed + patient_idx * 100_000 + tile_idx) % (2 ** 32)
        _, img = reconstruct_tile_random_noise(
            model,
            genomic_tensor,
            device,
            guidance_scale=guidance_scale,
            seed=seed,
            investigate=False,
        )
        images.append(img)
    return images


# ─────────────────────────────────────────────────────────────────────────────
#  ZIP helpers
# ─────────────────────────────────────────────────────────────────────────────

_IMG_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _save_images_to_zip(images: List[np.ndarray], zip_path: Path, patient_id: str) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for idx, img in enumerate(images):
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="PNG")
            zf.writestr(f"{patient_id}_{idx:05d}.png", buf.getvalue())


def _sample_tiles_from_zip(
    src_zip: Path,
    n_tiles: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Sample up to ``n_tiles`` random tiles from a patient source ZIP."""
    with zipfile.ZipFile(src_zip, "r") as zf:
        members = [m for m in zf.namelist() if Path(m).suffix.lower() in _IMG_SUFFIXES]
        if not members:
            return []
        chosen = rng.choice(members, size=min(n_tiles, len(members)), replace=False)
        imgs = []
        for name in chosen:
            with zf.open(name) as fh:
                imgs.append(np.array(Image.open(io.BytesIO(fh.read())).convert("RGB")))
    return imgs


def _iter_zip_images(zip_path: Path) -> List[Image.Image]:
    """Return all PIL images from a ZIP (for feature extraction)."""
    imgs = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if Path(name).suffix.lower() in _IMG_SUFFIXES:
                with zf.open(name) as fh:
                    imgs.append(Image.open(io.BytesIO(fh.read())).convert("RGB"))
    return imgs


# ─────────────────────────────────────────────────────────────────────────────
#  Per-subtype tile collection
# ─────────────────────────────────────────────────────────────────────────────

def generate_subtype_tiles(
    model: torch.nn.Module,
    patient_ids: List[str],
    gene_store,  # GeneExpressionStore
    n_tiles_total: int,
    out_dir: Path,
    device: torch.device,
    guidance_scale: float,
    gen_batch_size: int,
    seed: Optional[int],
    skip_existing: bool = True,
) -> None:
    """Generate random-noise tiles for all patients of one subtype.

    Tiles are saved as per-patient ZIPs: ``out_dir/<patient_id>.zip``.
    n_tiles_per_patient is set so the total across all patients reaches
    approximately ``n_tiles_total``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_patients = len(patient_ids)
    n_tiles_per_patient = max(1, math.ceil(n_tiles_total / n_patients))
    logger.info(
        f"  {n_patients} patients × {n_tiles_per_patient} tiles = "
        f"~{n_patients * n_tiles_per_patient} tiles total"
    )

    for p_idx, pid in enumerate(patient_ids):
        out_zip = out_dir / f"{pid}.zip"
        if skip_existing and out_zip.exists():
            logger.info(f"  [{p_idx+1}/{n_patients}] {pid}: skip (exists)")
            continue

        genomic_np = gene_store.get_vector(pid)
        if genomic_np is None:
            logger.warning(f"  {pid}: no genomic vector, skipping")
            continue

        genomic_t = torch.from_numpy(genomic_np).to(device, dtype=torch.float32)
        logger.info(f"  [{p_idx+1}/{n_patients}] {pid}: generating {n_tiles_per_patient} tiles…")

        imgs = _generate_tiles_for_patient(
            model=model,
            genomic_tensor=genomic_t,
            n_tiles=n_tiles_per_patient,
            device=device,
            guidance_scale=guidance_scale,
            gen_batch_size=gen_batch_size,
            base_seed=seed,
            patient_idx=p_idx,
        )
        _save_images_to_zip(imgs, out_zip, pid)
        logger.info(f"    → saved {len(imgs)} tiles to {out_zip.name}")


def collect_real_tiles(
    patient_ids: List[str],
    tiles_dir: Path,
    n_tiles_total: int,
    out_dir: Path,
    seed: Optional[int],
    skip_existing: bool = True,
) -> None:
    """Sample real tiles from the original patient ZIPs.

    Saves per-patient ZIPs to ``out_dir/<patient_id>.zip``.
    n_tiles_per_patient is chosen to reach ~n_tiles_total across all patients.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_patients = len(patient_ids)
    n_per_patient = max(1, math.ceil(n_tiles_total / n_patients))

    for p_idx, pid in enumerate(patient_ids):
        out_zip = out_dir / f"{pid}.zip"
        if skip_existing and out_zip.exists():
            continue

        # find the source ZIP (patient ID matching, case-insensitive)
        candidates = [
            p for p in tiles_dir.iterdir()
            if p.suffix.lower() == ".zip" and pid.upper() in p.stem.upper()
        ]
        if not candidates:
            logger.warning(f"  No source ZIP found for {pid} in {tiles_dir}")
            continue

        src_zip = candidates[0]
        imgs = _sample_tiles_from_zip(src_zip, n_per_patient, rng)
        if not imgs:
            logger.warning(f"  {pid}: no tiles found in {src_zip.name}")
            continue

        _save_images_to_zip(imgs, out_zip, pid)

    n_written = sum(1 for p in patient_ids if (out_dir / f"{p}.zip").exists())
    logger.info(f"  Real tiles ready for {n_written}/{n_patients} patients → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_inception_features(
    tile_dir: Path,
    extractor: _InceptionV3Extractor,
    batch_size: int = 32,
    label: str = "",
) -> np.ndarray:
    """Extract InceptionV3 pool3 features for all tiles in a directory of ZIPs.

    Returns float32 array of shape (N_tiles_total, 2048).
    """
    zip_files = sorted(tile_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No ZIP files found in {tile_dir}")

    all_features: List[np.ndarray] = []
    batch: List[Image.Image] = []
    n_tiles = 0

    for zp in zip_files:
        for img in _iter_zip_images(zp):
            batch.append(img)
            if len(batch) == batch_size:
                all_features.append(extractor.encode_batch(batch))
                n_tiles += len(batch)
                batch = []

    if batch:
        all_features.append(extractor.encode_batch(batch))
        n_tiles += len(batch)

    if not all_features:
        raise ValueError(f"No tiles found under {tile_dir}")

    features = np.concatenate(all_features, axis=0)
    logger.info(f"  {label}: {n_tiles} tiles → features {features.shape}")
    return features.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Main orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run(
    checkpoint_path: str,
    experiment_dir: str,
    gene_csv: str,
    tiles_dir: str,
    output_dir: str,
    gene_list_path: str,
    subtypes: Tuple[str, str] = ("LumA", "Basal"),
    subtype_col: str = "Majority_Subtype_mRNA",
    patient_col: str = "Patient_ID",
    n_tiles: int = 10_000,
    test_mode: bool = False,
    n_tiles_per_patient_test: int = 5,
    guidance_scale: float = 1.0,
    gen_batch_size: int = 1,
    inception_batch_size: int = 32,
    device: Optional[str] = None,
    seed: int = 42,
    skip_generate: bool = False,
    skip_real_tiles: bool = False,
    skip_features: bool = False,
) -> Dict:
    """Run the full FID subtype evaluation pipeline.

    Returns the FD matrix dict:
        {
          "real_LumA_gen_LumA": float,
          "real_LumA_gen_Basal": float,
          "real_Basal_gen_LumA": float,
          "real_Basal_gen_Basal": float,
        }
    """
    import pandas as pd

    from src.reconstruction.reconstruct_tiles import load_checkpoint, load_gene_expression_store

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    checkpoint_path = str(_resolve_path(checkpoint_path))
    exp_dir        = _resolve_path(experiment_dir)
    gene_csv       = str(_resolve_path(gene_csv))
    tiles_dir      = str(_resolve_path(tiles_dir))
    gene_list_path = str(_resolve_path(gene_list_path))

    # output_dir may not exist yet — resolve parent only
    out_dir = _WORKSPACE_ROOT / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subtype_a, subtype_b = subtypes

    # ── Patient splits ──────────────────────────────────────────────────────
    splits_path = exp_dir / "patient_splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"patient_splits.json not found at {splits_path}")
    with open(splits_path) as f:
        splits = json.load(f)
    test_entry = splits.get("test", {})
    test_patients_raw = (
        test_entry["patients"] if isinstance(test_entry, dict) else test_entry
    )
    test_patients: set[str] = set(p.upper() for p in test_patients_raw)
    logger.info(f"Test split: {len(test_patients)} patients")

    # ── Load gene CSV with subtype labels ────────────────────────────────────
    df = pd.read_csv(gene_csv)
    from src.reconstruction.utils import extract_patient_id

    df["_pid"] = df[patient_col].apply(lambda x: extract_patient_id(str(x)).upper())
    df_test = df[df["_pid"].isin(test_patients)]

    patients_a = sorted(df_test[df_test[subtype_col] == subtype_a]["_pid"].unique())
    patients_b = sorted(df_test[df_test[subtype_col] == subtype_b]["_pid"].unique())
    logger.info(f"{subtype_a} test patients: {len(patients_a)}")
    logger.info(f"{subtype_b} test patients: {len(patients_b)}")

    if not patients_a:
        raise ValueError(f"No {subtype_a} patients found in test split")
    if not patients_b:
        raise ValueError(f"No {subtype_b} patients found in test split")

    # ── Effective tile counts ────────────────────────────────────────────────
    if test_mode:
        tiles_a = len(patients_a) * n_tiles_per_patient_test
        tiles_b = len(patients_b) * n_tiles_per_patient_test
        effective_n_a = n_tiles_per_patient_test
        effective_n_b = n_tiles_per_patient_test
        logger.info(
            f"TEST MODE: {n_tiles_per_patient_test} tiles/patient → "
            f"{tiles_a} {subtype_a}, {tiles_b} {subtype_b}"
        )
    else:
        effective_n_a = n_tiles  # total for subtype; script distributes across patients
        effective_n_b = n_tiles
        logger.info(f"FULL MODE: targeting {n_tiles} tiles per subtype")

    # ── Load normalization stats & model ─────────────────────────────────────
    norm_stats_path = exp_dir / "normalization_stats.json"
    norm_means = norm_stds = None
    apply_log1p = None
    if norm_stats_path.exists():
        with open(norm_stats_path) as f:
            ns = json.load(f)
        norm_means = np.array(ns["means"], dtype=np.float64)
        norm_stds = np.array(ns["stds"], dtype=np.float64)
        apply_log1p = bool(ns["apply_log1p"])
        logger.info(f"Loaded normalization stats ({len(norm_means)} genes)")

    if not skip_generate:
        logger.info("Loading checkpoint…")
        model, _conf, joint_cfg = load_checkpoint(checkpoint_path, device_obj)
        model.eval()

        # ── Gene expression ──────────────────────────────────────────────────
        all_patients = sorted(set(patients_a) | set(patients_b))
        gene_store = load_gene_expression_store(
            gene_csv,
            patient_ids=all_patients,
            patient_col=patient_col,
            label_col=subtype_col,
            gene_list_path=gene_list_path,
            norm_means=norm_means,
            norm_stds=norm_stds,
            apply_log1p=apply_log1p,
        )

        # ── Generate tiles ───────────────────────────────────────────────────
        for subtype, patient_ids, eff_n in [
            (subtype_a, patients_a, effective_n_a),
            (subtype_b, patients_b, effective_n_b),
        ]:
            gen_dir = out_dir / "generated" / subtype
            logger.info(f"Generating {subtype} tiles → {gen_dir}")
            if test_mode:
                # In test mode, eff_n is per-patient, not total
                _gen_n_total = eff_n * len(patient_ids)
            else:
                _gen_n_total = eff_n
            generate_subtype_tiles(
                model=model,
                patient_ids=patient_ids,
                gene_store=gene_store,
                n_tiles_total=_gen_n_total,
                out_dir=gen_dir,
                device=device_obj,
                guidance_scale=guidance_scale,
                gen_batch_size=gen_batch_size,
                seed=seed,
                skip_existing=True,
            )
    else:
        logger.info("--skip-generate: skipping tile generation")

    # ── Collect real tiles ────────────────────────────────────────────────────
    if not skip_real_tiles:
        tiles_dir_path = Path(tiles_dir)
        for subtype, patient_ids, eff_n in [
            (subtype_a, patients_a, effective_n_a),
            (subtype_b, patients_b, effective_n_b),
        ]:
            real_dir = out_dir / "real" / subtype
            logger.info(f"Collecting real {subtype} tiles → {real_dir}")
            if test_mode:
                _real_n_total = eff_n * len(patient_ids)
            else:
                _real_n_total = eff_n
            collect_real_tiles(
                patient_ids=patient_ids,
                tiles_dir=tiles_dir_path,
                n_tiles_total=_real_n_total,
                out_dir=real_dir,
                seed=seed,
                skip_existing=True,
            )
    else:
        logger.info("--skip-real-tiles: skipping real tile collection")

    # ── Extract Inception features ────────────────────────────────────────────
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    distributions = {
        f"real_{subtype_a}": out_dir / "real" / subtype_a,
        f"real_{subtype_b}": out_dir / "real" / subtype_b,
        f"gen_{subtype_a}":  out_dir / "generated" / subtype_a,
        f"gen_{subtype_b}":  out_dir / "generated" / subtype_b,
    }

    features: Dict[str, np.ndarray] = {}

    if not skip_features:
        extractor = _InceptionV3Extractor(device=device)
        for key, tile_dir in distributions.items():
            npz_path = feat_dir / f"inception_{key}.npz"
            if npz_path.exists():
                logger.info(f"  {key}: loading cached features from {npz_path.name}")
                features[key] = np.load(npz_path)["features"]
            else:
                logger.info(f"  Extracting features for {key}…")
                feats = extract_inception_features(
                    tile_dir, extractor, batch_size=inception_batch_size, label=key
                )
                np.savez_compressed(npz_path, features=feats)
                features[key] = feats
                logger.info(f"    saved → {npz_path.name}")
    else:
        logger.info("--skip-features: loading cached features")
        for key in distributions:
            npz_path = feat_dir / f"inception_{key}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(
                    f"Feature file not found: {npz_path}. "
                    "Run without --skip-features first."
                )
            features[key] = np.load(npz_path)["features"]

    # ── Compute FD matrix ─────────────────────────────────────────────────────
    logger.info("Computing Fréchet Distances…")
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for key, feats in features.items():
        logger.info(f"  fitting Gaussian for {key} ({feats.shape[0]} tiles, {feats.shape[1]}-d)")
        stats[key] = compute_statistics(feats)

    rows = [f"real_{subtype_a}", f"real_{subtype_b}"]
    cols = [f"gen_{subtype_a}",  f"gen_{subtype_b}"]

    fd_matrix: Dict[str, float] = {}
    for row_key in rows:
        for col_key in cols:
            mu1, sig1 = stats[row_key]
            mu2, sig2 = stats[col_key]
            fd = frechet_distance(mu1, sig1, mu2, sig2)
            label = f"{row_key}_vs_{col_key}"
            fd_matrix[label] = fd
            logger.info(f"  FD({row_key}, {col_key}) = {fd:.2f}")

    # ── Save results ──────────────────────────────────────────────────────────
    tile_counts = {k: int(v.shape[0]) for k, v in features.items()}
    results = {
        "fd_matrix": fd_matrix,
        "tile_counts": tile_counts,
        "subtypes": list(subtypes),
        "test_mode": test_mode,
        "n_tiles_target": n_tiles,
        "guidance_scale": guidance_scale,
        "checkpoint": str(checkpoint_path),
        "rows": rows,
        "cols": cols,
    }
    results_path = out_dir / "fd_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {results_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        from src.visualization.fid_matrix_plot import plot_fd_matrix
        plot_path = out_dir / "fd_matrix.png"
        plot_fd_matrix(results, plot_path)
        logger.info(f"Plot saved → {plot_path}")
    except Exception as exc:
        logger.warning(f"Plotting failed: {exc}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FID-based subtype separability evaluation (LumA vs Basal)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="Path to joint training .ckpt")
    p.add_argument("--experiment-dir", required=True,
                   help="Experiment dir containing normalization_stats.json and patient_splits.json")
    p.add_argument("--gene-csv", required=True,
                   help="Gene expression CSV with subtype column (e.g. brca_gene_expression_with_subtypes.csv)")
    p.add_argument("--tiles-dir", required=True,
                   help="Directory with per-patient ZIP tile archives (real tiles)")
    p.add_argument("--output-dir", required=True, help="Output directory for all results")
    p.add_argument("--gene-list", required=True,
                   help="Path to gene_list.txt selecting the 512 training genes "
                        "(e.g. experiments/20260424.../gene_list.txt)")
    p.add_argument("--subtypes", nargs=2, default=["LumA", "Basal"],
                   metavar=("SUBTYPE_A", "SUBTYPE_B"))
    p.add_argument("--subtype-col", default="Majority_Subtype_mRNA")
    p.add_argument("--patient-col", default="Patient_ID")
    p.add_argument("--n-tiles", type=int, default=10_000,
                   help="Target total tiles per subtype for full run")
    p.add_argument("--test", action="store_true",
                   help="Test mode: 5 tiles per patient (~5-10 min on GPU)")
    p.add_argument("--n-tiles-per-patient-test", type=int, default=5)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--inception-batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-generate", action="store_true",
                   help="Skip tile generation (use existing ZIPs in output-dir/generated/)")
    p.add_argument("--skip-real-tiles", action="store_true",
                   help="Skip real tile collection (use existing ZIPs in output-dir/real/)")
    p.add_argument("--skip-features", action="store_true",
                   help="Skip feature extraction (use cached .npz in output-dir/features/)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    run(
        checkpoint_path=args.checkpoint,
        experiment_dir=args.experiment_dir,
        gene_csv=args.gene_csv,
        tiles_dir=args.tiles_dir,
        output_dir=args.output_dir,
        gene_list_path=args.gene_list,
        subtypes=tuple(args.subtypes),
        subtype_col=args.subtype_col,
        patient_col=args.patient_col,
        n_tiles=args.n_tiles,
        test_mode=args.test,
        n_tiles_per_patient_test=args.n_tiles_per_patient_test,
        guidance_scale=args.guidance_scale,
        inception_batch_size=args.inception_batch_size,
        device=args.device,
        seed=args.seed,
        skip_generate=args.skip_generate,
        skip_real_tiles=args.skip_real_tiles,
        skip_features=args.skip_features,
    )


if __name__ == "__main__":
    main()
