#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TFD Class Separability — Pairwise Topological Fréchet Distance Between PAM50 Subtypes

Repurposes the Topological Fréchet Distance from a *generation quality* metric
(reference vs synthetic) into a *class separability* metric: each PAM50 subtype is
treated as a distribution of cell spatial topologies.  Large pairwise TFD means the
two subtypes have topologically distinct cell arrangements; small TFD means they look
topologically similar.

Pipeline
--------
1. [auto_segment]  Run DeepCMorph on ``tiles_zip_dir`` (BRCA-tumor-tiles-corrected)
                   to produce per-patient mask ZIPs in ``masks_dir``.
                   Skipped when masks already exist and ``force_resegment=False``.
2. [load]          Read ``metadata_csv`` to map patient IDs → PAM50 subtypes.
                   For each class, stream ``_cls.npy`` masks from the relevant patient
                   ZIPs, applying ``max_tiles_by_subtype`` caps for class balance.
3. [fit]           Run ``compute_class_distribution()`` once per class to fit a
                   multivariate Gaussian over the 100-dim persistence landscape vectors.
4. [compare]       Compute pairwise Fréchet distance between all class Gaussians
                   → produces a symmetric N×N distance matrix.
5. [noise floor]   Optionally estimate the intra-class TFD by splitting each class
                   in half ``n_noise_floor_splits`` times.  Inter-class TFD should
                   substantially exceed this floor.
6. [output]        Save JSON results and a heatmap PNG.

Usage
-----
python run_pipeline.py --config src/config.yaml --stage tfd_separability
"""

from __future__ import annotations

import itertools
import json
import logging
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .topological_frechet_distance import (
    ClassDistribution,
    _calculate_frechet_distance,
    compute_class_distribution,
    compute_persistence_diagram_1d,
    extract_centres_multichannel,
    vectorise_persistence_landscape,
)
from .utils import extract_patient_id

logger = logging.getLogger(__name__)


# ===================================================================
# § 1  Result dataclass
# ===================================================================

@dataclass
class TFDSeparabilityResult:
    """Pairwise topological separability between histological classes.

    Attributes
    ----------
    class_names : list of str
        Sorted list of class labels analysed.
    pairwise_tfd : dict[(str, str), float]
        Mean TFD (averaged over cell-type channels) for each unordered pair.
    pairwise_per_channel : dict[(str, str), dict[int, float]]
        Per-channel TFD for each pair.
    noise_floor : dict[str, float] or None
        Intra-class TFD estimate (mean over random 50/50 splits).
        ``None`` when noise-floor computation is disabled.
    n_samples : dict[str, int]
        Number of masks loaded per class.
    """
    class_names: List[str]
    pairwise_tfd: Dict[Tuple[str, str], float]
    pairwise_per_channel: Dict[Tuple[str, str], Dict[int, float]]
    noise_floor: Optional[Dict[str, float]] = None
    n_samples: Dict[str, int] = field(default_factory=dict)

    def as_matrix(self) -> Tuple[List[str], np.ndarray]:
        """Return ``(class_names, symmetric_distance_matrix)``."""
        n = len(self.class_names)
        mat = np.zeros((n, n))
        idx = {c: i for i, c in enumerate(self.class_names)}
        for (a, b), d in self.pairwise_tfd.items():
            i, j = idx[a], idx[b]
            mat[i, j] = mat[j, i] = d
        return self.class_names, mat

    def to_dict(self) -> dict:
        def _safe(v: float) -> Optional[float]:
            return float(v) if np.isfinite(v) else None

        return {
            "class_names": self.class_names,
            "pairwise_tfd": {
                f"{a}__vs__{b}": _safe(v)
                for (a, b), v in self.pairwise_tfd.items()
            },
            "pairwise_per_channel": {
                f"{a}__vs__{b}": {str(k): _safe(v) for k, v in ch.items()}
                for (a, b), ch in self.pairwise_per_channel.items()
            },
            "noise_floor": (
                {k: _safe(v) for k, v in self.noise_floor.items()}
                if self.noise_floor is not None else None
            ),
            "n_samples": self.n_samples,
        }


# ===================================================================
# § 2  Core computation
# ===================================================================

def compute_tfd_separability(
    masks_per_class: Dict[str, List[np.ndarray]],
    *,
    use_alpha: bool = True,
    n_landscape_bins: int = 100,
    n_landscape_layers: int = 1,
    min_cells: int = 3,
    compute_noise_floor: bool = True,
    n_noise_floor_splits: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> TFDSeparabilityResult:
    """Compute pairwise TFD between histological classes.

    Parameters
    ----------
    masks_per_class : dict[str, list of np.ndarray]
        Class name → list of segmentation masks ``(H, W)`` or ``(H, W, C)``.
        Must be plain lists (not lazy generators) when ``compute_noise_floor=True``.
    compute_noise_floor : bool
        When True, estimate an intra-class TFD for each class by randomly
        splitting its masks in half ``n_noise_floor_splits`` times.
        Inter-class TFD should substantially exceed this noise floor.
    n_noise_floor_splits : int
        Number of random 50/50 splits averaged for each class noise floor.
    seed : int
        Random seed for reproducible noise-floor splits.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    TFDSeparabilityResult
    """
    class_names = sorted(masks_per_class.keys())
    rng = np.random.default_rng(seed)

    tda_kwargs = dict(
        use_alpha=use_alpha,
        n_landscape_bins=n_landscape_bins,
        n_landscape_layers=n_landscape_layers,
        min_cells=min_cells,
    )

    # --- Phase 1: per-class Gaussian distributions ---
    distributions: Dict[str, ClassDistribution] = {}
    for cls in class_names:
        if verbose:
            print(f"  [{cls}] Fitting persistence landscape distribution…")
        dist = compute_class_distribution(
            iter(masks_per_class[cls]),
            class_name=cls,
            **tda_kwargs,
        )
        distributions[cls] = dist
        if verbose:
            n = dist.n_samples.get(0, 0)
            print(f"  [{cls}] {n} tiles processed, {dist.n_channels} channels")

    # --- Phase 2: pairwise Fréchet distances ---
    pairwise_tfd: Dict[Tuple[str, str], float] = {}
    pairwise_per_channel: Dict[Tuple[str, str], Dict[int, float]] = {}

    for cls_a, cls_b in itertools.combinations(class_names, 2):
        da, db = distributions[cls_a], distributions[cls_b]
        per_ch: Dict[int, float] = {}

        for ch in range(da.n_channels):
            na = da.n_samples.get(ch, 0)
            nb = db.n_samples.get(ch, 0)
            if na < 2 or nb < 2:
                logger.warning(
                    "ch%d: %s vs %s — not enough samples (%d, %d), skipping",
                    ch, cls_a, cls_b, na, nb,
                )
                per_ch[ch] = float("nan")
                continue
            per_ch[ch] = _calculate_frechet_distance(
                da.mean[ch], da.cov[ch],
                db.mean[ch], db.cov[ch],
            )

        valid = [v for v in per_ch.values() if np.isfinite(v)]
        avg = float(np.mean(valid)) if valid else float("nan")
        pair = (cls_a, cls_b)
        pairwise_tfd[pair] = avg
        pairwise_per_channel[pair] = per_ch
        if verbose:
            print(f"  TFD({cls_a}, {cls_b}) = {avg:.4f}")

    # --- Phase 3: noise floor (optional) ---
    noise_floor: Optional[Dict[str, float]] = None
    if compute_noise_floor:
        noise_floor = {}
        if verbose:
            print("\n  Computing intra-class noise floor…")
        for cls in class_names:
            masks = masks_per_class[cls]
            n = len(masks)
            if n < 4:
                logger.warning("%s: only %d masks — skipping noise floor", cls, n)
                noise_floor[cls] = float("nan")
                continue

            split_fds: List[float] = []
            for _ in range(n_noise_floor_splits):
                indices = np.arange(n)
                rng.shuffle(indices)
                half = n // 2
                half_a = [masks[i] for i in indices[:half]]
                half_b = [masks[i] for i in indices[half:]]

                da_nf = compute_class_distribution(iter(half_a), **tda_kwargs)
                db_nf = compute_class_distribution(iter(half_b), **tda_kwargs)

                per_ch_nf: Dict[int, float] = {}
                for ch in range(da_nf.n_channels):
                    na = da_nf.n_samples.get(ch, 0)
                    nb = db_nf.n_samples.get(ch, 0)
                    if na < 2 or nb < 2:
                        per_ch_nf[ch] = float("nan")
                        continue
                    per_ch_nf[ch] = _calculate_frechet_distance(
                        da_nf.mean[ch], da_nf.cov[ch],
                        db_nf.mean[ch], db_nf.cov[ch],
                    )
                valid_nf = [v for v in per_ch_nf.values() if np.isfinite(v)]
                if valid_nf:
                    split_fds.append(float(np.mean(valid_nf)))

            noise_floor[cls] = float(np.mean(split_fds)) if split_fds else float("nan")
            if verbose:
                print(f"  Noise floor ({cls}) = {noise_floor[cls]:.4f}")

    n_samples = {cls: len(masks_per_class[cls]) for cls in class_names}
    return TFDSeparabilityResult(
        class_names=class_names,
        pairwise_tfd=pairwise_tfd,
        pairwise_per_channel=pairwise_per_channel,
        noise_floor=noise_floor,
        n_samples=n_samples,
    )


# ===================================================================
# § 3  Data loading helpers
# ===================================================================

def _load_masks_for_class(
    masks_dir: Path,
    patient_ids: List[str],
    *,
    max_tiles_per_patient: Optional[int],
    cls_suffix: str = "_cls.npy",
    seed: int = 42,
) -> List[np.ndarray]:
    """Load classification masks from per-patient ZIP archives for one class.

    Parameters
    ----------
    masks_dir : Path
        Directory containing ``<patient_id>.zip`` archives produced by the
        segmentation stage.
    patient_ids : list of str
        Canonical patient IDs (e.g. ``'TCGA-AR-A2LK'``) to include.
    max_tiles_per_patient : int or None
        Maximum tiles drawn randomly from each patient ZIP.  ``None`` = use all.
    cls_suffix : str
        Filename suffix that identifies classification mask files inside ZIPs.
    """
    rng = np.random.default_rng(seed)
    masks: List[np.ndarray] = []
    missing = 0

    for pid in patient_ids:
        zip_path = masks_dir / f"{pid}.zip"
        if not zip_path.exists():
            missing += 1
            continue

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                tile_names = [n for n in zf.namelist() if n.endswith(cls_suffix)]
                if max_tiles_per_patient is not None and len(tile_names) > max_tiles_per_patient:
                    chosen = rng.choice(len(tile_names), max_tiles_per_patient, replace=False)
                    tile_names = [tile_names[i] for i in chosen]

                for name in tile_names:
                    try:
                        with zf.open(name) as f:
                            m = np.load(BytesIO(f.read()))
                        # Normalise channel axis: ensure (H, W, C)
                        if m.ndim == 3 and m.shape[0] < m.shape[1]:
                            m = np.transpose(m, (1, 2, 0))
                        masks.append(m)
                    except Exception:
                        logger.warning("Failed to load %s from %s", name, zip_path)
        except Exception:
            logger.warning("Failed to open ZIP: %s", zip_path)

    if missing:
        logger.info(
            "%d/%d patient ZIPs not found in %s (segmentation may not have run for them)",
            missing, len(patient_ids), masks_dir,
        )
    return masks


def _iter_masks_for_class(
    masks_dir: Path,
    patient_ids: List[str],
    *,
    max_tiles_per_patient: Optional[int],
    cls_suffix: str = "_cls.npy",
    seed: int = 42,
) -> Iterable[np.ndarray]:
    """Yield classification masks lazily from per-patient ZIP archives.

    This avoids materialising all masks in RAM at once, which can exceed host
    memory for large cohorts.
    """
    rng = np.random.default_rng(seed)
    missing = 0

    for pid in patient_ids:
        zip_path = masks_dir / f"{pid}.zip"
        if not zip_path.exists():
            missing += 1
            continue

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                tile_names = [n for n in zf.namelist() if n.endswith(cls_suffix)]
                if max_tiles_per_patient is not None and len(tile_names) > max_tiles_per_patient:
                    chosen = rng.choice(len(tile_names), max_tiles_per_patient, replace=False)
                    tile_names = [tile_names[i] for i in chosen]

                for name in tile_names:
                    try:
                        with zf.open(name) as f:
                            m = np.load(BytesIO(f.read()))
                        if m.ndim == 3 and m.shape[0] < m.shape[1]:
                            m = np.transpose(m, (1, 2, 0))
                        yield m
                    except Exception:
                        logger.warning("Failed to load %s from %s", name, zip_path)
        except Exception:
            logger.warning("Failed to open ZIP: %s", zip_path)

    if missing:
        logger.info(
            "%d/%d patient ZIPs not found in %s (segmentation may not have run for them)",
            missing, len(patient_ids), masks_dir,
        )


# ===================================================================
# § 3b  Landscape-vector collection helpers (memory-efficient noise floor)
# ===================================================================

def _collect_landscape_vectors(
    mask_iter: Iterable[np.ndarray],
    *,
    use_alpha: bool = True,
    n_landscape_bins: int = 100,
    n_landscape_layers: int = 1,
    min_cells: int = 3,
) -> Dict[int, np.ndarray]:
    """Stream masks and collect per-tile persistence landscape vectors per channel.

    Stores ~100 floats per channel per tile (≈5.6 KB/tile) rather than the raw
    mask (≈256 KB/tile), keeping peak RAM below ~200 MB for 21k tiles.

    Returns
    -------
    dict[int, np.ndarray]
        Channel index → float64 array of shape (N_tiles, vec_dim).
    """
    vec_dim = n_landscape_bins * n_landscape_layers
    n_channels: Optional[int] = None
    vectors_per_ch: Dict[int, List[np.ndarray]] = {}

    for mask in mask_iter:
        mask = np.asarray(mask)
        centres_per_ch = extract_centres_multichannel(mask)

        if n_channels is None:
            n_channels = len(centres_per_ch)
            vectors_per_ch = {c: [] for c in range(n_channels)}

        for ch, centres in enumerate(centres_per_ch):
            if centres.shape[0] < min_cells:
                vec = np.zeros(vec_dim, dtype=np.float64)
            else:
                diagram = compute_persistence_diagram_1d(centres, use_alpha=use_alpha)
                vec = vectorise_persistence_landscape(
                    diagram,
                    n_bins=n_landscape_bins,
                    n_layers=n_landscape_layers,
                )
            vectors_per_ch[ch].append(vec)

    if n_channels is None:
        return {}
    return {ch: np.array(vecs, dtype=np.float64) for ch, vecs in vectors_per_ch.items()}


def _fit_gaussian_from_vectors(
    vectors_per_ch: Dict[int, np.ndarray],
    class_name: str = "",
) -> ClassDistribution:
    """Fit a multivariate Gaussian per channel from stored landscape vectors.

    Parameters
    ----------
    vectors_per_ch : dict[int, np.ndarray]
        Channel index → array of shape (N_tiles, vec_dim).
    """
    if not vectors_per_ch:
        raise ValueError(f"No vectors for class '{class_name}'")

    n_channels = len(vectors_per_ch)
    vec_dim = next(iter(vectors_per_ch.values())).shape[1]

    mean_dict: Dict[int, np.ndarray] = {}
    cov_dict: Dict[int, np.ndarray] = {}
    n_dict: Dict[int, int] = {}

    for ch, vecs in vectors_per_ch.items():
        n = len(vecs)
        n_dict[ch] = n
        if n == 0:
            mean_dict[ch] = np.zeros(vec_dim)
            cov_dict[ch] = np.eye(vec_dim)
        elif n == 1:
            mean_dict[ch] = vecs[0]
            cov_dict[ch] = np.eye(vec_dim)
        else:
            mean_dict[ch] = vecs.mean(axis=0)
            cov_dict[ch] = np.cov(vecs.T)

    return ClassDistribution(
        class_name=class_name,
        n_channels=n_channels,
        vec_dim=vec_dim,
        mean=mean_dict,
        cov=cov_dict,
        n_samples=n_dict,
    )


# ===================================================================
# § 4  Pipeline entry point
# ===================================================================

def run_tfd_separability(cfg: Dict, verbose: bool = True) -> None:
    """Config-driven pipeline stage for TFD-based class separability.

    Expected config keys (``tfd_separability`` section of ``config.yaml``)
    -----------------------------------------------------------------------
    tiles_zip_dir : str
        Directory of per-patient tile ZIP archives (BRCA-tumor-tiles-corrected).
    metadata_csv : str
        CSV with patient → subtype mapping (same as used for training).
    patient_col : str
        Column name for patient IDs (default: ``'Patient_ID'``).
    subtype_col : str
        Column name for PAM50 subtypes (default: ``'Majority_Subtype_mRNA'``).
    classes : list or null
        Subset of classes to analyse.  ``null`` = all classes in the CSV.
    masks_dir : str
        Where to save / load per-patient segmentation mask ZIPs.
    output_dir : str
        Where to write results JSON and heatmap.
    auto_segment : bool
        If ``true`` (default), run DeepCMorph segmentation when ``masks_dir``
        is missing or empty.  Set ``false`` to assume masks already exist.
    force_resegment : bool
        If ``true``, re-run segmentation even when ``masks_dir`` already contains
        files.  Default: ``false``.

    Segmentation parameters (forwarded to DeepCMorph):
        num_classes, weights_dataset, device, grouping_json, batch_size

    Tile balancing:
        max_tiles_by_subtype : dict[str, int or null]
            Per-patient tile cap per class (mirrors ``mopadi_genomic_training``).
            Example: ``{LumA: 45, LumB: 120, Basal: 135, Her2: 300, Normal: null}``.
            ``null`` = no cap for that class.

    TDA parameters:
        use_alpha_complex, n_landscape_bins, n_landscape_layers, min_cells_per_image

    Noise floor:
        compute_noise_floor : bool (default: true)
        n_noise_floor_splits : int (default: 5)

    Visualisation:
        save_heatmap : bool (default: true)
        figsize : [width, height] (default: auto)
    """
    import pandas as pd

    tiles_zip_dir = Path(cfg["tiles_zip_dir"])
    masks_dir = Path(cfg["masks_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 – load patient → subtype mapping
    # (must happen before segmentation so we can pass per-patient caps)
    # ------------------------------------------------------------------
    metadata_csv = Path(cfg["metadata_csv"])
    patient_col = cfg.get("patient_col", "Patient_ID")
    subtype_col = cfg.get("subtype_col", "Majority_Subtype_mRNA")

    df = pd.read_csv(metadata_csv)
    df["_pid"] = df[patient_col].apply(extract_patient_id)
    patient_to_subtype: Dict[str, str] = dict(zip(df["_pid"], df[subtype_col]))

    available_classes = sorted(df[subtype_col].dropna().unique().tolist())
    requested: Optional[List[str]] = cfg.get("classes")
    if requested:
        unknown = [c for c in requested if c not in available_classes]
        if unknown:
            raise ValueError(
                f"Unknown classes {unknown}. Available in {subtype_col}: {available_classes}"
            )
        classes = requested
    else:
        classes = available_classes

    if verbose:
        print(f"\n[TopoFD-Sep] Classes: {classes}")

    class_to_patients: Dict[str, List[str]] = {cls: [] for cls in classes}
    for pid, subtype in patient_to_subtype.items():
        if subtype in class_to_patients:
            class_to_patients[subtype].append(pid)

    # ------------------------------------------------------------------
    # Step 2 – segmentation (DeepCMorph)
    # Resume-aware behavior:
    #   * if force_resegment=True: segment all requested patients again
    #   * else: segment only missing patient ZIPs
    # ------------------------------------------------------------------
    auto_segment = cfg.get("auto_segment", True)
    force_resegment = cfg.get("force_resegment", False)

    expected_pids: Set[str] = set()
    for cls in classes:
        expected_pids.update(class_to_patients[cls])

    existing_pids: Set[str] = set()
    if masks_dir.exists():
        existing_pids = {p.stem for p in masks_dir.glob("*.zip")}

    missing_pids = expected_pids - existing_pids

    if auto_segment:
        if force_resegment:
            _run_segmentation(
                cfg,
                tiles_zip_dir,
                masks_dir,
                classes=classes,
                patient_to_subtype=patient_to_subtype,
                target_pids=expected_pids,
                verbose=verbose,
            )
        elif missing_pids:
            if verbose:
                print(
                    f"[TopoFD-Sep] Found {len(existing_pids)}/{len(expected_pids)} patient mask ZIPs; "
                    f"resuming segmentation for {len(missing_pids)} missing patients."
                )
            _run_segmentation(
                cfg,
                tiles_zip_dir,
                masks_dir,
                classes=classes,
                patient_to_subtype=patient_to_subtype,
                target_pids=missing_pids,
                verbose=verbose,
            )
        elif verbose:
            print(f"[TopoFD-Sep] All expected masks already present in {masks_dir}; skipping segmentation.")
    elif verbose:
        if masks_dir.exists() and any(masks_dir.glob("*.zip")):
            n_zips = sum(1 for _ in masks_dir.glob("*.zip"))
            print(f"[TopoFD-Sep] auto_segment=False — using existing masks in {masks_dir} ({n_zips} ZIPs)")
        else:
            print(f"[TopoFD-Sep] auto_segment=False — expecting pre-computed masks in {masks_dir}")

    # ------------------------------------------------------------------
    # Step 3 – fit per-class distributions in streaming mode
    # (memory-safe: do not keep all masks in RAM)
    # ------------------------------------------------------------------
    max_tiles_by_subtype: Dict[str, Optional[int]] = cfg.get("max_tiles_by_subtype", {})
    use_alpha = cfg.get("use_alpha_complex", True)
    n_landscape_bins = cfg.get("n_landscape_bins", 100)
    n_landscape_layers = cfg.get("n_landscape_layers", 1)
    min_cells = cfg.get("min_cells_per_image", 3)

    distributions: Dict[str, ClassDistribution] = {}
    class_vectors: Dict[str, Dict[int, np.ndarray]] = {}
    n_samples: Dict[str, int] = {}
    for cls in classes:
        pids = class_to_patients[cls]
        max_per_patient = max_tiles_by_subtype.get(cls)
        if verbose:
            cap_str = str(max_per_patient) if max_per_patient is not None else "all"
            print(f"  [{cls}] Streaming masks for {len(pids)} patients (≤{cap_str} tiles/patient)…")

        tda_kw = dict(
            use_alpha=use_alpha,
            n_landscape_bins=n_landscape_bins,
            n_landscape_layers=n_landscape_layers,
            min_cells=min_cells,
        )
        vecs = _collect_landscape_vectors(
            _iter_masks_for_class(
                masks_dir,
                pids,
                max_tiles_per_patient=max_per_patient,
                seed=42,
            ),
            **tda_kw,
        )
        class_vectors[cls] = vecs
        dist = _fit_gaussian_from_vectors(vecs, class_name=cls)
        distributions[cls] = dist
        n_cls = dist.n_samples.get(0, 0)
        n_samples[cls] = n_cls
        if verbose:
            print(f"  [{cls}] {n_cls} masks processed")
        if n_cls < 10:
            logger.warning(
                "%s: only %d masks — covariance estimation may be unreliable "
                "(need at least ~100 for a stable 100×100 covariance matrix)",
                cls, n_cls,
            )

    # ------------------------------------------------------------------
    # Step 4 – compute pairwise separability from fitted distributions
    # ------------------------------------------------------------------
    if verbose:
        print("\n[TopoFD-Sep] Computing pairwise TFD…")

    pairwise_tfd: Dict[Tuple[str, str], float] = {}
    pairwise_per_channel: Dict[Tuple[str, str], Dict[int, float]] = {}
    for cls_a, cls_b in itertools.combinations(sorted(classes), 2):
        da, db = distributions[cls_a], distributions[cls_b]
        per_ch: Dict[int, float] = {}

        for ch in range(da.n_channels):
            na = da.n_samples.get(ch, 0)
            nb = db.n_samples.get(ch, 0)
            if na < 2 or nb < 2:
                logger.warning(
                    "ch%d: %s vs %s — not enough samples (%d, %d), skipping",
                    ch, cls_a, cls_b, na, nb,
                )
                per_ch[ch] = float("nan")
                continue
            per_ch[ch] = _calculate_frechet_distance(
                da.mean[ch], da.cov[ch],
                db.mean[ch], db.cov[ch],
            )

        valid = [v for v in per_ch.values() if np.isfinite(v)]
        avg = float(np.mean(valid)) if valid else float("nan")
        pair = (cls_a, cls_b)
        pairwise_tfd[pair] = avg
        pairwise_per_channel[pair] = per_ch
        if verbose:
            print(f"  TFD({cls_a}, {cls_b}) = {avg:.4f}")

    noise_floor: Optional[Dict[str, float]] = None
    if cfg.get("compute_noise_floor", True):
        n_splits = cfg.get("n_noise_floor_splits", 5)
        rng = np.random.default_rng(42)
        noise_floor = {}
        if verbose:
            print("\n[TopoFD-Sep] Computing intra-class noise floor…")

        for cls in sorted(classes):
            vecs = class_vectors[cls]
            if not vecs:
                noise_floor[cls] = float("nan")
                continue
            n = next(iter(vecs.values())).shape[0]
            if n < 4:
                logger.warning("%s: only %d tiles — skipping noise floor", cls, n)
                noise_floor[cls] = float("nan")
                continue

            split_fds: List[float] = []
            for _ in range(n_splits):
                idx = np.arange(n)
                rng.shuffle(idx)
                half = n // 2
                idx_a, idx_b = idx[:half], idx[half:]

                vecs_a = {ch: v[idx_a] for ch, v in vecs.items()}
                vecs_b = {ch: v[idx_b] for ch, v in vecs.items()}
                da_nf = _fit_gaussian_from_vectors(vecs_a)
                db_nf = _fit_gaussian_from_vectors(vecs_b)

                per_ch_nf: Dict[int, float] = {}
                for ch in range(da_nf.n_channels):
                    if da_nf.n_samples.get(ch, 0) < 2 or db_nf.n_samples.get(ch, 0) < 2:
                        per_ch_nf[ch] = float("nan")
                        continue
                    per_ch_nf[ch] = _calculate_frechet_distance(
                        da_nf.mean[ch], da_nf.cov[ch],
                        db_nf.mean[ch], db_nf.cov[ch],
                    )
                valid_nf = [v for v in per_ch_nf.values() if np.isfinite(v)]
                if valid_nf:
                    split_fds.append(float(np.mean(valid_nf)))

            noise_floor[cls] = float(np.mean(split_fds)) if split_fds else float("nan")
            if verbose:
                nf_str = f"{noise_floor[cls]:.4f}" if np.isfinite(noise_floor[cls]) else "—"
                print(f"  Noise floor ({cls}) = {nf_str}")

    result = TFDSeparabilityResult(
        class_names=sorted(classes),
        pairwise_tfd=pairwise_tfd,
        pairwise_per_channel=pairwise_per_channel,
        noise_floor=noise_floor,
        n_samples=n_samples,
    )

    # ------------------------------------------------------------------
    # Step 5 – save results
    # ------------------------------------------------------------------
    result_path = output_dir / "tfd_separability_results.json"
    with open(result_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    if verbose:
        print(f"\n[OK] Results saved to {result_path}")

    if cfg.get("save_heatmap", True):
        try:
            figsize = cfg.get("figsize")
            _save_heatmap(result, output_dir, figsize=figsize, verbose=verbose)
        except Exception as exc:
            logger.warning("Could not save heatmap: %s", exc)

    if verbose:
        _print_summary(result)


# ===================================================================
# § 5  Helpers
# ===================================================================

def _run_segmentation(
    cfg: Dict,
    tiles_zip_dir: Path,
    masks_dir: Path,
    classes: List[str],
    patient_to_subtype: Dict[str, str],
    target_pids: Optional[Set[str]] = None,
    verbose: bool = True,
) -> None:
    """Run DeepCMorph on a subtype-filtered, per-patient-capped subset of tiles.

    Rather than segmenting every tile in ``tiles_zip_dir`` and discarding most
    of them later, we:
      1. Build the set of patient IDs that belong to the requested ``classes``.
      2. Look up each patient's per-subtype tile cap from ``max_tiles_by_subtype``.
      3. Pass ``include_pids`` and ``per_zip_max_tiles`` directly to
         ``process_tiles_from_zips`` so only the tiles we will actually analyse
         are run through DeepCMorph.

    This typically cuts segmentation work by 5–10× compared to a naive
    "segment everything" approach.
    """
    import json as _json
    import torch

    from src.classifier.segment_and_classify_cells import (
        DeepCMorphSegmenter,
        process_tiles_from_zips,
    )

    masks_dir.mkdir(parents=True, exist_ok=True)

    # --- Build per-patient cap and patient filter ---
    max_tiles_by_subtype: Dict[str, Optional[int]] = cfg.get("max_tiles_by_subtype", {})

    include_pids: set = set()
    per_zip_max_tiles: Dict[str, Optional[int]] = {}
    for pid, subtype in patient_to_subtype.items():
        if subtype not in classes:
            continue
        if target_pids is not None and pid not in target_pids:
            continue
        include_pids.add(pid)
        # None means no cap for this patient (e.g. Normal class)
        per_zip_max_tiles[pid] = max_tiles_by_subtype.get(subtype)

    # Diagnose requested patients that have no source ZIP in tiles_zip_dir.
    source_zip_pids = {extract_patient_id(zp.stem) for zp in tiles_zip_dir.glob("*.zip")}
    missing_source = include_pids - source_zip_pids
    if missing_source:
        logger.warning(
            "%d requested patient(s) have no source tile ZIP in %s; they will be skipped.",
            len(missing_source),
            tiles_zip_dir,
        )

    if verbose:
        total_cap = sum(
            (v if v is not None else 9999)
            for v in per_zip_max_tiles.values()
        )
        print(
            f"\n[TopoFD-Sep] Segmenting tiles for {len(include_pids)} patients "
            f"across classes {classes}"
        )
        print(f"             Estimated upper bound: ~{total_cap:,} tiles")
        print(f"             Output: {masks_dir}")

    # --- Build segmenter ---
    device = cfg.get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    grouping: Optional[Dict] = None
    grouping_json = cfg.get("grouping_json")
    if grouping_json:
        with open(grouping_json) as f:
            grouping = _json.load(f)

    segmenter = DeepCMorphSegmenter(
        num_classes=cfg.get("num_classes", 32),
        weights_dataset=cfg.get("weights_dataset", "TCGA"),
        device=device,
        grouping=grouping,
    )

    # --- Run segmentation with per-patient caps ---
    n = process_tiles_from_zips(
        input_dir=tiles_zip_dir,
        output_dir=masks_dir,
        segmenter=segmenter,
        per_zip_max_tiles=per_zip_max_tiles,
        include_pids=include_pids,
        save_seg=False,  # skip nuclei masks — only classification maps needed for TFD
    )

    if verbose:
        print(f"[TopoFD-Sep] Segmentation complete — {n:,} tiles processed.")


def _save_heatmap(
    result: TFDSeparabilityResult,
    output_dir: Path,
    *,
    figsize: Optional[List[float]] = None,
    verbose: bool = True,
) -> None:
    """Save the pairwise TFD matrix as a labelled heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping heatmap")
        return

    class_names, mat = result.as_matrix()
    n = len(class_names)

    if figsize is None:
        figsize = (max(6, n + 2), max(5, n + 1))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("TFD", fontsize=11)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_title("Topological Fréchet Distance — PAM50 Class Separability", fontsize=12)

    # Annotate cells: diagonal shows noise floor, off-diagonal shows TFD
    for i in range(n):
        for j in range(n):
            text_color = "white" if mat[i, j] > mat.max() * 0.5 else "black"
            if i == j:
                # Diagonal: show noise floor
                if result.noise_floor and class_names[i] in result.noise_floor:
                    nf = result.noise_floor[class_names[i]]
                    label = f"NF\n{nf:.2f}" if np.isfinite(nf) else "NF\n—"
                else:
                    label = "—"
            else:
                val = mat[i, j]
                label = f"{val:.2f}" if np.isfinite(val) else "—"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=8, color=text_color)

    plt.tight_layout()
    out_path = output_dir / "tfd_separability_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"[OK] Heatmap saved to {out_path}")


def _print_summary(result: TFDSeparabilityResult) -> None:
    """Print a compact summary table to stdout."""
    print("\n" + "=" * 60)
    print("TFD CLASS SEPARABILITY SUMMARY")
    print("=" * 60)
    print(f"{'Pair':<30}  {'TFD':>10}")
    print("-" * 45)
    for (a, b), d in sorted(result.pairwise_tfd.items()):
        tag = f"{a} vs {b}"
        val_str = f"{d:.4f}" if np.isfinite(d) else "   —  "
        print(f"  {tag:<28}  {val_str:>10}")

    if result.noise_floor:
        print("\nIntra-class noise floor:")
        for cls, nf in sorted(result.noise_floor.items()):
            nf_str = f"{nf:.4f}" if np.isfinite(nf) else "—"
            print(f"  {cls:<20}  {nf_str}")

    print("\nSamples per class:")
    for cls, n in sorted(result.n_samples.items()):
        print(f"  {cls:<20}  {n:>6} masks")
    print("=" * 60)
