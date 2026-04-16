#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Topological Fréchet Distance (TopoFD) for Cell Layout Evaluation

This module computes the Topological Fréchet Distance between two sets of
cell segmentation masks.  The metric captures differences in spatial
organisation (topology) of cells between a reference set and a generated /
reconstructed set.

Pipeline (per cell type channel):
    1. Extract cell-centre coordinates from each labelled segmentation mask.
    2. Compute 1-dimensional persistent homology (loops) via an Alpha complex
       to obtain a persistence diagram per image.
    3. Vectorise each persistence diagram into a fixed-length Persistence
       Landscape vector.
    4. Fit a multivariate Gaussian to the collection of landscape vectors
       (mean + covariance) for both reference and generated sets.
    5. Compute the Fréchet distance between the two Gaussians.

The final TopoFD is the average over all cell-type channels.

Reference:
    Inspired by the eval_TopoFD implementation in TopoCellGen
    (https://github.com/Melon-Xu/TopoCellGen)

Dependencies:
    numpy, scipy, gudhi, giotto-tda (gtda)

Optional:
    matplotlib, persim  (for visualisation helpers)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import linalg
from scipy.ndimage import label, center_of_mass

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Lazy imports – heavy TDA libraries are loaded only when needed
# ---------------------------------------------------------------------------

_GUDHI_AVAILABLE: Optional[bool] = None
_GTDA_AVAILABLE: Optional[bool] = None


def _check_gudhi() -> None:
    global _GUDHI_AVAILABLE
    if _GUDHI_AVAILABLE is None:
        try:
            import gudhi  # noqa: F401
            from gudhi.wasserstein.barycenter import lagrangian_barycenter  # noqa: F401
            _GUDHI_AVAILABLE = True
        except ImportError:
            _GUDHI_AVAILABLE = False
    if not _GUDHI_AVAILABLE:
        raise ImportError(
            "gudhi is required for TopoFD.  Install with:  pip install gudhi"
        )


def _check_gtda() -> None:
    global _GTDA_AVAILABLE
    if _GTDA_AVAILABLE is None:
        try:
            from gtda.diagrams import PersistenceLandscape  # noqa: F401
            _GTDA_AVAILABLE = True
        except ImportError:
            _GTDA_AVAILABLE = False
    if not _GTDA_AVAILABLE:
        raise ImportError(
            "giotto-tda is required for TopoFD.  "
            "Install with:  pip install giotto-tda"
        )


# ===================================================================
# § 1  Cell-centre extraction
# ===================================================================

def extract_cell_centres(
    segmentation: np.ndarray,
) -> np.ndarray:
    """Extract (y, x) centre-of-mass coordinates from a 2-D labelled mask.

    The input can be either:
      * a *binary* mask (foreground > 0) – connected-component labelling is
        applied automatically, or
      * an *instance* segmentation mask where each cell already has a unique
        integer label.

    Parameters
    ----------
    segmentation : np.ndarray, shape (H, W)
        Single-channel segmentation mask.  Zero is background.

    Returns
    -------
    centres : np.ndarray, shape (N, 2)
        Array of (y, x) cell-centre coordinates.  Empty array with shape
        (0, 2) if no cells are found.
    """
    seg = np.asarray(segmentation)
    if seg.ndim != 2:
        raise ValueError(
            f"Expected a 2-D segmentation mask, got shape {seg.shape}"
        )

    # If the mask only contains 0/1 (or 0/255 etc.) we label it ourselves.
    unique_labels = np.unique(seg)
    unique_labels = unique_labels[unique_labels != 0]

    if len(unique_labels) == 0:
        return np.empty((0, 2), dtype=np.float64)

    if len(unique_labels) <= 2:
        # Likely a binary mask – label connected components
        labelled, n_features = label(seg > 0)  # type: ignore[misc]
    else:
        # Already an instance mask
        labelled = seg
        n_features = len(unique_labels)

    if n_features == 0:
        return np.empty((0, 2), dtype=np.float64)

    indices = range(1, int(labelled.max()) + 1)
    centres = np.array(
        [center_of_mass(seg, labelled, idx) for idx in indices if (labelled == idx).any()],
        dtype=np.float64,
    )

    if centres.ndim == 1:
        centres = centres.reshape(-1, 2)

    return centres


def extract_centres_multichannel(
    segmentation: np.ndarray,
) -> List[np.ndarray]:
    """Extract cell centres per channel from a multi-channel segmentation.

    Parameters
    ----------
    segmentation : np.ndarray
        Either (H, W) for single channel, or (H, W, C) for multi-channel.

    Returns
    -------
    list of np.ndarray
        One (N_c, 2) array of centres per channel.
    """
    seg = np.asarray(segmentation)
    
    # Heuristic: if floats or small max values, treat as soft probability maps
    is_soft = np.issubdtype(seg.dtype, np.floating) or (seg.max() <= 1.0)
    
    if seg.ndim == 2:
        mask = (seg > 0.5) if is_soft else seg
        return [extract_cell_centres(mask)]
    elif seg.ndim == 3:
        n_channels = seg.shape[2]
        
        if is_soft:
            # Determine hard class assignments via argmax to avoid overlapping classifications
            max_cls = np.argmax(seg, axis=-1)
            max_prob = np.max(seg, axis=-1)
            
            # Use dynamic threshold (0.5 for [0, 1] range) to identify foreground vs background
            threshold = 0.5 if seg.max() <= 1.0 else (seg.max() / 2.0)
            is_fg = max_prob > threshold
            
            centres_list = []
            for c in range(n_channels):
                # Isolate cells of this specific class
                class_mask = (max_cls == c) & is_fg
                centres_list.append(extract_cell_centres(class_mask))
            return centres_list
        else:
            return [extract_cell_centres(seg[..., c]) for c in range(n_channels)]
    else:
        raise ValueError(f"Unexpected segmentation shape: {seg.shape}")


# ===================================================================
# § 2  Persistent homology
# ===================================================================

def compute_persistence_diagram_1d(
    point_cloud: np.ndarray,
    use_alpha: bool = True,
) -> np.ndarray:
    """Compute the 1-dimensional persistence diagram for a point cloud.

    Parameters
    ----------
    point_cloud : np.ndarray, shape (N, 2)
        Cell-centre coordinates.
    use_alpha : bool
        If True use Alpha complex (exact, faster for low dimensions).
        Otherwise fall back to Rips complex.

    Returns
    -------
    diagram : np.ndarray, shape (K, 2)
        Array of (birth, death) pairs for homology dimension 1.
        Returns an empty (0, 2) array when there are fewer than 3 points
        or no 1-cycles are found.
    """
    _check_gudhi()
    import gudhi

    if point_cloud.shape[0] < 3:
        return np.empty((0, 2), dtype=np.float64)

    if use_alpha:
        cplx = gudhi.AlphaComplex(points=point_cloud.tolist())  # type: ignore
        st = cplx.create_simplex_tree()
    else:
        cplx = gudhi.RipsComplex(points=point_cloud.tolist())  # type: ignore
        st = cplx.create_simplex_tree(max_dimension=2)

    st.persistence()
    pairs = st.persistence_intervals_in_dimension(1)

    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.float64)

    diagram = np.array(pairs, dtype=np.float64)

    # Remove infinite-death features (unbounded cycles)
    finite_mask = np.isfinite(diagram[:, 1])
    diagram = diagram[finite_mask]

    return diagram if diagram.shape[0] > 0 else np.empty((0, 2), dtype=np.float64)


# ===================================================================
# § 3  Persistence Landscape vectorisation
# ===================================================================

def vectorise_persistence_landscape(
    diagram: np.ndarray,
    n_bins: int = 100,
    n_layers: int = 1,
) -> np.ndarray:
    """Convert a persistence diagram into a fixed-length landscape vector.

    Uses giotto-tda's ``PersistenceLandscape`` transformer.

    Parameters
    ----------
    diagram : np.ndarray, shape (K, 2)
        Birth-death pairs (1-dim homology).
    n_bins : int
        Number of sampling points for the landscape.
    n_layers : int
        Number of landscape layers to use.

    Returns
    -------
    vector : np.ndarray, shape (n_bins * n_layers,)
        Flattened persistence landscape vector.
    """
    _check_gtda()
    from gtda.diagrams import PersistenceLandscape

    if diagram.shape[0] == 0:
        return np.zeros(n_bins * n_layers, dtype=np.float64)

    # giotto-tda expects shape (n_diagrams, n_points, 3) with last column =
    # homology dimension.
    dgm = np.column_stack([
        diagram,
        np.ones(diagram.shape[0], dtype=np.float64),  # dim = 1
    ])
    dgm = dgm[np.newaxis, ...]  # batch dim

    pl = PersistenceLandscape(n_bins=n_bins, n_layers=n_layers)
    vector = pl.fit_transform(dgm).flatten()

    return vector.astype(np.float64)


# ===================================================================
# § 4  Fréchet distance between Gaussian-fitted landscape vectors
# ===================================================================

def _calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Fréchet distance between two multivariate Gaussians N(mu1, sigma1) and N(mu2, sigma2).

    Parameters
    ----------
    mu1, mu2 : np.ndarray, shape (D,)
        Mean vectors.
    sigma1, sigma2 : np.ndarray, shape (D, D)
        Covariance matrices.
    eps : float
        Regularisation added to the diagonal when covmean is singular.

    Returns
    -------
    fd : float
        Fréchet distance (non-negative).
    """
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Mean vectors differ in length"
    assert sigma1.shape == sigma2.shape, "Covariance matrices differ in shape"

    diff = mu1 - mu2

    # Scipy's sqrtm is known to be numerically unstable for highly rank-deficient
    # matrices (like when we have 16 samples for a 100-dim vector space).
    covmean_raw, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    covmean: np.ndarray = np.asarray(covmean_raw)

    if not np.isfinite(covmean).all():
        logger.warning(
            "Fréchet distance: singular product – adding eps=%s to diagonal", eps
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = np.asarray(linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset)))

    # Numerical precision errors often introduce an imaginary component
    # for PSD matrices. Standard FID practice is to drop the imaginary part.
    if np.iscomplexobj(covmean):
        max_imag = np.max(np.abs(covmean.imag))
        if max_imag > 1e-3:
            logger.warning(
                "Fréchet distance: large imaginary component (%.4f) detected during matrix square root. "
                "This is typical for highly rank-deficient covariance matrices (e.g., small sample sizes). "
                "Dropping the imaginary component to compute the real Fréchet Distance.", max_imag
            )
        covmean = covmean.real
        
    # In some extreme rank-deficient edge cases, the trace can still be slightly negative
    # due to numerical approximation.
    tr_covmean = np.trace(covmean)
    
    fd = float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    
    # Due to floating point imprecision, fd might technically end up as -0.00001
    return max(0.0, fd)


# ===================================================================
# § 5  Public API – compute_topofd
# ===================================================================

@dataclass
class TopoFDResult:
    """Container for Topological Fréchet Distance results.

    Attributes
    ----------
    topofd : float
        Mean TopoFD averaged over all channels.
    per_channel : dict[int, float]
        TopoFD for each channel index.
    n_reference : int
        Number of reference segmentation masks used.
    n_generated : int
        Number of generated segmentation masks used.
    n_channels : int
        Number of cell-type channels evaluated.
    """
    topofd: float
    per_channel: Dict[int, float]
    n_reference: int
    n_generated: int
    n_channels: int

    def __repr__(self) -> str:
        ch_str = ", ".join(
            f"ch{k}={v:.4f}" for k, v in sorted(self.per_channel.items())
        )
        return (
            f"TopoFDResult(topofd={self.topofd:.4f}, {ch_str}, "
            f"n_ref={self.n_reference}, n_gen={self.n_generated})"
        )


class _OnlineMeanCov:
    """Online mean and covariance estimator (Welford algorithm).

    This avoids keeping all vectors in memory, which is critical for large
    datasets when computing TopoFD.
    """

    def __init__(self, dim: int):
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros((dim, dim), dtype=np.float64)

    def update(self, x: np.ndarray):
        """Update with a new sample vector x."""
        x = x.astype(np.float64)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += np.outer(delta, delta2)

    def finalize(self):
        """Return (mean, covariance) with Bessel's correction."""
        if self.n < 2:
            cov = np.zeros_like(self.M2)
        else:
            cov = self.M2 / (self.n - 1)
        return self.mean, cov, self.n


from collections.abc import Iterable
import itertools


def compute_topofd(
    reference_segmentations: Iterable[np.ndarray],
    generated_segmentations: Iterable[np.ndarray],
    *,
    use_alpha: bool = True,
    n_landscape_bins: int = 100,
    n_landscape_layers: int = 1,
    min_cells: int = 3,
) -> TopoFDResult:
    """Compute the Topological Fréchet Distance between two sets of segmentations.

    This implementation uses an online mean/covariance estimator so that we do
    not need to store all persistence landscape vectors in memory.

    Parameters
    ----------
    reference_segmentations : iterable of np.ndarray
        Reference (ground-truth) segmentation masks.  Each element is either
        (H, W) for a single cell type or (H, W, C) for *C* cell types.
    generated_segmentations : iterable of np.ndarray
        Generated / reconstructed segmentation masks (same format).
    use_alpha : bool
        Use Alpha complex (True) or Rips complex (False) for persistent
        homology.
    n_landscape_bins : int
        Number of bins for the persistence landscape vectorisation.
    n_landscape_layers : int
        Number of landscape layers to keep.
    min_cells : int
        Minimum number of cell centres required per image to compute a
        persistence diagram.  Images with fewer cells are skipped.

    Returns
    -------
    TopoFDResult
        Dataclass containing overall and per-channel TopoFD values.

    Raises
    ------
    ValueError
        If inputs are empty or have incompatible channel counts.
    ImportError
        If gudhi or giotto-tda are not installed.

    Examples
    --------
    >>> import numpy as np
    >>> ref = [np.random.randint(0, 5, (256, 256)) for _ in range(20)]
    >>> gen = [np.random.randint(0, 5, (256, 256)) for _ in range(20)]
    >>> result = compute_topofd(ref, gen)
    >>> print(result)
    """
    _check_gudhi()
    _check_gtda()

    # Make sure iterables are re-iterable
    ref_iter = iter(reference_segmentations)
    gen_iter = iter(generated_segmentations)

    try:
        first_ref = next(ref_iter)
    except StopIteration:
        raise ValueError("Reference set is empty")
    try:
        first_gen = next(gen_iter)
    except StopIteration:
        raise ValueError("Generated set is empty")

    # Determine number of channels from the first reference sample
    sample_ref = np.asarray(first_ref)
    n_channels = sample_ref.shape[2] if sample_ref.ndim == 3 else 1

    vec_dim = n_landscape_bins * n_landscape_layers

    # -------- helper: process one set of segmentations --------
    def _process_set(
        first: np.ndarray, remaining: Iterable[np.ndarray]
    ) -> Tuple[Dict[int, _OnlineMeanCov], int]:
        """Return (stats_per_channel, n_samples)."""
        stats: Dict[int, _OnlineMeanCov] = {
            c: _OnlineMeanCov(vec_dim) for c in range(n_channels)
        }
        count = 0

        def _process(seg):
            nonlocal count
            count += 1
            centres_per_ch = extract_centres_multichannel(np.asarray(seg))

            # Validate channel count
            if len(centres_per_ch) != n_channels:
                logger.warning(
                    "Skipping segmentation with %d channels (expected %d)",
                    len(centres_per_ch),
                    n_channels,
                )
                return

            for ch_idx, centres in enumerate(centres_per_ch):
                if centres.shape[0] < min_cells:
                    stats[ch_idx].update(np.zeros(vec_dim, dtype=np.float64))
                    continue

                diagram = compute_persistence_diagram_1d(
                    centres, use_alpha=use_alpha
                )
                vec = vectorise_persistence_landscape(
                    diagram,
                    n_bins=n_landscape_bins,
                    n_layers=n_landscape_layers,
                )
                stats[ch_idx].update(vec)

        # Process the first element we already pulled
        _process(first)
        # Process the rest
        for seg in remaining:
            _process(seg)

        return stats, count

    logger.info("Processing reference segmentations…")
    ref_stats, n_ref = _process_set(first_ref, ref_iter)

    logger.info("Processing generated segmentations…")
    gen_stats, n_gen = _process_set(first_gen, gen_iter)

    if n_ref == 0 or n_gen == 0:
        raise ValueError("Both reference and generated sets must be non-empty.")

    # -------- Fréchet distance per channel --------
    per_channel: Dict[int, float] = {}

    for ch in range(n_channels):
        mean_ref, cov_ref, n_ref_ch = ref_stats[ch].finalize()
        mean_gen, cov_gen, n_gen_ch = gen_stats[ch].finalize()

        if n_ref_ch < 2 or n_gen_ch < 2:
            logger.warning(
                "Channel %d: not enough samples (ref=%d, gen=%d) – skipping.",
                ch, n_ref_ch, n_gen_ch,
            )
            per_channel[ch] = float("nan")
            continue

        fd = _calculate_frechet_distance(mean_ref, cov_ref, mean_gen, cov_gen)
        per_channel[ch] = fd
        logger.info("Channel %d  TopoFD = %.4f", ch, fd)

    # Average (ignoring NaN channels)
    valid = [v for v in per_channel.values() if np.isfinite(v)]
    topofd = float(np.mean(valid)) if valid else float("nan")

    return TopoFDResult(
        topofd=topofd,
        per_channel=per_channel,
        n_reference=n_ref,
        n_generated=n_gen,
        n_channels=n_channels,
    )

    # -------- Fréchet distance per channel --------
    per_channel: Dict[int, float] = {}

    for ch in range(n_channels):
        mean_ref, cov_ref, n_ref = ref_stats[ch].finalize()
        mean_gen, cov_gen, n_gen = gen_stats[ch].finalize()

        if n_ref < 2 or n_gen < 2:
            logger.warning(
                "Channel %d: not enough samples (ref=%d, gen=%d) – skipping.",
                ch, n_ref, n_gen,
            )
            per_channel[ch] = float("nan")
            continue

        fd = _calculate_frechet_distance(mean_ref, cov_ref, mean_gen, cov_gen)
        per_channel[ch] = fd
        logger.info("Channel %d  TopoFD = %.4f", ch, fd)

    # Average (ignoring NaN channels)
    valid = [v for v in per_channel.values() if np.isfinite(v)]
    topofd = float(np.mean(valid)) if valid else float("nan")

    return TopoFDResult(
        topofd=topofd,
        per_channel=per_channel,
        n_reference=len(reference_segmentations),
        n_generated=len(generated_segmentations),
        n_channels=n_channels,
    )

    # -------- Fréchet distance per channel --------
    per_channel: Dict[int, float] = {}

    for ch in range(n_channels):
        ref_mat = np.array(ref_vectors[ch])  # (N_ref, vec_dim)
        gen_mat = np.array(gen_vectors[ch])  # (N_gen, vec_dim)

        if ref_mat.shape[0] < 2 or gen_mat.shape[0] < 2:
            logger.warning(
                "Channel %d: not enough samples (ref=%d, gen=%d) – skipping.",
                ch, ref_mat.shape[0], gen_mat.shape[0],
            )
            per_channel[ch] = float("nan")
            continue

        mu_ref = np.mean(ref_mat, axis=0)
        cov_ref = np.cov(ref_mat, rowvar=False)

        mu_gen = np.mean(gen_mat, axis=0)
        cov_gen = np.cov(gen_mat, rowvar=False)

        fd = _calculate_frechet_distance(mu_ref, cov_ref, mu_gen, cov_gen)
        per_channel[ch] = fd
        logger.info("Channel %d  TopoFD = %.4f", ch, fd)

    # Average (ignoring NaN channels)
    valid = [v for v in per_channel.values() if np.isfinite(v)]
    topofd = float(np.mean(valid)) if valid else float("nan")

    return TopoFDResult(
        topofd=topofd,
        per_channel=per_channel,
        n_reference=len(reference_segmentations),
        n_generated=len(generated_segmentations),
        n_channels=n_channels,
    )


# ===================================================================
# § 6  Convenience – compute from folders of .npy files
# ===================================================================

def compute_topofd_from_folders(
    reference_dir: Union[str, Path],
    generated_dir: Union[str, Path],
    *,
    glob_pattern: str = "*.npy",
    batch_size: int = 500,
    **kwargs,
) -> TopoFDResult:
    """Compute TopoFD by loading segmentation masks from two directories.

    Each ``.npy`` file should contain an array of shape (H, W) or (H, W, C).
    
    For large datasets, masks are processed in batches to avoid memory exhaustion.
    Broadcasting all persistence landscape vectors and fitting Gaussians is more
    memory-efficient than loading thousands of raw mask arrays.

    Parameters
    ----------
    reference_dir, generated_dir : path-like
        Directories containing segmentation mask files.
    glob_pattern : str
        Glob pattern for discovering mask files.
    batch_size : int
        Number of masks to process in memory per batch (default: 500).
        Increase if OOM occurs; decrease on memory-constrained systems.
    **kwargs
        Forwarded to :func:`compute_topofd`.

    Returns
    -------
    TopoFDResult
    """
    ref_dir = Path(reference_dir)
    gen_dir = Path(generated_dir)

    ref_files = sorted(ref_dir.glob(glob_pattern))
    gen_files = sorted(gen_dir.glob(glob_pattern))

    if not ref_files:
        raise FileNotFoundError(f"No files matching '{glob_pattern}' in {ref_dir}")
    if not gen_files:
        raise FileNotFoundError(f"No files matching '{glob_pattern}' in {gen_dir}")

    logger.info("Streaming %d reference masks from %s (batch_size=%d)", len(ref_files), ref_dir, batch_size)
    ref_segs = _load_masks_batched(ref_files, batch_size)

    logger.info("Streaming %d generated masks from %s (batch_size=%d)", len(gen_files), gen_dir, batch_size)
    gen_segs = _load_masks_batched(gen_files, batch_size)

    return compute_topofd(ref_segs, gen_segs, **kwargs)


def run_topofd(
    reference_dir: Union[str, Path],
    generated_dir: Union[str, Path],
    config: Dict,
    verbose: bool = True,
) -> None:
    """
    Run Topological Fréchet Distance computation from config parameters.
    
    Parameters
    ----------
    reference_dir : path-like
        Directory with reference segmentation masks.
    generated_dir : path-like
        Directory with generated segmentation masks.
    config : dict
        Configuration dictionary with optional keys:
        - output_dir: where to save results (JSON/visualizations)
        - save_detailed_report: whether to save per-channel JSON report
        - save_visualizations: whether to save visualization plots
        - n_landscape_bins: persistence landscape resolution
        - n_landscape_layers: number of landscape layers
        - use_alpha_complex: use Alpha complex (True) vs Rips (False)
        - min_cells_per_image: skip images with fewer cells
    verbose : bool
        Whether to print progress information.
    """
    from pathlib import Path
    
    ref_dir = Path(reference_dir)
    gen_dir = Path(generated_dir)
    
    output_dir = config.get("output_dir")
    save_report = config.get("save_detailed_report", True)
    save_vis = config.get("save_visualizations", False)
    
    if verbose:
        print(f"\n[INFO] Computing TopoFD...")
        print(f"  Reference masks: {ref_dir}")
        print(f"  Generated masks: {gen_dir}")
    
    # Build kwargs for compute_topofd_from_folders
    kwargs = {}
    if config.get("use_alpha_complex") is not None:
        kwargs["use_alpha"] = config["use_alpha_complex"]
    if config.get("n_landscape_bins"):
        kwargs["n_landscape_bins"] = config["n_landscape_bins"]
    if config.get("n_landscape_layers"):
        kwargs["n_landscape_layers"] = config["n_landscape_layers"]
    if config.get("min_cells_per_image"):
        kwargs["min_cells"] = config["min_cells_per_image"]
    
    # Use only classification maps for TopoFD
    kwargs["glob_pattern"] = "*_cls.npy"
    
    # Compute TopoFD
    result = compute_topofd_from_folders(
        ref_dir,
        gen_dir,
        **kwargs,
    )
    
    if verbose:
        print(f"\n[RESULT] {result}")
    
    # Save results if output_dir specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary to JSON
        summary = {
            "topofd": float(result.topofd),
            "per_channel": {str(k): float(v) for k, v in result.per_channel.items()},
            "n_reference": result.n_reference,
            "n_generated": result.n_generated,
            "n_channels": result.n_channels,
        }
        
        summary_path = output_dir / "topofd_summary.json"
        with open(summary_path, "w") as f:
            import json
            json.dump(summary, f, indent=2)
        
        if verbose:
            print(f"[OK] Saved summary to {summary_path}")
        
        # Optionally save detailed report
        if save_report:
            detailed_path = output_dir / "topofd_detailed.json"
            with open(detailed_path, "w") as f:
                import json
                json.dump(summary, f, indent=2)
            if verbose:
                print(f"[OK] Saved detailed report to {detailed_path}")



def _load_masks_batched(
    file_paths: List[Path],
    batch_size: int = 500,
) -> Iterable[np.ndarray]:
    """Yield masks from files in batches to avoid memory exhaustion.

    This avoids materializing the full list of masks by yielding them one by
    one as they are loaded.

    Parameters
    ----------
    file_paths : list of Path
        Paths to .npy mask files
    batch_size : int
        Number of files to load in memory at once
    """
    n_files = len(file_paths)

    for batch_start in range(0, n_files, batch_size):
        batch_end = min(batch_start + batch_size, n_files)
        batch_files = file_paths[batch_start:batch_end]

        batch_masks = []
        for f in batch_files:
            try:
                m = np.load(f)
                # Ensure shape is (H, W, C) - DeepCMorph might save as (C, H, W)
                if m.ndim == 3 and m.shape[0] < m.shape[1]:
                    m = np.transpose(m, (1, 2, 0))
                batch_masks.append(m)
            except Exception as e:
                logger.warning(f"Failed to load {f.name}: {e}")

        logger.debug(
            f"Loaded batch {batch_start//batch_size + 1}: {len(batch_masks)} files "
            f"({batch_start + 1}–{batch_end} / {n_files})"
        )

        for m in batch_masks:
            yield m


# ===================================================================
# § 7  CLI entry-point
# ===================================================================

def _cli() -> None:
    """Command-line interface for computing TopoFD."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute Topological Fréchet Distance between two sets of "
                    "cell segmentation masks."
    )
    parser.add_argument(
        "--reference-dir", type=str, required=True,
        help="Directory with reference segmentation .npy files.",
    )
    parser.add_argument(
        "--generated-dir", type=str, required=True,
        help="Directory with generated segmentation .npy files.",
    )
    parser.add_argument(
        "--glob", type=str, default="*.npy",
        help="Glob pattern for mask files (default: '*.npy').",
    )
    parser.add_argument(
        "--use-rips", action="store_true",
        help="Use Rips complex instead of Alpha complex.",
    )
    parser.add_argument(
        "--n-bins", type=int, default=100,
        help="Persistence landscape bins (default: 100).",
    )
    parser.add_argument(
        "--n-layers", type=int, default=1,
        help="Persistence landscape layers (default: 1).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=500,
        help="Masks per batch during loading (default: 500). Lower = less memory, slower.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to save results as JSON.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    result = compute_topofd_from_folders(
        reference_dir=args.reference_dir,
        generated_dir=args.generated_dir,
        glob_pattern=args.glob,
        batch_size=args.batch_size,
        use_alpha=not args.use_rips,
        n_landscape_bins=args.n_bins,
        n_landscape_layers=args.n_layers,
    )

    print(result)

    if args.output:
        import json

        out = {
            "topofd": result.topofd,
            "per_channel": {str(k): v for k, v in result.per_channel.items()},
            "n_reference": result.n_reference,
            "n_generated": result.n_generated,
            "n_channels": result.n_channels,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    _cli()
