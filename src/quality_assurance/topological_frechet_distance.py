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

    indices = range(1, labelled.max() + 1)
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
    if seg.ndim == 2:
        return [extract_cell_centres(seg)]
    elif seg.ndim == 3:
        n_channels = seg.shape[2]
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
        cplx = gudhi.AlphaComplex(points=point_cloud.tolist())
        st = cplx.create_simplex_tree()
    else:
        cplx = gudhi.RipsComplex(points=point_cloud.tolist())
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

    covmean_raw, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    covmean: np.ndarray = np.asarray(covmean_raw)

    if not np.isfinite(covmean).all():
        logger.warning(
            "Fréchet distance: singular product – adding eps=%s to diagonal", eps
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = np.asarray(linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset)))

    # Discard small imaginary artefacts
    if np.iscomplexobj(covmean):
        max_imag = np.max(np.abs(covmean.imag))
        if max_imag > 1e-3:
            raise ValueError(
                f"Fréchet distance: significant imaginary component ({max_imag})"
            )
        covmean = covmean.real

    return float(
        diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    )


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


def compute_topofd(
    reference_segmentations: Sequence[np.ndarray],
    generated_segmentations: Sequence[np.ndarray],
    *,
    use_alpha: bool = True,
    n_landscape_bins: int = 100,
    n_landscape_layers: int = 1,
    min_cells: int = 3,
) -> TopoFDResult:
    """Compute the Topological Fréchet Distance between two sets of segmentations.

    Parameters
    ----------
    reference_segmentations : sequence of np.ndarray
        Reference (ground-truth) segmentation masks.  Each element is either
        (H, W) for a single cell type or (H, W, C) for *C* cell types.
    generated_segmentations : sequence of np.ndarray
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

    if len(reference_segmentations) == 0 or len(generated_segmentations) == 0:
        raise ValueError("Both reference and generated sets must be non-empty.")

    # Determine number of channels from first element
    sample_ref = np.asarray(reference_segmentations[0])
    n_channels = sample_ref.shape[2] if sample_ref.ndim == 3 else 1

    vec_dim = n_landscape_bins * n_landscape_layers

    # -------- helper: process one set of segmentations --------
    def _process_set(
        segmentations: Sequence[np.ndarray],
    ) -> Dict[int, List[np.ndarray]]:
        """Return {channel_idx: [landscape_vector, ...]}."""
        vectors: Dict[int, List[np.ndarray]] = {c: [] for c in range(n_channels)}

        for seg in segmentations:
            centres_per_ch = extract_centres_multichannel(np.asarray(seg))

            # Validate channel count
            if len(centres_per_ch) != n_channels:
                logger.warning(
                    "Skipping segmentation with %d channels (expected %d)",
                    len(centres_per_ch),
                    n_channels,
                )
                continue

            for ch_idx, centres in enumerate(centres_per_ch):
                if centres.shape[0] < min_cells:
                    # Too few cells for meaningful topology – use zero vector
                    vectors[ch_idx].append(np.zeros(vec_dim, dtype=np.float64))
                    continue

                diagram = compute_persistence_diagram_1d(
                    centres, use_alpha=use_alpha
                )
                vec = vectorise_persistence_landscape(
                    diagram,
                    n_bins=n_landscape_bins,
                    n_layers=n_landscape_layers,
                )
                vectors[ch_idx].append(vec)

        return vectors

    logger.info("Processing reference segmentations (%d)…", len(reference_segmentations))
    ref_vectors = _process_set(reference_segmentations)

    logger.info("Processing generated segmentations (%d)…", len(generated_segmentations))
    gen_vectors = _process_set(generated_segmentations)

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
    **kwargs,
) -> TopoFDResult:
    """Compute TopoFD by loading segmentation masks from two directories.

    Each ``.npy`` file should contain an array of shape (H, W) or (H, W, C).

    Parameters
    ----------
    reference_dir, generated_dir : path-like
        Directories containing segmentation mask files.
    glob_pattern : str
        Glob pattern for discovering mask files.
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

    logger.info("Loading %d reference masks from %s", len(ref_files), ref_dir)
    ref_segs = [np.load(f) for f in ref_files]

    logger.info("Loading %d generated masks from %s", len(gen_files), gen_dir)
    gen_segs = [np.load(f) for f in gen_files]

    return compute_topofd(ref_segs, gen_segs, **kwargs)


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
