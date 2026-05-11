#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization and pipeline integration for spatial metrics.

Functions:
  - plot_ripley_L_by_subtype: Scale-dependent clustering trends per subtype
  - plot_voronoi_distribution: Cell-type dispersion as Voronoi area distributions
  - plot_knn_metrics_comparison: Degree, clustering coeff, component size distributions
  - compute_spatial_metrics_per_subtype: Tile→patient→subtype aggregation with stats
  - run_spatial_metrics_pipeline: End-to-end pipeline entry point
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..statistics.spatial_metrics import (
    compute_ripley_L_bootstrap,
    compute_voronoi_areas,
    compute_knn_metrics,
)

logger = logging.getLogger(__name__)


def compute_spatial_metrics_per_tile(
    mask: np.ndarray,
    channel_idx: int,
    bounding_box: Optional[Tuple[float, float]] = None,
    n_bootstrap: int = 99,
) -> Dict[str, float | np.ndarray]:
    """Extract cell centroids from a single-channel mask and compute spatial metrics.
    
    Parameters
    ----------
    mask : (H, W) array
        Binary or probability map for one cell type (already extracted from multi-channel).
    channel_idx : int
        Channel ID (for logging).
    bounding_box : (width, height) or None
    
    Returns
    -------
    metrics : dict
        'n_cells': int — cell count in tile
        'ripley_radii': (M,) array
        'ripley_L': (M,) array
        'ripley_L_lower': (M,) array
        'ripley_L_upper': (M,) array
        'voronoi_areas': (N,) array
        'voronoi_median_area': float
        'voronoi_std_area': float
        'knn_degree': (N,) array
        'knn_mean_degree': float
        'knn_clustering_coeff': (N,) array
        'knn_mean_clustering': float
        'knn_largest_component_frac': float
    """
    # Extract cell centroids (connected component centroids or probability peaks)
    # Simple approach: find local maxima via CWT or just use all above-threshold pixels
    from scipy.ndimage import label, center_of_mass
    
    mask = np.asarray(mask, dtype=float)
    if mask.max() < 1e-6:
        return {
            'n_cells': 0,
            'ripley_radii': np.array([]),
            'ripley_L': np.array([]),
            'ripley_L_lower': np.array([]),
            'ripley_L_upper': np.array([]),
            'voronoi_areas': np.array([]),
            'voronoi_median_area': np.nan,
            'voronoi_std_area': np.nan,
            'knn_degree': np.array([]),
            'knn_mean_degree': np.nan,
            'knn_clustering_coeff': np.array([]),
            'knn_mean_clustering': np.nan,
            'knn_largest_component_frac': np.nan,
        }
    
    # Threshold and label connected components
    binary = mask > 0.5
    labeled, n_cells = label(binary)
    
    if n_cells < 3:
        return {
            'n_cells': n_cells,
            'ripley_radii': np.array([]),
            'ripley_L': np.array([]),
            'ripley_L_lower': np.array([]),
            'ripley_L_upper': np.array([]),
            'voronoi_areas': np.array([]),
            'voronoi_median_area': np.nan,
            'voronoi_std_area': np.nan,
            'knn_degree': np.array([]),
            'knn_mean_degree': np.nan,
            'knn_clustering_coeff': np.array([]),
            'knn_mean_clustering': np.nan,
            'knn_largest_component_frac': np.nan,
        }
    
    # Get centroids
    centroids = np.array(center_of_mass(binary, labeled, range(1, n_cells + 1)))
    # Swap to (x, y) from (row, col)
    points = centroids[:, [1, 0]]
    
    if bounding_box is None:
        bounding_box = (mask.shape[1], mask.shape[0])
    
    # Compute all metrics
    radii, L_obs, L_lower, L_upper = compute_ripley_L_bootstrap(
        points, radii=np.arange(10, min(200, bounding_box[0] // 2), 10),
        n_bootstrap=n_bootstrap, seed=42
    )
    
    voronoi_areas = compute_voronoi_areas(points, bounding_box)
    voronoi_areas_valid = voronoi_areas[np.isfinite(voronoi_areas)]
    
    knn_metrics = compute_knn_metrics(points, k=min(5, n_cells - 1))
    
    return {
        'n_cells': n_cells,
        'ripley_radii': radii,
        'ripley_L': L_obs,
        'ripley_L_lower': L_lower,
        'ripley_L_upper': L_upper,
        'voronoi_areas': voronoi_areas_valid,
        'voronoi_median_area': float(np.median(voronoi_areas_valid)) if len(voronoi_areas_valid) > 0 else np.nan,
        'voronoi_std_area': float(np.std(voronoi_areas_valid)) if len(voronoi_areas_valid) > 0 else np.nan,
        'knn_degree': knn_metrics['degree'],
        'knn_mean_degree': knn_metrics['mean_degree'],
        'knn_clustering_coeff': knn_metrics['clustering_coeff'],
        'knn_mean_clustering': knn_metrics['mean_clustering_coeff'],
        'knn_largest_component_frac': knn_metrics['largest_component_frac'],
    }


def aggregate_metrics_per_patient(
    tile_metrics_list: List[Dict],
) -> Dict[str, float]:
    """Aggregate per-tile topology metrics to a single patient-level summary.

    Parameters
    ----------
    tile_metrics_list : list[dict]
        Output from :func:`compute_spatial_metrics_per_tile` for each tile
        belonging to one patient.

    Returns
    -------
    dict
        Keys: ``{metric}_median``, ``{metric}_q1``, ``{metric}_q3``,
        ``{metric}_count``, and ``n_tiles`` for the five scalar topology
        metrics (Voronoi area, kNN clustering, etc.).
    """
    if not tile_metrics_list:
        return {}

    agg: Dict[str, float] = {}
    for key in [
        "voronoi_median_area", "voronoi_std_area",
        "knn_mean_degree", "knn_mean_clustering",
        "knn_largest_component_frac",
    ]:
        vals = [m[key] for m in tile_metrics_list if np.isfinite(m.get(key, np.nan))]
        if vals:
            agg[f"{key}_median"] = float(np.median(vals))
            agg[f"{key}_q1"]     = float(np.percentile(vals, 25))
            agg[f"{key}_q3"]     = float(np.percentile(vals, 75))
            agg[f"{key}_count"]  = len(vals)

    agg["n_tiles"] = len(tile_metrics_list)
    return agg


# ---------------------------------------------------------------------------
# BH FDR helper (statsmodels preferred; Bonferroni fallback)
# ---------------------------------------------------------------------------

try:
    from statsmodels.stats.multitest import multipletests as _sm_multipletests

    def _fdr_bh(pvals: List[float]) -> np.ndarray:
        if not pvals:
            return np.array([])
        _, q, _, _ = _sm_multipletests(pvals, method="fdr_bh")
        return np.asarray(q)

except ImportError:
    def _fdr_bh(pvals: List[float]) -> np.ndarray:  # type: ignore[misc]
        """Bonferroni correction as fallback when statsmodels is unavailable."""
        if not pvals:
            return np.array([])
        return np.minimum(np.asarray(pvals) * len(pvals), 1.0)


def run_subtype_tests(
    patient_vals_by_subtype: Dict[str, np.ndarray],
    metric_name: str = "",
    alpha: float = 0.05,
) -> Dict:
    """Kruskal-Wallis + pairwise Mann-Whitney U with Benjamini-Hochberg FDR.

    The unit of observation is **one value per patient** (typically the
    median of tile-level measurements within that patient).  This avoids
    pseudo-replication from pooling tiles directly.

    Parameters
    ----------
    patient_vals_by_subtype : {subtype: (n_patients,)}
        One finite scalar per patient per subtype.  NaN entries are dropped.
        Groups with fewer than 3 valid patients are excluded.
    metric_name : str
        Label stored in the result dict for traceability.
    alpha : float
        Significance threshold for ``significant_pairs``.

    Returns
    -------
    dict with keys:
        ``metric``, ``n_patients``, ``kruskal_H``, ``kruskal_p``,
        ``pairwise_q`` ({"{s1}_vs_{s2}": q}), ``significant_pairs``.
    """
    from scipy.stats import kruskal, mannwhitneyu

    valid = {
        s: arr[np.isfinite(arr)]
        for s, arr in patient_vals_by_subtype.items()
        if np.isfinite(np.asarray(arr, dtype=float)).sum() >= 3
    }
    if len(valid) < 2:
        return {"metric": metric_name, "error": "fewer than 2 groups with ≥3 patients"}

    subtypes = list(valid.keys())
    groups   = [valid[s] for s in subtypes]

    h_stat, p_kw = kruskal(*groups)

    pairs:  List[tuple] = []
    raw_p:  List[float] = []
    for i, s1 in enumerate(subtypes):
        for s2 in subtypes[i + 1:]:
            _, p = mannwhitneyu(valid[s1], valid[s2], alternative="two-sided")
            pairs.append((s1, s2))
            raw_p.append(float(p))

    q_vals      = _fdr_bh(raw_p)
    pairwise_q  = {f"{a}_vs_{b}": float(q) for (a, b), q in zip(pairs, q_vals)}

    return {
        "metric":           metric_name,
        "n_patients":       {s: int(len(v)) for s, v in valid.items()},
        "kruskal_H":        float(h_stat),
        "kruskal_p":        float(p_kw),
        "pairwise_q":       pairwise_q,
        "significant_pairs": [k for k, q in pairwise_q.items() if q < alpha],
    }


def compare_metrics_across_subtypes(
    subtype_patient_metrics: Dict[str, Dict[str, Dict[str, float]]],
    metrics: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> Dict[str, Dict]:
    """Statistical comparison of topology metrics across PAM50 subtypes.

    Parameters
    ----------
    subtype_patient_metrics : {subtype: {pid: aggregate_dict}}
        Per-patient aggregated metrics from :func:`aggregate_metrics_per_patient`,
        grouped by subtype.  Each inner dict is the output for one patient.
    metrics : list[str] or None
        Metric base names to test.  Defaults to the four key topology scalars.
    alpha : float

    Returns
    -------
    {metric_name: :func:`run_subtype_tests` result dict}
    """
    if metrics is None:
        metrics = [
            "voronoi_median_area",
            "voronoi_std_area",
            "knn_mean_clustering",
            "knn_largest_component_frac",
        ]

    results: Dict[str, Dict] = {}
    for metric in metrics:
        key = f"{metric}_median"
        patient_vals: Dict[str, np.ndarray] = {
            subtype: np.array([
                pid_dict[key]
                for pid_dict in pid_to_dict.values()
                if np.isfinite(pid_dict.get(key, np.nan))
            ])
            for subtype, pid_to_dict in subtype_patient_metrics.items()
        }
        results[metric] = run_subtype_tests(patient_vals, metric_name=metric, alpha=alpha)
    return results


def _tile_list(subtype_result: Union[List[Dict], Dict]) -> List[Dict]:
    """Extract the flat tile list from either format.

    Accepts the old format (plain ``list[dict]``) or the new enriched format
    (``{"tiles": list[dict], "per_patient": {pid: list[dict]}}``).
    """
    if isinstance(subtype_result, dict):
        return subtype_result.get("tiles", [])
    return subtype_result  # type: ignore[return-value]


def plot_ripley_L_by_subtype(
    results: Dict[str, Union[List[Dict], Dict]],
    output_dir: Union[str, Path],
    dpi: int = 150,
) -> None:
    """Plot Ripley's L(r) curves with CSR envelope for each subtype.

    Parameters
    ----------
    results : dict[str, list[dict] | dict]
        Per-subtype tile-level metrics from
        :func:`~tfd_separability._collect_spatial_topology_for_subtypes`.
        Accepts both the old flat-list format and the new enriched dict format.
    output_dir : path
    dpi : int
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping Ripley plot.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_subtypes = len(results)
    fig, axes = plt.subplots(1, n_subtypes, figsize=(n_subtypes * 4, 4), sharex=True, sharey=True)
    if n_subtypes == 1:
        axes = [axes]

    for ax, (subtype, subtype_data) in zip(axes, sorted(results.items())):
        tile_list = _tile_list(subtype_data)
        # Collect L curves from all tiles
        radii_all = []
        L_obs_all = []
        L_lower_all = []
        L_upper_all = []
        
        for tile_dict in tile_list:
            if tile_dict['ripley_L'].size == 0:
                continue
            radii_all.append(tile_dict['ripley_radii'])
            L_obs_all.append(tile_dict['ripley_L'])
            L_lower_all.append(tile_dict['ripley_L_lower'])
            L_upper_all.append(tile_dict['ripley_L_upper'])
        
        if not L_obs_all:
            ax.set_title(f"{subtype} (no data)")
            continue
        
        # Average L and envelope across tiles
        radii = radii_all[0]  # assume same for all tiles
        L_obs_mean = np.mean(L_obs_all, axis=0)
        L_obs_std = np.std(L_obs_all, axis=0)
        L_lower_mean = np.mean(L_lower_all, axis=0)
        L_upper_mean = np.mean(L_upper_all, axis=0)
        
        # Plot
        ax.plot(radii, L_obs_mean, 'b-', linewidth=2, label='Observed L(r)')
        ax.fill_between(radii, L_lower_mean, L_upper_mean, alpha=0.2, color='blue', label='CSR envelope (95%)')
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_title(f"{subtype} (n={len(L_obs_all)} tiles)")
        ax.set_xlabel("Radius r (px)")
        if ax == axes[0]:
            ax.set_ylabel("L(r)")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Ripley's L(r) by Subtype: Clustering vs. Inhibition at Multiple Scales", fontsize=11)
    fig.tight_layout()
    out = output_dir / "spatial_ripley_L_by_subtype.png"
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Ripley plot → {out}")


def plot_voronoi_distribution(
    results: Dict[str, Union[List[Dict], Dict]],
    output_dir: Union[str, Path],
    dpi: int = 150,
) -> None:
    """Plot Voronoi cell area distributions per subtype.

    Parameters
    ----------
    results : dict[str, list[dict] | dict]
        Per-subtype tile metrics (flat list or enriched dict — see
        :func:`_tile_list`).
    output_dir : path
    dpi : int
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available; skipping Voronoi plot.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for subtype, subtype_data in results.items():
        for tile_dict in _tile_list(subtype_data):
            for area in tile_dict["voronoi_areas"]:
                if np.isfinite(area):
                    records.append({"Subtype": subtype, "Voronoi Area (px²)": area})

    if not records:
        logger.warning("No Voronoi data to plot.")
        return

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=df, x="Subtype", y="Voronoi Area (px²)", ax=ax, palette="Set2")
    ax.set_ylabel("Voronoi Cell Area (px²)")
    ax.set_title("Cell Dispersion: Voronoi Area Distribution per PAM50 Subtype")
    fig.tight_layout()
    out = output_dir / "spatial_voronoi_areas_by_subtype.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Voronoi plot → {out}")


def plot_knn_metrics_comparison(
    results: Dict[str, Union[List[Dict], Dict]],
    output_dir: Union[str, Path],
    dpi: int = 150,
) -> None:
    """Plot kNN graph metrics (clustering coeff, largest component) per subtype.

    Parameters
    ----------
    results : dict[str, list[dict] | dict]
        Per-subtype tile metrics (flat list or enriched dict — see
        :func:`_tile_list`).
    output_dir : path
    dpi : int
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available; skipping kNN plot.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for subtype, subtype_data in results.items():
        for tile_dict in _tile_list(subtype_data):
            records.append({
                "Subtype":                  subtype,
                "kNN Mean Clustering Coeff": tile_dict["knn_mean_clustering"],
                "kNN Mean Degree":           tile_dict["knn_mean_degree"],
                "Largest Component Frac":    tile_dict["knn_largest_component_frac"],
            })

    if not records:
        logger.warning("No kNN data to plot.")
        return

    df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(
        axes,
        ["kNN Mean Clustering Coeff", "kNN Mean Degree", "Largest Component Frac"],
    ):
        sns.boxplot(data=df, x="Subtype", y=col, ax=ax, palette="Set2")
        ax.set_ylabel(col)
        ax.set_title(f"{col} by Subtype")

    fig.suptitle("kNN Graph Metrics per PAM50 Subtype", fontsize=11, y=1.02)
    fig.tight_layout()
    out = output_dir / "spatial_knn_metrics_by_subtype.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved kNN metrics plot → {out}")
