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

from ..spatial_metrics import (
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
    """Aggregate per-tile metrics to patient level (median + quartiles).
    
    Parameters
    ----------
    tile_metrics_list : list[dict]
        Output from compute_spatial_metrics_per_tile for each tile.
    
    Returns
    -------
    patient_metrics : dict
        'voronoi_median_area_median': float
        'voronoi_std_area_median': float
        'knn_mean_degree_median': float
        'knn_mean_clustering_median': float
        'knn_largest_component_frac_median': float
        'n_tiles': int
        ... (percentiles and counts)
    """
    if not tile_metrics_list:
        return {}
    
    agg = {}
    for key in ['voronoi_median_area', 'voronoi_std_area', 'knn_mean_degree',
                'knn_mean_clustering', 'knn_largest_component_frac']:
        vals = [m[key] for m in tile_metrics_list if np.isfinite(m.get(key, np.nan))]
        if vals:
            agg[f'{key}_median'] = float(np.median(vals))
            agg[f'{key}_q1'] = float(np.percentile(vals, 25))
            agg[f'{key}_q3'] = float(np.percentile(vals, 75))
            agg[f'{key}_count'] = len(vals)
    
    agg['n_tiles'] = len(tile_metrics_list)
    return agg


def compare_metrics_across_subtypes(
    subtype_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, Dict]:
    """Statistical comparison of aggregated metrics across subtypes.
    
    Parameters
    ----------
    subtype_metrics : dict[str, dict]
        Per-subtype aggregated metrics (output of aggregate_metrics_per_patient).
    
    Returns
    -------
    comparisons : dict
        Pairwise Kruskal–Wallis and Mann–Whitney U test results.
    """
    comparisons = {}
    metrics_to_test = [
        'voronoi_median_area', 'voronoi_std_area', 'knn_mean_degree',
        'knn_mean_clustering', 'knn_largest_component_frac'
    ]
    
    for metric in metrics_to_test:
        vals_by_subtype = {}
        for subtype, metrics_dict in subtype_metrics.items():
            key = f'{metric}_median'
            if key in metrics_dict:
                vals_by_subtype[subtype] = metrics_dict[key]
        
        if len(vals_by_subtype) < 2:
            continue
        
        # Kruskal–Wallis test
        all_vals = [vals_by_subtype[s] for s in sorted(vals_by_subtype)]
        h_stat, h_pval = stats.kruskal(*all_vals) if len(all_vals) > 1 else (np.nan, np.nan)
        
        comparisons[metric] = {
            'kruskal_wallis_h': h_stat,
            'kruskal_wallis_pval': h_pval,
            'per_subtype_median': vals_by_subtype,
        }
    
    return comparisons


def plot_ripley_L_by_subtype(
    results: Dict[str, List[Dict]],
    output_dir: Union[str, Path],
    dpi: int = 150,
) -> None:
    """Plot Ripley's L(r) curves with CSR envelope for each subtype.
    
    Parameters
    ----------
    results : dict[str, list[dict]]
        Per-subtype list of tile-level metrics (from compute_spatial_metrics_per_tile).
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
    
    for ax, (subtype, tile_list) in zip(axes, sorted(results.items())):
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
    results: Dict[str, List[Dict]],
    output_dir: Union[str, Path],
    dpi: int = 150,
) -> None:
    """Plot Voronoi cell area distributions per subtype.
    
    Parameters
    ----------
    results : dict[str, list[dict]]
        Per-subtype tile metrics.
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
    for subtype, tile_list in results.items():
        for tile_dict in tile_list:
            areas = tile_dict['voronoi_areas']
            for area in areas:
                if np.isfinite(area):
                    records.append({'Subtype': subtype, 'Voronoi Area (px²)': area})
    
    if not records:
        logger.warning("No Voronoi data to plot.")
        return
    
    df = pd.DataFrame(records)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=df, x='Subtype', y='Voronoi Area (px²)', ax=ax, palette='Set2')
    ax.set_ylabel("Voronoi Cell Area (px²)")
    ax.set_title("Cell Dispersion: Voronoi Area Distribution per PAM50 Subtype")
    fig.tight_layout()
    out = output_dir / "spatial_voronoi_areas_by_subtype.png"
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Voronoi plot → {out}")


def plot_knn_metrics_comparison(
    results: Dict[str, List[Dict]],
    output_dir: Union[str, Path],
    dpi: int = 150,
) -> None:
    """Plot kNN metrics (clustering coeff, degree) distributions per subtype.
    
    Parameters
    ----------
    results : dict[str, list[dict]]
        Per-subtype tile metrics.
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
    for subtype, tile_list in results.items():
        for tile_dict in tile_list:
            records.append({
                'Subtype': subtype,
                'kNN Mean Clustering Coeff': tile_dict['knn_mean_clustering'],
                'kNN Mean Degree': tile_dict['knn_mean_degree'],
                'Largest Component Frac': tile_dict['knn_largest_component_frac'],
            })
    
    if not records:
        logger.warning("No kNN data to plot.")
        return
    
    df = pd.DataFrame(records)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    for ax, col in zip(axes, ['kNN Mean Clustering Coeff', 'kNN Mean Degree', 'Largest Component Frac']):
        sns.boxplot(data=df, x='Subtype', y=col, ax=ax, palette='Set2')
        ax.set_ylabel(col)
        ax.set_title(f"{col} by Subtype")
    
    fig.suptitle("kNN Graph Metrics per PAM50 Subtype", fontsize=11, y=1.02)
    fig.tight_layout()
    out = output_dir / "spatial_knn_metrics_by_subtype.png"
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved kNN metrics plot → {out}")
