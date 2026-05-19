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

from .core import CATEGORICAL_CMAP, build_label_palette

from ..statistics.spatial_metrics import (
    compute_ripley_L_bootstrap,
    compute_voronoi_areas,
    compute_knn_metrics,
)

logger = logging.getLogger(__name__)

PAM50_ORDER = ["Basal", "Her2", "LumA", "LumB", "Normal"]


def _ordered_subtypes(values: List[str]) -> List[str]:
    """Return PAM50-canonical subtype order with unknown labels appended."""
    ordered = [s for s in PAM50_ORDER if s in values]
    ordered += [s for s in values if s not in PAM50_ORDER]
    return ordered


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
        n_bootstrap=n_bootstrap, seed=42, window=bounding_box,
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
        # Patient-level aggregation: each patient contributes one mean L(r) curve
        # regardless of how many tiles were sampled from them.
        per_patient_dict = (
            subtype_data.get("per_patient", {})
            if isinstance(subtype_data, dict)
            else {}
        )

        def _tile_curves(tiles):
            L_obs, L_lower, L_upper, radii = [], [], [], []
            for td in tiles:
                if td['ripley_L'].size == 0:
                    continue
                L_obs.append(td['ripley_L'])
                L_lower.append(td['ripley_L_lower'])
                L_upper.append(td['ripley_L_upper'])
                radii.append(td['ripley_radii'])
            return L_obs, L_lower, L_upper, radii

        if per_patient_dict:
            # One mean L(r) curve per patient, then summarise across patients
            pat_L_obs = []
            radii_ref = None
            for pid, tiles in per_patient_dict.items():
                L_obs, _, _, radii_list = _tile_curves(tiles)
                if not L_obs:
                    continue
                pat_L_obs.append(np.mean(L_obs, axis=0))
                if radii_ref is None:
                    radii_ref = radii_list[0]
            n_label = f"n={len(pat_L_obs)} patients"
            L_obs_all = pat_L_obs
        else:
            # Fallback: tile-level aggregation (old flat-list format)
            tile_list = _tile_list(subtype_data)
            L_obs_all, _, _, radii_list = _tile_curves(tile_list)
            radii_ref = radii_list[0] if radii_list else None
            n_label = f"n={len(L_obs_all)} tiles"

        if not L_obs_all or radii_ref is None:
            ax.set_title(f"{subtype} (no data)")
            continue

        radii = radii_ref
        L_obs_arr = np.array(L_obs_all)          # (n_patients, n_radii)
        L_obs_mean = L_obs_arr.mean(axis=0)
        n_obs = len(L_obs_arr)
        L_obs_sem = (
            L_obs_arr.std(axis=0, ddof=1) / np.sqrt(n_obs)
            if n_obs > 1 else np.zeros_like(L_obs_mean)
        )

        # Plot mean curve + ±1 SEM band across patients
        ax.plot(radii, L_obs_mean, 'b-', linewidth=2, label='Mean L(r)')
        ax.fill_between(
            radii,
            L_obs_mean - L_obs_sem,
            L_obs_mean + L_obs_sem,
            alpha=0.25, color='blue', label='±1 SEM across patients',
        )
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='CSR (L = 0)')
        ax.set_title(f"{subtype} ({n_label})")
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


def _auc_ripley_L(L: np.ndarray, radii: np.ndarray, r_min: float, r_max: float) -> float:
    """Trapezoidal AUC of L(r) over [r_min, r_max]."""
    mask = (radii >= r_min) & (radii <= r_max)
    if mask.sum() < 2:
        return np.nan
    return float(np.trapz(L[mask], radii[mask]))


def _draw_significance_bars(
    ax: plt.Axes,
    subtypes: List[str],
    pairwise_q: Dict[str, float],
    y_top: float,
    y_range: float,
) -> None:
    """Annotate significant pairs with bracketed asterisks above the boxplot."""
    sig: List[tuple] = []
    for key, q in pairwise_q.items():
        if q >= 0.05:
            continue
        idx = key.find("_vs_")
        if idx < 0:
            continue
        s1, s2 = key[:idx], key[idx + 4:]
        if s1 not in subtypes or s2 not in subtypes:
            continue
        stars = "***" if q < 0.001 else "**" if q < 0.01 else "*"
        sig.append((subtypes.index(s1), subtypes.index(s2), stars))

    # Stack bars narrowest-span first to minimise overlap
    sig.sort(key=lambda t: abs(t[1] - t[0]))
    step = 0.08 * y_range
    for level, (i1, i2, stars) in enumerate(sig):
        y = y_top + step * level
        x1, x2 = min(i1, i2), max(i1, i2)
        bar_h = step * 0.35
        ax.plot([x1, x1, x2, x2], [y, y + bar_h, y + bar_h, y], "k-", lw=1.0)
        ax.text((x1 + x2) / 2, y + bar_h, stars, ha="center", va="bottom", fontsize=9)

    if sig:
        ax.set_ylim(top=y_top + step * (len(sig) + 1.5))


def plot_ripley_L_auc_comparison(
    results: Dict[str, Union[List[Dict], Dict]],
    output_dir: Union[str, Path],
    r_min: float = 20.0,
    r_max: float = 100.0,
    cmap_name: str = CATEGORICAL_CMAP,
    dpi: int = 150,
) -> Dict:
    """Reduce per-patient Ripley L(r) to a scalar AUC and compare across subtypes.

    For each patient the mean tile-level L(r) curve is reduced to its area
    under the curve (AUC) over ``[r_min, r_max]`` px.  A Kruskal-Wallis test
    followed by pairwise Mann-Whitney U tests (Benjamini-Hochberg FDR) is then
    run across subtypes.  The figure shows a boxplot with individual patient
    dots and significance brackets for all q < 0.05 pairs.

    Parameters
    ----------
    results : dict[str, list[dict] | dict]
        Same format as :func:`plot_ripley_L_by_subtype`.
    output_dir : path
    r_min, r_max : float
        Radius range (px) for the AUC integration.
    cmap_name, dpi : see other plot functions.

    Returns
    -------
    dict
        Output of :func:`run_subtype_tests` — Kruskal-Wallis H/p, pairwise q.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available; skipping AUC plot.")
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Collect one AUC scalar per patient per subtype
    # ------------------------------------------------------------------
    patient_auc: Dict[str, np.ndarray] = {}
    for subtype, subtype_data in sorted(results.items()):
        per_patient_dict = (
            subtype_data.get("per_patient", {})
            if isinstance(subtype_data, dict) else {}
        )
        aucs: List[float] = []
        if per_patient_dict:
            for pid, tiles in per_patient_dict.items():
                L_curves = [td["ripley_L"] for td in tiles if td["ripley_L"].size > 0]
                radii_list = [td["ripley_radii"] for td in tiles if td["ripley_radii"].size > 0]
                if not L_curves:
                    continue
                mean_L = np.mean(L_curves, axis=0)
                auc = _auc_ripley_L(mean_L, radii_list[0], r_min, r_max)
                if np.isfinite(auc):
                    aucs.append(auc)
        else:
            # Fallback: treat all tiles as one virtual patient
            tile_list = _tile_list(subtype_data)
            L_curves = [td["ripley_L"] for td in tile_list if td["ripley_L"].size > 0]
            radii_all = [td["ripley_radii"] for td in tile_list if td["ripley_radii"].size > 0]
            if L_curves:
                mean_L = np.mean(L_curves, axis=0)
                auc = _auc_ripley_L(mean_L, radii_all[0], r_min, r_max)
                if np.isfinite(auc):
                    aucs.append(auc)
        patient_auc[subtype] = np.array(aucs)

    # ------------------------------------------------------------------
    # Statistical test
    # ------------------------------------------------------------------
    metric_name = f"ripley_L_auc_{int(r_min)}-{int(r_max)}px"
    test_results = run_subtype_tests(patient_auc, metric_name=metric_name)
    logger.info("Ripley L AUC test: %s", test_results)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    records = [
        {"Subtype": subtype, "AUC L(r)": auc}
        for subtype, aucs in patient_auc.items()
        for auc in aucs
    ]
    if not records:
        logger.warning("No AUC data to plot.")
        return test_results

    df = pd.DataFrame(records)
    subtypes = _ordered_subtypes(df["Subtype"].unique().tolist())
    palette = build_label_palette(np.array(subtypes), cmap_name=cmap_name)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df, x="Subtype", y="AUC L(r)", hue="Subtype",
        order=subtypes, palette=palette, legend=False,
        width=0.55, fliersize=0, ax=ax,
    )
    sns.stripplot(
        data=df, x="Subtype", y="AUC L(r)",
        order=subtypes, color="black", alpha=0.35, size=3, jitter=True, ax=ax,
    )

    # n-labels below the x-axis ticks
    y_lo, y_hi = ax.get_ylim()
    for i, s in enumerate(subtypes):
        n = len(patient_auc.get(s, []))
        ax.text(i, y_lo - 0.03 * (y_hi - y_lo), f"n={n}",
                ha="center", va="top", fontsize=8, color="dimgray")

    # Significance bars
    pairwise_q = test_results.get("pairwise_q", {})
    _draw_significance_bars(ax, subtypes, pairwise_q, y_top=y_hi, y_range=(y_hi - y_lo))

    # Title with Kruskal-Wallis result
    kw_p = test_results.get("kruskal_p", float("nan"))
    kw_h = test_results.get("kruskal_H", float("nan"))
    ax.set_title(
        f"Ripley L(r) AUC [{r_min:.0f}–{r_max:.0f} px] per Patient by PAM50 Subtype\n"
        f"Kruskal-Wallis H={kw_h:.2f}, p={kw_p:.3g}",
        fontsize=10,
    )
    ax.set_xlabel("PAM50 Subtype")
    ax.set_ylabel(f"AUC of L(r)  [{r_min:.0f}–{r_max:.0f} px]")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = output_dir / "spatial_ripley_L_auc_by_subtype.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info("Saved Ripley L AUC plot → %s", out)

    return test_results


def plot_voronoi_distribution(
    results: Dict[str, Union[List[Dict], Dict]],
    output_dir: Union[str, Path],
    cmap_name: str = CATEGORICAL_CMAP,
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
    subtypes = _ordered_subtypes(df["Subtype"].dropna().astype(str).unique().tolist())
    palette = build_label_palette(np.array(subtypes), cmap_name=cmap_name)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(
        data=df,
        x="Subtype",
        y="Voronoi Area (px²)",
        hue="Subtype",
        order=subtypes,
        palette=palette,
        legend=False,
        ax=ax,
    )
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
    cmap_name: str = CATEGORICAL_CMAP,
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
    subtypes = _ordered_subtypes(df["Subtype"].dropna().astype(str).unique().tolist())
    palette = build_label_palette(np.array(subtypes), cmap_name=cmap_name)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(
        axes,
        ["kNN Mean Clustering Coeff", "kNN Mean Degree", "Largest Component Frac"],
    ):
        metric_df = df[np.isfinite(df[col])]
        if metric_df.empty:
            ax.set_title(f"{col} by Subtype")
            ax.set_xlabel("")
            ax.set_ylabel(col)
            ax.text(
                0.5,
                0.5,
                "no finite data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="grey",
            )
            continue

        sns.boxplot(
            data=metric_df,
            x="Subtype",
            y=col,
            hue="Subtype",
            order=subtypes,
            palette=palette,
            legend=False,
            ax=ax,
        )
        ax.set_ylabel(col)
        ax.set_title(f"{col} by Subtype")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle("kNN Graph Metrics per PAM50 Subtype", fontsize=11, y=1.02)
    fig.tight_layout()
    out = output_dir / "spatial_knn_metrics_by_subtype.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved kNN metrics plot → {out}")
