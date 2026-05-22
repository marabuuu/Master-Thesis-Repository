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
        Significance threshold for ``significant_pairs`` (applied to the
        within-call BH-corrected ``pairwise_q``).

    Returns
    -------
    dict with keys:
        ``metric``, ``n_patients``,
        ``kruskal_H``, ``kruskal_p``, ``eta_sq``,
        ``pairwise_p_raw``  — raw Mann-Whitney p-values before any correction,
        ``pairwise_q``      — BH-corrected q-values (within this call),
        ``pairwise_r``      — rank-biserial correlations ∈ [−1, 1],
        ``significant_pairs``.

    Notes
    -----
    ``pairwise_q`` here is corrected *within* the set of pairs for this
    single metric.  When multiple metrics are tested in one analysis, call
    :func:`_globally_correct_stats` afterwards to replace ``pairwise_q``
    with a globally pooled BH correction and to add ``kruskal_q``.

    Effect sizes
    ------------
    *η²* (eta-squared) = H / (n − 1).  Cohen's conventions: 0.01 small,
    0.06 medium, 0.14 large.

    *Rank-biserial r* = 1 − 2U₁ / (n₁ × n₂).  Cohen's conventions:
    |r| ≥ 0.1 small, ≥ 0.3 medium, ≥ 0.5 large.
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
    n_total  = sum(len(g) for g in groups)

    h_stat, p_kw = kruskal(*groups)
    eta_sq = float(h_stat / (n_total - 1)) if n_total > 1 else 0.0

    pairs:        List[tuple] = []
    raw_p:        List[float] = []
    r_vals:       List[float] = []

    for i, s1 in enumerate(subtypes):
        for s2 in subtypes[i + 1:]:
            res = mannwhitneyu(valid[s1], valid[s2], alternative="two-sided")
            n1, n2 = len(valid[s1]), len(valid[s2])
            r_rb = float(1 - 2 * res.statistic / (n1 * n2)) if n1 * n2 > 0 else 0.0
            pairs.append((s1, s2))
            raw_p.append(float(res.pvalue))
            r_vals.append(r_rb)

    q_vals     = _fdr_bh(raw_p)
    pair_keys  = [f"{a}_vs_{b}" for a, b in pairs]
    pairwise_p = {k: float(p) for k, p in zip(pair_keys, raw_p)}
    pairwise_q = {k: float(q) for k, q in zip(pair_keys, q_vals)}
    pairwise_r = {k: float(r) for k, r in zip(pair_keys, r_vals)}

    return {
        "metric":           metric_name,
        "n_patients":       {s: int(len(v)) for s, v in valid.items()},
        "kruskal_H":        float(h_stat),
        "kruskal_p":        float(p_kw),
        "eta_sq":           eta_sq,
        "pairwise_p_raw":   pairwise_p,
        "pairwise_q":       pairwise_q,
        "pairwise_r":       pairwise_r,
        "significant_pairs": [k for k, q in pairwise_q.items() if q < alpha],
    }


def _globally_correct_stats(results: Dict[str, Dict]) -> Dict[str, Dict]:
    """Hierarchical BH correction across all metrics in *results*.

    Implements a two-family gatekeeping procedure:

    **Family 1 — omnibus tests.**
    All Kruskal-Wallis p-values are pooled and corrected jointly with BH.
    The corrected values are stored as ``kruskal_q``.

    **Gate.**
    Pairwise Mann-Whitney tests are considered only for metrics where the
    omnibus test is significant (``kruskal_q < 0.05``).  For metrics that
    do not pass the gate, ``pairwise_q`` is set to 1.0 and
    ``significant_pairs`` is cleared — no pairwise conclusion is drawn.

    **Family 2 — pairwise tests.**
    All Mann-Whitney raw p-values from gate-surviving metrics are pooled and
    corrected jointly with BH.  The corrected values overwrite ``pairwise_q``
    and ``significant_pairs`` is recomputed.

    This approach (Westfall & Young hierarchical FDR) is statistically valid
    because the pairwise tests are logically subordinate to the omnibus test:
    it is incoherent to claim a pairwise difference exists when the omnibus
    test finds no evidence of any group difference.  Pooling only the
    surviving pairwise p-values also avoids penalising comparisons for
    metrics where no correction was warranted.

    Parameters
    ----------
    results : dict[metric_name, run_subtype_tests result]
        Output of multiple :func:`run_subtype_tests` calls keyed by label.

    Returns
    -------
    The same dict (mutated in-place) so callers can chain the call.
    """
    # --- Family 1: BH across all KW omnibus p-values ---
    kw_keys = [k for k, v in results.items() if "kruskal_p" in v]
    kw_p    = [float(results[k]["kruskal_p"]) for k in kw_keys]
    kw_q    = _fdr_bh(kw_p)
    for k, q in zip(kw_keys, kw_q):
        results[k]["kruskal_q"] = float(q)

    # --- Gate: only metrics where KW passes enter Family 2 ---
    gated = {k for k in kw_keys if results[k]["kruskal_q"] < 0.05}

    # Metrics that did not pass the gate: zero out pairwise conclusions.
    for metric_key, res in results.items():
        if metric_key not in gated:
            res["pairwise_q"]      = {k: 1.0 for k in res.get("pairwise_q", {})}
            res["significant_pairs"] = []

    # --- Family 2: BH across all pairwise raw p-values from gated metrics ---
    entries: List[tuple] = []   # (metric_key, pair_key, raw_p)
    for metric_key in gated:
        for pair_key, p in results[metric_key].get("pairwise_p_raw", {}).items():
            entries.append((metric_key, pair_key, float(p)))

    if entries:
        global_q = _fdr_bh([e[2] for e in entries])
        for (metric_key, pair_key, _), q in zip(entries, global_q):
            results[metric_key]["pairwise_q"][pair_key] = float(q)
        for metric_key in gated:
            res = results[metric_key]
            res["significant_pairs"] = [
                k for k, q in res["pairwise_q"].items() if q < 0.05
            ]

    return results


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
    _globally_correct_stats(results)
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
    auc_r_min: float = 20.0,
    auc_r_max: float = 100.0,
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
    auc_r_min, auc_r_max : float
        Radius range (px) used for the AUC-based pairwise significance test.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping Ripley plot.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # ------------------------------------------------------------------
    # Pass 1: collect per-patient mean L(r) curves + compute AUC for stats
    # ------------------------------------------------------------------
    sorted_items = sorted(results.items())
    patient_aucs: Dict[str, np.ndarray] = {}
    panel_data: Dict[str, Dict] = {}  # stores pre-computed arrays per subtype

    for subtype, subtype_data in sorted_items:
        per_patient_dict = (
            subtype_data.get("per_patient", {})
            if isinstance(subtype_data, dict)
            else {}
        )

        if per_patient_dict:
            pat_L_obs = []
            pat_aucs: List[float] = []
            radii_ref = None
            for pid, tiles in per_patient_dict.items():
                L_obs, _, _, radii_list = _tile_curves(tiles)
                if not L_obs:
                    continue
                pat_mean = np.mean(L_obs, axis=0)
                pat_L_obs.append(pat_mean)
                if radii_ref is None:
                    radii_ref = radii_list[0]
                auc = _auc_ripley_L(pat_mean, radii_ref, auc_r_min, auc_r_max)
                if np.isfinite(auc):
                    pat_aucs.append(auc)
            n_label = f"n={len(pat_L_obs)} patients"
            L_obs_all = pat_L_obs
        else:
            tile_list = _tile_list(subtype_data)
            L_obs_all, _, _, radii_list = _tile_curves(tile_list)
            radii_ref = radii_list[0] if radii_list else None
            n_label = f"n={len(L_obs_all)} tiles"
            pat_aucs = []
            if radii_ref is not None:
                for L in L_obs_all:
                    auc = _auc_ripley_L(np.asarray(L), radii_ref, auc_r_min, auc_r_max)
                    if np.isfinite(auc):
                        pat_aucs.append(auc)

        panel_data[subtype] = {
            "L_obs_all": L_obs_all,
            "radii_ref": radii_ref,
            "n_label": n_label,
        }
        if pat_aucs:
            patient_aucs[subtype] = np.array(pat_aucs)

    # ------------------------------------------------------------------
    # Pairwise significance of L(r) AUC across subtypes
    # ------------------------------------------------------------------
    auc_stats = run_subtype_tests(patient_aucs, metric_name="Ripley_L_AUC")
    pairwise_q: Dict[str, float] = auc_stats.get("pairwise_q", {})
    pairwise_r: Dict[str, float] = auc_stats.get("pairwise_r", {})
    kruskal_p: float = auc_stats.get("kruskal_p", float("nan"))
    _R_THRESHOLD = 0.1  # |rank-biserial r| minimum for annotation

    def _sig_diffs(subtype: str) -> str:
        """Compact annotation of subtypes that differ from *subtype* in AUC.

        Requires both BH q < 0.05 and |rank-biserial r| >= 0.1 so that
        differences detectable only via large n do not generate annotations.
        """
        diffs: List[str] = []
        for key, q in sorted(pairwise_q.items(), key=lambda kv: kv[1]):
            if q >= 0.05:
                continue
            if abs(pairwise_r.get(key, 0.0)) < _R_THRESHOLD:
                continue
            idx = key.find("_vs_")
            if idx < 0:
                continue
            s1, s2 = key[:idx], key[idx + 4:]
            other = s2 if s1 == subtype else (s1 if s2 == subtype else None)
            if other is None:
                continue
            star = "***" if q < 0.001 else "**" if q < 0.01 else "*"
            diffs.append(f"≠{other}{star}")
        return "  ".join(diffs)

    # ------------------------------------------------------------------
    # Pass 2: draw panels
    # ------------------------------------------------------------------
    n_subtypes = len(sorted_items)
    fig, axes = plt.subplots(1, n_subtypes, figsize=(n_subtypes * 4, 4), sharex=True, sharey=True)
    if n_subtypes == 1:
        axes = [axes]

    for ax, (subtype, _) in zip(axes, sorted_items):
        data = panel_data[subtype]
        L_obs_all = data["L_obs_all"]
        radii_ref = data["radii_ref"]
        n_label = data["n_label"]

        if not L_obs_all or radii_ref is None:
            ax.set_title(f"{subtype} (no data)")
            continue

        radii = radii_ref
        L_obs_arr = np.array(L_obs_all)
        L_obs_mean = L_obs_arr.mean(axis=0)
        n_obs = len(L_obs_arr)
        L_obs_sem = (
            L_obs_arr.std(axis=0, ddof=1) / np.sqrt(n_obs)
            if n_obs > 1 else np.zeros_like(L_obs_mean)
        )

        ax.plot(radii, L_obs_mean, 'b-', linewidth=2, label='Mean L(r)')
        ax.fill_between(
            radii,
            L_obs_mean - L_obs_sem,
            L_obs_mean + L_obs_sem,
            alpha=0.25, color='blue', label='±1 SEM across patients',
        )
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='CSR (L = 0)')

        diffs_str = _sig_diffs(subtype)
        title = f"{subtype} ({n_label})"
        if diffs_str:
            title += f"\n{diffs_str}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Radius r (px)")
        if ax == axes[0]:
            ax.set_ylabel("L(r)")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    kw_str = (
        f"Kruskal-Wallis on L(r) AUC [{auc_r_min:.0f}–{auc_r_max:.0f} px]: "
        f"p = {kruskal_p:.3g}"
        if np.isfinite(kruskal_p) else ""
    )
    fig.suptitle(
        "Ripley's L(r) by Subtype: Clustering vs. Inhibition at Multiple Scales\n"
        + kw_str,
        fontsize=10,
    )
    fig.tight_layout()
    fig.text(
        0.5, -0.01,
        "Panel titles show pairwise AUC significance (BH FDR).  "
        "* p<0.05  ** p<0.01  *** p<0.001",
        ha="center", va="top", fontsize=7, color="#555555", style="italic",
    )
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
    ax: "plt.Axes",
    subtypes: List[str],
    pairwise_q: Dict[str, float],
    y_top: float,
    y_range: float,
    pairwise_r: Optional[Dict[str, float]] = None,
    r_threshold: float = 0.1,
) -> None:
    """Annotate significant pairs with bracketed asterisks above the boxplot.

    Parameters
    ----------
    pairwise_r : dict or None
        Rank-biserial correlations from :func:`run_subtype_tests`.  When
        provided, a bracket is drawn only if |r| ≥ *r_threshold* in addition
        to q < 0.05 — this prevents purely sample-size-driven significance
        from cluttering the plot.
    r_threshold : float
        Minimum |rank-biserial r| for drawing a bracket (default 0.1,
        "small effect" by Cohen's convention).
    """
    sig: List[tuple] = []
    for key, q in pairwise_q.items():
        if q >= 0.05:
            continue
        if pairwise_r is not None and abs(pairwise_r.get(key, 0.0)) < r_threshold:
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

    # Significance bars — gate on |r| >= 0.1 to avoid n-driven false positives
    pairwise_q = test_results.get("pairwise_q", {})
    pairwise_r = test_results.get("pairwise_r", {})
    _draw_significance_bars(
        ax, subtypes, pairwise_q, y_top=y_hi, y_range=(y_hi - y_lo),
        pairwise_r=pairwise_r, r_threshold=0.1,
    )

    # Title with Kruskal-Wallis result and effect size
    kw_p   = test_results.get("kruskal_p", float("nan"))
    kw_h   = test_results.get("kruskal_H", float("nan"))
    eta_sq = test_results.get("eta_sq", float("nan"))
    ax.set_title(
        f"Ripley L(r) AUC [{r_min:.0f}–{r_max:.0f} px] per Patient by PAM50 Subtype\n"
        f"Kruskal-Wallis H={kw_h:.2f}, p={kw_p:.3g}, η²={eta_sq:.3f}  "
        f"(brackets: BH q<0.05 and |r|≥0.1)",
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
