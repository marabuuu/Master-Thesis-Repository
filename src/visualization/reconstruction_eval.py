#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reconstruction evaluation visualizations

This module also provides a simple command-line interface for generating
plots from paired zip archives containing original and reconstructed image
tiles. Example usage:

    python -m visualization.reconstruction_eval \
        --real-zip-dir /path/to/real/zips \
        --recon-zip-dir /path/to/recon/zips \
        --out-dir /path/to/output/plots \
        --plot metrics_summary          # or comparison_grid, per_patient

Additional options include ``--max-pairs`` to limit the number of tile
pairs in the comparison grid and ``--show`` to display plots interactively.

"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
import zipfile
from io import BytesIO
import os

# metrics returned by ``quality_assurance.metrics.compute_all_metrics`` may
# contain scalar floats as well as per‑channel ``np.ndarray`` values. the
# various lists/dicts that store those results therefore need to be typed
# accordingly. ``Dict`` is invariant so we can't pretend all values are
# ``float``; instead create a dedicated alias that includes ``ndarray``.
MetricDict = Dict[str, Union[float, np.ndarray]]

try:
    import matplotlib.gridspec as gridspec
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

try:
    from .core import (
        _check_matplotlib,
        get_categorical_colors,
        get_crameri_cmap,
        save_figure,
        setup_style,
        show_or_save,
        HAS_CRAMERI,
        DIVERGING_CMAP,
        HEATMAP_CMAP,
        SEQUENTIAL_CMAP,
    )
    _HAS_CORE = True
except Exception:
    _HAS_CORE = False
    HAS_CRAMERI = False
    DIVERGING_CMAP = "RdBu_r"
    HEATMAP_CMAP = "hot"
    SEQUENTIAL_CMAP = "viridis"

    def get_crameri_cmap(name: str):  # type: ignore[misc]
        import matplotlib as _mpl
        return _mpl.colormaps.get(name, _mpl.colormaps["viridis"])
    
    def _check_matplotlib():
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization. Install it with: pip install matplotlib")
    
    def setup_style():
        _check_matplotlib()
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "seaborn-whitegrid")
        plt.rcParams.update({
            "figure.figsize": (12, 8),
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        })
    
    def save_figure(fig: "Figure", save_path: Union[str, Path], close: bool = True) -> None:
        _check_matplotlib()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"[OK] Saved figure: {save_path}")
        if close:
            plt.close(fig)
    
    def show_or_save(fig, save_path=None, show=True, close=True):
        if save_path:
            save_figure(fig, save_path, close=False)
        if show:
            plt.show()
        if close:
            plt.close(fig)


def _get_colors():
    """Return a short palette of colors to use for MSE/PSNR/SSIM.

    Uses Crameri scientific colormaps when available via ``core.py``.
    """
    if HAS_CRAMERI:
        try:
            cmap = get_crameri_cmap("batlow")
            return [cmap(0.25), cmap(0.55), cmap(0.85)]
        except Exception:
            pass
    return ["#3498db", "#2ecc71", "#e74c3c"]


def _build_subtype_colors(subtypes: List[str]) -> Dict[str, Any]:
    """Return a mapping from subtype string to a distinct matplotlib colour.

    Uses tab10 so that up to 10 subtypes get visually distinct colours.
    A stable sort is applied so that the mapping is deterministic across calls.
    """
    unique = sorted(set(s for s in subtypes if s and str(s).lower() not in ("nan", "none", "")))
    if not unique:
        return {}
    try:
        cmap = plt.cm.get_cmap("tab10", max(len(unique), 1))
    except Exception:
        cmap = plt.cm.tab10  # type: ignore[attr-defined]
    return {s: cmap(i % 10) for i, s in enumerate(unique)}


def _subtype_label(patient_id: Optional[str], subtype_map: Optional[Dict[str, str]]) -> str:
    """Return a short display string like ``'LumA'`` for a patient, or ``''``."""
    if not subtype_map or not patient_id:
        return ""
    subtype = subtype_map.get(str(patient_id), "")
    return str(subtype) if subtype and str(subtype).lower() not in ("nan", "none") else ""


def _compute_ssim_diff_map(
    original: "Image.Image",
    reconstructed: "Image.Image",
) -> np.ndarray:
    """Compute a per-pixel SSIM difference map between two images.

    Returns a 2-D float array in [0, 1] where 1 = identical.
    """
    from skimage.metrics import structural_similarity

    orig_arr = np.asarray(original.convert("RGB"), dtype=np.float64) / 255.0
    recon_arr = np.asarray(reconstructed.convert("RGB"), dtype=np.float64) / 255.0

    _, ssim_map = structural_similarity(
        orig_arr,
        recon_arr,
        channel_axis=2,
        data_range=1.0,
        full=True,
    )
    # Average across colour channels → single spatial map
    return np.mean(ssim_map, axis=2)


def plot_metrics_summary(tile_results: List[MetricDict], save_path: Optional[Union[str, Path]] = None, show: bool = False, figsize: Tuple[float, float] = (14, 10)) -> Optional["Figure"]:
    _check_matplotlib()
    setup_style()
    if not tile_results:
        print("[WARN] No results to plot")
        return None

    mse_values = np.array([t["mse"] for t in tile_results])
    psnr_values = np.array([t["psnr"] for t in tile_results])
    ssim_values = np.array([t["ssim"] for t in tile_results])
    psnr_finite = psnr_values[np.isfinite(psnr_values)]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("Reconstruction Quality Metrics", fontsize=14, fontweight="bold")

    colors = _get_colors()

    axes[0, 0].hist(mse_values, bins=30, color=colors[0], alpha=0.7, edgecolor="white")
    axes[0, 0].axvline(np.mean(mse_values), color="red", linestyle="--", label=f"Mean: {np.mean(mse_values):.2f}")
    axes[0, 0].axvline(np.median(mse_values), color="orange", linestyle=":", label=f"Median: {np.median(mse_values):.2f}")
    axes[0, 0].set_xlabel("MSE")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Mean Squared Error Distribution")
    axes[0, 0].legend()

    axes[0, 1].hist(psnr_finite, bins=30, color=colors[1], alpha=0.7, edgecolor="white")
    axes[0, 1].axvline(np.mean(psnr_finite), color="red", linestyle="--", label=f"Mean: {np.mean(psnr_finite):.2f} dB")
    axes[0, 1].axvline(np.median(psnr_finite), color="orange", linestyle=":", label=f"Median: {np.median(psnr_finite):.2f} dB")
    axes[0, 1].set_xlabel("PSNR (dB)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Peak Signal-to-Noise Ratio Distribution")
    axes[0, 1].legend()

    axes[0, 2].hist(ssim_values, bins=30, color=colors[2], alpha=0.7, edgecolor="white")
    axes[0, 2].axvline(np.mean(ssim_values), color="red", linestyle="--", label=f"Mean: {np.mean(ssim_values):.4f}")
    axes[0, 2].axvline(np.median(ssim_values), color="orange", linestyle=":", label=f"Median: {np.median(ssim_values):.4f}")
    axes[0, 2].set_xlabel("SSIM")
    axes[0, 2].set_ylabel("Count")
    axes[0, 2].set_title("Structural Similarity Index Distribution")
    axes[0, 2].legend()

    bp1 = axes[1, 0].boxplot([mse_values], patch_artist=True)
    bp1["boxes"][0].set_facecolor(colors[0])
    bp1["boxes"][0].set_alpha(0.7)
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].set_xticklabels(["MSE"])
    axes[1, 0].set_title(f"MSE Box Plot (n={len(mse_values)})")

    bp2 = axes[1, 1].boxplot([psnr_finite], patch_artist=True)
    bp2["boxes"][0].set_facecolor(colors[1])
    bp2["boxes"][0].set_alpha(0.7)
    axes[1, 1].set_ylabel("PSNR (dB)")
    axes[1, 1].set_xticklabels(["PSNR"])
    axes[1, 1].set_title(f"PSNR Box Plot (n={len(psnr_finite)})")

    bp3 = axes[1, 2].boxplot([ssim_values], patch_artist=True)
    bp3["boxes"][0].set_facecolor(colors[2])
    bp3["boxes"][0].set_alpha(0.7)
    axes[1, 2].set_ylabel("SSIM")
    axes[1, 2].set_xticklabels(["SSIM"])
    axes[1, 2].set_title(f"SSIM Box Plot (n={len(ssim_values)})")

    plt.tight_layout(rect=(0, 0, 1, 0.96))  # type: ignore

    if save_path:
        save_figure(fig, save_path, close=not show)
    if show:
        plt.show()
        return None
    return fig


def plot_per_patient_metrics(patient_results: Dict[str, Any], save_path: Optional[Union[str, Path]] = None, show: bool = False, figsize: Optional[Tuple[float, float]] = None, max_patients: int = 30, subtype_map: Optional[Dict[str, str]] = None) -> Optional["Figure"]:
    _check_matplotlib()
    setup_style()
    if not patient_results:
        print("[WARN] No patient results to plot")
        return None

    summaries = []
    for pid, result in patient_results.items():
        summary = result.get_summary()
        if summary:
            summaries.append({"patient_id": pid, **summary})

    if not summaries:
        return None

    summaries = sorted(summaries, key=lambda x: x.get("ssim_mean", 0), reverse=True)
    if len(summaries) > max_patients:
        summaries = summaries[:max_patients]

    n_patients = len(summaries)
    if figsize is None:
        figsize = (max(12, n_patients * 0.5), 10)

    # Build per-patient bar colours from subtype when available
    patient_ids = [s["patient_id"] for s in summaries]
    patient_subtypes = [_subtype_label(s["patient_id"], subtype_map) for s in summaries]
    subtype_colors = _build_subtype_colors(patient_subtypes)
    bar_colors = [subtype_colors.get(st, _get_colors()[0]) if subtype_colors else _get_colors()[0]
                  for st in patient_subtypes]

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    title = "Per-Patient Reconstruction Quality"
    if subtype_map:
        title += "  (bars coloured by Majority_Subtype_mRNA)"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    x = np.arange(n_patients)

    mse_means = [s.get("mse_mean", 0) for s in summaries]
    mse_stds = [s.get("mse_std", 0) for s in summaries]
    axes[0].bar(x, mse_means, yerr=mse_stds, capsize=3, color=bar_colors, alpha=0.85)
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Mean Squared Error per Patient")
    axes[0].axhline(np.mean(mse_means), color="red", linestyle="--", alpha=0.7, label=f"Overall mean: {np.mean(mse_means):.2f}")
    axes[0].legend()

    psnr_means = [s.get("psnr_mean", 0) for s in summaries]
    psnr_stds = [s.get("psnr_std", 0) for s in summaries]
    psnr_means_filtered = [p if np.isfinite(p) else 0 for p in psnr_means]
    psnr_stds_filtered = [p if np.isfinite(p) else 0 for p in psnr_stds]
    axes[1].bar(x, psnr_means_filtered, yerr=psnr_stds_filtered, capsize=3, color=bar_colors, alpha=0.85)
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Peak Signal-to-Noise Ratio per Patient")
    valid_psnr = [p for p in psnr_means if np.isfinite(p)]
    if valid_psnr:
        axes[1].axhline(np.mean(valid_psnr), color="red", linestyle="--", alpha=0.7, label=f"Overall mean: {np.mean(valid_psnr):.2f} dB")
        axes[1].legend()

    ssim_means = [s.get("ssim_mean", 0) for s in summaries]
    ssim_stds = [s.get("ssim_std", 0) for s in summaries]
    axes[2].bar(x, ssim_means, yerr=ssim_stds, capsize=3, color=bar_colors, alpha=0.85)
    axes[2].set_ylabel("SSIM")
    axes[2].set_title("Structural Similarity Index per Patient")
    axes[2].set_ylim([0, 1])
    axes[2].axhline(np.mean(ssim_means), color="red", linestyle="--", alpha=0.7, label=f"Overall mean: {np.mean(ssim_means):.4f}")
    axes[2].legend()

    # X-axis labels: append subtype in parentheses when known
    xticklabels = [
        f"{pid}\n({st})" if st else pid
        for pid, st in zip(patient_ids, patient_subtypes)
    ]
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=8)
    axes[2].set_xlabel("Patient ID")

    # Subtype legend (colour patches)
    if subtype_colors:
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=c, label=st) for st, c in subtype_colors.items()]
        axes[0].legend(handles=patches + axes[0].get_legend_handles_labels()[0][:1],
                       fontsize=8, loc="upper right", title="Subtype")

    plt.tight_layout(rect=(0, 0, 1, 0.96))  # type: ignore
    if save_path:
        save_figure(fig, save_path, close=not show)
    if show:
        plt.show()
        return None
    return fig


def plot_comparison_grid(
    tile_pairs: List[Any],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    num_cols: int = 4,
    include_metrics: bool = True,
    include_diff: bool = True,
    subtype_map: Optional[Dict[str, str]] = None,
    mode: str = "image_guided",
) -> Optional["Figure"]:
    """Grid of Original | Reconstructed | SSIM-diff triptychs.

    Parameters
    ----------
    include_diff : bool
        Add a third column per pair showing the per-pixel SSIM map with a
        Crameri diverging colourmap (default *True*).
    subtype_map : dict, optional
        Mapping from patient_id → ``Majority_Subtype_mRNA`` string. When
        provided the subtype is annotated on each tile group.
    mode : str
        ``'image_guided'`` (default) or ``'random_noise'``. In
        ``'image_guided'`` mode cross-patient conditioning is detected
        automatically when a pair exposes a ``conditioning_patient_id``
        attribute that differs from ``patient_id``.
    """
    _check_matplotlib()
    setup_style()
    if not tile_pairs:
        print("[WARN] No tile pairs to plot")
        return None

    from quality_assurance.metrics import compute_all_metrics

    n_pairs = len(tile_pairs)
    panels_per_pair = 3 if include_diff else 2
    num_rows = (n_pairs + num_cols - 1) // num_cols
    if figsize is None:
        figsize = (num_cols * panels_per_pair * 2.5, num_rows * 3.5)

    fig = plt.figure(figsize=figsize)
    mode_label = "Image-Guided" if mode == "image_guided" else mode.replace("_", " ").title()
    fig.suptitle(f"Original vs Reconstructed Tile Comparison  [{mode_label}]", fontsize=14, fontweight="bold")

    diff_cmap = get_crameri_cmap(DIVERGING_CMAP) if include_diff else None

    for idx, pair in enumerate(tile_pairs):
        row = idx // num_cols
        col = idx % num_cols
        base_col = col * panels_per_pair

        pid = getattr(pair, "patient_id", None)
        cond_pid = getattr(pair, "conditioning_patient_id", None)
        subtype = _subtype_label(pid, subtype_map)
        is_cross = cond_pid is not None and cond_pid != pid

        ax_orig = plt.subplot2grid((num_rows, num_cols * panels_per_pair), (row, base_col))
        ax_recon = plt.subplot2grid((num_rows, num_cols * panels_per_pair), (row, base_col + 1))

        # Original panel
        orig_title = "Original"
        if pid:
            orig_title += f"\n{pid}"
        if subtype:
            orig_title += f"  [{subtype}]"
        ax_orig.imshow(pair.original)
        ax_orig.set_title(orig_title, fontsize=7, pad=2)
        ax_orig.axis("off")

        # Reconstructed panel
        if is_cross:
            cond_subtype = _subtype_label(cond_pid, subtype_map)
            recon_title = f"Recon (cond: {cond_pid}"
            if cond_subtype:
                recon_title += f"  [{cond_subtype}]"
            recon_title += ")"
        else:
            recon_title = "Reconstructed"
        ax_recon.imshow(pair.reconstructed)
        ax_recon.set_title(recon_title, fontsize=7, pad=2)
        ax_recon.axis("off")

        if include_diff:
            ax_diff = plt.subplot2grid((num_rows, num_cols * panels_per_pair), (row, base_col + 2))
            try:
                ssim_map = _compute_ssim_diff_map(pair.original, pair.reconstructed)
                ax_diff.imshow(ssim_map, cmap=diff_cmap, vmin=0, vmax=1)
            except Exception:
                orig_arr = np.asarray(pair.original.convert("RGB") if hasattr(pair.original, "convert") else pair.original, dtype=np.float32)
                recon_arr = np.asarray(pair.reconstructed.convert("RGB") if hasattr(pair.reconstructed, "convert") else pair.reconstructed, dtype=np.float32)
                diff = np.mean(np.abs(orig_arr - recon_arr), axis=2) / 255.0
                ax_diff.imshow(diff, cmap=get_crameri_cmap(HEATMAP_CMAP), vmin=0, vmax=1)
            ax_diff.set_title("SSIM Map", fontsize=7, pad=2)
            ax_diff.axis("off")

        if include_metrics:
            metrics = compute_all_metrics(pair.original, pair.reconstructed)
            metric_text = (f"MSE: {metrics['mse']:.1f} | " f"PSNR: {metrics['psnr']:.1f}dB | " f"SSIM: {metrics['ssim']:.3f}")
            target_ax = ax_diff if include_diff else ax_recon
            target_ax.text(0.5, -0.08, metric_text, transform=target_ax.transAxes, ha="center", va="top", fontsize=6, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout(rect=(0, 0.02, 1, 0.96))  # type: ignore
    if save_path:
        save_figure(fig, save_path, close=not show)
    if show:
        plt.show()
        return None
    return fig


def plot_metric_correlation(tile_results: List[MetricDict], save_path: Optional[Union[str, Path]] = None, show: bool = False, figsize: Tuple[float, float] = (12, 4)) -> Optional["Figure"]:
    _check_matplotlib()
    setup_style()
    if not tile_results:
        return None
    mse = np.array([t["mse"] for t in tile_results])
    psnr = np.array([t["psnr"] for t in tile_results])
    ssim = np.array([t["ssim"] for t in tile_results])
    mask = np.isfinite(psnr)
    mse_f, psnr_f, ssim_f = mse[mask], psnr[mask], ssim[mask]
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("Metric Correlations", fontsize=14, fontweight="bold")
    axes[0].scatter(mse_f, psnr_f, alpha=0.5, s=10)
    axes[0].set_xlabel("MSE")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("MSE vs PSNR")
    axes[1].scatter(mse_f, ssim_f, alpha=0.5, s=10, color="green")
    axes[1].set_xlabel("MSE")
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("MSE vs SSIM")
    axes[2].scatter(psnr_f, ssim_f, alpha=0.5, s=10, color="red")
    axes[2].set_xlabel("PSNR (dB)")
    axes[2].set_ylabel("SSIM")
    axes[2].set_title("PSNR vs SSIM")
    plt.tight_layout(rect=(0, 0, 1, 0.96))  # type: ignore
    if save_path:
        save_figure(fig, save_path, close=not show)
    if show:
        plt.show()
        return None
    return fig


def plot_single_comparison(
    original: "Image.Image",
    reconstructed: "Image.Image",
    title: str = "",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    figsize: Tuple[float, float] = (15, 5),
    include_diff: bool = True,
    subtype: Optional[str] = None,
    conditioning_subtype: Optional[str] = None,
    patient_id: Optional[str] = None,
    conditioning_patient_id: Optional[str] = None,
    mode: str = "image_guided",
) -> Optional["Figure"]:
    """Plot Original | Reconstructed | SSIM diff heatmap for one tile pair.

    Parameters
    ----------
    include_diff : bool
        If *True* (default) a third panel shows the per-pixel SSIM map
        rendered with the Crameri *vik* diverging colourmap.
    subtype : str, optional
        ``Majority_Subtype_mRNA`` of the tile's patient.
    conditioning_subtype : str, optional
        ``Majority_Subtype_mRNA`` of the conditioning patient (cross-patient
        ``image_guided`` mode only).
    patient_id : str, optional
        ID of the patient whose tile is shown.
    conditioning_patient_id : str, optional
        ID of the patient whose genomic conditioning was used.
    mode : str
        ``'image_guided'`` (default) or ``'random_noise'``.
    """
    _check_matplotlib()
    setup_style()
    from quality_assurance.metrics import compute_all_metrics
    metrics = compute_all_metrics(original, reconstructed)

    is_cross = (
        conditioning_patient_id is not None
        and patient_id is not None
        and conditioning_patient_id != patient_id
    )

    ncols = 3 if include_diff else 2
    fig, axes = plt.subplots(1, ncols, figsize=figsize)

    # Build figure title
    mode_label = "Image-Guided" if mode == "image_guided" else mode.replace("_", " ").title()
    auto_title = title or f"Tile Comparison  [{mode_label}]"
    if patient_id:
        auto_title += f"  —  {patient_id}"
    if subtype:
        auto_title += f"  [{subtype}]"
    fig.suptitle(auto_title, fontsize=12, fontweight="bold")

    # Original panel label
    orig_label = "Original"
    if subtype:
        orig_label += f"\nSubtype: {subtype}"
    axes[0].imshow(original)
    axes[0].set_title(orig_label, fontsize=10)
    axes[0].axis("off")

    # Reconstructed panel label
    if is_cross:
        recon_label = f"Reconstructed\n(conditioned on {conditioning_patient_id}"
        if conditioning_subtype:
            recon_label += f"  [{conditioning_subtype}]"
        recon_label += ")"
    else:
        recon_label = "Reconstructed"
        if subtype:
            recon_label += f"\n(conditioned on same patient)"
    axes[1].imshow(reconstructed)
    axes[1].set_title(recon_label, fontsize=10)
    axes[1].axis("off")

    if include_diff:
        try:
            ssim_map = _compute_ssim_diff_map(original, reconstructed)
            diff_cmap = get_crameri_cmap(DIVERGING_CMAP)
            im = axes[2].imshow(ssim_map, cmap=diff_cmap, vmin=0, vmax=1)
            axes[2].set_title("SSIM Map")
            axes[2].axis("off")
            fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        except Exception:
            # Fallback: plain absolute-difference image
            orig_arr = np.asarray(original.convert("RGB"), dtype=np.float32)
            recon_arr = np.asarray(reconstructed.convert("RGB"), dtype=np.float32)
            diff = np.mean(np.abs(orig_arr - recon_arr), axis=2) / 255.0
            diff_cmap = get_crameri_cmap(HEATMAP_CMAP)
            im = axes[2].imshow(diff, cmap=diff_cmap, vmin=0, vmax=1)
            axes[2].set_title("Absolute Diff")
            axes[2].axis("off")
            fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    metric_text = (f"MSE: {metrics['mse']:.2f}  |  " f"PSNR: {metrics['psnr']:.2f} dB  |  " f"SSIM: {metrics['ssim']:.4f}")
    fig.text(0.5, 0.02, metric_text, ha="center", va="bottom", fontsize=11, bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8))
    plt.tight_layout(rect=(0, 0.08, 1, 0.95))  # type: ignore
    if save_path:
        save_figure(fig, save_path, close=not show)
    if show:
        plt.show()
        return None
    return fig


def plot_random_noise_grid(
    tiles: List[Any],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    num_cols: int = 4,
    subtype_map: Optional[Dict[str, str]] = None,
) -> Optional["Figure"]:
    """Grid of tiles generated from *random noise* via genomic conditioning.

    Because there is no ground-truth original to compare against, each cell
    shows only the generated tile together with the patient ID and histological
    subtype (``Majority_Subtype_mRNA``) that drove the conditioning.

    Parameters
    ----------
    tiles : list
        Objects exposing ``.reconstructed`` (``PIL.Image``) and
        ``.patient_id`` (``str``).  Any attribute absent on an element is
        replaced with a safe default.
    subtype_map : dict, optional
        Mapping from patient_id → ``Majority_Subtype_mRNA`` string.
    """
    _check_matplotlib()
    setup_style()
    if not tiles:
        print("[WARN] No tiles to plot")
        return None

    n = len(tiles)
    num_rows = (n + num_cols - 1) // num_cols
    if figsize is None:
        figsize = (num_cols * 2.8, num_rows * 3.2)

    fig, axes_grid = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)
    fig.suptitle("Random-Noise Reconstructions  (conditioned on patient genomics)", fontsize=13, fontweight="bold")

    # Collect subtypes for colouring tile borders
    subtypes = [_subtype_label(getattr(t, "patient_id", None), subtype_map) for t in tiles]
    subtype_colors = _build_subtype_colors(subtypes)

    for idx, tile in enumerate(tiles):
        row, col = divmod(idx, num_cols)
        ax = axes_grid[row][col]

        img = getattr(tile, "reconstructed", None) or getattr(tile, "image", None)
        pid = getattr(tile, "patient_id", None)
        subtype = _subtype_label(pid, subtype_map)

        if img is not None:
            ax.imshow(img)
        else:
            ax.set_facecolor("#cccccc")
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)

        cell_title = str(pid) if pid else f"Tile {idx}"
        if subtype:
            cell_title += f"\n[{subtype}]"
        ax.set_title(cell_title, fontsize=8, pad=3)
        ax.axis("off")

        # Draw a coloured border matching the subtype
        if subtype and subtype in subtype_colors:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(3)
                spine.set_edgecolor(subtype_colors[subtype])

    # Hide unused axes
    for idx in range(n, num_rows * num_cols):
        row, col = divmod(idx, num_cols)
        axes_grid[row][col].axis("off")

    # Subtype legend
    if subtype_colors:
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=c, label=st) for st, c in subtype_colors.items()]
        fig.legend(handles=patches, title="Majority_Subtype_mRNA", loc="lower center",
                   ncol=min(len(patches), 6), fontsize=9, bbox_to_anchor=(0.5, 0.0))
        plt.tight_layout(rect=(0, 0.06, 1, 0.96))
    else:
        plt.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path:
        save_figure(fig, save_path, close=not show)
    if show:
        plt.show()
        return None
    return fig


def plot_cross_patient_comparison(
    tile_pairs: List[Any],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    num_cols: int = 3,
    include_metrics: bool = True,
    subtype_map: Optional[Dict[str, str]] = None,
) -> Optional["Figure"]:
    """Specialised grid for *cross-patient* image-guided reconstructions.

    Each row shows a single tile triplet:

    ``Original (patient A)``  |  ``Reconstructed (conditioned on patient B)``  |  ``SSIM diff map``

    Both the tile patient and the conditioning patient are clearly labelled
    together with their ``Majority_Subtype_mRNA`` subtype, making it easy to
    spot visually how genomic conditioning from a *different* patient alters
    the reconstruction compared to the original tile.

    Parameters
    ----------
    tile_pairs : list
        Objects exposing ``.original``, ``.reconstructed``, ``.patient_id``
        and ``.conditioning_patient_id``.
    subtype_map : dict, optional
        Mapping from patient_id → ``Majority_Subtype_mRNA`` string.
    num_cols : int
        Number of triplets per row (default 3).
    """
    _check_matplotlib()
    setup_style()
    if not tile_pairs:
        print("[WARN] No tile pairs to plot")
        return None

    from quality_assurance.metrics import compute_all_metrics

    n_pairs = len(tile_pairs)
    panels_per_pair = 3  # original | reconstructed | diff
    num_rows = (n_pairs + num_cols - 1) // num_cols
    if figsize is None:
        figsize = (num_cols * panels_per_pair * 2.8, num_rows * 3.8)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        "Cross-Patient Image-Guided Reconstruction\n"
        "(Original tile from patient A  ·  Reconstruction conditioned on patient B)",
        fontsize=13, fontweight="bold",
    )

    diff_cmap = get_crameri_cmap(DIVERGING_CMAP)
    total_cols = num_cols * panels_per_pair

    for idx, pair in enumerate(tile_pairs):
        row = idx // num_cols
        col = idx % num_cols
        base_col = col * panels_per_pair

        pid = getattr(pair, "patient_id", None)
        cond_pid = getattr(pair, "conditioning_patient_id", None)
        subtype_a = _subtype_label(pid, subtype_map)
        subtype_b = _subtype_label(cond_pid, subtype_map)

        ax_orig = plt.subplot2grid((num_rows, total_cols), (row, base_col))
        ax_recon = plt.subplot2grid((num_rows, total_cols), (row, base_col + 1))
        ax_diff = plt.subplot2grid((num_rows, total_cols), (row, base_col + 2))

        # -- Original --
        orig_title = "Original tile\n"
        orig_title += f"{pid or 'Patient A'}"
        if subtype_a:
            orig_title += f"  [{subtype_a}]"
        ax_orig.imshow(pair.original)
        ax_orig.set_title(orig_title, fontsize=8, pad=3, color="#1a237e")
        ax_orig.axis("off")

        # -- Reconstructed (cross-patient) --
        recon_title = "Reconstructed\n(cond: "
        recon_title += f"{cond_pid or 'Patient B'}"
        if subtype_b:
            recon_title += f"  [{subtype_b}]"
        recon_title += ")"
        ax_recon.imshow(pair.reconstructed)
        ax_recon.set_title(recon_title, fontsize=8, pad=3, color="#880e4f")
        ax_recon.axis("off")

        # -- SSIM diff --
        try:
            ssim_map = _compute_ssim_diff_map(pair.original, pair.reconstructed)
            im = ax_diff.imshow(ssim_map, cmap=diff_cmap, vmin=0, vmax=1)
        except Exception:
            orig_arr = np.asarray(
                pair.original.convert("RGB") if hasattr(pair.original, "convert") else pair.original,
                dtype=np.float32,
            )
            recon_arr = np.asarray(
                pair.reconstructed.convert("RGB") if hasattr(pair.reconstructed, "convert") else pair.reconstructed,
                dtype=np.float32,
            )
            diff = np.mean(np.abs(orig_arr - recon_arr), axis=2) / 255.0
            im = ax_diff.imshow(diff, cmap=get_crameri_cmap(HEATMAP_CMAP), vmin=0, vmax=1)
        ax_diff.set_title("SSIM Map\n(1 = identical)", fontsize=8, pad=3)
        ax_diff.axis("off")
        fig.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04, format="%.1f")

        if include_metrics:
            try:
                metrics = compute_all_metrics(pair.original, pair.reconstructed)
                metric_text = (
                    f"MSE: {metrics['mse']:.1f}  |  "
                    f"PSNR: {metrics['psnr']:.1f} dB  |  "
                    f"SSIM: {metrics['ssim']:.3f}"
                )
                ax_diff.text(
                    0.5, -0.08, metric_text,
                    transform=ax_diff.transAxes,
                    ha="center", va="top", fontsize=6.5,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
                )
            except Exception:
                pass

    plt.tight_layout(rect=(0, 0.0, 1, 0.93))

    if save_path:
        save_figure(fig, save_path, close=not show)
    if show:
        plt.show()
        return None
    return fig


def _canonical_patient_id(name: str) -> str:
    """Extract TCGA-XX-XXXX patient ID, handling both long SVS and short names."""
    stem = Path(name).stem.upper()
    import re as _re
    m = _re.match(r"(TCGA-[A-Z0-9]+-[A-Z0-9]+)", stem)
    if m:
        return m.group(1)
    for sep in ("_", "."):
        stem = stem.replace(sep, "-")
    while "--" in stem:
        stem = stem.replace("--", "-")
    parts = stem.split("-")
    if len(parts) >= 3 and parts[0].startswith("TCGA"):
        return "-".join(parts[:3])
    return stem


_COORD_RE_VIZ = __import__("re").compile(r"\(([\d.]+)[,_\s]+([\d.]+)\)")


def _extract_coords_viz(name: str) -> Optional[Tuple[float, float]]:
    """Extract ``(x, y)`` tile coordinates from a filename."""
    m = _COORD_RE_VIZ.search(name)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return None


def _load_image_from_zip(zpath: Path, member: str) -> Image.Image:
    with zipfile.ZipFile(zpath, "r") as zf:
        with zf.open(member) as fh:
            return Image.open(BytesIO(fh.read())).convert("RGB")


class PatientResult:
    def __init__(self, pid: str, metrics_list: List[MetricDict]):
        self.pid = pid
        self.metrics_list = metrics_list

    def get_summary(self) -> Dict[str, float]:
        if not self.metrics_list:
            return {}
        import numpy as _np
        mse = _np.array([m['mse'] for m in self.metrics_list])
        psnr = _np.array([m['psnr'] for m in self.metrics_list])
        ssim = _np.array([m['ssim'] for m in self.metrics_list])
        return {
            'mse_mean': float(mse.mean()), 'mse_std': float(mse.std()),
            'psnr_mean': float(_np.nanmean(psnr)), 'psnr_std': float(_np.nanstd(psnr)),
            'ssim_mean': float(_np.nanmean(ssim)), 'ssim_std': float(_np.nanstd(ssim)),
        }


def _load_subtype_map(csv_path: Optional[str], subtype_col: str = "Majority_Subtype_mRNA", patient_col: str = "Patient_ID") -> Dict[str, str]:
    """Load a patient_id → subtype mapping from a CSV file.

    Falls back gracefully if the file or columns are missing.
    """
    if not csv_path:
        return {}
    try:
        import csv as _csv
        subtype_map: Dict[str, str] = {}
        with open(csv_path, newline="") as fh:
            reader = _csv.DictReader(fh)
            for row in reader:
                pid = row.get(patient_col, "").strip()
                subtype = row.get(subtype_col, "").strip()
                if pid:
                    subtype_map[_canonical_patient_id(pid)] = subtype
        print(f"[OK] Loaded subtypes for {len(subtype_map)} patients from {csv_path}")
        return subtype_map
    except Exception as exc:
        print(f"[WARN] Could not load subtype map from {csv_path}: {exc}")
        return {}


def cli_main():
    parser = argparse.ArgumentParser(description="Create reconstruction plots from zip archives")
    parser.add_argument("--real-zip-dir", help="Directory with per-patient input tile zips (required for image_guided mode)")
    parser.add_argument("--recon-zip-dir", required=True, help="Directory with per-patient reconstructed tile zips")
    parser.add_argument("--out-dir", required=True, help="Directory to save generated plots")
    parser.add_argument(
        "--plot",
        choices=["metrics_summary", "comparison_grid", "per_patient", "random_noise_grid", "cross_patient", "all"],
        default="metrics_summary",
        help="Which plot to generate (use 'all' to save every applicable plot)",
    )
    parser.add_argument("--max-pairs", type=int, default=64, help="Max tile pairs to use for comparison/random-noise grid")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument(
        "--mode",
        choices=["image_guided", "random_noise"],
        default="image_guided",
        help="Reconstruction mode used (affects which plots are generated; default: image_guided)",
    )
    parser.add_argument("--metadata-csv", default=None, help="Path to patient metadata CSV containing subtype column")
    parser.add_argument("--subtype-col", default="Majority_Subtype_mRNA", help="Column name in metadata CSV for subtype (default: Majority_Subtype_mRNA)")
    parser.add_argument("--patient-col", default="Patient_ID", help="Column name in metadata CSV for patient ID (default: Patient_ID)")
    args = parser.parse_args()

    recon_dir = Path(args.recon_zip_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subtype_map = _load_subtype_map(args.metadata_csv, args.subtype_col, args.patient_col)

    # ------------------------------------------------------------------ #
    #  random_noise mode: no originals needed, just show generated tiles  #
    # ------------------------------------------------------------------ #
    if args.mode == "random_noise" or args.plot == "random_noise_grid":
        recon_zips = sorted(recon_dir.glob("*.zip"))
        if not recon_zips:
            raise RuntimeError(f"No zip files found in {recon_dir}")

        generated_tiles = []
        for zpath in recon_zips:
            pid = _canonical_patient_id(zpath.name)
            with zipfile.ZipFile(zpath, "r") as zf:
                names = sorted(n for n in zf.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg")))
                for name in names[: args.max_pairs]:
                    img = Image.open(BytesIO(zf.read(name))).convert("RGB")
                    generated_tiles.append(type("GT", (), {"patient_id": pid, "reconstructed": img})())
            if len(generated_tiles) >= args.max_pairs:
                break

        fig = plot_random_noise_grid(
            generated_tiles[: args.max_pairs],
            save_path=out_dir / "random_noise_grid.png",
            show=args.show,
            subtype_map=subtype_map or None,
        )
        if fig is not None or args.show:
            print(f"[OK] Saved random_noise_grid → {out_dir}")
        return

    # ------------------------------------------------------------------ #
    #  image_guided mode: pair originals with reconstructions             #
    # ------------------------------------------------------------------ #
    if not args.real_zip_dir:
        raise ValueError("--real-zip-dir is required for image_guided mode")

    real_dir = Path(args.real_zip_dir)
    real_zips = {_canonical_patient_id(p.name): p for p in real_dir.glob("*.zip")}
    recon_zips_map = {_canonical_patient_id(p.name): p for p in recon_dir.glob("*.zip")}

    # Detect cross-patient zips: filenames like {pid}__cond_{cond_pid}.zip
    import re as _re
    _CROSS_RE = _re.compile(r"^(.+?)__cond_(.+?)\.zip$", _re.IGNORECASE)
    cross_zips: List[Tuple[str, str, Path]] = []  # (tile_pid, cond_pid, zip_path)
    for zpath in recon_dir.glob("*.zip"):
        m = _CROSS_RE.match(zpath.name)
        if m:
            cross_zips.append((_canonical_patient_id(m.group(1)), _canonical_patient_id(m.group(2)), zpath))

    common = sorted(set(real_zips) & set(recon_zips_map))
    if not common and not cross_zips:
        raise RuntimeError(f"No matching zip files between {real_dir} and {recon_dir}")

    from quality_assurance.metrics import compute_all_metrics

    tile_pairs = []
    tile_results: List[MetricDict] = []
    patient_metrics: Dict[str, List[MetricDict]] = {}

    def _collect_pairs(pid: str, rzip: Path, gzip: Path, cond_pid: Optional[str] = None) -> None:
        with zipfile.ZipFile(rzip, "r") as rz, zipfile.ZipFile(gzip, "r") as gz:
            real_names = [n for n in rz.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg"))]
            recon_names = [n for n in gz.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg"))]

            real_basenames = {Path(n).name: n for n in real_names}
            recon_basenames = {Path(n).name: n for n in recon_names}
            common_basenames = set(real_basenames) & set(recon_basenames)

            paired: List[Tuple[str, str]] = []
            if common_basenames:
                for b in sorted(common_basenames):
                    paired.append((real_basenames[b], recon_basenames[b]))
            else:
                orig_by_coord: Dict[Tuple[float, float], str] = {}
                for n in real_names:
                    c = _extract_coords_viz(Path(n).name)
                    if c is not None:
                        orig_by_coord[c] = n
                for n in sorted(recon_names):
                    c = _extract_coords_viz(Path(n).name)
                    if c is not None and c in orig_by_coord:
                        paired.append((orig_by_coord[c], n))
                if not paired and real_names and recon_names:
                    print(f"[WARN] No basename/coord match for patient {pid}; pairing by file order")
                    for rn, gn in zip(sorted(real_names), sorted(recon_names)):
                        paired.append((rn, gn))

            for real_member, recon_member in paired[: args.max_pairs]:
                orig = Image.open(BytesIO(rz.read(real_member))).convert("RGB")
                recon = Image.open(BytesIO(gz.read(recon_member))).convert("RGB")
                tp = type("TP", (), {
                    "original": orig,
                    "reconstructed": recon,
                    "patient_id": pid,
                    "conditioning_patient_id": cond_pid or pid,
                })()
                tile_pairs.append(tp)
                metrics = compute_all_metrics(orig, recon)
                tile_results.append(metrics)
                patient_metrics.setdefault(pid, []).append(metrics)

    for pid in common:
        _collect_pairs(pid, real_zips[pid], recon_zips_map[pid])

    # Also load cross-patient pairs (when present)
    cross_tile_pairs = []
    for tile_pid, cond_pid, gzip in cross_zips:
        if tile_pid in real_zips:
            _collect_pairs(tile_pid, real_zips[tile_pid], gzip, cond_pid=cond_pid)
            cross_tile_pairs.extend(
                p for p in tile_pairs
                if getattr(p, "conditioning_patient_id", None) == cond_pid
                and getattr(p, "patient_id", None) == tile_pid
            )

    # Generate requested plot(s)
    saved_any = False
    is_cross_run = bool(cross_zips) and not common
    effective_mode = "image_guided"

    if args.plot == "metrics_summary" or args.plot == "all":
        if tile_results:
            fig = plot_metrics_summary(tile_results, save_path=out_dir / "metrics_summary.png", show=args.show)
            saved_any = saved_any or (fig is not None)
        else:
            print("[WARN] metrics_summary skipped (no metrics available)")

    if args.plot == "comparison_grid" or args.plot == "all":
        pairs_to_plot = cross_tile_pairs if is_cross_run else tile_pairs
        if pairs_to_plot:
            fig = plot_comparison_grid(
                pairs_to_plot,
                save_path=out_dir / "comparison_grid.png",
                show=args.show,
                subtype_map=subtype_map or None,
                mode=effective_mode,
            )
            saved_any = saved_any or (fig is not None)

    if args.plot == "cross_patient" or (args.plot == "all" and cross_tile_pairs):
        if cross_tile_pairs:
            fig = plot_cross_patient_comparison(
                cross_tile_pairs,
                save_path=out_dir / "cross_patient_comparison.png",
                show=args.show,
                subtype_map=subtype_map or None,
            )
            saved_any = saved_any or (fig is not None)
        else:
            print("[WARN] cross_patient plot skipped (no cross-patient zip files detected)")

    if args.plot == "per_patient" or args.plot == "all":
        if patient_metrics:
            patient_results = {pid: PatientResult(pid, metrics) for pid, metrics in patient_metrics.items()}
            fig = plot_per_patient_metrics(
                patient_results,
                save_path=out_dir / "per_patient.png",
                show=args.show,
                subtype_map=subtype_map or None,
            )
            saved_any = saved_any or (fig is not None)

    if args.plot == "all" and tile_results:
        fig = plot_metric_correlation(tile_results, save_path=out_dir / "metric_correlation.png", show=args.show)
        saved_any = saved_any or (fig is not None)

    if saved_any and not args.show:
        print(f"Saved plot(s) to {out_dir}")


if __name__ == "__main__":
    # Allow running this module directly to create plots from zip dirs
    cli_main()
