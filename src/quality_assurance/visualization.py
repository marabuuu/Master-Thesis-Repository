#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization Module for Reconstruction Quality Evaluation

This module provides plotting functions for visualizing:
- Metric distributions (histograms, box plots)
- Per-patient metric comparisons
- Side-by-side tile comparisons (original vs reconstructed)
- Correlation plots

All functions support saving to user-specified directories.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL import Image

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def setup_style():
    """Set up consistent plot styling."""
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


def save_figure(
    fig: "Figure",
    save_path: Union[str, Path],
    close: bool = True,
) -> None:
    """
    Save a matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure object
        save_path: Path to save the figure (supports .png, .pdf, .svg)
        close: If True, close the figure after saving to free memory
    """
    _check_matplotlib()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(save_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"[OK] Saved figure: {save_path}")
    
    if close:
        plt.close(fig)


def plot_metrics_summary(
    tile_results: List[Dict],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    figsize: Tuple[float, float] = (14, 10),
) -> Optional["Figure"]:
    """
    Plot summary distributions of all metrics.
    
    Creates a 2x3 grid with:
    - Row 1: Histograms of MSE, PSNR, SSIM
    - Row 2: Box plots of same metrics
    
    Args:
        tile_results: List of dictionaries with 'mse', 'psnr', 'ssim' keys
        save_path: Path to save figure (optional)
        show: If True, display the figure
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure if show=False and save_path=None
    """
    _check_matplotlib()
    setup_style()
    
    if not tile_results:
        print("[WARN] No results to plot")
        return None
    
    # Extract metrics
    mse_values = np.array([t["mse"] for t in tile_results])
    psnr_values = np.array([t["psnr"] for t in tile_results])
    ssim_values = np.array([t["ssim"] for t in tile_results])
    
    # Filter infinite PSNR values for plotting
    psnr_finite = psnr_values[np.isfinite(psnr_values)]
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("Reconstruction Quality Metrics", fontsize=14, fontweight="bold")
    
    # Color palette
    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    
    # Row 1: Histograms
    # MSE Histogram
    axes[0, 0].hist(mse_values, bins=30, color=colors[0], alpha=0.7, edgecolor="white")
    axes[0, 0].axvline(np.mean(mse_values), color="red", linestyle="--", 
                       label=f"Mean: {np.mean(mse_values):.2f}")
    axes[0, 0].axvline(np.median(mse_values), color="orange", linestyle=":", 
                       label=f"Median: {np.median(mse_values):.2f}")
    axes[0, 0].set_xlabel("MSE")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Mean Squared Error Distribution")
    axes[0, 0].legend()
    
    # PSNR Histogram
    axes[0, 1].hist(psnr_finite, bins=30, color=colors[1], alpha=0.7, edgecolor="white")
    axes[0, 1].axvline(np.mean(psnr_finite), color="red", linestyle="--", 
                       label=f"Mean: {np.mean(psnr_finite):.2f} dB")
    axes[0, 1].axvline(np.median(psnr_finite), color="orange", linestyle=":", 
                       label=f"Median: {np.median(psnr_finite):.2f} dB")
    axes[0, 1].set_xlabel("PSNR (dB)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Peak Signal-to-Noise Ratio Distribution")
    axes[0, 1].legend()
    
    # SSIM Histogram
    axes[0, 2].hist(ssim_values, bins=30, color=colors[2], alpha=0.7, edgecolor="white")
    axes[0, 2].axvline(np.mean(ssim_values), color="red", linestyle="--", 
                       label=f"Mean: {np.mean(ssim_values):.4f}")
    axes[0, 2].axvline(np.median(ssim_values), color="orange", linestyle=":", 
                       label=f"Median: {np.median(ssim_values):.4f}")
    axes[0, 2].set_xlabel("SSIM")
    axes[0, 2].set_ylabel("Count")
    axes[0, 2].set_title("Structural Similarity Index Distribution")
    axes[0, 2].legend()
    
    # Row 2: Box plots
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


def plot_per_patient_metrics(
    patient_results: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    max_patients: int = 30,
) -> Optional["Figure"]:
    """
    Plot metrics comparison across patients.
    
    Creates bar charts showing mean metrics per patient with error bars.
    
    Args:
        patient_results: Dictionary mapping patient_id to PatientResult objects
        save_path: Path to save figure (optional)
        show: If True, display the figure
        figsize: Figure size tuple (auto-calculated if None)
        max_patients: Maximum number of patients to show
        
    Returns:
        Matplotlib figure if show=False and save_path=None
    """
    _check_matplotlib()
    setup_style()
    
    if not patient_results:
        print("[WARN] No patient results to plot")
        return None
    
    # Get patient summaries
    summaries = []
    for pid, result in patient_results.items():
        summary = result.get_summary()
        if summary:
            summaries.append({
                "patient_id": pid,
                **summary,
            })
    
    if not summaries:
        return None
    
    # Sort by SSIM and limit
    summaries = sorted(summaries, key=lambda x: x.get("ssim_mean", 0), reverse=True)
    if len(summaries) > max_patients:
        summaries = summaries[:max_patients]
    
    n_patients = len(summaries)
    
    # Calculate figure size
    if figsize is None:
        figsize = (max(12, n_patients * 0.5), 10)
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle("Per-Patient Reconstruction Quality", fontsize=14, fontweight="bold")
    
    patient_ids = [s["patient_id"] for s in summaries]
    x = np.arange(n_patients)
    
    # MSE plot
    mse_means = [s.get("mse_mean", 0) for s in summaries]
    mse_stds = [s.get("mse_std", 0) for s in summaries]
    axes[0].bar(x, mse_means, yerr=mse_stds, capsize=3, color="#3498db", alpha=0.7)
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Mean Squared Error per Patient")
    axes[0].axhline(np.mean(mse_means), color="red", linestyle="--", alpha=0.7, 
                    label=f"Overall mean: {np.mean(mse_means):.2f}")
    axes[0].legend()
    
    # PSNR plot
    psnr_means = [s.get("psnr_mean", 0) for s in summaries]
    psnr_stds = [s.get("psnr_std", 0) for s in summaries]
    # Filter NaN values
    psnr_means_filtered = [p if np.isfinite(p) else 0 for p in psnr_means]
    psnr_stds_filtered = [p if np.isfinite(p) else 0 for p in psnr_stds]
    axes[1].bar(x, psnr_means_filtered, yerr=psnr_stds_filtered, capsize=3, 
                color="#2ecc71", alpha=0.7)
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Peak Signal-to-Noise Ratio per Patient")
    valid_psnr = [p for p in psnr_means if np.isfinite(p)]
    if valid_psnr:
        axes[1].axhline(np.mean(valid_psnr), color="red", linestyle="--", alpha=0.7,
                        label=f"Overall mean: {np.mean(valid_psnr):.2f} dB")
        axes[1].legend()
    
    # SSIM plot
    ssim_means = [s.get("ssim_mean", 0) for s in summaries]
    ssim_stds = [s.get("ssim_std", 0) for s in summaries]
    axes[2].bar(x, ssim_means, yerr=ssim_stds, capsize=3, color="#e74c3c", alpha=0.7)
    axes[2].set_ylabel("SSIM")
    axes[2].set_title("Structural Similarity Index per Patient")
    axes[2].set_ylim([0, 1])
    axes[2].axhline(np.mean(ssim_means), color="red", linestyle="--", alpha=0.7,
                    label=f"Overall mean: {np.mean(ssim_means):.4f}")
    axes[2].legend()
    
    # X-axis labels
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(patient_ids, rotation=45, ha="right", fontsize=8)
    axes[2].set_xlabel("Patient ID")
    
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
) -> Optional["Figure"]:
    """
    Plot a grid comparing original and reconstructed tiles side by side.
    
    Each cell shows:
    - Original tile (left)
    - Reconstructed tile (right)
    - Metrics below (if include_metrics=True)
    
    Args:
        tile_pairs: List of TilePair objects
        save_path: Path to save figure (optional)
        show: If True, display the figure
        figsize: Figure size tuple (auto-calculated if None)
        num_cols: Number of comparison columns
        include_metrics: If True, show metrics below each pair
        
    Returns:
        Matplotlib figure if show=False and save_path=None
    """
    _check_matplotlib()
    setup_style()
    
    if not tile_pairs:
        print("[WARN] No tile pairs to plot")
        return None
    
    from .metrics import compute_all_metrics
    
    n_pairs = len(tile_pairs)
    num_rows = (n_pairs + num_cols - 1) // num_cols
    
    # Calculate figure size
    if figsize is None:
        figsize = (num_cols * 5, num_rows * 3)
    
    fig = plt.figure(figsize=figsize)
    fig.suptitle("Original vs Reconstructed Tile Comparison", fontsize=14, fontweight="bold")
    
    for idx, pair in enumerate(tile_pairs):
        row = idx // num_cols
        col = idx % num_cols
        
        # Create subplot for this pair (2 images side by side)
        ax_orig = plt.subplot2grid(
            (num_rows, num_cols * 2),
            (row, col * 2),
        )
        ax_recon = plt.subplot2grid(
            (num_rows, num_cols * 2),
            (row, col * 2 + 1),
        )
        
        # Plot images
        ax_orig.imshow(pair.original)
        ax_orig.set_title(f"Original", fontsize=9)
        ax_orig.axis("off")
        
        ax_recon.imshow(pair.reconstructed)
        ax_recon.set_title(f"Reconstructed", fontsize=9)
        ax_recon.axis("off")
        
        # Add metrics annotation
        if include_metrics:
            metrics = compute_all_metrics(pair.original, pair.reconstructed)
            metric_text = (
                f"MSE: {metrics['mse']:.1f} | "
                f"PSNR: {metrics['psnr']:.1f}dB | "
                f"SSIM: {metrics['ssim']:.3f}"
            )
            # Add text below the reconstructed image
            ax_recon.text(
                0.5, -0.1, metric_text,
                transform=ax_recon.transAxes,
                ha="center", va="top",
                fontsize=7,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
    
    plt.tight_layout(rect=(0, 0.02, 1, 0.96))  # type: ignore
    
    if save_path:
        save_figure(fig, save_path, close=not show)
    
    if show:
        plt.show()
        return None
    
    return fig


def plot_metric_correlation(
    tile_results: List[Dict],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    figsize: Tuple[float, float] = (12, 4),
) -> Optional["Figure"]:
    """
    Plot correlations between different metrics.
    
    Creates scatter plots showing relationships between MSE, PSNR, and SSIM.
    
    Args:
        tile_results: List of dictionaries with 'mse', 'psnr', 'ssim' keys
        save_path: Path to save figure (optional)
        show: If True, display the figure
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure if show=False and save_path=None
    """
    _check_matplotlib()
    setup_style()
    
    if not tile_results:
        return None
    
    mse = np.array([t["mse"] for t in tile_results])
    psnr = np.array([t["psnr"] for t in tile_results])
    ssim = np.array([t["ssim"] for t in tile_results])
    
    # Filter finite values
    mask = np.isfinite(psnr)
    mse_f, psnr_f, ssim_f = mse[mask], psnr[mask], ssim[mask]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("Metric Correlations", fontsize=14, fontweight="bold")
    
    # MSE vs PSNR
    axes[0].scatter(mse_f, psnr_f, alpha=0.5, s=10)
    axes[0].set_xlabel("MSE")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("MSE vs PSNR")
    
    # MSE vs SSIM
    axes[1].scatter(mse_f, ssim_f, alpha=0.5, s=10, color="green")
    axes[1].set_xlabel("MSE")
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("MSE vs SSIM")
    
    # PSNR vs SSIM
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
    figsize: Tuple[float, float] = (10, 5),
) -> Optional["Figure"]:
    """
    Plot a single original-reconstructed comparison with metrics.
    
    Args:
        original: Original PIL image
        reconstructed: Reconstructed PIL image
        title: Optional title for the plot
        save_path: Path to save figure (optional)
        show: If True, display the figure
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure if show=False and save_path=None
    """
    _check_matplotlib()
    setup_style()
    
    from .metrics import compute_all_metrics
    
    metrics = compute_all_metrics(original, reconstructed)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    axes[1].imshow(reconstructed)
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")
    
    # Add metrics as figure text
    metric_text = (
        f"MSE: {metrics['mse']:.2f}  |  "
        f"PSNR: {metrics['psnr']:.2f} dB  |  "
        f"SSIM: {metrics['ssim']:.4f}"
    )
    fig.text(
        0.5, 0.02, metric_text,
        ha="center", va="bottom",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )
    
    plt.tight_layout(rect=(0, 0.08, 1, 0.95))  # type: ignore
    
    if save_path:
        save_figure(fig, save_path, close=not show)
    
    if show:
        plt.show()
        return None
    
    return fig
