"""
Quality Assurance Module for Tile Reconstruction Evaluation

This module provides tools to evaluate the quality of diffusion model
reconstructions conditioned on genomic vectors.

Metrics:
    - MSE (Mean Squared Error): Pixel-level reconstruction error
    - PSNR (Peak Signal-to-Noise Ratio): Signal quality measure in dB
    - SSIM (Structural Similarity Index): Perceptual similarity measure
    - TopoFD (Topological Fréchet Distance): Topology-aware cell layout similarity

Usage:
    from quality_assurance import evaluate_reconstruction, metrics, visualization
"""

from .metrics import (
    compute_mse,
    compute_psnr,
    compute_ssim,
    compute_all_metrics,
)
from .evaluate_reconstruction import (
    ReconstructionEvaluator,
    evaluate_patient_tiles,
)
from .visualization import (
    plot_metrics_summary,
    plot_comparison_grid,
    plot_per_patient_metrics,
    plot_metric_correlation,
    plot_single_comparison,
    save_figure,
)
from .topological_frechet_distance import (
    compute_topofd,
    compute_topofd_from_folders,
    TopoFDResult,
)

__all__ = [
    # Metrics
    "compute_mse",
    "compute_psnr",
    "compute_ssim",
    "compute_all_metrics",
    # Evaluation
    "ReconstructionEvaluator",
    "evaluate_patient_tiles",
    # Visualization
    "plot_metrics_summary",
    "plot_comparison_grid",
    "plot_per_patient_metrics",
    "plot_metric_correlation",
    "plot_single_comparison",
    "save_figure",
    # Topological Fréchet Distance
    "compute_topofd",
    "compute_topofd_from_folders",
    "TopoFDResult",
]
