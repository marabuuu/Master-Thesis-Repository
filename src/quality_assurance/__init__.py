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

try:
    from .evaluate_reconstruction import (
        ReconstructionEvaluator,
        evaluate_patient_tiles,
    )
except Exception:
    pass

try:
    from visualization.reconstruction_eval import (
        plot_metrics_summary,
        plot_comparison_grid,
        plot_per_patient_metrics,
        plot_metric_correlation,
        plot_single_comparison,
        save_figure,
    )
except Exception:
    pass

from .topological_frechet_distance import (
    compute_topofd,
    compute_topofd_from_folders,
    TopoFDResult,
    ClassDistribution,
    compute_class_distribution,
)
from .tfd_separability import (
    TFDSeparabilityResult,
    compute_tfd_separability,
    run_tfd_separability,
)

try:
    from .run_evaluation import (
        run_evaluation,
        load_config,
    )
except Exception:
    pass

__all__ = [
    # Metrics
    "compute_mse",
    "compute_psnr",
    "compute_ssim",
    "compute_all_metrics",
    # Evaluation
    "ReconstructionEvaluator",
    "evaluate_patient_tiles",
    "run_evaluation",
    "load_config",
    # Visualization
    "plot_metrics_summary",
    "plot_comparison_grid",
    "plot_per_patient_metrics",
    "plot_metric_correlation",
    "plot_single_comparison",
    "save_figure",
    # Topological Fréchet Distance (ref vs gen)
    "compute_topofd",
    "compute_topofd_from_folders",
    "TopoFDResult",
    # TFD Class Separability
    "ClassDistribution",
    "compute_class_distribution",
    "TFDSeparabilityResult",
    "compute_tfd_separability",
    "run_tfd_separability",
]
