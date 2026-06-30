"""
Master Thesis Repository - Source Package

Modules:
    encoding: VAE-based genomic feature encoding (CSV → latent vectors)
    finetune_diffusion: Genomic-conditioned diffusion helpers
    preprocessing: Data loading and preprocessing utilities
    statistics: Evaluation metrics (FID, training curves)
    classifier: Genomic classification helpers (train/evaluate)
    quality_assurance: Reconstruction quality metrics and visualization
"""

__all__ = [
    "encoding",
    "preprocessing",
    "statistics",
    "classifier",
    "quality_assurance",
    "visualization",
]

