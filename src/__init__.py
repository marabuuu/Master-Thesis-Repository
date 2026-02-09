"""
Master Thesis Repository - Source Package

Modules:
    encoding: VAE-based genomic feature encoding (CSV → latent vectors)
    finetune_diffusion: Genomic-conditioned diffusion model fine-tuning
    preprocessing: Data loading and preprocessing utilities
    statistics: Evaluation metrics (FID, training curves)
    scripts: Utility scripts (classifiers, evaluation)
"""

__all__ = [
    "encoding",
    "finetune_diffusion",
    "preprocessing",
    "statistics",
    "scripts",
]
