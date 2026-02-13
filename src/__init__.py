"""
Master Thesis Repository - Source Package

Modules:
    encoding: VAE-based genomic feature encoding (CSV → latent vectors)
    finetune_diffusion: Genomic-conditioned diffusion model fine-tuning
    preprocessing: Data loading and preprocessing utilities
    statistics: Evaluation metrics (FID, training curves)
    classifier: Genomic classification helpers (train/evaluate)
    quality_assurance: Reconstruction quality metrics and visualization
"""

from . import encoding
from . import finetune_diffusion
from . import preprocessing
from . import statistics
from . import classifier
from . import quality_assurance

__all__ = [
    "encoding",
    "finetune_diffusion",
    "preprocessing",
    "statistics",
    "classifier",
    "quality_assurance",
]

