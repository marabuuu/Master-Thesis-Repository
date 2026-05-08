"""
Statistics and Metrics for Generative Models
=============================================

Sub-modules
-----------
training_curves
    Parse training metrics from diffusion fine-tuning runs
    (TensorBoard events, Lightning stdout/stderr logs).
fid_score
    Compute FID between real and generated images.
"""

from .fid_score import compute_fid
from .training_curves import load_scalars, plot_training_stats

__all__ = [
    "load_scalars",
    "plot_training_stats",
    "compute_fid",
]
