"""
Statistics and metrics for generative models.

Modules:
    training_curves: Plot training loss curves from checkpoints
    fid_score: Compute FID between real and generated images
"""

from .training_curves import plot_training_curves
from .fid_score import compute_fid

__all__ = ["plot_training_curves", "compute_fid"]
