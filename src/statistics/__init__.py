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
from .training_curves import (
    parse_experiment_dir,
    parse_lightning_log,
    parse_stderr_log,
    parse_tensorboard_events,
    print_training_summary,
)

__all__ = [
    # training-curve parsing
    "parse_experiment_dir",
    "parse_tensorboard_events",
    "parse_lightning_log",
    "parse_stderr_log",
    "print_training_summary",
    # FID
    "compute_fid",
]
