"""
Genomic Feature Encoding Module

This module provides a VAE-based approach to encode high-dimensional genomic data
(e.g., gene expression from CSV files) into compact latent feature vectors suitable
for downstream tasks like conditional image generation.

Main components:
    - train: Training script for the VAE encoder
    - config: Configuration utilities
    - architecture: VAE model components (encoder, decoder, loss)

Usage:
    python -m src.encoding.train --csv /path/to/genomic.csv --out-dir ./output
"""

from .architecture import ProbabilisticEncoder, ProbabilisticDecoder, VAE
from .architecture import compute_mmd, MMDLoss, FullyConnectedLayer

__all__ = [
    "ProbabilisticEncoder",
    "ProbabilisticDecoder", 
    "VAE",
    "compute_mmd",
    "MMDLoss",
    "FullyConnectedLayer",
]
