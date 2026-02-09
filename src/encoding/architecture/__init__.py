"""
VAE Architecture Components

This submodule contains the neural network building blocks for the
Variational Autoencoder used to encode genomic features.
"""

from .layers import FullyConnectedLayer
from .encoder import ProbabilisticEncoder
from .decoder import ProbabilisticDecoder
from .vae import VAE
from .loss import compute_mmd, compute_kernel, MMDLoss

__all__ = [
    "FullyConnectedLayer",
    "ProbabilisticEncoder",
    "ProbabilisticDecoder",
    "VAE",
    "compute_mmd",
    "compute_kernel",
    "MMDLoss",
]
