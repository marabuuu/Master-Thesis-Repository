"""
GenomicConfig — extends GenomicTrainConfig for clean genomic diffusion training.

Philosophy: Use MoPaDi's built-in dual-conditioning infrastructure (resnet_two_cond=True).
Genomic features are passed as the feature conditioning stream without any
additional wrappers or modifications to the base UNet.

This module simply extends GenomicTrainConfig and enables resnet_two_cond=True.
All actual training logic is inherited from the parent GenomicLitModel.
"""

from __future__ import annotations

from mopadi_genomic_crossattn.genomic_config import GenomicTrainConfig


class GenomicConfig(GenomicTrainConfig):
    """Alias for GenomicTrainConfig with resnet_two_cond=True baked in.
    
    This is the main configuration for clean genomic diffusion training.
    No extra cross-attention wrappers or auxiliary losses. Just pure MoPaDi
    with genomic features flowing through the existing conditioning pathway.
    """

    def __post_init__(self):
        super().__post_init__()
        # Already set in parent, but emphasize that we use the built-in mechanism
        self.net_beatgans_resnet_two_cond = True
