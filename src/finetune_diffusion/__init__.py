"""
Diffusion Model Fine-tuning Module

This module provides tools for fine-tuning diffusion models with genomic conditioning.

Components:
    - finetune_diffusion_with_genomic: Fine-tune a diffusion model with genomic vectors
    - projection_head_genomic: Train a projection head to map genomic to image space
    - sample_tiles_from_genomic: Generate tiles conditioned on genomic features

Pipeline Overview:
    1. Train a projection head to map genomic features to the conditioning space.
    2. Fine-tune the diffusion model with genomic conditioning.
    3. Generate synthetic tiles using the fine-tuned model in two modes:
        - Random noise generation
        - Encode-decode with real tiles

Note:
    The sampling script requires mopadi to be installed. Import individual components
    directly from their respective modules when needed:

        from src.finetune_diffusion.sample_tiles_from_genomic import ProjectionHead

Usage:
    python -m src.finetune_diffusion.sample_tiles_from_genomic --help
"""

# Note: We don't import sample_tiles_from_genomic at package level because it
# requires mopadi at import time. Import directly from the module instead.

