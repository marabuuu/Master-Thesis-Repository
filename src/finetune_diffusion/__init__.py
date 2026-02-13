"""
Diffusion Model Fine-tuning Module

This module provides tools for fine-tuning diffusion models with genomic conditioning.

Components:
    - finetune_diffusion_with_genomic: Fine-tune a diffusion model with genomic vectors
    - projection_head_genomic: Train a projection head to map genomic to image space
    - sample_tiles_from_genomic: Generate tiles conditioned on genomic features

Note:
    The sampling script requires mopadi to be installed. Import individual components
    directly from their respective modules when needed:
    
        from src.finetune_diffusion.sample_tiles_from_genomic import ProjectionHead

Usage:
    python -m src.finetune_diffusion.sample_tiles_from_genomic --help
"""

# Note: We don't import sample_tiles_from_genomic at package level because it
# requires mopadi at import time. Import directly from the module instead.

