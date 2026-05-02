"""
GenomicCrossAttnConfig — extends GenomicTrainConfig with cross-attention params.

All dataset/diffusion/optimizer fields are inherited unchanged.
New fields control the CrossAttentionUNetWrapper and the genomic-guided loss.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict

from mopadi_genomic_crossattn.genomic_config import GenomicTrainConfig


@dataclass
class GenomicCrossAttnConfig(GenomicTrainConfig):
    """GenomicTrainConfig extended with cross-attention and genomic-guided loss.

    Cross-attention injects genomic features as a spatial residual on the UNet
    input (image patches attend to the genomic conditioning vector as a single
    K/V token).  The genomic-guided loss adds a denoising objective at high
    timesteps (t ∈ [0.8T, T)) where x_t ≈ N(0,I) and the model must rely
    entirely on the genomic conditioning to predict the noise direction.

    Extra fields
    ------------
    cross_attn_heads:
        Number of attention heads in the cross-attention module.
    cross_attn_dim_per_head:
        Dimension per head; embed_dim = heads × dim_per_head.
    cross_attn_patch_size:
        Non-overlapping patch side length.  Must divide img_size evenly.
        Smaller → more patches, finer spatial resolution, more memory.
    cross_attn_lr:
        Learning rate for the cross-attention wrapper parameters.
        Should be >= unet_lr since wrapper layers are zero-initialized.
    unet_lr:
        Learning rate for the base UNet parameters.  Matches conf.lr when
        training from scratch; lower it only when warm-starting from a
        pre-trained diffusion checkpoint.
    genomic_guided_loss_weight:
        Weight λ for the high-t denoising loss.  Total loss =
        L_diffusion + λ * L_genomic_guided.  Set to 0 to disable.
    genomic_guided_high_t_frac:
        Lower boundary of the high-timestep range as a fraction of T.
        Default 0.8 → t ∈ [0.8T, T).
    """

    # ── Cross-attention ───────────────────────────────────────────────────
    cross_attn_heads: int = 4
    cross_attn_dim_per_head: int = 64
    cross_attn_patch_size: int = 16

    # ── Per-component learning rates ──────────────────────────────────────
    cross_attn_lr: float = 1e-4
    unet_lr: float = 1e-4

    # ── Genomic-guided high-t loss ────────────────────────────────────────
    genomic_guided_loss_weight: float = 0.3
    genomic_guided_high_t_frac: float = 0.8
    compute_high_t_loss_during_training: bool = False  # Skip high-t loss during training; still compute at validation
    # ── Optional genomic reconstruction loss (MSE from pooled mid features)
    genomic_recon_weight: float = 0.05

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "GenomicCrossAttnConfig":
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in cfg.items() if k in known_fields}

        _NESTED_ALIASES = {"net_ch_mult": "net_ch_mult", "net_attn": "net_attn"}
        for alias, field_name in _NESTED_ALIASES.items():
            if alias in cfg and alias not in filtered:
                filtered[field_name] = cfg[alias]

        return cls(**filtered)
