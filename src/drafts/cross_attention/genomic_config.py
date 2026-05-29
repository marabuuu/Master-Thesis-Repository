"""
GenomicCaConfig — extends GenomicTrainConfig with cross-attention and pred-gap fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from src.mopadi_genomic_crossattn.genomic_config import GenomicTrainConfig


@dataclass
class GenomicCaConfig(GenomicTrainConfig):
    """GenomicTrainConfig + bottleneck cross-attention + prediction-gap loss fields.

    New fields
    ----------
    use_genomic_cross_attn:
        Whether to attach GenomicCrossAttentionBlock to the UNet bottleneck.
    genomic_ca_heads:
        Number of attention heads (spatial_channels must be divisible by this).
    genomic_ca_n_tokens:
        How many gene-summary tokens to project the genomic vector into.
    pred_gap_lambda:
        Weight on the prediction-gap hinge: relu(pred_gap_margin - ||eps_neg - eps_ref||^2).
    pred_gap_margin:
        Target minimum MSE between neg and ref predicted noise.
    """

    # Cross-attention
    use_genomic_cross_attn: bool = True
    genomic_ca_heads: int = 8
    genomic_ca_n_tokens: int = 4

    # Classifier-free guidance dropout (fraction of batches that use null feats)
    cfg_dropout: float = 0.15

    # Kept for backward-compat with old configs; unused in CFG training
    pred_gap_lambda: float = 0.0
    pred_gap_margin: float = 0.01

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "GenomicCaConfig":
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in cfg.items() if k in known})
