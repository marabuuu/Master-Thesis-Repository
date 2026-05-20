"""
GDAConfig — config for Genomic Diffusion Adapter training from scratch.

Extends GenomicTrainConfig with adapter-specific fields.
No pretrained checkpoint required; both backbone and adapter initialise
from random weights and train jointly.

Key design:
  backbone_lr   — LR for the main MoPaDi UNet (always receives cond=zeros)
  adapter_lr    — LR for the adapter + genomic encoder (higher, learns faster)
  cfg_dropout   — fraction of steps where adapter receives null token instead
                  of real genomic tokens, enabling CFG at inference
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from src.drafts.mopadi_genomic.config import GenomicTrainConfig


@dataclass
class GDAConfig(GenomicTrainConfig):
    # ── Adapter architecture ──────────────────────────────────────────
    adapter_base_ch: int = 64     # base channels; doubles at each down level
    adapter_n_levels: int = 3     # number of stride-2 downsamplings
    adapter_n_tokens: int = 8     # genomic token sequence length for cross-attn
    adapter_token_dim: int = 256  # dim of each token vector
    adapter_t_dim: int = 256      # sinusoidal timestep embedding dim
    adapter_n_heads: int = 4      # attention heads in cross-attention layers

    # ── CFG ──────────────────────────────────────────────────────────
    # Override parent's cond_dropout_prob with a clearer name.
    # 15 % null dropout: enough for CFG at inference while giving the adapter
    # more gradient signal on real tokens during early training.
    cfg_dropout: float = 0.15

    # ── Subtype contrastive loss ──────────────────────────────────────
    # Supervised contrastive loss (SupCon) on mean-pooled g_tokens grouped
    # by PAM50 subtype.  Pulls same-subtype token vectors together and pushes
    # different-subtype vectors apart, directly regularising the bottleneck.
    # Set to 0.0 to disable.
    contrastive_weight: float = 0.01
    contrastive_temp: float = 0.1

    # ── Per-component learning rates ─────────────────────────────────
    # Parent's conf.lr is used for the backbone; adapter gets its own LR.
    backbone_lr: float = 1e-4
    adapter_lr: float = 3e-4

    def __post_init__(self):
        # Skip GenomicTrainConfig.__post_init__ feat_dim == style_ch check —
        # in GDA the backbone still uses style_ch-sized cond vectors (zeros),
        # but the adapter uses adapter_token_dim, which is independent.
        # We call the grandparent post_init (TrainConfig) instead.
        from mopadi.configs.config import TrainConfig
        TrainConfig.__post_init__(self)

    @classmethod
    def from_dict(cls, cfg: Dict) -> "GDAConfig":
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in cfg.items() if k in known}
        return cls(**filtered)
