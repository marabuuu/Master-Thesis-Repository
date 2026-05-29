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
from typing import Dict

from src.drafts.mopadi_genomic.config import GenomicTrainConfig


@dataclass
class GDAConfig(GenomicTrainConfig):
    # ── Adapter architecture ──────────────────────────────────────────
    adapter_base_ch: int = 64     # base channels; doubles at each down level
    adapter_n_tokens: int = 8     # genomic token sequence length for cross-attn
    adapter_token_dim: int = 256  # dim of each token vector
    adapter_t_dim: int = 256      # sinusoidal timestep embedding dim
    adapter_n_heads: int = 4      # attention heads in cross-attention layers

    # ── CFG ──────────────────────────────────────────────────────────
    cfg_dropout: float = 0.15

    # ── Per-component learning rates ─────────────────────────────────
    backbone_lr: float = 1e-4
    adapter_lr: float = 3e-4

    # ── Adapter loss ─────────────────────────────────────────────────
    # Weight for the delta-encouragement term: -weight * ||Δε_own - Δε_null||².
    # Prevents adapter from ignoring token input when null_token is fixed at zeros.
    # 0.0 = disabled. Keep small (0.001) so MSE still dominates.
    delta_encouragement_weight: float = 0.0

    # ── Frozen backbone ───────────────────────────────────────────────
    # When True, backbone weights are frozen at training start.
    # backbone_lr is ignored when freeze_backbone=True.
    freeze_backbone: bool = False

    # Optional checkpoint used to initialize the backbone before freezing.
    # Only backbone weights are loaded; conditioning path can be reinitialized.
    backbone_ckpt_path: str = ""

    # If True, the adapter, genomic encoder, and null token are reset to fresh
    # initial weights after loading the backbone checkpoint.
    reinit_adapter: bool = False

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
