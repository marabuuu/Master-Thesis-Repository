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

    # ── Subtype contrastive loss (SupCon) ────────────────────────────────
    # Supervised contrastive loss on mean-pooled g_tokens.
    # Replaced by cohort_weight (cross-entropy) for binary tasks; set to
    # 0.0 to disable.
    contrastive_weight: float = 0.01
    contrastive_temp: float = 0.1

    # ── Cohort classification head ────────────────────────────────────────
    # Cross-entropy loss on mean-pooled g_tokens against cohort integer labels.
    # Creates a direct gradient path from cohort identity through the genomic
    # encoder, forcing discriminative token embeddings.  Weight 1.0 makes this
    # signal comparable to the MSE loss (~0.07 at convergence), overcoming the
    # mean-encoding collapse that defeats the weaker SupCon signal.
    # n_cohorts must equal the number of distinct subtype strings in the data.
    # Set cohort_weight=0.0 to disable (default, backward-compatible).
    cohort_weight: float = 0.0
    n_cohorts: int = 2

    # ── Per-component learning rates ─────────────────────────────────
    # Parent's conf.lr is used for the backbone; adapter gets its own LR.
    backbone_lr: float = 1e-4
    adapter_lr: float = 3e-4

    # ── Delta encouragement loss ──────────────────────────────────────
    # Adds −delta_encouragement_weight × log(guidance_delta + ε) to the
    # training loss, directly penalising near-zero adapter divergence.
    # guidance_delta = E[‖Δε_own − Δε_null‖²].
    # Set to 0.0 to disable (default, backward-compatible).
    # Recommended starting value: 1e-3.
    delta_encouragement_weight: float = 0.0

    # ── Pairwise Δε loss (unsupervised) ──────────────────────────────
    # Adds −pairwise_delta_weight × log(pairwise_delta + ε) to the loss,
    # where pairwise_delta = E[‖Δε_own − Δε_perm‖²] and Δε_perm is the
    # adapter output for the same noisy images but with cyclically shifted
    # patient tokens.  Forces the adapter to produce different corrections
    # for different patients' RNA-seq — no cohort labels required.
    # Requires delta_encouragement_weight > 0 (shares the d_own forward pass).
    # Set to 0.0 to disable (default, backward-compatible).
    pairwise_delta_weight: float = 0.0

    # ── Split backbone/adapter loss (stop-gradient) ───────────────────
    # When True, the loss is split into two separate terms:
    #   L_backbone = MSE(eps_backbone, noise)
    #   L_adapter  = MSE(eps_backbone.detach() + d_train, noise)
    # This forces the backbone to learn denoising independently (cannot
    # free-ride on the adapter), while the adapter trains on the residual
    # from a stopped backbone — no competing gradients, since backbone and
    # adapter are separate networks.
    # When False (default), uses the original joint loss:
    #   MSE(eps_backbone + d_train, noise)
    split_backbone_adapter_loss: bool = False

    # ── Frozen backbone (v15+) ────────────────────────────────────────
    # When True, backbone weights are frozen at training start.  Only the
    # adapter, genomic encoder, and null_token receive gradients.
    # Eliminates backbone–adapter competition entirely: use when resuming
    # from a well-trained backbone checkpoint (e.g. v13 epoch-13).
    # backbone_lr is ignored when freeze_backbone=True.
    freeze_backbone: bool = False

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
