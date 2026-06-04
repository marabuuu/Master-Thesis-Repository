"""
GDAConfig — config for Genomic Diffusion Adapter training.

Extends MoPaDi's TrainConfig with genomic dataset fields (previously in
GenomicTrainConfig) and adapter-specific fields.  No dependency on src/drafts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from mopadi.configs.config import TrainConfig


@dataclass
class GDAConfig(TrainConfig):
    # ── Genomic dataset ───────────────────────────────────────────────────
    zip_dir: Optional[str] = None
    genomic_feature_dir: Optional[str] = None
    patient_splits_path: Optional[str] = None
    max_tiles_by_subtype: Optional[Dict[str, Optional[int]]] = None
    tile_sampling_seed: int = 42
    cache_pickle_tiles_path: Optional[str] = None

    # ── Validation ────────────────────────────────────────────────────────
    # Cap val batches directly (Lightning's limit_val_batches is unreliable
    # with integer val_check_interval in 2.5.x).
    val_limit_batches: int = 100

    # Sample image logging cadence. Keep this much slower than validation so
    # the samples directory does not fill up during fast runs.
    sample_every_samples: int = 250_000

    # ── Image pre-processing ──────────────────────────────────────────────
    do_normalize: bool = True
    do_resize: bool = False

    # ── Adapter architecture ──────────────────────────────────────────────
    adapter_base_ch: int = 64
    adapter_n_tokens: int = 8
    adapter_token_dim: int = 256
    adapter_t_dim: int = 256
    adapter_n_heads: int = 4

    # ── Conditioning type ─────────────────────────────────────────────────
    # "real"    — RNA-seq from H5 files (requires genomic_feature_dir)
    # "zeros"   — 512-dim zero vector (unconditional baseline)
    # "noise"   — fresh unit-sphere random vector per sample (random baseline)
    # "one_hot" — fixed orthogonal unit vector per cohort (BRCA→e₁, LIHC→e₂)
    conditioning_type: str = "real"

    # ── CFG ───────────────────────────────────────────────────────────────
    cfg_dropout: float = 0.15

    # ── Per-component learning rates ──────────────────────────────────────
    backbone_lr: float = 1e-4
    adapter_lr: float = 3e-4

    # ── Adapter loss ──────────────────────────────────────────────────────
    # Weight for -weight * ||Δε_own - Δε_null||².  Keep small so MSE dominates.
    delta_encouragement_weight: float = 0.0

    # Weight for genomic reconstruction loss: MSE(decoder(g_tokens), feats).
    # Bootstraps encoder diversity when cross-attention is not yet active.
    genomic_recon_weight: float = 0.0

    # ── Frozen backbone ───────────────────────────────────────────────────
    freeze_backbone: bool = False
    backbone_ckpt_path: str = ""
    reinit_adapter: bool = False

    def __post_init__(self):
        # GDA: backbone sees cond=zeros (not feat_dim-sized genomic features),
        # adapter uses adapter_token_dim for cross-attention (independent of style_ch).
        # Skip the GenomicTrainConfig feat_dim == style_ch check; call TrainConfig directly.
        TrainConfig.__post_init__(self)

    @classmethod
    def from_dict(cls, cfg: Dict) -> "GDAConfig":
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in cfg.items() if k in known}
        return cls(**filtered)
