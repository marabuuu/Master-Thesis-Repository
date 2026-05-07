"""
GenomicTrainConfig — extends MoPaDi's TrainConfig with genomic-specific fields.

All model/diffusion/optimizer parameters are inherited unchanged from TrainConfig.
The additional fields configure the ZIP-based genomic dataset and tile balancing.

Usage
-----
    from mopadi_genomic_crossattn.genomic_config import GenomicTrainConfig
    conf = GenomicTrainConfig.from_dict(cfg_dict)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from mopadi.configs.config import TrainConfig


@dataclass
class GenomicTrainConfig(TrainConfig):
    """TrainConfig extended for genomic conditioning without image feature extractors.

    Extra fields
    ------------
    zip_dir:
        Path to the directory containing per-patient tile ZIP archives,
        e.g. ``/data/BRCA-tumor-tiles-corrected``.
    genomic_feature_dir:
        Path to the directory containing per-patient H5 files produced by
        ``data_prep.build_genomic_features`` (``{patient_id}.h5``).
    patient_splits_path:
        Path to ``patient_splits.json`` produced by
        ``data_prep.build_genomic_features``.
    max_tiles_by_subtype:
        Per-subtype cap on tiles per patient, e.g.
        ``{"LumA": 45, "LumB": 120, "Basal": 135, "Her2": 300, "Normal": None}``.
        ``None`` for a subtype means no cap is applied.
        If ``None`` overall, no capping is applied to any subtype.
    tile_sampling_seed:
        Random seed for the per-patient tile sampling when caps are applied.
    do_normalize:
        Whether to apply diffusion-style image normalisation (mean=0.5, std=0.5).
        Defaults to True.
    do_resize:
        Whether to resize tiles.  Defaults to False since there is no image
        feature extractor that requires a specific input resolution.
    """

    # ── Genomic dataset ───────────────────────────────────────────────────
    zip_dir: Optional[str] = None
    genomic_feature_dir: Optional[str] = None
    patient_splits_path: Optional[str] = None
    max_tiles_by_subtype: Optional[Dict[str, Optional[int]]] = None
    tile_sampling_seed: int = 42

    # ── Validation ───────────────────────────────────────────────────────
    # Cap val batches in val_dataloader() directly (Lightning's limit_val_batches
    # is unreliable with integer val_check_interval in 2.5.x).
    val_limit_batches: int = 100

    # ── Image pre-processing overrides ───────────────────────────────────
    do_normalize: bool = True
    do_resize: bool = False

    def __post_init__(self):
        super().__post_init__()
        # Enforce that feat_dim matches style_ch so the 512-dim gene vector
        # slots into the UNet's conditioning pathway without mismatch.
        if self.feat_dim != self.style_ch:
            raise ValueError(
                f"GenomicTrainConfig requires feat_dim == style_ch "
                f"(got feat_dim={self.feat_dim}, style_ch={self.style_ch}). "
                "Both should be 512 to match the gene-expression vector size."
            )
        
        # Enable dual conditioning: time + genomic features
        # This is MoPaDi's built-in mechanism; no wrapper needed.
        self.net_beatgans_resnet_two_cond = True

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, cfg: Dict) -> "GenomicTrainConfig":
        """Construct a GenomicTrainConfig from a plain dict (e.g. from YAML).

        Unknown keys are silently ignored so the config section in YAML can
        contain documentation keys without breaking the constructor.
        """
        # Collect only the field names that GenomicTrainConfig (and its
        # parents) declare, to avoid passing unexpected keyword arguments.
        import dataclasses
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in cfg.items() if k in known_fields}

        # Handle nested model config keys that make_model_conf() needs.
        _NESTED_ALIASES = {
            "net_ch_mult": "net_ch_mult",
            "net_attn": "net_attn",
        }
        for alias, field_name in _NESTED_ALIASES.items():
            if alias in cfg and alias not in filtered:
                filtered[field_name] = cfg[alias]

        return cls(**filtered)
