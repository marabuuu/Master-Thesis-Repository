"""
FrozenBackboneCaConfig — extends GenomicCaConfig for the frozen-backbone CA experiment.

Key differences from GenomicCaConfig (v11 CFG):
  pretrained_backbone_ckpt:
      Path to the v11 final checkpoint.  Backbone weights are loaded from here
      and immediately frozen.  Only the CA block is trained.
  cfg_dropout: 0.30
      30 % null batches (up from 15 %) — stronger incentive for CA to use
      conditioning on the 70 % real-feats batches when backbone is frozen.
  genomic_ca_n_tokens: 8
      Double the gene-token count (was 4) — more representational capacity
      for the CA block to encode subtype directions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from src.cross_attention.genomic_config import GenomicCaConfig


@dataclass
class FrozenBackboneCaConfig(GenomicCaConfig):
    # Path to a finished checkpoint whose backbone we load and freeze.
    pretrained_backbone_ckpt: str = ""

    # Whether to freeze the backbone (always True in normal usage).
    frozen_backbone: bool = True

    # Higher CFG dropout: 30 % null batches strengthen the conditioning signal.
    cfg_dropout: float = 0.30

    # More gene tokens for richer CA representations.
    genomic_ca_n_tokens: int = 8

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "FrozenBackboneCaConfig":
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in cfg.items() if k in known})
