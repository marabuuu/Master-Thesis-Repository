from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GeneTokenTransformerConfig:
    gene_list_path: str | None = None
    seq_len: int | None = None
    value_embedding: str = "mlp"  # mlp | bins
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    pooling: str = "mean"  # cls | mean | attn_pool
    cond_dim: int = 512
    freeze_transformer_steps: int = 0
    transformer_lr: float = 1e-4


def parse_gene_token_transformer_config(joint_cfg: dict[str, Any]) -> GeneTokenTransformerConfig:
    cfg = joint_cfg.get("gene_token_transformer", {})
    value_embedding = cfg.get("value_embedding", "mlp")
    if value_embedding not in {"mlp", "bins"}:
        raise ValueError("gene_token_transformer.value_embedding must be one of: mlp, bins")

    pooling = cfg.get("pooling", "mean")
    if pooling not in {"cls", "mean", "attn_pool"}:
        raise ValueError("gene_token_transformer.pooling must be one of: cls, mean, attn_pool")

    return GeneTokenTransformerConfig(
        gene_list_path=cfg.get("gene_list_path"),
        seq_len=cfg.get("seq_len"),
        value_embedding=value_embedding,
        d_model=int(cfg.get("d_model", 256)),
        n_heads=int(cfg.get("n_heads", 8)),
        n_layers=int(cfg.get("n_layers", 4)),
        ff_mult=int(cfg.get("ff_mult", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        pooling=pooling,
        cond_dim=int(cfg.get("cond_dim", 512)),
        freeze_transformer_steps=int(cfg.get("freeze_transformer_steps", 0)),
        transformer_lr=float(cfg.get("transformer_lr", 1e-4)),
    )
