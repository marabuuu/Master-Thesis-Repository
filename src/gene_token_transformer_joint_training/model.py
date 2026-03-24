from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GeneTokenTransformerConfig, parse_gene_token_transformer_config

try:
    from joint_training.model import JointLitModel, build_conf
except ImportError:  # pragma: no cover
    from src.joint_training.model import JointLitModel, build_conf  # type: ignore[import-not-found]


class GeneTokenTransformerEncoder(nn.Module):
    """Token/value embedding + transformer encoder for genomic conditioning."""

    def __init__(self, n_genes: int, cfg: GeneTokenTransformerConfig):
        super().__init__()
        self.n_genes = n_genes
        self.d_model = cfg.d_model
        self.seq_len = cfg.seq_len or n_genes

        self.gene_embedding = nn.Embedding(n_genes, cfg.d_model)
        self.value_projection = nn.Linear(1, cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.pooling = cfg.pooling
        if self.pooling == "attn_pool":
            self.pool_query = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=cfg.d_model,
                num_heads=cfg.n_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )

    def _pool(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return x[:, 0]

        if self.pooling == "attn_pool":
            query = self.pool_query.expand(x.size(0), -1, -1)
            key_padding_mask = ~attention_mask.bool()
            pooled, _ = self.pool_attn(query, x, x, key_padding_mask=key_padding_mask)
            return pooled[:, 0]

        mask = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
        masked_x = x * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        return masked_x.sum(dim=1) / denom

    def forward(self, gene_ids: torch.Tensor, gene_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embed = self.gene_embedding(gene_ids)
        value_embed = self.value_projection(gene_values.unsqueeze(-1))
        x = token_embed + value_embed
        x = F.layer_norm(x, normalized_shape=(self.d_model,))

        key_padding_mask = ~attention_mask.bool()
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        pooled = self._pool(encoded, attention_mask)
        return pooled


class GeneTokenTransformerJointLitModel(JointLitModel):  # type: ignore[misc]
    """Joint model variant with gene-token transformer genomic encoder."""

    def __init__(self, conf, joint_cfg: dict, n_genes: int):
        super().__init__(conf, joint_cfg, n_genes)
        self.gtt_cfg = parse_gene_token_transformer_config(joint_cfg)
        self.gene_token_encoder = GeneTokenTransformerEncoder(n_genes=n_genes, cfg=self.gtt_cfg)
        self.cond_projection = nn.Sequential(
            nn.LayerNorm(self.gtt_cfg.d_model),
            nn.Linear(self.gtt_cfg.d_model, int(joint_cfg.get("cond_dim", conf.feat_dim))),
        )

        self.encoder = self.gene_token_encoder
        self.projection = self.cond_projection

        self._cached_gene_ids: torch.Tensor | None = None

        n_encoder = sum(p.numel() for p in self.gene_token_encoder.parameters())
        n_proj = sum(p.numel() for p in self.cond_projection.parameters())
        n_unet = sum(p.numel() for p in self.model.parameters())
        print(
            f"[GeneTokenJoint] Encoder: {n_encoder:,}  Proj: {n_proj:,}  "
            f"UNet: {n_unet:,}  Total: {n_encoder + n_proj + n_unet:,}"
        )

        self.save_hyperparameters({
            "gene_token_transformer": self.gtt_cfg.__dict__,
            "joint_variant": "gene_token_transformer_joint_training",
        })

    def _tokenize_genomic(self, genomic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if genomic.ndim != 2:
            raise ValueError(f"Expected genomic shape [B, L], got {tuple(genomic.shape)}")

        batch_size, n_genes = genomic.shape
        if self._cached_gene_ids is None or self._cached_gene_ids.numel() != n_genes:
            self._cached_gene_ids = torch.arange(n_genes, device=genomic.device, dtype=torch.long)

        seq_len = min(self.gtt_cfg.seq_len or n_genes, n_genes)
        gene_ids = self._cached_gene_ids[:seq_len].unsqueeze(0).expand(batch_size, -1)
        gene_values = genomic[:, :seq_len]
        attention_mask = torch.ones((batch_size, seq_len), device=genomic.device, dtype=torch.bool)
        return gene_ids, gene_values, attention_mask

    def encode_genomic(self, genomic):
        gene_ids, gene_values, attention_mask = self._tokenize_genomic(genomic)
        pooled = self.gene_token_encoder(gene_ids, gene_values, attention_mask)
        cond = self.cond_projection(pooled)
        return cond


def build_gene_token_transformer_conf(joint_cfg: dict):
    """Reuse baseline joint config builder for diffusion/training defaults."""
    return build_conf(joint_cfg)
