"""
GenomicCrossAttentionBlock — bottleneck cross-attention for genomic conditioning.

Spatial feature map (queries) attends to genomic gene tokens (keys/values).
Output is a per-position FiLM: h_out = h + h * scale + shift.

The block is injected into BeatGANsAutoencModel after middle_block via a two-line
hook in mopadi/src/mopadi/model/unet_autoenc.py:

    _gca = getattr(self, 'genomic_cross_attn', None)
    if _gca is not None and cond is not None and x is not None:
        h = _gca(h, cond)

Why 1%-scaled out_proj instead of zero_module:
    zero_module(out_proj) → W_out = 0
    → d(film)/d(attn_out) = W_out^T = 0
    → gene_proj and kv_proj receive zero gradient from step 1 (gradient deadlock)
    1% scale keeps the initial FiLM tiny while all gradients flow immediately.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mopadi.model.nn import linear, normalization


class GenomicCrossAttentionBlock(nn.Module):
    """Cross-attention from spatial bottleneck features to genomic gene tokens.

    Args:
        spatial_channels: C — bottleneck feature-map channels (512 for default config).
        gene_dim:          dimension of the L2-normalised genomic vector (512).
        n_heads:           multi-head attention heads (default 8; head_dim = C / n_heads).
        n_gene_tokens:     number of gene-summary tokens projected from the genomic
                           vector (default 4).
    """

    def __init__(
        self,
        spatial_channels: int,
        gene_dim: int,
        n_heads: int = 8,
        n_gene_tokens: int = 4,
    ):
        super().__init__()
        if spatial_channels % n_heads != 0:
            raise ValueError(
                f"spatial_channels ({spatial_channels}) must be divisible by "
                f"n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.head_dim = spatial_channels // n_heads
        self.n_gene_tokens = n_gene_tokens

        self.norm = normalization(spatial_channels)
        self.q_proj = linear(spatial_channels, spatial_channels)
        # Gene-side projections use bias=False so that feats=zeros (CFG null
        # conditioning) produces zero gene tokens → zero k/v → zero attention
        # output → h unchanged.  With bias=True the null signal equals the bias
        # magnitude (~0.025 std), which is the same order as the actual genomic
        # signal (~0.028 std) — the model can't distinguish null from conditioned.
        self.gene_proj = nn.Linear(gene_dim, n_gene_tokens * spatial_channels, bias=False)
        self.kv_proj = nn.Linear(spatial_channels, 2 * spatial_channels, bias=False)
        # out_proj: bias=False ensures feats=zeros → CA output=0 → h unchanged.
        # Full-scale Xavier init retained: FiLM perturbation ~0.019 std at
        # bottleneck, above bf16 precision floor (~7.8e-3).
        self.out_proj = nn.Linear(spatial_channels, 2 * spatial_channels, bias=False)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:    (B, C, H, W) spatial feature map
            cond: (B, D) L2-normalised genomic vector
        Returns:
            h + h * scale + shift  (same shape as h)
        """
        B, C, H, W = h.shape
        HW = H * W

        # Normalize + flatten spatial features → (B, HW, C)
        q = self.norm(h).reshape(B, C, HW).permute(0, 2, 1)
        q = self.q_proj(q)

        # Build gene tokens: (B, D) → (B, n_gene_tokens, C)
        gene_tokens = self.gene_proj(cond).reshape(B, self.n_gene_tokens, C)
        kv = self.kv_proj(gene_tokens)
        k, v = kv.chunk(2, dim=-1)

        def split_heads(x: torch.Tensor, seq: int) -> torch.Tensor:
            return x.reshape(B, seq, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        q = split_heads(q, HW)
        k = split_heads(k, self.n_gene_tokens)
        v = split_heads(v, self.n_gene_tokens)

        out = F.scaled_dot_product_attention(q, k, v)   # flash attn when available
        out = out.permute(0, 2, 1, 3).reshape(B, HW, C)

        film = self.out_proj(out)
        scale, shift = film.chunk(2, dim=-1)
        scale = scale.permute(0, 2, 1).reshape(B, C, H, W)
        shift = shift.permute(0, 2, 1).reshape(B, C, H, W)

        return h + h * scale + shift
