"""
GenomicResidualAdapter — lightweight UNet that predicts Δε(x_t, t, g).

Architecture (3-level encoder-decoder with cross-attention on genomic tokens):

  x_t  ──► in_conv(3→ch)
           enc0(ch) + CA(ch, ctx) ──────────────────────────────────┐
           down1 (stride-2, ch→2ch)                                  │ skip
           enc1(2ch) + CA(2ch, ctx) ─────────────────────────────┐  │
           down2 (stride-2, 2ch→4ch)                              │  │
           enc2(4ch) + CA(4ch, ctx) ────────────────────────┐     │  │
           mid(4ch) + CA(4ch, ctx)                           │     │  │
           up2 (4ch→2ch) + cat(skip) ◄──────────────────────┘     │  │
           dec2(4ch→2ch)                                            │  │
           up1 (2ch→ch) + cat(skip)  ◄──────────────────────────── ┘  │
           dec1(2ch→ch)                                                │
           cat(skip)     ◄───────────────────────────────────────────── ┘
           dec0(2ch→ch)
           zero_conv(ch→3) = Δε

The final conv is zero-initialised: Δε=0 at training start, so the combined
model (backbone + adapter) begins from the same distribution as the purely
unconditional backbone.

Timestep conditioning: sinusoidal embedding → MLP → AdaGN in every ResBlock.
Genomic conditioning: cross-attention on token sequence from GenomicTokenEncoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Timestep utilities
# ---------------------------------------------------------------------------

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class AdapterResBlock(nn.Module):
    """ResBlock with AdaGN for timestep conditioning."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        groups = min(32, in_ch)
        out_groups = min(32, out_ch)
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(out_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, 2 * out_ch)  # scale + shift
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale_shift = self.t_proj(F.silu(t_emb))[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        h = F.silu(self.norm2(h) * (1 + scale) + shift)
        h = self.conv2(h)
        return h + self.skip(x)


class SpatialCrossAttention(nn.Module):
    """
    Cross-attention between spatial feature map and genomic token sequence.
    Query: flattened spatial positions (B, H*W, C).
    Key/Value: genomic tokens (B, n_tokens, token_dim).
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        n_heads: int = 4,
        zero_init_output: bool = False,
    ):
        super().__init__()
        assert query_dim % n_heads == 0, "query_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = query_dim // n_heads
        self.norm = nn.GroupNorm(min(32, query_dim), query_dim)
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.out_proj = nn.Linear(query_dim, query_dim, bias=False)
        if zero_init_output:
            # Zero-init lets new CA layers start as identity (no residual effect),
            # so they can be added to a mid-run checkpoint without disruption.
            nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h_flat = h.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)

        Q = self.to_q(h_flat).view(B, H * W, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.to_k(context).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.to_v(context).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(Q, K, V)
        attn = attn.transpose(1, 2).reshape(B, H * W, C)
        out = self.out_proj(attn).permute(0, 2, 1).view(B, C, H, W)
        return x + out  # residual


class GenomicTokenEncoder(nn.Module):
    """Projects raw genomic features to a fixed-length token sequence for cross-attention."""

    def __init__(self, genomic_in: int, n_tokens: int, token_dim: int):
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.net = nn.Sequential(
            nn.Linear(genomic_in, 256),
            nn.SiLU(),
            nn.Linear(256, n_tokens * token_dim),
        )

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.net(g).view(g.shape[0], self.n_tokens, self.token_dim)


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class GenomicResidualAdapter(nn.Module):
    """
    Predicts Δε(x_t, t, g_tokens) — the genomic correction added on top of
    the unconditional backbone's epsilon prediction.

    Parameters
    ----------
    in_ch : int
        Input/output image channels (3 for RGB).
    base_ch : int
        Base channel count; encoder levels use base_ch, 2*base_ch, 4*base_ch.
    t_dim : int
        Sinusoidal timestep embedding dimension.
    token_dim : int
        Dimension of each genomic token (context_dim for cross-attention).
    n_heads : int
        Number of attention heads in cross-attention layers.
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 64,
        t_dim: int = 256,
        token_dim: int = 256,
        n_heads: int = 4,
    ):
        super().__init__()
        ch = base_ch
        self.t_dim = t_dim

        # Timestep MLP (sinusoidal → projected)
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        # ── Encoder ────────────────────────────────────────────────────────
        self.in_conv = nn.Conv2d(in_ch, ch, 3, padding=1)

        self.enc0 = AdapterResBlock(ch, ch, t_dim)
        self.ca0 = SpatialCrossAttention(ch, token_dim, n_heads)

        self.down1 = nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)
        self.enc1 = AdapterResBlock(ch * 2, ch * 2, t_dim)
        self.ca1 = SpatialCrossAttention(ch * 2, token_dim, n_heads)

        self.down2 = nn.Conv2d(ch * 2, ch * 4, 3, stride=2, padding=1)
        self.enc2 = AdapterResBlock(ch * 4, ch * 4, t_dim)
        self.ca2 = SpatialCrossAttention(ch * 4, token_dim, n_heads)

        # ── Bottleneck ─────────────────────────────────────────────────────
        self.mid = AdapterResBlock(ch * 4, ch * 4, t_dim)
        self.mid_ca = SpatialCrossAttention(ch * 4, token_dim, n_heads)

        # ── Decoder (skip connections: enc level N matches decoder level N) ──
        # up2(hm) at H/4 → H/2, concat with h1 (H/2): (2ch + 2ch) = 4ch in
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)
        self.dec2 = AdapterResBlock(ch * 4, ch * 2, t_dim)
        # Decoder CA re-attends to g_tokens after each ResBlock so that
        # genomic information is not diluted by the spatial skip-connection path.
        # zero_init_output=True lets these layers resume from a pre-existing
        # checkpoint without disrupting the already-trained weights.
        self.ca_dec2 = SpatialCrossAttention(ch * 2, token_dim, n_heads, zero_init_output=True)

        # up1(d2) at H/2 → H, concat with h0 (H): (ch + ch) = 2ch in
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        self.dec1 = AdapterResBlock(ch * 2, ch, t_dim)
        self.ca_dec1 = SpatialCrossAttention(ch, token_dim, n_heads, zero_init_output=True)

        # Zero-init final conv: Δε=0 at start → stable initialisation
        self.out_conv = nn.Conv2d(ch, in_ch, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        g_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        xt : (B, 3, H, W)
        timesteps : (B,) integer timestep indices
        g_tokens : (B, n_tokens, token_dim) — use null_token for CFG null pass
        """
        t_emb = self.t_mlp(sinusoidal_embedding(timesteps, self.t_dim))

        # Encoder
        h0 = self.ca0(self.enc0(self.in_conv(xt), t_emb), g_tokens)   # (B, ch, H, W)
        h1 = self.ca1(self.enc1(self.down1(h0), t_emb), g_tokens)     # (B, 2ch, H/2, W/2)
        h2 = self.ca2(self.enc2(self.down2(h1), t_emb), g_tokens)     # (B, 4ch, H/4, W/4)

        # Bottleneck
        hm = self.mid_ca(self.mid(h2, t_emb), g_tokens)               # (B, 4ch, H/4, W/4)

        # Decoder — skip from level N pairs with upsample from level N+1
        d2 = self.ca_dec2(
            self.dec2(torch.cat([self.up2(hm), h1], dim=1), t_emb), g_tokens
        )  # (B, 2ch, H/2, W/2)
        d1 = self.ca_dec1(
            self.dec1(torch.cat([self.up1(d2), h0], dim=1), t_emb), g_tokens
        )  # (B, ch, H, W)

        return self.out_conv(d1)  # (B, 3, H, W) = Δε
