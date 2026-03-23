#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-attention variant of joint_training.
- Reuses JointLitModel from joint_training
- Wraps UNet with lightweight multi-level cross-attention to genomic cond
- Adds x_T and cond dropout hooks to force conditioning reliance
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast

try:
    from joint_training.model import JointLitModel, build_conf
except ImportError:  # pragma: no cover
    from src.joint_training.model import JointLitModel, build_conf  # type: ignore[import-not-found]


class CrossAttentionBlock(nn.Module):
    """Cross-attention from image tokens to a single cond token."""

    def __init__(self, cond_dim: int, heads: int, dim_head: int):
        super().__init__()
        embed_dim = heads * dim_head
        self.q_proj = nn.LazyLinear(embed_dim)
        self.k_proj = nn.Linear(cond_dim, embed_dim)
        self.v_proj = nn.Linear(cond_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        # Project back to RGB channels (input x has 3 channels)
        self.out_proj = nn.Linear(embed_dim, 3)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), cond: (B, cond_dim)
        b, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        q = self.q_proj(tokens)
        k = self.k_proj(cond).unsqueeze(1)  # (B, 1, E)
        v = self.v_proj(cond).unsqueeze(1)
        attn_out, _ = self.attn(q, k, v)
        tokens_out = self.out_proj(attn_out)  # (B, HW, 3)
        out = tokens_out.reshape(b, h, w, 3).permute(0, 3, 1, 2)
        return out


class CrossAttentionUNetWrapper(nn.Module):
    """Wraps base UNet, injecting cross-attn residuals at multiple scales."""

    def __init__(self, base_unet: nn.Module, cond_dims: List[int], heads: int, dim_head: int):
        super().__init__()
        self.base_unet = base_unet
        self.cond_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(cond_dims[0], d),
            nn.GELU(),
            nn.Linear(d, d),
        ) for d in cond_dims])
        self.attn_blocks = nn.ModuleList([CrossAttentionBlock(d, heads, dim_head) for d in cond_dims])
        self.scales = [1, 2, 4][: len(cond_dims)]

    def _make_cond_multi(self, cond: torch.Tensor) -> List[torch.Tensor]:
        return [proj(cond) for proj in self.cond_heads]

    def forward(self, x: torch.Tensor, t: torch.Tensor, *, cond: torch.Tensor, cond_multi: Optional[List[torch.Tensor]] = None, **kwargs):
        cond_list = cond_multi or self._make_cond_multi(cond)
        h = x
        for scale, attn, cvec in zip(self.scales, self.attn_blocks, cond_list):
            if scale > 1:
                h_down = F.avg_pool2d(h, kernel_size=scale, stride=scale)
            else:
                h_down = h
            delta = attn(h_down, cvec)
            if scale > 1:
                delta = F.interpolate(delta, size=h.shape[2:], mode="bilinear", align_corners=False)
            h = h + delta
        return self.base_unet(x=h, t=t, cond=cond, **kwargs)

    def make_cond_multi(self, cond: torch.Tensor) -> List[torch.Tensor]:
        return self._make_cond_multi(cond)


class CrossAttentionJointLitModel(JointLitModel):
    """Joint model with cross-attn UNet and conditioning/noise dropouts."""

    def __init__(self, conf, joint_cfg: dict, n_genes: int):
        cross_cfg = joint_cfg.get("cross_attention", {})
        heads = int(cross_cfg.get("heads", 4))
        dim_head = int(cross_cfg.get("dim_per_head", 64))
        cond_dims = cross_cfg.get("cond_dims", [512, 256, 128])
        super().__init__(conf, joint_cfg, n_genes)
        self.cross_cfg = {
            "heads": heads,
            "dim_head": dim_head,
            "cond_dims": cond_dims,
            "xT_dropout_prob": float(joint_cfg.get("xT_dropout_prob", 0.05)),
            "cond_dropout_prob": float(joint_cfg.get("cond_dropout_prob", 0.05)),
            "cond_feature_dropout": float(joint_cfg.get("cond_feature_dropout", 0.05)),
        }
        # Replace UNet with cross-attn wrapper (keep sampler interface intact)
        self.model = CrossAttentionUNetWrapper(self.model, cond_dims=cond_dims, heads=heads, dim_head=dim_head)
        # Save extended hparams
        self.save_hyperparameters({"cross_cfg": self.cross_cfg})

    def training_step(self, batch, batch_idx):
        with autocast(device_type='cuda', enabled=self.conf.fp16):
            imgs = batch['img'].to(self.device)
            genomic = batch['genomic'].to(self.device, dtype=torch.float32)

            cond = self.encode_genomic(genomic)

            # Conditioning dropout
            p_cond = self.cross_cfg.get("cond_dropout_prob", 0.0)
            if p_cond > 0:
                mask = torch.rand(cond.shape[0], device=cond.device) < p_cond
                if mask.any():
                    cond = cond.clone()
                    cond[mask] = 0
            p_feat = self.cross_cfg.get("cond_feature_dropout", 0.0)
            if p_feat > 0:
                cond = F.dropout(cond, p=p_feat, training=self.training)

            cond_multi = self.model.make_cond_multi(cond)

            # x_T dropout: replace some x_start with pure noise
            x_start = imgs
            p_x = self.cross_cfg.get("xT_dropout_prob", 0.0)
            if p_x > 0:
                mask_x = torch.rand(imgs.shape[0], device=imgs.device) < p_x
                if mask_x.any():
                    noise = torch.randn_like(imgs)
                    x_start = x_start.clone()
                    x_start[mask_x] = noise[mask_x]

            t, _ = self.T_sampler.sample(len(imgs), imgs.device)
            losses = self.sampler.training_losses(
                model=self.model,
                x_start=x_start,
                cond=cond,
                t=t,
                model_kwargs={"cond": cond, "cond_multi": cond_multi},
            )
            loss = losses['loss'].mean()

        self.log('loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(imgs))
        self.log('loss_step', loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=len(imgs))

        if self.global_rank == 0 and hasattr(self, 'logger') and hasattr(self.logger, 'experiment'):
            self.logger.experiment.add_scalar('loss', loss.item(), self.num_samples)  # type: ignore[union-attr]

        return {'loss': loss}


def build_cross_conf(joint_cfg: dict):
    """Reuse build_conf from joint_training (helper for symmetry)."""
    return build_conf(joint_cfg)
