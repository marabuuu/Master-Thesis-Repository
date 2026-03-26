#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-attention variant of joint_training.
- Reuses JointLitModel from joint_training
- Wraps UNet with lightweight multi-level cross-attention to genomic cond
- Adds x_T and cond dropout hooks to force conditioning reliance
"""

from __future__ import annotations

from typing import List, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast

try:
    from joint_training.model import JointLitModel, build_conf
except ImportError:  # pragma: no cover
    from src.joint_training.model import JointLitModel, build_conf  # type: ignore[import-not-found]

JointLitModelBase = cast(type, JointLitModel)


def _build_swapped_indices(batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
    if batch_size < 2:
        return None
    shift = int(torch.randint(1, batch_size, (1,), device=device).item())
    return torch.roll(torch.arange(batch_size, device=device), shifts=shift)


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


class CrossAttentionJointLitModel(JointLitModelBase):
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
            "counterfactual_loss_weight": float(joint_cfg.get("counterfactual_loss_weight", 0.0)),
            "counterfactual_margin": float(joint_cfg.get("counterfactual_margin", 0.02)),
            "counterfactual_monitor_every_n_steps": int(joint_cfg.get("counterfactual_monitor_every_n_steps", 200)),
            "counterfactual_zero_threshold": float(joint_cfg.get("counterfactual_zero_threshold", 1e-4)),
        }
        # Replace UNet with cross-attn wrapper (keep sampler interface intact)
        self.model = CrossAttentionUNetWrapper(self.model, cond_dims=cond_dims, heads=heads, dim_head=dim_head)
        # Save extended hparams
        self.save_hyperparameters({
            "cross_cfg": self.cross_cfg,
            "joint_variant": "cross_attention_joint_training",
        })

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

            # x_T dropout (curriculum): send a subset to highest timestep.
            #
            # Rationale:
            # Replacing x_start with pure random noise and then applying q_sample
            # again can create "double-noise" inputs with weak learning signal.
            # Instead, keep x_start=real image and force t=T-1 for dropped samples,
            # which approximates the intended hard denoising regime while keeping
            # the diffusion objective well-formed.
            x_start = imgs
            t, _ = self.T_sampler.sample(len(imgs), imgs.device)
            p_x = self.cross_cfg.get("xT_dropout_prob", 0.0)
            if p_x > 0:
                mask_x = torch.rand(imgs.shape[0], device=imgs.device) < p_x
                if mask_x.any():
                    t = t.clone()
                    t[mask_x] = int(self.conf.T - 1)

            losses = self.sampler.training_losses(
                model=self.model,
                x_start=x_start,
                cond=cond,
                t=t,
                model_kwargs={"cond": cond, "cond_multi": cond_multi},
            )
            main_loss_per_sample = losses['loss']
            main_loss = main_loss_per_sample.mean()

            cf_weight = float(self.cross_cfg.get("counterfactual_loss_weight", 0.0))
            cf_margin_loss = torch.zeros((), device=imgs.device, dtype=main_loss.dtype)
            swap_gap = torch.zeros((), device=imgs.device, dtype=main_loss.dtype)
            swap_idx = _build_swapped_indices(cond.shape[0], cond.device)
            if cf_weight > 0.0 and swap_idx is not None:
                cond_swapped = cond[swap_idx]
                cond_multi_swapped = self.model.make_cond_multi(cond_swapped)
                losses_swapped = self.sampler.training_losses(
                    model=self.model,
                    x_start=x_start,
                    cond=cond_swapped,
                    t=t,
                    model_kwargs={"cond": cond_swapped, "cond_multi": cond_multi_swapped},
                )
                swapped_loss_per_sample = losses_swapped['loss']
                margin = float(self.cross_cfg.get("counterfactual_margin", 0.02))
                cf_margin_loss = F.relu(
                    margin + main_loss_per_sample.detach() - swapped_loss_per_sample
                ).mean()
                swap_gap = (swapped_loss_per_sample.detach() - main_loss_per_sample.detach()).mean()

            loss = main_loss + cf_weight * cf_margin_loss

        self.log('loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(imgs))
        self.log('loss_step', loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=len(imgs))
        self.log('loss_main_step', main_loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=len(imgs))
        self.log('loss_cf_margin_step', cf_margin_loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=len(imgs))
        self.log('cf_swap_gap_step', swap_gap, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=len(imgs))

        monitor_every = int(self.cross_cfg.get("counterfactual_monitor_every_n_steps", 200))
        zero_threshold = float(self.cross_cfg.get("counterfactual_zero_threshold", 1e-4))
        if (
            self.global_rank == 0
            and monitor_every > 0
            and self.global_step % monitor_every == 0
            and self.cross_cfg.get("counterfactual_loss_weight", 0.0) > 0.0
        ):
            cf_margin_value = float(cf_margin_loss.detach().item())
            message = (
                f"[CF-TUNING] step={self.global_step} total={float(loss.detach().item()):.6f} "
                f"main={float(main_loss.detach().item()):.6f} cf_margin={cf_margin_value:.6f} "
                f"swap_gap={float(swap_gap.detach().item()):.6f} "
                f"weight={float(self.cross_cfg.get('counterfactual_loss_weight', 0.0)):.3f} "
                f"margin={float(self.cross_cfg.get('counterfactual_margin', 0.0)):.3f}"
            )
            if cf_margin_value <= zero_threshold:
                message += " | hint: loss_cf_margin_step≈0 -> try margin=0.05 or weight=0.3"
            else:
                message += " | hint: if training/image quality becomes unstable, lower weight to 0.1"
            print(message)

        if self.global_rank == 0 and hasattr(self, 'logger') and hasattr(self.logger, 'experiment'):
            self.logger.experiment.add_scalar('loss', loss.item(), self.num_samples)  # type: ignore[union-attr]

        return {'loss': loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Override EMA update to handle wrapped model.
        
        Since self.model is a CrossAttentionUNetWrapper, we need to pass
        the base_unet to the EMA function to match keys in ema_model.
        """
        # Import here to avoid circular imports
        from mopadi.train_diff_autoenc import ema
        
        # Pass base_unet (unwrapped) to EMA for proper key matching
        ema(self.model.base_unet, self.ema_model, self.conf.ema_decay)


def build_cross_conf(joint_cfg: dict):
    """Reuse build_conf from joint_training (helper for symmetry)."""
    return build_conf(joint_cfg)
