#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-attention variant of joint_training.
- Reuses JointLitModel from joint_training
- Wraps UNet with a patchified single-scale cross-attention module
- Input image is tokenised via a learned Conv2d patchifier (semantically
  richer than raw RGB pixel values); spatial tokens attend to the genomic
  conditioning vector as a single K/V token.
- The cross-attention output is a zero-initialised residual added to the
  UNet input, so training starts from the identity and the attention path
  is learned gradually.
- Adds cond/x_T dropout hooks and an optional counterfactual loss to force
  conditioning reliance.

Design notes (multi-scale → single-scale change)
-------------------------------------------------
The previous implementation applied cross-attention at multiple spatial
scales of the raw RGB input (1×, ½×, ¼× resolution via avg-pool).
Downsampling the RGB image before attention is problematic because:
  1. RGB pixel values are not semantic features; attention computed from
     3-channel queries has little discriminative power regardless of scale.
  2. Downsampling + bilinear upsampling introduces unnecessary smoothing
     without adding semantic multi-scale structure.
  3. The "multiple scales" corresponded to scales of the *input image*, not
     to different depths in the UNet, so the intended multi-level conditioning
     was not achieved.

The new design uses a single Conv2d patchifier (patch_size × patch_size,
stride = patch_size) to convert the input into spatially meaningful tokens
before cross-attention.  This is equivalent to a shallow ViT patch embedding.
Since K and V are a single genomic token, attention complexity is O(N) in the
number of patches — so even small patch sizes (e.g. 8) are feasible.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from typing import cast

try:
    from joint_training.model import JointLitModel, build_conf
except ImportError:  # pragma: no cover
    from src.joint_training.model import JointLitModel, build_conf  # type: ignore[import-not-found]

try:
    from mopadi.configs.choices import OptimizerType
except ImportError:  # pragma: no cover
    from configs.choices import OptimizerType  # type: ignore[import-not-found]

JointLitModelBase = cast(type, JointLitModel)


def _build_swapped_indices(batch_size: int, device: torch.device):
    if batch_size < 2:
        return None
    shift = int(torch.randint(1, batch_size, (1,), device=device).item())
    return torch.roll(torch.arange(batch_size, device=device), shifts=shift)


class CrossAttentionUNetWrapper(nn.Module):
    """Wraps the base UNet with a single-scale patchified genomic cross-attention.

    The input image is split into non-overlapping patches via a learned Conv2d
    patchifier, giving semantically richer tokens than raw RGB pixel values.
    These spatial tokens attend to the genomic conditioning vector (used as a
    single key-value token) via MultiheadAttention.  The attended output is
    projected back to image-space patches and added as a zero-initialised
    residual to the UNet input before forwarding through the base UNet.

    Parameters
    ----------
    base_unet:
        The mopadi UNet model to wrap.
    cond_dim:
        Dimension of the genomic conditioning vector (output of projection head).
    patch_size:
        Side length of non-overlapping patches.  Must evenly divide the image
        height and width.  Smaller → more patches, finer spatial resolution,
        more memory.  Default 16 gives 32×32 = 1 024 patches for 512×512 images.
    heads:
        Number of attention heads.
    dim_head:
        Dimension per attention head; embed_dim = heads × dim_head.
    """

    def __init__(
        self,
        base_unet: nn.Module,
        cond_dim: int,
        patch_size: int = 16,
        heads: int = 4,
        dim_head: int = 64,
    ):
        super().__init__()
        self.base_unet = base_unet
        self.patch_size = patch_size
        embed_dim = heads * dim_head
        self.embed_dim = embed_dim

        # Learned patchifier: (B, 3, H, W) → (B, embed_dim, H/ps, W/ps)
        self.patch_proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Project genomic conditioning to K and V (single token per sample)
        self.k_proj = nn.Linear(cond_dim, embed_dim)
        self.v_proj = nn.Linear(cond_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        # Decode attended tokens back to image patches.
        # Zero-init so the wrapper starts as a pure identity.
        self.out_proj = nn.Linear(embed_dim, patch_size * patch_size * 3)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def wrapper_parameters(self):
        """Yield only the cross-attention wrapper parameters (not base_unet).

        Use this to build a dedicated optimizer param group so that wrapper
        layers (which start zero-initialized) can be trained at a higher LR
        than the pre-trained base UNet.
        """
        yield from self.patch_proj.parameters()
        yield from self.k_proj.parameters()
        yield from self.v_proj.parameters()
        yield from self.attn.parameters()
        yield from self.out_proj.parameters()

    def forward(self, x: torch.Tensor, t: torch.Tensor, *, cond: torch.Tensor, **kwargs):
        b, _c, h, w = x.shape
        ps = self.patch_size
        if h % ps != 0 or w % ps != 0:
            raise ValueError(
                f"Image size ({h}×{w}) is not divisible by patch_size={ps}. "
                f"Set patch_size to a divisor of img_size in the cross_attention config."
            )
        # Patchify: (B, 3, H, W) → (B, embed_dim, H/ps, W/ps) → (B, N, embed_dim)
        patches = self.patch_proj(x)
        ph, pw = patches.shape[2], patches.shape[3]
        tokens = patches.permute(0, 2, 3, 1).reshape(b, ph * pw, self.embed_dim)
        # Cross-attention: image tokens (Q) attend to genomic cond (K, V — single token)
        k = self.k_proj(cond).unsqueeze(1)  # (B, 1, embed_dim)
        v = self.v_proj(cond).unsqueeze(1)
        attn_out, _ = self.attn(tokens, k, v)  # (B, N, embed_dim)
        # Decode to image-space residual
        delta_flat = self.out_proj(attn_out)  # (B, N, ps*ps*3)
        delta = (
            delta_flat.reshape(b, ph, pw, ps, ps, 3)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(b, 3, h, w)
        )
        x_mod = x + delta
        # Pass modified input and original cond to the base UNet (AdaGN path unchanged)
        return self.base_unet(x=x_mod, t=t, cond=cond, **kwargs)


class CrossAttentionJointLitModel(JointLitModelBase):
    """Joint model with patchified cross-attn UNet and conditioning/noise dropouts.

    Inherits encoder, projection, dataset, and optimizer from JointLitModel.
    Overrides:
      - ``__init__``: wraps UNet with CrossAttentionUNetWrapper and re-initialises
        EMA to track the full wrapped model.
      - ``training_step``: adds cond dropout, feature dropout, x_T dropout, and
        an optional counterfactual margin loss.
      - ``on_train_batch_end``: runs EMA update on the full wrapped model.
    """

    def __init__(self, conf, joint_cfg: dict, n_genes: int):
        cross_cfg = joint_cfg.get("cross_attention", {})
        heads = int(cross_cfg.get("heads", 4))
        dim_head = int(cross_cfg.get("dim_per_head", 64))
        patch_size = int(cross_cfg.get("patch_size", 16))
        super().__init__(conf, joint_cfg, n_genes)
        cond_dim = int(joint_cfg.get("cond_dim", conf.feat_dim))

        self.cross_cfg = {
            "heads": heads,
            "dim_head": dim_head,
            "patch_size": patch_size,
            "xT_dropout_prob": float(joint_cfg.get("xT_dropout_prob", 0.05)),
            "cond_dropout_prob": float(joint_cfg.get("cond_dropout_prob", 0.05)),
            "cond_feature_dropout": float(joint_cfg.get("cond_feature_dropout", 0.05)),
            "counterfactual_loss_weight": float(joint_cfg.get("counterfactual_loss_weight", 0.0)),
            "counterfactual_margin": float(joint_cfg.get("counterfactual_margin", 0.02)),
            "counterfactual_monitor_every_n_steps": int(joint_cfg.get("counterfactual_monitor_every_n_steps", 200)),
            "counterfactual_zero_threshold": float(joint_cfg.get("counterfactual_zero_threshold", 1e-4)),
        }

        # Wrap base UNet with patchified cross-attention
        self.model = CrossAttentionUNetWrapper(
            self.model, cond_dim=cond_dim, patch_size=patch_size, heads=heads, dim_head=dim_head
        )
        # Re-initialise EMA to track the FULL wrapped model (cross-attn layers included).
        # The EMA model initialised in mopadi's LitModel.__init__ only tracked the base
        # UNet; replacing it here ensures cross-attention weights are also smoothed by EMA
        # and available at inference time.
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        self.save_hyperparameters({
            "cross_cfg": self.cross_cfg,
            "joint_variant": "cross_attention_joint_training",
        })

    def configure_optimizers(self):
        """Override to give the cross-attention wrapper layers their own LR group.

        The wrapper layers (patch_proj, k_proj, v_proj, attn, out_proj) start
        zero-initialized or from scratch and need a higher LR than the base UNet.
        ``cross_attn_lr`` in joint_cfg controls this; it defaults to ``encoder_lr``.
        """
        conf = self.conf
        jcfg = self.joint_cfg
        lr = float(conf.lr)
        cross_attn_lr = float(jcfg.get("cross_attn_lr", jcfg.get("encoder_lr", lr)))

        param_groups = [
            {"params": list(self.model.base_unet.parameters()), "lr": float(jcfg.get("unet_lr", lr))},
            {"params": list(self.model.wrapper_parameters()), "lr": cross_attn_lr},
            {"params": list(self.encoder.parameters()), "lr": float(jcfg.get("encoder_lr", lr))},
            {"params": list(self.projection.parameters()), "lr": float(jcfg.get("proj_lr", lr))},
        ]

        if conf.optimizer == OptimizerType.adamw:
            optim = torch.optim.AdamW(
                param_groups, betas=(0.9, 0.99), eps=1e-6,
                weight_decay=conf.weight_decay,
            )
        else:
            optim = torch.optim.Adam(param_groups, weight_decay=conf.weight_decay)

        epochs = int(jcfg.get("epochs", getattr(conf, "max_epochs", 100)))
        total_steps = max(1, epochs * int(conf.steps_per_epoch))
        warmup_steps = int(conf.warmup)

        if warmup_steps > 0:
            warmup = LambdaLR(optim, lr_lambda=lambda s: min(s + 1, warmup_steps) / warmup_steps)
            cosine = CosineAnnealingLR(optim, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6)
            sched = SequentialLR(optim, schedulers=[warmup, cosine], milestones=[warmup_steps])
        else:
            sched = CosineAnnealingLR(optim, T_max=total_steps, eta_min=1e-6)

        return {"optimizer": optim, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def training_step(self, batch, batch_idx):
        with autocast(device_type='cuda', enabled=self.conf.fp16):
            imgs = batch['img'].to(self.device)
            genomic = batch['genomic'].to(self.device, dtype=torch.float32)

            # Stochastic encoding (reparameterisation trick) during training acts
            # as an additional regulariser on the genomic latent space.
            cond = self.encode_genomic(genomic)

            # ── Conditioning dropout ──────────────────────────────────────
            p_cond = self.cross_cfg.get("cond_dropout_prob", 0.0)
            if p_cond > 0:
                mask = torch.rand(cond.shape[0], device=cond.device) < p_cond
                if mask.any():
                    cond = cond.clone()
                    cond[mask] = 0

            p_feat = self.cross_cfg.get("cond_feature_dropout", 0.0)
            if p_feat > 0:
                cond = F.dropout(cond, p=p_feat, training=self.training)

            # ── x_T dropout (curriculum) ──────────────────────────────────
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
                model_kwargs={"cond": cond},
            )
            main_loss_per_sample = losses['loss']
            main_loss = main_loss_per_sample.mean()

            # ── Counterfactual margin loss ────────────────────────────────
            cf_weight = float(self.cross_cfg.get("counterfactual_loss_weight", 0.0))
            cf_margin_loss = torch.zeros((), device=imgs.device, dtype=main_loss.dtype)
            swap_gap = torch.zeros((), device=imgs.device, dtype=main_loss.dtype)
            swap_idx = _build_swapped_indices(cond.shape[0], cond.device)
            if cf_weight > 0.0 and swap_idx is not None:
                cond_swapped = cond[swap_idx]
                losses_swapped = self.sampler.training_losses(
                    model=self.model,
                    x_start=x_start,
                    cond=cond_swapped,
                    t=t,
                    model_kwargs={"cond": cond_swapped},
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
        """EMA update for the full wrapped model (cross-attn layers included)."""
        from mopadi.train_diff_autoenc import ema

        if self.is_last_accum(batch_idx):
            # EMA tracks the full CrossAttentionUNetWrapper, not just base_unet.
            ema(self.model, self.ema_model, self.conf.ema_decay)

            with torch.no_grad():
                genomic = batch['genomic'].to(self.device, dtype=torch.float32)
                cond = self.encode_genomic(genomic)

            self.log_sample(x_start=batch['img'], cond=cond)
            self.evaluate_scores()


def build_cross_conf(joint_cfg: dict):
    """Reuse build_conf from joint_training (helper for symmetry)."""
    return build_conf(joint_cfg)
