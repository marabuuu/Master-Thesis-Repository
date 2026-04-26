"""
GenomicCrossAttnLitModel — MoPaDi genomic training with patchified cross-attention.

Extends GenomicLitModel with two additions:

1. CrossAttentionUNetWrapper: wraps self.model so that, before each UNet
   forward pass, image patches attend to the genomic conditioning vector as a
   single K/V token and a zero-initialised residual is added to the input.
   Style conditioning (AdaGN) is unchanged and provides the global genomic
   signal as before; the cross-attention layer adds spatial specificity on top.

2. Genomic-guided high-t loss: in addition to the standard diffusion loss
   (L1, computed by the parent at uniformly sampled t), a second forward pass
   at t ∈ [high_t_frac·T, T) is computed and added as λ·L2.  At these high
   timesteps x_t ≈ N(0,I) so the model cannot rely on image content and must
   use the genomic conditioning to predict the noise direction.  This creates
   an explicit genomic-guided learning regime.

Total loss = L1 + λ·L2, with λ = conf.genomic_guided_loss_weight.
"""

from __future__ import annotations

import copy
import logging

import torch

try:
    from mopadi_genomic.train import GenomicLitModel
    from cross_attention_joint_training.model import CrossAttentionUNetWrapper
    from mopadi_genomic_crossattn.config import GenomicCrossAttnConfig
except ImportError:
    from src.mopadi_genomic.train import GenomicLitModel
    from src.cross_attention_joint_training.model import CrossAttentionUNetWrapper
    from src.mopadi_genomic_crossattn.config import GenomicCrossAttnConfig

log = logging.getLogger(__name__)


class GenomicCrossAttnLitModel(GenomicLitModel):
    """MoPaDi genomic diffusion model with patchified cross-attention + dual loss.

    Inherits from GenomicLitModel:
      - setup(): creates ZipTilesWithGenomicFeatures datasets (unchanged)
      - val_dataloader(): validation loader with batch cap (unchanged)
      - validation_step(): logs loss/val, loss/val_shuffled, cond/gap (unchanged)
      - on_validation_epoch_end(): per-epoch validation summary (unchanged)
      - on_fit_start(): sanity check on ZIP dataset (unchanged)
      - evaluate_scores(): no-op (unchanged)
      - log_sample(): skip at num_samples==0 (unchanged)

    Overrides:
      - __init__: wraps self.model with CrossAttentionUNetWrapper, re-inits EMA
      - configure_optimizers: splits params into UNet / cross-attn LR groups
      - training_step: parent L1 + genomic-guided high-t L2
    """

    def __init__(self, conf: GenomicCrossAttnConfig):
        super().__init__(conf)
        self.conf: GenomicCrossAttnConfig = conf

        # Wrap the base UNet with cross-attention.  The wrapper's out_proj is
        # zero-initialised so training starts from the identity — the attention
        # path is learned gradually without destabilising the UNet.
        self.model = CrossAttentionUNetWrapper(
            base_unet=self.model,
            cond_dim=conf.style_ch,
            patch_size=conf.cross_attn_patch_size,
            heads=conf.cross_attn_heads,
            dim_head=conf.cross_attn_dim_per_head,
        )

        # Re-initialise EMA to track the full wrapped model (base_unet +
        # cross-attn layers).  Without this, EMA would point to the unwrapped
        # model and the on_train_batch_end EMA update would fail silently.
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        # Exclude EMA from DDP's initial parameter broadcast and gradient buckets.
        # EMA starts as a deepcopy of self.model, which DDP already syncs from
        # rank 0 — re-broadcasting 139 M extra params is redundant and triggers
        # a CUDA illegal-memory-access crash during _sync_params_and_buffers on
        # multi-GPU setups.  ema_model remains a registered submodule so PL
        # still includes it in checkpoints automatically.
        self._ddp_params_and_buffers_to_ignore = [
            f"ema_model.{n}" for n, _ in self.ema_model.named_parameters()
        ] + [
            f"ema_model.{n}" for n, _ in self.ema_model.named_buffers()
        ]

        log.info(
            "CrossAttentionUNetWrapper applied: patch_size=%d, heads=%d, dim_per_head=%d",
            conf.cross_attn_patch_size,
            conf.cross_attn_heads,
            conf.cross_attn_dim_per_head,
        )

    # ------------------------------------------------------------------
    # Optimizer: separate LR groups for UNet and cross-attention wrapper
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Split parameters into two LR groups.

        UNet params: trained at unet_lr.  When training from scratch this
        equals conf.lr; reduce it only when warm-starting from a pre-trained
        diffusion checkpoint.

        Wrapper params (patch_proj / k_proj / v_proj / attn / out_proj):
        trained at cross_attn_lr.  These layers are zero-initialised and need
        a comparable or higher LR to learn meaningful attention patterns within
        a reasonable number of steps.
        """
        unet_lr = float(getattr(self.conf, "unet_lr", self.conf.lr))
        cross_attn_lr = float(getattr(self.conf, "cross_attn_lr", self.conf.lr))

        param_groups = [
            {"params": list(self.model.base_unet.parameters()), "lr": unet_lr},
            {"params": list(self.model.wrapper_parameters()), "lr": cross_attn_lr},
        ]

        optim = torch.optim.Adam(param_groups, weight_decay=self.conf.weight_decay)
        return {"optimizer": optim}

    # ------------------------------------------------------------------
    # Training step: L1 (standard diffusion) + L2 (genomic-guided high-t)
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """Compute dual loss: standard diffusion (L1) + high-t genomic (L2).

        L1 is computed by GenomicLitModel.training_step at uniformly sampled t,
        logged as loss/train.  L2 is an additional forward pass at t ∈ [high_t, T)
        where x_t ≈ N(0,I) — the model cannot exploit image structure and must
        use the genomic conditioning to predict noise.  Logged as loss/genomic_guided.

        Total backprop loss = L1 + λ·L2, with λ = genomic_guided_loss_weight.
        """
        # L1: standard diffusion loss at uniform t (image-guided regime)
        out = super().training_step(batch, batch_idx)

        genomic_weight = float(getattr(self.conf, "genomic_guided_loss_weight", 0.0))
        if genomic_weight <= 0.0:
            return out

        # L2: denoising at high t (genomic-guided regime)
        imgs = batch["img"].to(self.device)
        feats = batch["feat"].to(self.device, dtype=torch.float32)

        high_t_frac = float(getattr(self.conf, "genomic_guided_high_t_frac", 0.8))
        T = self.conf.T
        t_lo = int(high_t_frac * T)
        high_t = torch.randint(t_lo, T, (len(imgs),), device=imgs.device)

        losses_high_t = self.sampler.training_losses(
            model=self.model,
            x_start=imgs,
            cond=feats,
            t=high_t,
            model_kwargs={"cond": feats},
        )
        genomic_loss = losses_high_t["loss"].mean()

        loss_l1 = out["loss"] if isinstance(out, dict) else out
        total_loss = loss_l1 + genomic_weight * genomic_loss

        if isinstance(out, dict):
            out["loss"] = total_loss
        else:
            out = total_loss

        self.log(
            "loss/genomic_guided",
            genomic_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

        return out
