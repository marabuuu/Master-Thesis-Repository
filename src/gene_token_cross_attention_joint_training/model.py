from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

try:
    from gene_token_transformer_joint_training.model import (
        GeneTokenTransformerJointLitModel,
        build_gene_token_transformer_conf,
    )
except ImportError:  # pragma: no cover
    from src.gene_token_transformer_joint_training.model import (  # type: ignore[import-not-found]
        GeneTokenTransformerJointLitModel,
        build_gene_token_transformer_conf,
    )

try:
    from cross_attention_joint_training.model import CrossAttentionUNetWrapper
except ImportError:  # pragma: no cover
    from src.cross_attention_joint_training.model import CrossAttentionUNetWrapper  # type: ignore[import-not-found]

try:
    from mopadi.configs.choices import OptimizerType
except ImportError:  # pragma: no cover
    from configs.choices import OptimizerType  # type: ignore[import-not-found]


def _build_swapped_indices(batch_size: int, device: torch.device):
    if batch_size < 2:
        return None
    shift = int(torch.randint(1, batch_size, (1,), device=device).item())
    return torch.roll(torch.arange(batch_size, device=device), shifts=shift)


class GeneTokenCrossAttentionJointLitModel(GeneTokenTransformerJointLitModel):  # type: ignore[misc]
    """Hybrid model: gene-token transformer encoder + patchified cross-attention UNet wrapper.

    Combines:
      - GeneTokenTransformerJointLitModel: treats each gene as a token and uses a
        transformer encoder to capture gene-gene interactions.
      - CrossAttentionUNetWrapper: patchifies the UNet input via a learned Conv2d and
        lets spatial tokens attend to the genomic conditioning vector, allowing
        spatially non-uniform conditioning beyond the global AdaGN path.

    EMA tracks the full wrapped model (transformer encoder weights are not in
    the UNet wrapper, but the cross-attention wrapper layers are now properly
    included in EMA via re-initialisation after wrapping).
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
            "genomic_recon_loss_weight": float(joint_cfg.get("genomic_recon_loss_weight", 0.0)),
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

        # Genomic auto-reconstruction decoder.
        # A small MLP that maps the conditioning vector back to the original gene
        # expression space.  Training it jointly forces the transformer encoder +
        # cond_projection to preserve gene-level information instead of collapsing
        # to a degenerate constant.  Activated when genomic_recon_loss_weight > 0.
        self.genomic_decoder = nn.Sequential(
            nn.Linear(cond_dim, self.gtt_cfg.d_model),
            nn.GELU(),
            nn.Linear(self.gtt_cfg.d_model, n_genes),
        )

        self.save_hyperparameters({
            "cross_cfg": self.cross_cfg,
            "joint_variant": "gene_token_cross_attention_joint_training",
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
            {"params": list(self.gene_token_encoder.parameters()), "lr": float(jcfg.get("encoder_lr", lr))},
            {"params": list(self.cond_projection.parameters()), "lr": float(jcfg.get("proj_lr", lr))},
            {"params": list(self.genomic_decoder.parameters()), "lr": float(jcfg.get("proj_lr", lr))},
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
        with autocast(device_type="cuda", enabled=self.conf.fp16):
            imgs = batch["img"].to(self.device)
            genomic = batch["genomic"].to(self.device, dtype=torch.float32)

            # Stochastic encoding (reparameterisation trick) during training acts
            # as an additional regulariser on the genomic latent space.
            cond = self.encode_genomic(genomic)

            # ── Genomic auto-reconstruction loss (before any dropout) ─────
            # Computed on the clean cond so dropout doesn't corrupt supervision.
            # Forces the encoder to preserve gene-level information.
            genomic_recon_weight = float(self.cross_cfg.get("genomic_recon_loss_weight", 0.0))
            genomic_recon_loss = torch.zeros((), device=imgs.device, dtype=cond.dtype)
            if genomic_recon_weight > 0.0:
                genomic_pred = self.genomic_decoder(cond)
                genomic_recon_loss = F.mse_loss(genomic_pred, genomic)

            # ── Three mutually exclusive training modes ───────────────────
            # A single draw assigns each sample to exactly one mode so that
            # cond-dropout and xT-forcing never cancel each other out:
            #
            #   [0, p_cond)            cond dropout  → trains unconditional (CFG)
            #   [p_cond, p_cond+p_xt)  xT forcing    → t near-T, x_t≈N(0,I),
            #                                           model must use genomic cond
            #   [p_cond+p_xt, 1.0)     normal        → standard reconstruction
            p_cond = float(self.cross_cfg.get("cond_dropout_prob", 0.0))
            p_xt   = float(self.cross_cfg.get("xT_dropout_prob", 0.0))
            r = torch.rand(cond.shape[0], device=cond.device)
            cond_drop_mask = r < p_cond
            xt_force_mask  = (r >= p_cond) & (r < p_cond + p_xt)

            if cond_drop_mask.any():
                cond = cond.clone()
                cond[cond_drop_mask] = 0

            p_feat = float(self.cross_cfg.get("cond_feature_dropout", 0.0))
            if p_feat > 0:
                cond = F.dropout(cond, p=p_feat, training=self.training)

            # ── xT forcing: override t → near-T so x_t ≈ N(0,I) ─────────
            x_start = imgs
            t, _ = self.T_sampler.sample(len(imgs), imgs.device)
            if xt_force_mask.any():
                T_max = int(getattr(self.conf, "T", 1000))
                t_high = torch.randint(
                    int(T_max * 0.8), T_max,
                    (int(xt_force_mask.sum().item()),),
                    device=imgs.device,
                )
                t = t.clone()
                t[xt_force_mask] = t_high

            # Sample noise ONCE and reuse for both main and CF forward passes.
            # Using the same noise ensures x_t is identical in both passes, so
            # (swapped_loss - main_loss) reflects only the conditioning difference,
            # not variance from independent noise draws.  With independent noise the
            # CF loss expectation is dominated by variance and collapses toward 0.
            shared_noise = torch.randn_like(x_start)

            losses = self.sampler.training_losses(
                model=self.model,
                x_start=x_start,
                cond=cond,
                t=t,
                noise=shared_noise,
                model_kwargs={"cond": cond},
            )
            main_loss_per_sample = losses["loss"]
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
                    noise=shared_noise,  # same x_t, only cond differs
                    model_kwargs={"cond": cond_swapped},
                )
                swapped_loss_per_sample = losses_swapped["loss"]
                margin = float(self.cross_cfg.get("counterfactual_margin", 0.02))
                cf_margin_loss = F.relu(
                    margin + main_loss_per_sample.detach() - swapped_loss_per_sample
                ).mean()
                swap_gap = (swapped_loss_per_sample.detach() - main_loss_per_sample.detach()).mean()

            loss = main_loss + cf_weight * cf_margin_loss + genomic_recon_weight * genomic_recon_loss

        self.log("loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(imgs))
        self.log("loss_step", loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=len(imgs))
        self.log("loss_main_step", main_loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=len(imgs))
        self.log("loss_cf_margin_step", cf_margin_loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=len(imgs))
        self.log("cf_swap_gap_step", swap_gap, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=len(imgs))
        self.log("loss_genomic_recon_step", genomic_recon_loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=len(imgs))

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
                message += " | GOOD: swapped loss exceeds matched by >= margin (model uses conditioning)"
            else:
                message += " | LEARNING: cf_margin>0 means model not yet differentiating conditioning"
            print(message)

        if self.global_rank == 0 and hasattr(self, "logger") and hasattr(self.logger, "experiment"):
            self.logger.experiment.add_scalar("loss", loss.item(), self.num_samples)  # type: ignore[union-attr]

        return {"loss": loss}

    def on_fit_start(self):
        super().on_fit_start()
        self.genomic_decoder = self.genomic_decoder.to(self.device)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """EMA update for the full wrapped model (cross-attn layers included)."""
        from mopadi.train_diff_autoenc import ema

        if self.is_last_accum(batch_idx):
            # EMA tracks the full CrossAttentionUNetWrapper, not just base_unet.
            ema(self.model, self.ema_model, self.conf.ema_decay)

            with torch.no_grad():
                genomic = batch["genomic"].to(self.device, dtype=torch.float32)
                cond = self.encode_genomic(genomic)

            self.log_sample(x_start=batch["img"], cond=cond)
            self.evaluate_scores()


def build_gene_token_cross_attention_conf(joint_cfg: dict):
    """Reuse gene-token baseline config builder for diffusion/training defaults."""
    return build_gene_token_transformer_conf(joint_cfg)
