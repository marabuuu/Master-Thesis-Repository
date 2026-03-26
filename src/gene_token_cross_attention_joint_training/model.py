from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast

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


def _build_swapped_indices(batch_size: int, device: torch.device):
    if batch_size < 2:
        return None
    shift = int(torch.randint(1, batch_size, (1,), device=device).item())
    return torch.roll(torch.arange(batch_size, device=device), shifts=shift)


class GeneTokenCrossAttentionJointLitModel(GeneTokenTransformerJointLitModel):  # type: ignore[misc]
    """Hybrid model: gene-token transformer encoder + cross-attention UNet wrapper."""

    def __init__(self, conf, joint_cfg: dict, n_genes: int):
        cross_cfg = joint_cfg.get("cross_attention", {})
        heads = int(cross_cfg.get("heads", 4))
        dim_head = int(cross_cfg.get("dim_per_head", 64))
        cond_dims = [int(x) for x in cross_cfg.get("cond_dims", [512, 256, 128])]

        super().__init__(conf, joint_cfg, n_genes)

        cond_dim = int(joint_cfg.get("cond_dim", conf.feat_dim))
        if cond_dims and cond_dims[0] != cond_dim:
            raise ValueError(
                f"cross_attention.cond_dims[0] ({cond_dims[0]}) must match cond_dim ({cond_dim})"
            )

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

        self.model = CrossAttentionUNetWrapper(
            self.model, cond_dims=cond_dims, heads=heads, dim_head=dim_head
        )
        self.save_hyperparameters({
            "cross_cfg": self.cross_cfg,
            "joint_variant": "gene_token_cross_attention_joint_training",
        })

    def training_step(self, batch, batch_idx):
        with autocast(device_type="cuda", enabled=self.conf.fp16):
            imgs = batch["img"].to(self.device)
            genomic = batch["genomic"].to(self.device, dtype=torch.float32)

            cond = self.encode_genomic(genomic)

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
            main_loss_per_sample = losses["loss"]
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
                swapped_loss_per_sample = losses_swapped["loss"]
                margin = float(self.cross_cfg.get("counterfactual_margin", 0.02))
                cf_margin_loss = F.relu(
                    margin + main_loss_per_sample.detach() - swapped_loss_per_sample
                ).mean()
                swap_gap = (swapped_loss_per_sample.detach() - main_loss_per_sample.detach()).mean()

            loss = main_loss + cf_weight * cf_margin_loss

        self.log(
            "loss_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(imgs),
        )
        self.log(
            "loss_step",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
            batch_size=len(imgs),
        )
        self.log(
            "loss_main_step",
            main_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
            batch_size=len(imgs),
        )
        self.log(
            "loss_cf_margin_step",
            cf_margin_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
            batch_size=len(imgs),
        )
        self.log(
            "cf_swap_gap_step",
            swap_gap,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
            batch_size=len(imgs),
        )

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

        if self.global_rank == 0 and hasattr(self, "logger") and hasattr(self.logger, "experiment"):
            self.logger.experiment.add_scalar("loss", loss.item(), self.num_samples)  # type: ignore[union-attr]

        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """EMA update for wrapped UNet model."""
        from mopadi.train_diff_autoenc import ema

        if self.is_last_accum(batch_idx):
            ema(self.model.base_unet, self.ema_model, self.conf.ema_decay)

            with torch.no_grad():
                genomic = batch["genomic"].to(self.device, dtype=torch.float32)
                cond = self.encode_genomic(genomic)

            self.log_sample(x_start=batch["img"], cond=cond)
            self.evaluate_scores()


def build_gene_token_cross_attention_conf(joint_cfg: dict):
    """Reuse gene-token baseline config builder for diffusion/training defaults."""
    return build_gene_token_transformer_conf(joint_cfg)
