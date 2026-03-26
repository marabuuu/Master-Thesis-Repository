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
            loss = losses["loss"].mean()

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
