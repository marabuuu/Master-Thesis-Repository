"""
GenomicCaLitModel — MoPaDi with bottleneck cross-attention + classifier-free guidance.

Conditioning mechanism
----------------------
FiLM (AdaGN in every ResBlock) is permanently disabled: _stop_cond_grad=True zeroes
cond_out so the backbone is unconditional through FiLM.  All genomic conditioning
flows exclusively through the bottleneck GenomicCrossAttentionBlock (CA).

Training
--------
Classifier-free guidance (CFG) dropout: cfg_dropout fraction of batches replace
feats with zeros.  The model learns both conditional and unconditional denoising.
Single forward + backward per step (no two-pass split needed).

Sampling
--------
Run the model twice: once with real feats, once with null (zeros).
  eps_guided = eps_null + guidance_scale × (eps_cond - eps_null)
Scale > 1 amplifies the genomic conditioning effect.

Diagnostics
-----------
cond/guidance_delta = MSE(eps_cond, eps_null) logged every 500 steps.
Should grow monotonically as CA learns subtype-specific h_mid modifications.
If it grows, conditioning is working.  No more pred_gap / CFL.
"""
from __future__ import annotations

import copy
import logging

import torch
import torch.nn.functional as F

from src.mopadi_genomic_crossattn.genomic_train import GenomicLitModel as _BaseGenomicLitModel
from .genomic_config import GenomicCaConfig
from .genomic_cross_attn import GenomicCrossAttentionBlock

log = logging.getLogger(__name__)


class GenomicCaLitModel(_BaseGenomicLitModel):
    """MoPaDi + bottleneck CA + classifier-free guidance training."""

    automatic_optimization = False

    def __init__(self, conf: GenomicCaConfig):
        super().__init__(conf)
        self.conf: GenomicCaConfig = conf

        if conf.use_genomic_cross_attn:
            bottleneck_ch = conf.net_ch * max(conf.net_ch_mult)
            ca = GenomicCrossAttentionBlock(
                spatial_channels=bottleneck_ch,
                gene_dim=conf.feat_dim,
                n_heads=conf.genomic_ca_heads,
                n_gene_tokens=conf.genomic_ca_n_tokens,
            )
            self.model.genomic_cross_attn = ca
            self.ema_model.genomic_cross_attn = copy.deepcopy(ca)
            self.ema_model.requires_grad_(False)
            log.info(
                "GenomicCrossAttentionBlock: bottleneck_ch=%d, gene_dim=%d, "
                "heads=%d, n_tokens=%d",
                bottleneck_ch, conf.feat_dim, conf.genomic_ca_heads, conf.genomic_ca_n_tokens,
            )

        # Permanently disable AdaGN FiLM — CA is the sole conditioning path.
        for mod in list(self.model.modules()) + list(self.ema_model.modules()):
            if hasattr(mod, 'cond_emb_layers'):
                mod._stop_cond_grad = True

    def training_step(self, batch, batch_idx):
        """Single-pass reconstruction with CFG dropout.

        cfg_dropout fraction of batches use feats=zeros (null conditioning).
        Reconstruction gradient trains CA to produce subtype-specific h_mid
        modifications that improve denoising — no competing pred_gap objective.
        """
        opt = self.optimizers()
        accum = self.conf.accum_batches

        batch, _bag_n = self._flatten_bag_batch(batch)
        imgs = batch["img"].to(self.device)
        feats = batch["feat"].to(self.device, dtype=torch.float32)

        # CFG dropout: replace feats with zeros for cfg_dropout fraction of samples.
        cfg_dropout = float(self.conf.cfg_dropout)
        if cfg_dropout > 0:
            null_mask = torch.rand(len(feats), device=feats.device) < cfg_dropout
            feats_train = feats.clone()
            feats_train[null_mask] = 0.0
        else:
            feats_train = feats

        # ── Single reconstruction pass ────────────────────────────────────────
        with torch.autocast(device_type="cuda", enabled=self.conf.fp16):
            t, _ = self.T_sampler.sample(len(imgs), imgs.device)
            losses = self.sampler.training_losses(
                model=self.model,
                x_start=imgs,
                cond=feats_train,
                t=t,
                model_kwargs={"cond": feats_train},
            )
            main_loss = losses["loss"].mean()

        self.manual_backward(main_loss / accum)

        if self.is_last_accum(batch_idx):
            if self.conf.grad_clip > 0:
                self.clip_gradients(
                    opt,
                    gradient_clip_val=self.conf.grad_clip,
                    gradient_clip_algorithm="norm",
                )
            opt.step()
            opt.zero_grad()

        # ── Logging ───────────────────────────────────────────────────────────
        if self.global_rank == 0:
            ns = self.num_samples
            self.logger.experiment.add_scalar("loss", main_loss.item(), ns)
            self.logger.experiment.add_scalar("loss/train", main_loss.item(), ns)

            # guidance_delta: measures how much the CA block changes model output
            # relative to null conditioning.  Should grow monotonically if CA is
            # learning to use genomic features.  Key diagnostic — watch this.
            if self.global_step % 500 == 0:
                n_diag = min(4, len(imgs))
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", enabled=self.conf.fp16):
                        T = self.conf.T
                        t_diag = torch.randint(
                            T // 2, T, (n_diag,), device=imgs.device
                        )
                        noise_diag = torch.randn_like(imgs[:n_diag])
                        x_t_diag = self.sampler.q_sample(
                            imgs[:n_diag].detach(), t_diag, noise=noise_diag
                        )
                        t_scaled = self.sampler._scale_timesteps(t_diag)
                        eps_cond = self.model.forward(
                            x=x_t_diag, t=t_scaled, x_start=None,
                            cond=feats[:n_diag],
                        ).pred
                        feats_null = torch.zeros_like(feats[:n_diag])
                        eps_null = self.model.forward(
                            x=x_t_diag, t=t_scaled, x_start=None,
                            cond=feats_null,
                        ).pred
                        guidance_delta = F.mse_loss(eps_cond, eps_null)
                self.logger.experiment.add_scalar(
                    "cond/guidance_delta", guidance_delta.item(), ns
                )

            if self.global_step % 500 == 0:
                log.info(
                    "step %6d | samples %10d | loss %.4f",
                    self.global_step, self.num_samples, main_loss.item(),
                )

        self.log("loss", main_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log("loss/train", main_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return {"loss": main_loss}
