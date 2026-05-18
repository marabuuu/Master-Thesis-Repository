"""
GDALitModel — jointly trains backbone UNet + GenomicResidualAdapter from scratch.

Training mechanics
------------------
Backbone UNet  : always receives  cond = zeros(B, style_ch)
                 → learns unconditional denoising
                 → CANNOT see patient genomic features → CANNOT suppress adapter

Adapter        : receives (x_t, t, g_tokens) → predicts Δε
                 → CFG null dropout (cfg_dropout %) replaces g_tokens with null_token

Combined loss  : MSE(ε_backbone + Δε_adapter, noise)

Both components receive gradients from the same loss and train jointly.
The backbone optimises the unconditional component; the adapter learns the
genomic-specific residual. No pretrained checkpoint is loaded.

Monitoring
----------
cond/guidance_delta = E[‖Δε_own − Δε_null‖²]  logged every 500 steps.
This is the primary health indicator: it must grow as the adapter learns
subtype-specific corrections.  If it stays flat the adapter is not learning
a conditioning signal (independent of the backbone — backbone cannot cause this).

Inference (CFG)
---------------
ε_guided = ε_backbone(x_t, t, 0) + scale * (Δε_own − Δε_null)
where ε_backbone uses EMA weights and Δε uses EMA adapter weights.
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn.functional as F
from mopadi.utils.dist_utils import get_world_size

from src.drafts.mopadi_genomic.train import GenomicLitModel as _BaseGenomicLitModel

from .adapter import GenomicResidualAdapter, GenomicTokenEncoder
from .config import GDAConfig

log = logging.getLogger(__name__)


def _ema_update(ema_params, model_params, decay: float) -> None:
    with torch.no_grad():
        for p_ema, p in zip(ema_params, model_params):
            p_ema.mul_(decay).add_(p.data, alpha=1.0 - decay)


class GDALitModel(_BaseGenomicLitModel):
    """
    Jointly trains backbone UNet (unconditional) + genomic residual adapter.

    The backbone never receives genomic features (always cond=zeros),
    so it cannot learn to suppress the adapter.  The adapter is the sole
    pathway for patient-specific information → its guidance_delta cannot
    be suppressed by backbone gradient competition.
    """

    automatic_optimization = False

    def __init__(self, conf: GDAConfig):
        super().__init__(conf)
        self.conf: GDAConfig = conf

        # ── Genomic token encoder ─────────────────────────────────────────
        self.genomic_encoder = GenomicTokenEncoder(
            genomic_in=conf.feat_dim,
            n_tokens=conf.adapter_n_tokens,
            token_dim=conf.adapter_token_dim,
        )

        # Null token for CFG: learned, initialised to zeros
        self.null_token = torch.nn.Parameter(
            torch.zeros(conf.adapter_n_tokens, conf.adapter_token_dim)
        )

        # ── Adapter ───────────────────────────────────────────────────────
        self.adapter = GenomicResidualAdapter(
            in_ch=3,
            base_ch=conf.adapter_base_ch,
            t_dim=conf.adapter_t_dim,
            token_dim=conf.adapter_token_dim,
            n_heads=conf.adapter_n_heads,
        )

        # ── EMA for adapter + genomic encoder ─────────────────────────────
        # Backbone EMA is managed by the parent (self.ema_model).
        self._ema_adapter = copy.deepcopy(self.adapter)
        self._ema_adapter.requires_grad_(False)
        self._ema_genomic_encoder = copy.deepcopy(self.genomic_encoder)
        self._ema_genomic_encoder.requires_grad_(False)
        self._ema_null_token = copy.deepcopy(self.null_token.data)

        n_adapter = sum(p.numel() for p in self.adapter.parameters())
        n_enc = sum(p.numel() for p in self.genomic_encoder.parameters())
        log.info(
            "GDALitModel: adapter params=%d  genomic_encoder params=%d  "
            "null_token shape=%s",
            n_adapter, n_enc, list(self.null_token.shape),
        )

    # ------------------------------------------------------------------
    # Sample counter
    # ------------------------------------------------------------------

    @property
    def num_samples(self) -> int:
        # With automatic_optimization=False the Trainer uses accumulate_grad_batches=1,
        # so global_step counts every micro-batch (not every effective optimizer step).
        # The parent formula (global_step * batch_size_effective) would overcount by
        # accum_batches; use batch_size (= per-micro-batch global count) instead.
        return self.global_step * self.conf.batch_size

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_bag_batch(batch: dict) -> tuple:
        """Flatten a bag batch (B, N, C, H, W) → (B*N, C, H, W).

        Returns (flat_batch, N) where N=1 if the input was already flat.
        feat is tiled so each flattened image carries its patient's vector.
        """
        img = batch["img"]
        if img.dim() == 5:
            B, N, C, H, W = img.shape
            flat = dict(batch)
            flat["img"] = img.reshape(B * N, C, H, W)
            feat = batch["feat"]
            flat["feat"] = feat.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
            return flat, N
        return batch, 1

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt_backbone = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.conf.backbone_lr,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )
        adapter_params = (
            list(self.adapter.parameters())
            + list(self.genomic_encoder.parameters())
            + [self.null_token]
        )
        opt_adapter = torch.optim.AdamW(
            adapter_params,
            lr=self.conf.adapter_lr,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )
        return [opt_backbone, opt_adapter], []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        opt_backbone, opt_adapter = self.optimizers()
        accum = self.conf.accum_batches

        batch, _ = self._flatten_bag_batch(batch)
        imgs = batch["img"].to(self.device)
        feats = batch["feat"].to(self.device, dtype=torch.float32)
        B = imgs.shape[0]

        # ── Noise + noisy image ───────────────────────────────────────────
        t, _ = self.T_sampler.sample(B, imgs.device)
        noise = torch.randn_like(imgs)
        x_t = self.sampler.q_sample(imgs, t, noise=noise)

        # Backbone always receives zeros — unconditional path
        zeros_cond = torch.zeros(B, self.conf.feat_dim, device=self.device, dtype=torch.float32)

        # ── CFG null dropout for adapter ──────────────────────────────────
        null_mask = torch.rand(B, device=self.device) < self.conf.cfg_dropout
        g_tokens = self.genomic_encoder(feats)                          # (B, n, d)
        null_expanded = self.null_token.unsqueeze(0).expand(B, -1, -1) # (B, n, d)
        # Differentiable conditional replace (avoids in-place on leaf tensor)
        g_tokens_train = torch.where(null_mask[:, None, None], null_expanded, g_tokens)

        # ── Forward pass ─────────────────────────────────────────────────
        with torch.autocast(device_type="cuda", enabled=self.conf.fp16):
            # Backbone: unconditional ε prediction
            t_scaled = self.sampler._scale_timesteps(t)
            backbone_out = self.model.forward(
                x=x_t,
                t=t_scaled,
                x_start=imgs,
                cond=zeros_cond,
            )
            eps_backbone = backbone_out.pred                             # (B, 3, H, W)

            # Adapter: genomic correction Δε
            delta_eps = self.adapter(x_t, t, g_tokens_train)           # (B, 3, H, W)

            loss = F.mse_loss(eps_backbone + delta_eps, noise)

        self.manual_backward(loss / accum)

        if self.is_last_accum(batch_idx):
            if self.conf.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.conf.grad_clip)
                torch.nn.utils.clip_grad_norm_(
                    list(self.adapter.parameters()) + list(self.genomic_encoder.parameters()),
                    self.conf.grad_clip,
                )
            opt_backbone.step()
            opt_adapter.step()
            opt_backbone.zero_grad()
            opt_adapter.zero_grad()

        # ── Logging ───────────────────────────────────────────────────────
        if self.global_rank == 0:
            self.logger.experiment.add_scalar("loss/train", loss.item(), self.num_samples)

        # guidance_delta: E[‖Δε_own − Δε_null‖²]
        # Primary health indicator — must grow as adapter learns conditioning.
        if self.global_rank == 0 and (self.num_samples % (500 * self.conf.batch_size_effective) < self.conf.batch_size_effective):
            with torch.no_grad():
                b = min(4, B)
                d_own = self.adapter(x_t[:b].detach(), t[:b], g_tokens[:b].detach())
                d_null = self.adapter(x_t[:b].detach(), t[:b], null_expanded[:b].detach())
                guidance_delta = (d_own - d_null).pow(2).mean().item()
                self.logger.experiment.add_scalar(
                    "cond/guidance_delta", guidance_delta, self.num_samples
                )

        return loss

    # ------------------------------------------------------------------
    # EMA and epoch hooks
    # ------------------------------------------------------------------

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if not self.is_last_accum(batch_idx):
            return

        # Backbone EMA (from parent)
        from mopadi.utils.ema import ema as _ema
        _ema(self.model, self.ema_model, self.conf.ema_decay)

        # Adapter EMA (manual)
        decay = self.conf.ema_decay
        _ema_update(self._ema_adapter.parameters(), self.adapter.parameters(), decay)
        _ema_update(self._ema_genomic_encoder.parameters(), self.genomic_encoder.parameters(), decay)
        with torch.no_grad():
            self._ema_null_token.mul_(decay).add_(self.null_token.data, alpha=1.0 - decay)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        batch, _ = self._flatten_bag_batch(batch)
        imgs = batch["img"].to(self.device)
        feats = batch["feat"].to(self.device, dtype=torch.float32)
        B = imgs.shape[0]

        t, _ = self.T_sampler.sample(B, imgs.device)
        noise = torch.randn_like(imgs)
        x_t = self.sampler.q_sample(imgs, t, noise=noise)
        zeros_cond = torch.zeros(B, self.conf.feat_dim, device=self.device, dtype=torch.float32)

        # Use EMA models for validation
        with torch.no_grad():
            t_scaled = self.sampler._scale_timesteps(t)
            eps_backbone = self.ema_model.forward(
                x=x_t, t=t_scaled, x_start=imgs, cond=zeros_cond,
            ).pred
            g_tokens = self._ema_genomic_encoder(feats)
            delta_eps = self._ema_adapter(x_t, t, g_tokens)
            loss_val = F.mse_loss(eps_backbone + delta_eps, noise)

        self.log("loss/val", loss_val, on_step=False, on_epoch=True,
                 sync_dist=True, prog_bar=True)
        if self.global_rank == 0:
            self.logger.experiment.add_scalar("loss/val", loss_val.item(), self.num_samples)
        return loss_val

    # ------------------------------------------------------------------
    # Inference helper (used by sampling scripts)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_conditional(
        self,
        genomic_feats: torch.Tensor,
        guidance_scale: float = 5.0,
        n_steps: int = 20,
        device: str = "cuda",
        use_ema: bool = True,
    ) -> torch.Tensor:
        """
        Generate tiles conditioned on genomic_feats using CFG over the adapter.

        ε_guided = ε_backbone(x_t, t, 0) + scale * (Δε_own − Δε_null)

        Parameters
        ----------
        genomic_feats : (B, feat_dim) tensor of patient genomic features
        guidance_scale : CFG scale factor (1.0 = no guidance)
        n_steps : DDIM steps
        device : target device
        use_ema : if True, use EMA weights for both backbone and adapter
        """
        backbone = self.ema_model if use_ema else self.model
        adapter = self._ema_adapter if use_ema else self.adapter
        enc = self._ema_genomic_encoder if use_ema else self.genomic_encoder
        null_tok = self._ema_null_token if use_ema else self.null_token.data

        B = genomic_feats.shape[0]
        genomic_feats = genomic_feats.to(device, dtype=torch.float32)
        g_tokens = enc(genomic_feats)
        null_expanded = null_tok.unsqueeze(0).expand(B, -1, -1).to(device)

        img_size = self.conf.img_size
        x = torch.randn(B, 3, img_size, img_size, device=device)
        zeros_cond = torch.zeros(B, self.conf.feat_dim, device=device, dtype=torch.float32)

        sampler = self.conf._make_diffusion_conf(self.conf.T_eval).make_sampler()

        def model_fn(x_t, t, **kwargs):
            t_scaled = sampler._scale_timesteps(t)
            eps_base = backbone.forward(x=x_t, t=t_scaled, x_start=None, cond=zeros_cond).pred
            delta_own = adapter(x_t, t, g_tokens)
            delta_null = adapter(x_t, t, null_expanded)
            return eps_base + guidance_scale * (delta_own - delta_null)

        # DDIM sampling loop (uses the sampler's sample method)
        from mopadi.diffusion.base import DummyModel
        import dataclasses

        # Wrap our custom model_fn as a MoPaDi-compatible model
        class _WrappedModel(torch.nn.Module):
            def forward(self_, x, t, **kw):
                from mopadi.diffusion.base import ModelReturn
                pred = model_fn(x, t, **kw)
                return ModelReturn(pred=pred)

        out = sampler.sample(
            model=_WrappedModel(),
            shape=(B, 3, img_size, img_size),
            device=device,
            progress=True,
        )
        return out
