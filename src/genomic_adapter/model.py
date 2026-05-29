"""
GDALitModel — jointly trains backbone UNet + GenomicResidualAdapter from scratch.

Training mechanics
------------------
Backbone UNet  : always receives  cond = zeros(B, style_ch)
                 → learns unconditional denoising
                 → CANNOT see patient genomic features → CANNOT suppress adapter

Adapter        : receives (x_t, t, g_tokens) → predicts Δε
                 → CFG null dropout (cfg_dropout %) replaces g_tokens with null_token

Combined loss  : L_bb  = MSE(ε_backbone, noise)
                 L_ada = MSE(ε_backbone.detach() + Δε_adapter, noise)

Backbone and adapter each receive gradient only from their own loss term.
The backbone optimises unconditional denoising; the adapter learns the
genomic-specific residual on top of a detached backbone prediction.
No pretrained checkpoint is loaded.

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
from pathlib import Path

import torch
import torch.nn.functional as F

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

        # Null token for CFG: fixed at zeros — NOT learned.
        # A learnable null token converges toward the average real-token direction
        # (both minimise the same MSE residual), collapsing guidance_delta to zero.
        # Fixed zeros means null conditioning = identity cross-attention (K=V=0,
        # bias=False), giving a clean architectural null baseline.
        self.null_token = torch.nn.Parameter(
            torch.zeros(conf.adapter_n_tokens, conf.adapter_token_dim),
            requires_grad=False,
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
        self.register_buffer(
            "_ema_null_token",
            torch.zeros(conf.adapter_n_tokens, conf.adapter_token_dim),
            persistent=True,
        )

        if conf.backbone_ckpt_path:
            self._load_backbone_weights(conf.backbone_ckpt_path)
            if conf.reinit_adapter:
                self._reset_conditioning_modules()

        # Freeze backbone when requested — must happen after super().__init__
        # which creates self.model and self.ema_model.
        if conf.freeze_backbone:
            self.model.requires_grad_(False)
            # ema_model is already requires_grad=False; mark explicitly for clarity
            self.ema_model.requires_grad_(False)
            log.info("GDALitModel: backbone FROZEN — only adapter/encoder receive gradients")

        n_adapter = sum(p.numel() for p in self.adapter.parameters())
        n_enc = sum(p.numel() for p in self.genomic_encoder.parameters())
        log.info(
            "GDALitModel: adapter params=%d  genomic_encoder params=%d  "
            "null_token shape=%s",
            n_adapter, n_enc, list(self.null_token.shape),
        )

    def _load_backbone_weights(self, ckpt_path: str) -> None:
        """Load only backbone weights from a checkpoint into model and EMA model."""
        ckpt_file = Path(ckpt_path)
        if not ckpt_file.exists():
            raise FileNotFoundError(f"backbone_ckpt_path not found: {ckpt_path}")

        log.info("Loading backbone weights from checkpoint: %s", ckpt_file)
        checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        backbone_state = {
            k[len("model."):]: v
            for k, v in state_dict.items()
            if k.startswith("model.") and not k.startswith("model.adapter")
        }
        missing, unexpected = self.model.load_state_dict(backbone_state, strict=False)
        log.info(
            "Loaded backbone into model: missing=%d unexpected=%d",
            len(missing), len(unexpected),
        )

        ema_state = {
            k[len("ema_model."):]: v
            for k, v in state_dict.items()
            if k.startswith("ema_model.") and not k.startswith("ema_model.adapter")
        }
        missing_ema, unexpected_ema = self.ema_model.load_state_dict(ema_state, strict=False)
        log.info(
            "Loaded backbone into ema_model: missing=%d unexpected=%d",
            len(missing_ema), len(unexpected_ema),
        )

    def _reset_conditioning_modules(self) -> None:
        """Reinitialize adapter-side modules while keeping the backbone fixed."""
        log.info("Reinitializing adapter, genomic encoder, and null token from scratch.")
        self.genomic_encoder = GenomicTokenEncoder(
            genomic_in=self.conf.feat_dim,
            n_tokens=self.conf.adapter_n_tokens,
            token_dim=self.conf.adapter_token_dim,
        )
        self.null_token = torch.nn.Parameter(
            torch.zeros(self.conf.adapter_n_tokens, self.conf.adapter_token_dim),
            requires_grad=False,
        )
        self.adapter = GenomicResidualAdapter(
            in_ch=3,
            base_ch=self.conf.adapter_base_ch,
            t_dim=self.conf.adapter_t_dim,
            token_dim=self.conf.adapter_token_dim,
            n_heads=self.conf.adapter_n_heads,
        )

        self._ema_adapter = copy.deepcopy(self.adapter)
        self._ema_adapter.requires_grad_(False)
        self._ema_genomic_encoder = copy.deepcopy(self.genomic_encoder)
        self._ema_genomic_encoder.requires_grad_(False)
        self._ema_null_token.zero_()

    # ------------------------------------------------------------------
    # Checkpoint compatibility
    # ------------------------------------------------------------------

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # ── Model weights: fill in any new params not present in checkpoint ──
        # (e.g. decoder CA layers added after the first run)
        ckpt_sd = checkpoint.get("state_dict", {})

        if self.conf.backbone_ckpt_path and self.conf.reinit_adapter:
            ckpt_sd = {
                k: v
                for k, v in ckpt_sd.items()
                if not (
                    k.startswith("adapter.")
                    or k.startswith("genomic_encoder.")
                    or k.startswith("null_token")
                    or k.startswith("_ema_adapter.")
                    or k.startswith("_ema_genomic_encoder.")
                    or k.startswith("_ema_null_token")
                )
            }

        model_sd = self.state_dict()
        added = []
        for k, v in model_sd.items():
            if k not in ckpt_sd:
                ckpt_sd[k] = v
                added.append(k)
        if added:
            log.info("on_load_checkpoint: initialised %d new params from scratch: %s",
                     len(added), added[:8])

        # Always reset null_token and its EMA to zeros regardless of what the
        # checkpoint stored.  A previously-learned null_token (e.g. from v9) would
        # act as a biased non-zero null baseline, defeating the fixed-null design.
        for key in ("null_token", "_ema_null_token"):
            if key in ckpt_sd:
                ckpt_sd[key] = torch.zeros_like(ckpt_sd[key])

        checkpoint["state_dict"] = ckpt_sd

        # ── Optimizer states ──────────────────────────────────────────────
        # When the backbone is frozen (v15+) there is only one optimizer
        # (opt_adapter).  Old checkpoints (v13/v14) had two; restoring them
        # would cause a group-size mismatch.  Drop all optimizer states so
        # Adam momentum resets cleanly for the new training phase.
        if self.conf.freeze_backbone:
            checkpoint["optimizer_states"] = []
            return

        # Drop if param group sizes no longer match ──────────────────────
        # PyTorch raises ValueError if the saved group has a different number of
        # params than the current optimizer group.  This happens whenever new
        # parameters are added to an optimizer (e.g. decoder CA added to
        # opt_adapter).  Clearing optimizer_states makes Lightning skip restoring
        # them; model weights above are fully preserved, and only Adam momentum
        # buffers are lost (acceptable after an architecture change).
        opt_states = checkpoint.get("optimizer_states")
        if opt_states:
            # opt_backbone → self.model.parameters()  (group index 0)
            # opt_adapter  → adapter + genomic_encoder  (group index 1)
            # null_token is fixed (requires_grad=False) and not in any optimizer
            current_sizes = [
                sum(1 for _ in self.model.parameters()),
                sum(1 for _ in self.adapter.parameters())
                + sum(1 for _ in self.genomic_encoder.parameters()),
            ]
            for i, (saved_opt, expected) in enumerate(zip(opt_states, current_sizes)):
                saved_groups = saved_opt.get("param_groups", [])
                saved_size = len(saved_groups[0]["params"]) if saved_groups else 0
                if saved_size != expected:
                    log.warning(
                        "Optimizer %d param group size mismatch "
                        "(saved=%d, current=%d) — discarding optimizer states. "
                        "Model weights are preserved; Adam momentum resets.",
                        i, saved_size, expected,
                    )
                    checkpoint["optimizer_states"] = []
                    break

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
            if "subtype" in flat:
                flat["subtype"] = [s for s in flat["subtype"] for _ in range(N)]
            return flat, N
        return batch, 1

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        adapter_params = (
            list(self.adapter.parameters())
            + list(self.genomic_encoder.parameters())
            # null_token is fixed (requires_grad=False) — not in any optimizer
        )
        opt_adapter = torch.optim.AdamW(
            adapter_params,
            lr=self.conf.adapter_lr,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )
        if self.conf.freeze_backbone:
            # Single optimizer — backbone has no trainable parameters.
            return [opt_adapter], []
        opt_backbone = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.conf.backbone_lr,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )
        return [opt_backbone, opt_adapter], []

    # ------------------------------------------------------------------
    # Data loading — subtype-balanced sampling
    # ------------------------------------------------------------------

    def train_dataloader(self):
        """Subtype-balanced DataLoader.

        Each tile is weighted by 1 / (total tiles in its PAM50 subtype), so
        every subtype has equal expected frequency per batch regardless of how
        many patients/tiles it has in the training split.

        Without this, LumA (≈51 % of training patients) would dominate batch
        gradients and adapter corrections for rare subtypes (Her2, Normal)
        would be systematically undertrained.
        """
        from collections import Counter

        import torch.utils.data as tud

        from src.drafts.mopadi_genomic.dataset import patient_id_from_tile_path

        tile_paths = self.train_data.tile_paths
        subtype_map = self.train_data._subtype_map

        subtype_counts: Counter = Counter()
        tile_subtypes: list = []
        for path in tile_paths:
            pid = patient_id_from_tile_path(path)
            subtype = subtype_map.get(pid, "unknown")
            tile_subtypes.append(subtype)
            subtype_counts[subtype] += 1

        weights = torch.tensor(
            [1.0 / subtype_counts[s] for s in tile_subtypes],
            dtype=torch.float32,
        )
        sampler = tud.WeightedRandomSampler(
            weights, num_samples=len(tile_paths), replacement=True
        )

        if self.trainer.is_global_zero:
            log.info(
                "Subtype-balanced sampler: %s",
                {s: c for s, c in sorted(subtype_counts.items())},
            )

        conf = self.conf.clone()
        conf.batch_size = self.batch_size
        return conf.make_loader(self.train_data, drop_last=True, sampler=sampler)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        if self.conf.freeze_backbone:
            opt_adapter = self.optimizers()
        else:
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
        g_tokens_train = torch.where(null_mask[:, None, None], null_expanded, g_tokens)

        # ── Forward pass ─────────────────────────────────────────────────
        # NOTE: no inner torch.autocast block here — Lightning's precision plugin
        # (bf16-mixed or fp16-mixed) wraps training_step at the trainer level.
        t_scaled = self.sampler._scale_timesteps(t)
        backbone_out = self.model.forward(x=x_t, t=t_scaled, x_start=imgs, cond=zeros_cond)
        eps_backbone = backbone_out.pred                             # (B, 3, H, W)

        d_train = self.adapter(x_t, t, g_tokens_train)

        # ── Split losses: backbone and adapter train on separate objectives ──
        # Backbone: standard MSE on its own prediction.
        # Adapter:  MSE on the residual relative to a detached backbone.
        #   Detaching eps_backbone here ensures backbone gradients do not flow
        #   through the adapter loss — eliminating the gradient conflict where
        #   the backbone could learn to compensate for whatever the adapter adds.
        # Cast to float32: F.mse_loss is not autocast-eligible under bf16-mixed.
        noise_f = noise.float()
        loss_bb = F.mse_loss(eps_backbone.float(), noise_f)
        loss_ada = F.mse_loss((eps_backbone.detach().float() + d_train.float()), noise_f)

        # Delta encouragement: prevent the adapter from ignoring its token input.
        # With null_token fixed at zeros, Δε_null ≈ 0 (identity cross-attention).
        # This term penalises ||Δε_own - 0||², forcing the adapter to produce
        # non-zero output for real tokens. The MSE term then ensures these outputs
        # are actually useful residual corrections, not arbitrary noise.
        # No labels are used — the comparison is purely own-tokens vs null-tokens.
        if self.conf.delta_encouragement_weight > 0:
            d_null_train = self.adapter(x_t, t, null_expanded)
            delta_sq = (d_train - d_null_train.detach()).pow(2).mean()
            loss_ada = loss_ada - self.conf.delta_encouragement_weight * delta_sq

        loss = loss_bb + loss_ada

        self.manual_backward(loss / accum)

        if self.is_last_accum(batch_idx):
            if self.conf.grad_clip > 0:
                if not self.conf.freeze_backbone:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.conf.grad_clip)
                torch.nn.utils.clip_grad_norm_(
                    list(self.adapter.parameters()) + list(self.genomic_encoder.parameters()),
                    self.conf.grad_clip,
                )
            if not self.conf.freeze_backbone:
                opt_backbone.step()
                opt_backbone.zero_grad()
            opt_adapter.step()
            opt_adapter.zero_grad()

        if self.trainer.is_global_zero:
            self.logger.experiment.add_scalar("loss/train", loss.item(), self.num_samples)

        # ── Periodic diagnostics (no gradient) ────────────────────────────
        # guidance_delta = E[‖Δε_own − Δε_null‖²]: primary conditioning health metric.
        # g_token_diversity: are tokens different across patients (encoder not collapsed)?
        # g_vs_null_dist: are real tokens distinguishable from the null token?
        _gd_interval = 500 * self.conf.batch_size_effective
        if self.trainer.is_global_zero and (self.num_samples % _gd_interval < self.conf.batch_size):
            with torch.no_grad():
                bm = min(16, B)
                t_m = torch.randint(0, self.conf.T, (bm,), device=self.device)
                x_t_m = self.sampler.q_sample(
                    imgs[:bm].detach(), t_m, noise=torch.randn_like(imgs[:bm]),
                )
                g_tok_m = self.genomic_encoder(feats[:bm].detach())
                null_m = self.null_token.unsqueeze(0).expand(bm, -1, -1).detach()
                d_own_m = self.adapter(x_t_m, t_m, g_tok_m)
                d_null_m = self.adapter(x_t_m, t_m, null_m)
                guidance_delta = (d_own_m - d_null_m).pow(2).mean().item()
                g_token_diversity = (g_tok_m - g_tok_m.mean(dim=0, keepdim=True)).pow(2).mean().item()
                # delta_magnitude: mean squared norm of the adapter's own output.
                # With null_token fixed at zeros, d_null ≈ 0, so this ≈ guidance_delta.
                # Tracks whether the adapter produces non-zero corrections at all.
                delta_magnitude = d_own_m.pow(2).mean().item()
                self.logger.experiment.add_scalar("cond/guidance_delta", guidance_delta, self.num_samples)
                self.logger.experiment.add_scalar("cond/g_token_diversity", g_token_diversity, self.num_samples)
                self.logger.experiment.add_scalar("cond/delta_magnitude", delta_magnitude, self.num_samples)

        _si_interval = self.conf.reconstruct_every_samples
        if self.trainer.is_global_zero and (self.num_samples % _si_interval < self.conf.batch_size):
            self._log_sample_images(imgs.detach(), feats.detach())

        return loss

    # ------------------------------------------------------------------
    # EMA and epoch hooks
    # ------------------------------------------------------------------

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if not self.is_last_accum(batch_idx):
            return

        # Backbone EMA — skip when frozen; weights never change so EMA = backbone always
        if not self.conf.freeze_backbone:
            from mopadi.train_diff_autoenc import ema as _ema
            _ema(self.model, self.ema_model, self.conf.ema_decay)

        # Adapter EMA (manual)
        decay = self.conf.ema_decay
        _ema_update(self._ema_adapter.parameters(), self.adapter.parameters(), decay)
        _ema_update(self._ema_genomic_encoder.parameters(), self.genomic_encoder.parameters(), decay)
        # _ema_null_token is always zeros (null_token is fixed); no EMA update needed.

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
            noise_f = noise.float()
            loss_val = F.mse_loss((eps_backbone + delta_eps).float(), noise_f)

            # Shuffled-conditioning loss: same forward pass but with mismatched genomic
            # features. If loss_val_shuffled > loss_val the conditioning is carrying
            # subtype-specific information (cond/gap > 0 is the key signal to watch).
            perm = torch.randperm(B, device=self.device)
            g_tokens_shuffled = self._ema_genomic_encoder(feats[perm])
            delta_eps_shuffled = self._ema_adapter(x_t, t, g_tokens_shuffled)
            loss_val_shuffled = F.mse_loss((eps_backbone + delta_eps_shuffled).float(), noise_f)

        # logger=False: accumulate in callback_metrics without writing to TFBoard
        # per batch (which would create 100 entries per val run at the same step).
        # on_validation_epoch_end writes a single add_scalar at num_samples.
        self.log("_val_loss", loss_val, on_step=False, on_epoch=True,
                 sync_dist=True, prog_bar=True, logger=False)
        self.log("_val_loss_shuffled", loss_val_shuffled, on_step=False, on_epoch=True,
                 sync_dist=True, prog_bar=False, logger=False)
        return loss_val

    def on_validation_epoch_end(self) -> None:
        if self.trainer.state.stage == "sanity_check":
            return
        val_loss = self.trainer.callback_metrics.get("_val_loss")
        val_loss_shuffled = self.trainer.callback_metrics.get("_val_loss_shuffled")
        if val_loss is not None:
            if self.trainer.is_global_zero:
                self.logger.experiment.add_scalar(
                    "loss/val", val_loss.item(), self.num_samples
                )
                if val_loss_shuffled is not None:
                    self.logger.experiment.add_scalar(
                        "loss/val_shuffled", val_loss_shuffled.item(), self.num_samples
                    )
                    self.logger.experiment.add_scalar(
                        "cond/gap", val_loss_shuffled.item() - val_loss.item(), self.num_samples
                    )
            # sync_dist=False: already aggregated by validation_step's sync_dist=True
            self.log("loss/val_ckpt", val_loss, prog_bar=False, sync_dist=False)

    # ------------------------------------------------------------------
    # Training-time sample visualisation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _log_sample_images(self, imgs: torch.Tensor, feats: torch.Tensor) -> None:
        """
        Save a reconstruction grid to TensorBoard + disk.

        Rows (each column = one patient tile):
          1. clean original
          2. noisy at t=250 (~25 % of T)
          3. backbone-only reconstruction of x_0
          4. backbone + adapter (conditioned) reconstruction of x_0
          5. guidance delta Δε_own − Δε_null (scaled for visibility)

        If the sampler does not expose sqrt_alphas_cumprod the reconstruction
        rows are omitted and only clean / noisy / guidance are shown.

        A fixed sample batch is captured on the first call so the same patients
        appear at every logging interval, making cross-step comparison meaningful.
        """
        from torchvision.utils import make_grid, save_image

        # Capture and pin a fixed sample batch on the first call
        if not hasattr(self, "_sample_imgs"):
            b = min(8, imgs.shape[0])
            self._sample_imgs = imgs[:b].clone().cpu()
            self._sample_feats = feats[:b].clone().cpu()

        imgs_s = self._sample_imgs.to(self.device)
        feats_s = self._sample_feats.to(self.device, dtype=torch.float32)
        b = imgs_s.shape[0]

        t_vis = torch.full((b,), 250, device=self.device, dtype=torch.long)
        x_t_vis = self.sampler.q_sample(
            imgs_s, t_vis, noise=torch.randn_like(imgs_s)
        )
        t_scaled = self.sampler._scale_timesteps(t_vis)
        zeros_cond = torch.zeros(b, self.conf.feat_dim, device=self.device, dtype=torch.float32)

        g_tokens_vis = self.genomic_encoder(feats_s)
        null_vis = self.null_token.unsqueeze(0).expand(b, -1, -1).detach()

        # Cast to float32: arithmetic below (sac/somc arrays, make_grid) expects
        # consistent float32; model outputs may be bf16 under bf16-mixed precision.
        eps_back = self.model.forward(
            x=x_t_vis, t=t_scaled, x_start=imgs_s, cond=zeros_cond
        ).pred.float()
        d_own = self.adapter(x_t_vis, t_vis, g_tokens_vis).float()
        d_null = self.adapter(x_t_vis, t_vis, null_vis).float()

        def _guidance_vis(delta):
            g_min = delta.flatten(1).min(1).values.view(-1, 1, 1, 1)
            g_max = delta.flatten(1).max(1).values.view(-1, 1, 1, 1)
            return (2 * (delta - g_min) / (g_max - g_min + 1e-8) - 1).clamp(-1, 1)

        sac  = torch.as_tensor(
            self.sampler.sqrt_alphas_cumprod, device=self.device, dtype=torch.float32
        )
        somc = torch.as_tensor(
            self.sampler.sqrt_one_minus_alphas_cumprod, device=self.device, dtype=torch.float32
        )

        def _x0(x_t_, eps_, t_):
            a = sac[t_].view(-1, 1, 1, 1)
            b_ = somc[t_].view(-1, 1, 1, 1)
            return ((x_t_ - b_ * eps_) / a).clamp(-1, 1)

        rows = [
            imgs_s.clamp(-1, 1),
            x_t_vis.clamp(-1, 1),
            _x0(x_t_vis, eps_back, t_vis),
            _x0(x_t_vis, eps_back + d_own, t_vis),
            _guidance_vis(d_own - d_null),
        ]
        grid = make_grid(torch.cat(rows, dim=0), nrow=b, normalize=True,
                         value_range=(-1, 1), padding=2)

        self.logger.experiment.add_image("samples/train", grid, self.num_samples)
        samples_dir = Path(self.conf.logdir) / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        save_image(grid, samples_dir / f"samples_{self.num_samples:010d}.png")

        # ── EMA version ───────────────────────────────────────────────────
        eps_back_ema = self.ema_model.forward(
            x=x_t_vis, t=t_scaled, x_start=imgs_s, cond=zeros_cond
        ).pred.float()
        g_tok_ema = self._ema_genomic_encoder(feats_s)
        null_ema = self._ema_null_token.unsqueeze(0).expand(b, -1, -1)
        d_own_ema  = self._ema_adapter(x_t_vis, t_vis, g_tok_ema).float()
        d_null_ema = self._ema_adapter(x_t_vis, t_vis, null_ema).float()

        rows_ema = [
            imgs_s.clamp(-1, 1),
            x_t_vis.clamp(-1, 1),
            _x0(x_t_vis, eps_back_ema, t_vis),
            _x0(x_t_vis, eps_back_ema + d_own_ema, t_vis),
            _guidance_vis(d_own_ema - d_null_ema),
        ]

        grid_ema = make_grid(torch.cat(rows_ema, dim=0), nrow=b, normalize=True,
                             value_range=(-1, 1), padding=2)
        self.logger.experiment.add_image("samples/train_ema", grid_ema, self.num_samples)
        save_image(grid_ema, samples_dir / f"samples_ema_{self.num_samples:010d}.png")

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

        from mopadi.diffusion.base import DummyReturn

        class _WrappedModel(torch.nn.Module):
            def forward(self_, x, t, **kw):
                return DummyReturn(pred=model_fn(x, t, **kw))

        out = sampler.sample(
            model=_WrappedModel(),
            shape=(B, 3, img_size, img_size),
            device=device,
            progress=True,
        )
        return out
