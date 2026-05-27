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
from pathlib import Path

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
        self.register_buffer(
            "_ema_null_token",
            torch.zeros(conf.adapter_n_tokens, conf.adapter_token_dim),
            persistent=False,
        )

        # ── Cohort classification head ─────────────────────────────────────
        # Linear probe on mean-pooled g_tokens.  Forces the genomic encoder to
        # produce cohort-discriminative embeddings; gradient flows only through
        # genomic_encoder, never through the backbone.
        if conf.cohort_weight > 0:
            self.cohort_head: torch.nn.Linear | None = torch.nn.Linear(
                conf.adapter_token_dim, conf.n_cohorts
            )
        else:
            self.cohort_head = None

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

    # ------------------------------------------------------------------
    # Checkpoint compatibility
    # ------------------------------------------------------------------

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # ── Model weights: fill in any new params not present in checkpoint ──
        # (e.g. decoder CA layers added after the first run)
        ckpt_sd = checkpoint.get("state_dict", {})
        model_sd = self.state_dict()
        added = []
        for k, v in model_sd.items():
            if k not in ckpt_sd:
                ckpt_sd[k] = v
                added.append(k)
        if added:
            log.info("on_load_checkpoint: initialised %d new params from scratch: %s",
                     len(added), added[:8])
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
            # opt_adapter  → adapter + genomic_encoder + null_token  (group index 1)
            current_sizes = [
                sum(1 for _ in self.model.parameters()),
                sum(1 for _ in self.adapter.parameters())
                + sum(1 for _ in self.genomic_encoder.parameters())
                + 1  # null_token
                + (sum(1 for _ in self.cohort_head.parameters()) if self.cohort_head is not None else 0),
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

    @staticmethod
    def _subtype_contrastive_loss(
        g_tokens: torch.Tensor,
        subtypes: list,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """Supervised contrastive loss (SupCon) on mean-pooled g_tokens.

        Pulls same-subtype token vectors together and pushes different-subtype
        vectors apart.  Returns zero if the batch has fewer than 2 distinct
        subtypes or no anchor has a same-subtype neighbour.
        """
        # Mean-pool tokens per sample and L2-normalise → (B, D)
        z = F.normalize(g_tokens.float().mean(dim=1), dim=-1)
        B = z.shape[0]

        unique_subtypes = list(dict.fromkeys(subtypes))
        if len(unique_subtypes) < 2:
            return z.new_zeros(())

        label_map = {s: i for i, s in enumerate(unique_subtypes)}
        labels = torch.tensor([label_map[s] for s in subtypes], device=z.device)

        mask_self = torch.eye(B, dtype=torch.bool, device=z.device)
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask_self  # (B, B)

        if not mask_pos.any():
            return z.new_zeros(())

        sim = torch.matmul(z, z.T) / temperature                                 # (B, B)
        log_denom = torch.logsumexp(
            sim.masked_fill(mask_self, -1e9), dim=1, keepdim=True
        )                                                                         # (B, 1)
        log_prob = sim - log_denom                                               # (B, B)

        n_pos = mask_pos.float().sum(dim=1)                                      # (B,)
        loss_per_anchor = -(log_prob * mask_pos.float()).sum(dim=1) / n_pos.clamp(min=1)
        return loss_per_anchor[n_pos > 0].mean()

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        adapter_params = (
            list(self.adapter.parameters())
            + list(self.genomic_encoder.parameters())
            + [self.null_token]
            + (list(self.cohort_head.parameters()) if self.cohort_head is not None else [])
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

        # ── Subtype contrastive loss on real g_tokens (float32, before autocast) ──
        # Computed here so gradients reach genomic_encoder in full precision.
        # Uses real tokens only (not null-replaced) so the loss reflects the
        # encoder's actual subtype separation, not the CFG-dropout mixture.
        if self.conf.contrastive_weight > 0 and "subtype" in batch:
            c_loss = self._subtype_contrastive_loss(
                g_tokens, batch["subtype"], self.conf.contrastive_temp
            )
        else:
            c_loss = feats.new_zeros(())

        # ── Cohort classification head ──────────────────────────────────────
        # Cross-entropy on mean-pooled g_tokens → cohort integer label.
        # Weight ~1.0 makes this signal competitive with the MSE loss (~0.07)
        # and overcomes the mean-encoding collapse seen at contrastive_weight=0.01.
        # Gradient path: cohort_head → g_tokens → genomic_encoder only;
        # backbone receives zero gradient from this term.
        if self.conf.cohort_weight > 0 and self.cohort_head is not None and "subtype" in batch:
            subtypes = batch["subtype"]
            # Sort alphabetically for consistent label assignment across batches.
            # TCGA-BRCA → 0, TCGA-LIHC → 1 (and any future cohort keeps order).
            unique_sorted = sorted(set(subtypes))
            label_map = {s: i for i, s in enumerate(unique_sorted)}
            cohort_labels = torch.tensor(
                [label_map[s] for s in subtypes], device=self.device, dtype=torch.long
            )
            g_pooled = g_tokens.float().mean(dim=1)          # (B, token_dim)
            cohort_logits = self.cohort_head(g_pooled)       # (B, n_cohorts)
            cohort_loss = F.cross_entropy(cohort_logits, cohort_labels)
            with torch.no_grad():
                cohort_acc = (cohort_logits.argmax(dim=-1) == cohort_labels).float().mean()
        else:
            cohort_loss = feats.new_zeros(())
            cohort_acc = feats.new_zeros(())

        # ── Forward pass ─────────────────────────────────────────────────
        # NOTE: no inner torch.autocast block here — Lightning's precision plugin
        # (bf16-mixed or fp16-mixed) wraps training_step at the trainer level.
        # An inner autocast(enabled=False) would disable the outer bf16 context,
        # causing dtype mismatches when tensors produced outside (under bf16) are
        # passed into model layers with the outer context unexpectedly disabled.
        t_scaled = self.sampler._scale_timesteps(t)
        backbone_out = self.model.forward(
            x=x_t,
            t=t_scaled,
            x_start=imgs,
            cond=zeros_cond,
        )
        eps_backbone = backbone_out.pred                             # (B, 3, H, W)

        use_delta    = self.conf.delta_encouragement_weight > 0
        use_pairwise = self.conf.pairwise_delta_weight > 0
        use_split    = self.conf.split_backbone_adapter_loss

        if use_delta or use_pairwise or use_split:
            # Need d_own and d_null separately for delta loss, pairwise loss, or stop-grad.
            d_own  = self.adapter(x_t, t, g_tokens)       # (B, 3, H, W) real tokens
            d_null = self.adapter(x_t, t, null_expanded)  # (B, 3, H, W) null token
            # Reconstruct CFG-dropout mix from the two precomputed outputs
            d_train = torch.where(null_mask[:, None, None, None], d_null, d_own)
        else:
            # Original single-pass path (backward-compatible)
            g_tokens_train = torch.where(null_mask[:, None, None], null_expanded, g_tokens)
            d_train = self.adapter(x_t, t, g_tokens_train)

        # Cast to float32 for loss: F.mse_loss is not autocast-eligible and
        # will fail with mixed bf16/float32 inputs under bf16-mixed precision.
        noise_f = noise.float()
        if use_split:
            mse_backbone = F.mse_loss(eps_backbone.float(), noise_f)
            mse_adapter  = F.mse_loss(eps_backbone.detach().float() + d_train.float(), noise_f)
            mse_loss = mse_backbone + mse_adapter
        else:
            mse_loss = F.mse_loss((eps_backbone + d_train).float(), noise_f)

        # Delta encouragement loss (fp32 for numerical stability with small values)
        if use_delta:
            gd = (d_own.float() - d_null.float()).pow(2).mean()
            delta_loss = -self.conf.delta_encouragement_weight * torch.log(gd + 1e-8)
        else:
            delta_loss = feats.new_zeros(())

        # ── Pairwise Δε loss (unsupervised) ──────────────────────────────────
        # Compare adapter output for the same noisy images but cyclically shifted
        # patient tokens: forces the adapter to produce different corrections for
        # different patients' RNA-seq without requiring any cohort labels.
        # Cyclic shift guarantees perm[i] ≠ i for all i (no trivial self-pairs).
        if use_pairwise:
            perm = (torch.arange(B, device=self.device) + 1) % B
            d_perm = self.adapter(x_t, t, g_tokens[perm])
            pd = (d_own.float() - d_perm.float()).pow(2).mean()
            pairwise_delta_loss = -self.conf.pairwise_delta_weight * torch.log(pd + 1e-8)
        else:
            pairwise_delta_loss = feats.new_zeros(())

        loss = (
            mse_loss
            + delta_loss
            + pairwise_delta_loss
            + self.conf.contrastive_weight * c_loss
            + self.conf.cohort_weight * cohort_loss
        )

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

        # ── Logging ───────────────────────────────────────────────────────
        if self.trainer.is_global_zero:
            self.logger.experiment.add_scalar("loss/train", loss.item(), self.num_samples)
            if self.conf.split_backbone_adapter_loss:
                self.logger.experiment.add_scalar(
                    "loss/backbone", mse_backbone.item(), self.num_samples
                )
                self.logger.experiment.add_scalar(
                    "loss/adapter", mse_adapter.item(), self.num_samples
                )
            if self.conf.delta_encouragement_weight > 0:
                self.logger.experiment.add_scalar(
                    "cond/delta_loss", delta_loss.item(), self.num_samples
                )
                self.logger.experiment.add_scalar(
                    "cond/guidance_delta_train", gd.item(), self.num_samples
                )
            if self.conf.pairwise_delta_weight > 0:
                self.logger.experiment.add_scalar(
                    "cond/pairwise_delta_loss", pairwise_delta_loss.item(), self.num_samples
                )
                self.logger.experiment.add_scalar(
                    "cond/pairwise_delta", pd.item(), self.num_samples
                )
            if self.conf.contrastive_weight > 0:
                self.logger.experiment.add_scalar(
                    "cond/contrastive_loss", c_loss.item(), self.num_samples
                )
            if self.conf.cohort_weight > 0:
                self.logger.experiment.add_scalar(
                    "cond/cohort_loss", cohort_loss.item(), self.num_samples
                )
                self.logger.experiment.add_scalar(
                    "cond/cohort_acc", cohort_acc.item(), self.num_samples
                )

        # guidance_delta: E[‖Δε_own − Δε_null‖²]
        # Fires exactly once per interval (< batch_size, not < batch_size_effective).
        # Uses fresh x_t with randomly drawn timesteps so consecutive measurements
        # are independent of the training batch's t distribution.
        _gd_interval = 500 * self.conf.batch_size_effective
        if self.trainer.is_global_zero and (self.num_samples % _gd_interval < self.conf.batch_size):
            with torch.no_grad():
                bm = min(16, B)
                t_m = torch.randint(0, self.conf.T, (bm,), device=self.device)
                x_t_m = self.sampler.q_sample(
                    imgs[:bm].detach(), t_m,
                    noise=torch.randn_like(imgs[:bm]),
                )
                g_tok_m = self.genomic_encoder(feats[:bm].detach())
                null_m = self.null_token.unsqueeze(0).expand(bm, -1, -1).detach()
                d_own_m = self.adapter(x_t_m, t_m, g_tok_m)
                d_null_m = self.adapter(x_t_m, t_m, null_m)
                guidance_delta = (d_own_m - d_null_m).pow(2).mean().item()
                self.logger.experiment.add_scalar(
                    "cond/guidance_delta", guidance_delta, self.num_samples
                )

                # Encoder health: are g_tokens diverse across patients?
                # If g_token_diversity ≈ 0, the genomic encoder has collapsed.
                # If g_vs_null_dist ≈ 0, real tokens ≈ null token → adapter can't distinguish.
                g_tok_mean = g_tok_m.mean(dim=0, keepdim=True)
                g_token_diversity = (g_tok_m - g_tok_mean).pow(2).mean().item()
                g_vs_null = (g_tok_m - null_m).pow(2).mean().item()
                self.logger.experiment.add_scalar(
                    "cond/g_token_diversity", g_token_diversity, self.num_samples
                )
                self.logger.experiment.add_scalar(
                    "cond/g_vs_null_dist", g_vs_null, self.num_samples
                )

        # Sample image grid: clean | noisy | backbone_recon | cond_recon | guidance_delta_vis
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

        rows = [imgs_s.clamp(-1, 1), x_t_vis.clamp(-1, 1)]

        # Reconstruct x_0 from eps if the sampler exposes the alphas
        try:
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

            rows += [_x0(x_t_vis, eps_back, t_vis), _x0(x_t_vis, eps_back + d_own, t_vis)]
        except AttributeError:
            pass

        rows.append(_guidance_vis(d_own - d_null))
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

        rows_ema = [imgs_s.clamp(-1, 1), x_t_vis.clamp(-1, 1)]
        try:
            rows_ema += [_x0(x_t_vis, eps_back_ema, t_vis),
                         _x0(x_t_vis, eps_back_ema + d_own_ema, t_vis)]
        except NameError:
            pass
        rows_ema.append(_guidance_vis(d_own_ema - d_null_ema))

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
