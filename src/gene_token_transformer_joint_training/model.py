from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from .config import GeneTokenTransformerConfig, parse_gene_token_transformer_config

try:
    from joint_training.model import JointLitModel, build_conf
except ImportError:  # pragma: no cover
    from src.joint_training.model import JointLitModel, build_conf  # type: ignore[import-not-found]

try:
    from mopadi.configs.choices import OptimizerType
except ImportError:
    from src.mopadi.configs.choices import OptimizerType  # type: ignore[import-not-found]


def _build_swapped_indices(batch_size: int, device: torch.device):
    """Return a cyclic permutation of batch indices (shift by random amount ≥1)."""
    if batch_size < 2:
        return None
    shift = int(torch.randint(1, batch_size, (1,), device=device).item())
    return torch.roll(torch.arange(batch_size, device=device), shifts=shift)


class GeneTokenTransformerEncoder(nn.Module):
    """Token/value embedding + transformer encoder for genomic conditioning."""

    def __init__(self, n_genes: int, cfg: GeneTokenTransformerConfig):
        super().__init__()
        self.n_genes = n_genes
        self.d_model = cfg.d_model
        self.seq_len = cfg.seq_len or n_genes

        self.gene_embedding = nn.Embedding(n_genes, cfg.d_model)
        self.value_projection = nn.Linear(1, cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.pooling = cfg.pooling
        if self.pooling == "attn_pool":
            self.pool_query = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=cfg.d_model,
                num_heads=cfg.n_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )

    def _pool(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return x[:, 0]

        if self.pooling == "attn_pool":
            query = self.pool_query.expand(x.size(0), -1, -1)
            key_padding_mask = ~attention_mask.bool()
            pooled, _ = self.pool_attn(query, x, x, key_padding_mask=key_padding_mask)
            return pooled[:, 0]

        mask = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
        masked_x = x * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        return masked_x.sum(dim=1) / denom

    def forward(self, gene_ids: torch.Tensor, gene_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embed = self.gene_embedding(gene_ids)
        value_embed = self.value_projection(gene_values.unsqueeze(-1))
        x = token_embed + value_embed
        x = F.layer_norm(x, normalized_shape=(self.d_model,))

        key_padding_mask = ~attention_mask.bool()
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        pooled = self._pool(encoded, attention_mask)
        return pooled


class GeneTokenTransformerJointLitModel(JointLitModel):  # type: ignore[misc]
    """Joint model variant with gene-token transformer genomic encoder."""

    def __init__(self, conf, joint_cfg: dict, n_genes: int):
        super().__init__(conf, joint_cfg, n_genes)
        self.gtt_cfg = parse_gene_token_transformer_config(joint_cfg)
        self.gene_token_encoder = GeneTokenTransformerEncoder(n_genes=n_genes, cfg=self.gtt_cfg)
        self.cond_projection = nn.Sequential(
            nn.LayerNorm(self.gtt_cfg.d_model),
            nn.Linear(self.gtt_cfg.d_model, int(joint_cfg.get("cond_dim", conf.feat_dim))),
        )

        # Remove the ProbabilisticEncoder and ProjectionHead created by
        # super().__init__() — they are unused in this variant.  Deleting them
        # from the module registry prevents duplicate state-dict keys and keeps
        # checkpoint files clean.  on_fit_start and configure_optimizers are
        # overridden below to reference gene_token_encoder / cond_projection
        # directly, so the base-class references to self.encoder / self.projection
        # are never reached.
        del self.encoder
        del self.projection

        self._cached_gene_ids: torch.Tensor | None = None

        n_encoder = sum(p.numel() for p in self.gene_token_encoder.parameters())
        n_proj = sum(p.numel() for p in self.cond_projection.parameters())
        n_unet = sum(p.numel() for p in self.model.parameters())
        print(
            f"[GeneTokenJoint] Encoder: {n_encoder:,}  Proj: {n_proj:,}  "
            f"UNet: {n_unet:,}  Total: {n_encoder + n_proj + n_unet:,}"
        )

        self.save_hyperparameters({
            "gene_token_transformer": self.gtt_cfg.__dict__,
            "joint_variant": "gene_token_transformer_joint_training",
        })

    # ──────────────────────────────────────────────────────────────────
    #  Device placement (override: base references self.encoder/projection)
    # ──────────────────────────────────────────────────────────────────

    def on_fit_start(self):
        self.gene_token_encoder = self.gene_token_encoder.to(self.device)
        self.cond_projection = self.cond_projection.to(self.device)
        if self.global_rank == 0:
            print(f"[GeneTokenJoint] Moved gene_token_encoder and cond_projection to device: {self.device}")

    # ──────────────────────────────────────────────────────────────────
    #  Optimizer (override: base references self.encoder/projection)
    # ──────────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        conf = self.conf
        jcfg = self.joint_cfg
        lr = float(conf.lr)

        param_groups = [
            {"params": list(self.model.parameters()), "lr": float(jcfg.get("unet_lr", lr))},
            {"params": list(self.gene_token_encoder.parameters()), "lr": float(jcfg.get("encoder_lr", lr))},
            {"params": list(self.cond_projection.parameters()), "lr": float(jcfg.get("proj_lr", lr))},
        ]

        if conf.optimizer == OptimizerType.adamw:
            optim = torch.optim.AdamW(
                param_groups, betas=(0.9, 0.99), eps=1e-6,
                weight_decay=conf.weight_decay,
            )
        else:
            optim = torch.optim.Adam(param_groups, weight_decay=conf.weight_decay)

        epochs = int(jcfg.get("epochs", conf.max_epochs if hasattr(conf, "max_epochs") else 100))
        steps_per_epoch = int(conf.steps_per_epoch)
        total_steps = max(1, epochs * steps_per_epoch)
        warmup_steps = int(conf.warmup)

        if warmup_steps > 0:
            warmup = LambdaLR(
                optim, lr_lambda=lambda s: min(s + 1, warmup_steps) / warmup_steps
            )
            cosine = CosineAnnealingLR(
                optim, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6
            )
            sched = SequentialLR(optim, schedulers=[warmup, cosine], milestones=[warmup_steps])
        else:
            sched = CosineAnnealingLR(optim, T_max=total_steps, eta_min=1e-6)

        return {"optimizer": optim, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    # ──────────────────────────────────────────────────────────────────
    #  Encoding
    # ──────────────────────────────────────────────────────────────────

    def _tokenize_genomic(self, genomic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if genomic.ndim != 2:
            raise ValueError(f"Expected genomic shape [B, L], got {tuple(genomic.shape)}")

        batch_size, n_genes = genomic.shape
        if self._cached_gene_ids is None or self._cached_gene_ids.numel() != n_genes:
            self._cached_gene_ids = torch.arange(n_genes, device=genomic.device, dtype=torch.long)

        seq_len = min(self.gtt_cfg.seq_len or n_genes, n_genes)
        gene_ids = self._cached_gene_ids[:seq_len].unsqueeze(0).expand(batch_size, -1)
        gene_values = genomic[:, :seq_len]
        attention_mask = torch.ones((batch_size, seq_len), device=genomic.device, dtype=torch.bool)
        return gene_ids, gene_values, attention_mask

    def encode_genomic(self, genomic, deterministic: bool = False):
        gene_ids, gene_values, attention_mask = self._tokenize_genomic(genomic)
        pooled = self.gene_token_encoder(gene_ids, gene_values, attention_mask)
        cond = self.cond_projection(pooled)
        return cond

    @torch.no_grad()
    def encode(self, genomic: torch.Tensor) -> torch.Tensor:
        """Encode gene expression → diffusion conditioning (deterministic)."""
        self.eval()
        return self.encode_genomic(genomic)

    @torch.no_grad()
    def save_latent_features(self, out_dir: str, split: str = "all") -> str:
        """Override base implementation for gene-token encoder API.

        The base class assumes a VAE encoder returning ``(mu, log_var)``.
        ``GeneTokenTransformerEncoder`` instead takes ``(gene_ids, gene_values,
        attention_mask)`` and returns a single pooled tensor, so we must
        tokenise the input before calling it.
        """
        import h5py
        import numpy as np

        os.makedirs(out_dir, exist_ok=True)
        self.eval()

        datasets = []
        if split in ("train", "all") and hasattr(self, "train_data"):
            datasets.append(("train", self.train_data))
        if split in ("val", "all") and hasattr(self, "val_data"):
            datasets.append(("val", self.val_data))
        if split == "test":
            try:
                from joint_training.dataset import GenomicTileDataset, load_split
            except ImportError:
                from src.joint_training.dataset import GenomicTileDataset, load_split  # type: ignore[import-not-found]
            split_path = os.path.join(self.conf.base_dir, "patient_splits.json")
            if os.path.exists(split_path):
                splits = load_split(split_path)
                cfg = self.joint_cfg
                _nm, _ns, _ali = (
                    self.train_data.get_normalization_state()
                    if hasattr(self, "train_data") else (None, None, None)
                )
                test_ds = GenomicTileDataset(
                    csv_path=cfg["csv_path"],
                    tiles_zip_dir=cfg["tiles_zip_dir"],
                    img_size=self.conf.img_size,
                    patient_col=cfg.get("patient_col", "Patient_ID"),
                    label_col=cfg.get("label_col"),
                    patient_ids=splits["test"],
                    norm_means=_nm,
                    norm_stds=_ns,
                    apply_log1p=_ali,
                )
                datasets.append(("test", test_ds))

        seen_patients: set[str] = set()
        saved = 0

        for split_name, ds in datasets:
            raw_ds = ds.dataset if hasattr(ds, "dataset") else ds
            for pid, genomic_vec in raw_ds._genomic.items():
                if pid in seen_patients:
                    continue
                if hasattr(raw_ds, "patient_ids") and pid not in raw_ds.patient_ids:
                    continue
                seen_patients.add(pid)

                g = genomic_vec.unsqueeze(0).to(self.device)
                gene_ids, gene_values, attention_mask = self._tokenize_genomic(g)
                # pooled: (1, d_model) — transformer output before cond_projection
                z = self.gene_token_encoder(gene_ids, gene_values, attention_mask)

                h5_path = os.path.join(out_dir, f"{pid}.h5")
                with h5py.File(h5_path, "w") as f:
                    f.create_dataset("feats", data=z.cpu().numpy().astype(np.float32))
                    f.attrs["patient_id"] = pid
                    f.attrs["split"] = split_name
                saved += 1

        print(f"[GeneTokenJoint] Saved latent features for {saved} patients to {out_dir}")
        return out_dir

    # ──────────────────────────────────────────────────────────────────
    #  Training step — adds cond dropout for CFG support at inference
    # ──────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        """Diffusion loss with three mutually exclusive training modes + CF loss.

        A single uniform draw ``r ~ U[0,1)`` assigns each sample in the batch
        to exactly one mode:

          [0, p_cond)              → **cond dropout**  (zeros genomic cond)
              Required for classifier-free guidance (CFG) at inference.
              Teaches the model an unconditional distribution alongside the
              conditioned one.

          [p_cond, p_cond+p_xt)   → **xT forcing**  (overrides t to near-T)
              At timestep t ≈ T, x_t ≈ N(0,I) regardless of x_start, so the
              model receives almost no image-content signal.  It must rely on
              the genomic conditioning vector to predict the noise direction.
              Directly trains Goals 2 (cross-conditioning) and 3 (random-noise
              generation): the model learns *which direction to denoise* from
              pure noise given a genomic cond vector.

          [p_cond+p_xt, 1.0)      → **normal**  (real image + real cond)
              Standard diffusion loss for image-guided reconstruction (Goal 1).

        ``cond_feature_dropout`` is applied after mode selection as a
        per-dimension regulariser on the conditioning pathway (not mutually
        exclusive with the mode above, but only active in cond+xt modes since
        the cond-dropout mode already zeros the whole vector).

        **Counterfactual margin loss** (when ``counterfactual_loss_weight > 0``):
        A second forward pass is run with shuffled conditioning vectors using the
        *same* noise sample so x_t is identical.  The margin loss penalises the
        model whenever swapped-cond loss ≤ matched-cond loss + margin, i.e. it
        explicitly forces the model to produce higher diffusion loss when the
        genomic conditioning does not match the image.  Both passes share noise
        so the gap is purely due to conditioning, not noise variance.
        """
        from torch.amp.autocast_mode import autocast

        with autocast(device_type="cuda", enabled=self.conf.fp16):
            imgs = batch["img"].to(self.device)
            genomic = batch["genomic"].to(self.device, dtype=torch.float32)

            cond = self.encode_genomic(genomic)

            p_cond = float(self.joint_cfg.get("cond_dropout_prob", 0.0))
            p_xt   = float(self.joint_cfg.get("xt_zero_prob", 0.0))

            # Single draw → mutually exclusive mode assignment
            r = torch.rand(len(imgs), device=imgs.device)
            cond_drop_mask = r < p_cond
            xt_force_mask  = (r >= p_cond) & (r < p_cond + p_xt)

            # Mode 1 — cond dropout (CFG training)
            if cond_drop_mask.any():
                cond = cond.clone()
                cond[cond_drop_mask] = 0.0

            # Per-dimension feature dropout (applied after cond dropout, mild regulariser)
            p_feat = float(self.joint_cfg.get("cond_feature_dropout", 0.0))
            if p_feat > 0.0:
                cond = F.dropout(cond, p=p_feat, training=self.training)

            # Mode 2 — xT forcing: override t to near-T so x_t ≈ N(0,I)
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

            # Sample noise ONCE — shared between main and CF forward passes so
            # x_t is identical and (swapped_loss - main_loss) reflects only the
            # conditioning difference, not independent noise variance.
            shared_noise = torch.randn_like(imgs)

            losses = self.sampler.training_losses(
                model=self.model,
                x_start=imgs,
                cond=cond,
                t=t,
                noise=shared_noise,
                model_kwargs={"cond": cond},
            )
            main_loss_per_sample = losses["loss"]
            main_loss = main_loss_per_sample.mean()

            # ── Counterfactual margin loss ────────────────────────────────
            cf_weight = float(self.joint_cfg.get("counterfactual_loss_weight", 0.0))
            cf_margin_loss = torch.zeros((), device=imgs.device, dtype=main_loss.dtype)
            swap_gap = torch.zeros((), device=imgs.device, dtype=main_loss.dtype)
            swap_idx = _build_swapped_indices(cond.shape[0], cond.device)
            if cf_weight > 0.0 and swap_idx is not None:
                cond_swapped = cond[swap_idx]
                losses_swapped = self.sampler.training_losses(
                    model=self.model,
                    x_start=imgs,
                    cond=cond_swapped,
                    t=t,
                    noise=shared_noise,  # same x_t, only cond differs
                    model_kwargs={"cond": cond_swapped},
                )
                swapped_loss_per_sample = losses_swapped["loss"]
                margin = float(self.joint_cfg.get("counterfactual_margin", 0.05))
                cf_margin_loss = F.relu(
                    margin + main_loss_per_sample.detach() - swapped_loss_per_sample
                ).mean()
                swap_gap = (swapped_loss_per_sample.detach() - main_loss_per_sample.detach()).mean()

            loss = main_loss + cf_weight * cf_margin_loss

        self.log("loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(imgs))
        self.log("loss_step", loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=len(imgs))
        self.log("loss_main_step", main_loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=len(imgs))
        self.log("loss_cf_margin_step", cf_margin_loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=len(imgs))
        self.log("cf_swap_gap_step", swap_gap, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=len(imgs))

        if self.global_rank == 0 and hasattr(self, "logger") and hasattr(self.logger, "experiment"):
            tb = self.logger.experiment  # type: ignore[union-attr]
            tb.add_scalar("loss", loss.item(), self.num_samples)
            tb.add_scalar("frac/cond_drop", cond_drop_mask.float().mean().item(), self.num_samples)
            tb.add_scalar("frac/xt_force",  xt_force_mask.float().mean().item(), self.num_samples)

        monitor_every = int(self.joint_cfg.get("counterfactual_monitor_every_n_steps", 200))
        zero_threshold = float(self.joint_cfg.get("counterfactual_zero_threshold", 1e-4))
        if (
            self.global_rank == 0
            and cf_weight > 0.0
            and monitor_every > 0
            and self.global_step % monitor_every == 0
        ):
            cf_margin_value = float(cf_margin_loss.detach().item())
            message = (
                f"[CF-TUNING] step={self.global_step} total={float(loss.detach().item()):.6f} "
                f"main={float(main_loss.detach().item()):.6f} cf_margin={cf_margin_value:.6f} "
                f"swap_gap={float(swap_gap.detach().item()):.6f} "
                f"weight={cf_weight:.3f} margin={float(self.joint_cfg.get('counterfactual_margin', 0.05)):.3f}"
            )
            if cf_margin_value <= zero_threshold:
                message += " | hint: cf_margin≈0 -> swapped loss already > matched + margin (good!)"
            else:
                message += " | hint: cf_margin>0 -> model not yet differentiating conditioning"
            print(message)

        return {"loss": loss}


def build_gene_token_transformer_conf(joint_cfg: dict):
    """Reuse baseline joint config builder for diffusion/training defaults."""
    return build_conf(joint_cfg)
