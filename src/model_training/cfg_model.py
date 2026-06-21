"""MoPaDi backbone trained with CFG dropout on genomic features.

The backbone is the conditioned denoiser — no separate adapter.
It receives genomic features via AdaGN (style conditioning) and CFG
dropout replaces them with zeros on a fraction of batches.

At inference: ε_guided = ε_null + s * (ε_cond − ε_null)

Health metrics:
  cond/signal  E[‖ε_cond − ε_null‖²] — must grow during training
  cond/gap     loss/val_shuffled − loss/val — positive means conditioning
               carries patient-specific information
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from collections import defaultdict
import numpy as np
import torch
from typing import Any, cast
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from mopadi.train_diff_autoenc import LitModel, ema as _ema_fn
from mopadi.utils.dist_utils import get_world_size

from .config import GDAConfig
from .dataset import ZipTilesWithGenomicFeatures, patient_id_from_tile_path, _make_orthogonal_binary_codes, COHORT_INDEX

log = logging.getLogger(__name__)


def _stratified_subset_indices(
    tile_paths: list[str],
    subtype_map: dict[str, str],
    limit: int,
    seed: int,
) -> list[int]:
    """Select a deterministic mixed subset of tile indices across subtypes."""
    by_subtype: dict[str, list[int]] = defaultdict(list)
    for idx, tile_path in enumerate(tile_paths):
        pid = patient_id_from_tile_path(tile_path)
        by_subtype[subtype_map.get(pid, "unknown")].append(idx)

    groups = [indices for subtype, indices in sorted(by_subtype.items()) if subtype != "unknown"]
    if not groups:
        return list(range(min(limit, len(tile_paths))))

    rng = random.Random(seed)
    for indices in groups:
        rng.shuffle(indices)

    selected: list[int] = []
    cursor = 0
    while len(selected) < limit:
        progressed = False
        for indices in groups:
            if cursor < len(indices):
                selected.append(indices[cursor])
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
        cursor += 1

    if len(selected) < limit:
        remaining = [idx for indices in groups for idx in indices[cursor:]]
        rng.shuffle(remaining)
        selected.extend(remaining[: limit - len(selected)])

    return sorted(selected[:limit])


def _build_subtype_mean_feats(
    genomic_cache: dict,
    subtype_map: dict,
    normalize: bool,
) -> dict:
    """Mean-pool genomic features per subtype; optionally L2-normalise."""
    from collections import defaultdict
    buckets: dict = defaultdict(list)
    for pid, feat in genomic_cache.items():
        subtype = subtype_map.get(pid, "unknown")
        if subtype != "unknown":
            buckets[subtype].append(feat)
    result = {}
    for subtype, feats in buckets.items():
        mean = torch.stack(feats).mean(0)
        if normalize:
            mean = F.normalize(mean, p=2, dim=-1)
        result[subtype] = mean
    return result


class CfgBackboneLitModel(LitModel):
    """
    Standard CFG training on MoPaDi backbone with genomic style conditioning.
    No adapter — the backbone handles both denoising and genomic conditioning.
    """

    automatic_optimization = False

    def __init__(self, conf: GDAConfig):
        super().__init__(conf)
        self.conf: GDAConfig = conf

        if conf.backbone_ckpt_path:
            self._load_backbone_weights(conf.backbone_ckpt_path)

        if conf.conditioning_type == "class_embed":
            self.class_embedding = torch.nn.Embedding(conf.num_classes, conf.feat_dim)
            log.info("CfgBackboneLitModel: class_embedding (%d × %d)", conf.num_classes, conf.feat_dim)

        n_bb = sum(p.numel() for p in self.model.parameters())
        log.info("CfgBackboneLitModel: backbone params=%d", n_bb)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _load_backbone_weights(self, ckpt_path: str) -> None:
        ckpt_file = Path(ckpt_path)
        if not ckpt_file.exists():
            raise FileNotFoundError(f"backbone_ckpt_path not found: {ckpt_path}")

        log.info("Loading backbone weights from: %s", ckpt_file)
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)

        bb_state = {
            k[len("model."):]: v
            for k, v in state.items()
            if k.startswith("model.") and not k.startswith("model.adapter")
        }
        missing, unexpected = self.model.load_state_dict(bb_state, strict=False)
        log.info("model: missing=%d unexpected=%d", len(missing), len(unexpected))

        ema_state = {
            k[len("ema_model."):]: v
            for k, v in state.items()
            if k.startswith("ema_model.") and not k.startswith("ema_model.adapter")
        }
        if ema_state:
            m2, u2 = self.ema_model.load_state_dict(ema_state, strict=False)
            log.info("ema_model: missing=%d unexpected=%d", len(m2), len(u2))

        # Re-initialise the conditioning pathway so it learns from scratch
        # with real genomic inputs (the prior run used zeros_cond, so FiLM
        # layers were trained on a constant input — wrong initialisation for
        # variable genomic conditioning).
        self._reinit_cond_layers()

    def _reinit_cond_layers(self) -> None:
        """Reset style MLP and per-ResBlock FiLM projections to random weights."""
        n_reset = 0
        for model in (self.model, self.ema_model):
            # Style MLP in TimeStyleSeperateEmbed
            if hasattr(model, "time_embed") and hasattr(model.time_embed, "style"):
                for m in model.time_embed.style.modules():
                    if hasattr(m, "reset_parameters"):
                        m.reset_parameters()
                        n_reset += 1
            # cond_emb_layers in every ResBlock
            for module in model.modules():
                if hasattr(module, "cond_emb_layers"):
                    for m in module.cond_emb_layers.modules():
                        if hasattr(m, "reset_parameters"):
                            m.reset_parameters()
                            n_reset += 1
        log.info("Re-initialised %d conditioning sub-modules", n_reset)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        pass  # skip parent's WebDataset sanity check

    @property
    def num_samples(self) -> int:
        return self.global_step * self.conf.batch_size

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        ckpt_sd = checkpoint.get("state_dict", {})
        model_sd = self.state_dict()
        new_keys = model_sd.keys() - ckpt_sd.keys()
        if new_keys:
            ckpt_sd.update({k: model_sd[k] for k in new_keys})
            log.info("on_load_checkpoint: added %d new params from init", len(new_keys))
        checkpoint["state_dict"] = ckpt_sd
        # Reset optimizer states if param-group sizes changed
        opt_states = checkpoint.get("optimizer_states") or []
        if opt_states:
            expected = sum(1 for _ in self.model.parameters())
            saved = len((opt_states[0].get("param_groups") or [{}])[0].get("params", []))
            if saved != expected:
                log.warning("Optimizer state mismatch (saved=%d current=%d) — resetting Adam momentum.", saved, expected)
                checkpoint["optimizer_states"] = []

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def setup(self, stage=None) -> None:
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        kwargs = dict(
            zip_dir=self.conf.zip_dir,
            genomic_h5_dir=self.conf.genomic_feature_dir,
            patient_splits_path=self.conf.patient_splits_path,
            max_tiles_by_subtype=self.conf.max_tiles_by_subtype,
            tile_sampling_seed=self.conf.tile_sampling_seed,
            img_size=self.conf.img_size,
            do_resize=self.conf.do_resize,
            do_normalize=self.conf.do_normalize,
            conditioning_type=self.conf.conditioning_type,
            feat_dim=self.conf.feat_dim,
            normalize_feats=self.conf.normalize_feats,
        )
        self.train_data = ZipTilesWithGenomicFeatures(split="train", **kwargs)
        self.val_data   = ZipTilesWithGenomicFeatures(split="val",   **kwargs)
        if self.global_rank == 0:
            log.info("train=%d  val=%d  tiles", len(self.train_data), len(self.val_data))

        # Pre-compute subtype-balance weights once — these are fixed for the dataset.
        from collections import Counter
        subtype_map = self.train_data._subtype_map
        counts: Counter = Counter()
        for p in self.train_data.tile_paths:
            counts[subtype_map.get(patient_id_from_tile_path(p), "unknown")] += 1
        self._sampler_weights = torch.tensor(
            [1.0 / counts[subtype_map.get(patient_id_from_tile_path(p), "unknown")]
             for p in self.train_data.tile_paths],
            dtype=torch.float32,
        )
        self._sampler_counts = counts
        if self.global_rank == 0:
            log.info("Subtype-balanced sampler: %s", dict(sorted(counts.items())))

        if self.conf.conditioning_type == "real":
            self._subtype_mean_feats = _build_subtype_mean_feats(
                self.train_data._genomic_cache,
                self.train_data._subtype_map,
                normalize=self.conf.normalize_feats,
            )
            log.info("Subtype mean feats for sep metric: %s", sorted(self._subtype_mean_feats))
        elif self.conf.conditioning_type == "one_hot" and self.train_data._orthogonal_codes is not None:
            self._subtype_mean_feats = dict(self.train_data._orthogonal_codes)
            log.info("One-hot codes for sep metric: %s", sorted(self._subtype_mean_feats))

    def train_dataloader(self):
        import torch.utils.data as tud

        sampler = tud.WeightedRandomSampler(
            self._sampler_weights, num_samples=len(self.train_data.tile_paths), replacement=True
        )

        conf = self.conf.clone()
        conf.batch_size = self.batch_size
        return conf.make_loader(self.train_data, drop_last=True, sampler=sampler)

    def val_dataloader(self):
        import torch.utils.data as tud
        limit = self.conf.val_limit_batches * self.conf.batch_size
        if len(self.val_data) > limit:
            indices = _stratified_subset_indices(
                tile_paths=self.val_data.tile_paths,
                subtype_map=self.val_data._subtype_map,
                limit=limit,
                seed=self.conf.seed,
            )
            dataset = tud.Subset(self.val_data, indices)
        else:
            dataset = self.val_data
        conf = self.conf.clone()
        conf.batch_size = self.batch_size
        return conf.make_loader(dataset, shuffle=False, drop_last=False)

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.conf.backbone_lr,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )
        return [opt], []

    # ------------------------------------------------------------------
    # Conditioning helpers
    # ------------------------------------------------------------------

    def _resolve_feats(self, batch: dict) -> torch.Tensor:
        """Return a (B, feat_dim) float conditioning tensor for this batch.

        For class_embed: looks up the learnable embedding by class index.
        For all other types: casts the pre-built feat vector to float and
        optionally L2-normalises it.
        """
        if self.conf.conditioning_type == "class_embed":
            idx = batch["feat"].squeeze(-1).to(self.device, dtype=torch.long)
            return self.class_embedding(idx)
        feats = batch["feat"].to(self.device, dtype=torch.float32)
        if self.conf.normalize_feats:
            feats = F.normalize(feats, p=2, dim=-1)
        return feats

    def _class_embed_pair(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (e_brca, e_lihc) embedding vectors expanded to batch size n.

        Used by the brca_lihc_sep diagnostic for class_embed runs.
        """
        brca_idx = torch.zeros(n, device=self.device, dtype=torch.long)
        lihc_idx = torch.ones(n,  device=self.device, dtype=torch.long)
        return self.class_embedding(brca_idx), self.class_embedding(lihc_idx)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        accum = self.conf.accum_batches

        imgs = batch["img"].to(self.device)
        feats = self._resolve_feats(batch)
        B = imgs.shape[0]

        t, _ = self.T_sampler.sample(B, imgs.device)
        noise = torch.randn_like(imgs)
        x_t = self.sampler.q_sample(imgs, t, noise=noise)
        t_scaled = self.sampler._scale_timesteps(t)

        # CFG dropout: replace conditioning with zeros for cfg_dropout% of samples
        null_mask = (torch.rand(B, device=self.device) < self.conf.cfg_dropout)[:, None]
        cond = torch.where(null_mask, torch.zeros_like(feats), feats)

        eps_pred = self.model.forward(x=x_t, t=t_scaled, x_start=imgs, cond=cond).pred
        loss = F.mse_loss(eps_pred.float(), noise.float())

        self.manual_backward(loss / accum)

        if self.is_last_accum(batch_idx):
            if self.conf.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.conf.grad_clip)
            opt.step()
            opt.zero_grad()

        if self.trainer.is_global_zero:
            exp = getattr(self.logger, "experiment", None)
            if exp is not None:
                exp.add_scalar("loss/train", loss.item(), self.num_samples)

        # Conditioning health: ‖backbone(cond) − backbone(null)‖²
        # Must grow as backbone learns to use the genomic signal.
        _diag_interval = 500 * self.conf.batch_size_effective
        if self.trainer.is_global_zero and self.global_step > 0 and (self.num_samples % _diag_interval < self.conf.batch_size):
            with torch.no_grad():
                bm = min(16, B)
                # Reuse the already-sampled x_t and t rather than re-sampling.
                zeros = torch.zeros(bm, self.conf.feat_dim, device=self.device)
                eps_c = self.ema_model.forward(x=x_t[:bm], t=t_scaled[:bm], x_start=None, cond=feats[:bm].detach()).pred
                eps_n = self.ema_model.forward(x=x_t[:bm], t=t_scaled[:bm], x_start=None, cond=zeros).pred
                self.logger.experiment.add_scalar("cond/signal", (eps_c - eps_n).pow(2).mean().item(), self.num_samples)

                # Universal probe: fixed orthogonal codes for BRCA and LIHC.
                # Logged for ALL conditioning types so runs are directly comparable.
                # zero/noise → stays ~0 (no cohort signal); RNA/one_hot → should grow.
                if not hasattr(self, "_diag_codes"):
                    codes = _make_orthogonal_binary_codes(self.conf.feat_dim, normalize=True)
                    self._diag_codes = {k: v.to(self.device) for k, v in codes.items()}
                e_brca = self._diag_codes["TCGA-BRCA"].unsqueeze(0).expand(bm, -1)
                e_lihc = self._diag_codes["TCGA-LIHC"].unsqueeze(0).expand(bm, -1)
                eps_brca = self.ema_model.forward(x=x_t[:bm], t=t_scaled[:bm], x_start=None, cond=e_brca).pred
                eps_lihc = self.ema_model.forward(x=x_t[:bm], t=t_scaled[:bm], x_start=None, cond=e_lihc).pred
                self.logger.experiment.add_scalar("cond/brca_lihc_sep", (eps_brca - eps_lihc).pow(2).mean().item(), self.num_samples)

                # class_embed: probe via the learned embedding vectors instead
                if self.conf.conditioning_type == "class_embed":
                    e_brca_emb, e_lihc_emb = self._class_embed_pair(bm)
                    eps_brca_emb = self.ema_model.forward(x=x_t[:bm], t=t_scaled[:bm], x_start=None, cond=e_brca_emb).pred
                    eps_lihc_emb = self.ema_model.forward(x=x_t[:bm], t=t_scaled[:bm], x_start=None, cond=e_lihc_emb).pred
                    self.logger.experiment.add_scalar("cond/brca_lihc_sep_embed", (eps_brca_emb - eps_lihc_emb).pow(2).mean().item(), self.num_samples)

                # real/one_hot: probe with mean RNA features or orthogonal codes per subtype
                elif self.conf.conditioning_type in ("real", "one_hot") and hasattr(self, "_subtype_mean_feats"):
                    sf = self._subtype_mean_feats
                    if "TCGA-BRCA" in sf and "TCGA-LIHC" in sf:
                        e_brca_rna = sf["TCGA-BRCA"].to(self.device).unsqueeze(0).expand(bm, -1)
                        e_lihc_rna = sf["TCGA-LIHC"].to(self.device).unsqueeze(0).expand(bm, -1)
                        eps_brca_rna = self.ema_model.forward(x=x_t[:bm], t=t_scaled[:bm], x_start=None, cond=e_brca_rna).pred
                        eps_lihc_rna = self.ema_model.forward(x=x_t[:bm], t=t_scaled[:bm], x_start=None, cond=e_lihc_rna).pred
                        self.logger.experiment.add_scalar("cond/brca_lihc_sep_rna", (eps_brca_rna - eps_lihc_rna).pow(2).mean().item(), self.num_samples)
                    elif "Basal" in sf and "LumA" in sf:
                        e_basal = sf["Basal"].to(self.device).unsqueeze(0).expand(bm, -1)
                        e_luma  = sf["LumA"].to(self.device).unsqueeze(0).expand(bm, -1)
                        eps_basal = self.ema_model.forward(x=x_t[:bm], t=t_scaled[:bm], x_start=None, cond=e_basal).pred
                        eps_luma  = self.ema_model.forward(x=x_t[:bm], t=t_scaled[:bm], x_start=None, cond=e_luma).pred
                        self.logger.experiment.add_scalar("cond/basal_luma_sep", (eps_basal - eps_luma).pow(2).mean().item(), self.num_samples)

        _si_interval = max(
            self.conf.reconstruct_every_samples,
            getattr(self.conf, "sample_every_samples", self.conf.reconstruct_every_samples),
        )
        if self.trainer.is_global_zero and (self.num_samples % _si_interval < self.conf.batch_size):
            self._log_sample_images(imgs.detach(), feats.detach())

        return {"loss": loss}

    # ------------------------------------------------------------------
    # EMA + batch end
    # ------------------------------------------------------------------

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if self.is_last_accum(batch_idx):
            _ema_fn(self.model, self.ema_model, self.conf.ema_decay)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        imgs = batch["img"].to(self.device)
        feats = self._resolve_feats(batch)
        B = imgs.shape[0]

        t, _ = self.T_sampler.sample(B, imgs.device)
        noise = torch.randn_like(imgs)
        x_t = self.sampler.q_sample(imgs, t, noise=noise)
        t_scaled = self.sampler._scale_timesteps(t)
        zeros = torch.zeros(B, self.conf.feat_dim, device=self.device)

        subtypes = batch["subtype"]
        cross_feats = feats.clone()
        if self.conf.val_swap_basal_luma:
            # BRCA-only: give Basal tiles LumA features and vice versa.
            # Maximally wrong for the Basal/LumA contrast — the key diagnostic.
            basal_idx = torch.tensor([i for i, s in enumerate(subtypes) if s == "Basal"],
                                     device=self.device)
            luma_idx  = torch.tensor([i for i, s in enumerate(subtypes) if s == "LumA"],
                                     device=self.device)
            if len(basal_idx) > 0 and len(luma_idx) > 0:
                cross_feats[basal_idx] = feats[luma_idx[torch.randint(len(luma_idx), (len(basal_idx),), device=self.device)]]
                cross_feats[luma_idx]  = feats[basal_idx[torch.randint(len(basal_idx), (len(luma_idx),), device=self.device)]]
            else:
                cross_feats = feats[torch.randperm(B, device=self.device)]
        else:
            # PoC (BRCA+LIHC): swap organ-level features for a strong mismatch signal.
            brca_idx = torch.tensor([i for i, s in enumerate(subtypes) if "BRCA" in s],
                                     device=self.device)
            lihc_idx = torch.tensor([i for i, s in enumerate(subtypes) if "LIHC" in s],
                                     device=self.device)
            if len(brca_idx) > 0 and len(lihc_idx) > 0:
                cross_feats[brca_idx] = feats[lihc_idx[torch.randint(len(lihc_idx), (len(brca_idx),), device=self.device)]]
                cross_feats[lihc_idx] = feats[brca_idx[torch.randint(len(brca_idx), (len(lihc_idx),), device=self.device)]]
            else:
                cross_feats = feats[torch.randperm(B, device=self.device)]

        with torch.no_grad():
            eps_cond     = self.ema_model.forward(x=x_t, t=t_scaled, x_start=imgs, cond=feats).pred
            eps_shuffled = self.ema_model.forward(x=x_t, t=t_scaled, x_start=imgs,
                                                   cond=cross_feats).pred
            noise_f = noise.float()
            loss_val      = F.mse_loss(eps_cond.float(), noise_f)
            loss_shuffled = F.mse_loss(eps_shuffled.float(), noise_f)

        self.log("_val_loss",          loss_val,      on_step=False, on_epoch=True, sync_dist=True, prog_bar=True,  logger=False)
        self.log("_val_loss_shuffled", loss_shuffled, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False, logger=False)
        return loss_val

    def on_validation_epoch_end(self) -> None:
        if self.trainer.state.stage == "sanity_check":
            return
        val_loss      = self.trainer.callback_metrics.get("_val_loss")
        val_shuffled  = self.trainer.callback_metrics.get("_val_loss_shuffled")
        if val_loss is not None and self.trainer.is_global_zero:
            self.logger.experiment.add_scalar("loss/val",          val_loss.item(),     self.num_samples)
            if val_shuffled is not None:
                self.logger.experiment.add_scalar("loss/val_shuffled", val_shuffled.item(), self.num_samples)
                self.logger.experiment.add_scalar("cond/gap",
                    val_shuffled.item() - val_loss.item(), self.num_samples)
        if val_loss is not None:
            self.log("loss/val_ckpt", val_loss, prog_bar=False, sync_dist=False)

    # ------------------------------------------------------------------
    # Sample visualisation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _log_sample_images(self, imgs: torch.Tensor, feats: torch.Tensor) -> None:
        if not hasattr(self, "_sample_imgs"):
            b = min(8, imgs.shape[0])
            self._sample_imgs  = imgs[:b].clone().cpu()
            self._sample_feats = feats[:b].clone().cpu()

        imgs_s  = self._sample_imgs.to(self.device)
        # _sample_feats are already resolved floats (embedding vectors for class_embed,
        # or raw/normalised genomic vectors for other types). No re-lookup needed.
        feats_s = self._sample_feats.to(self.device, dtype=torch.float32)
        if self.conf.normalize_feats and self.conf.conditioning_type != "class_embed":
            feats_s = F.normalize(feats_s, p=2, dim=-1)
        b = imgs_s.shape[0]
        zeros = torch.zeros(b, self.conf.feat_dim, device=self.device)

        t_vis = torch.full((b,), 750, device=self.device, dtype=torch.long)
        x_t_vis = self.sampler.q_sample(imgs_s, t_vis, noise=torch.randn_like(imgs_s))
        t_sc = self.sampler._scale_timesteps(t_vis)

        sac  = torch.as_tensor(self.sampler.sqrt_alphas_cumprod,           device=self.device, dtype=torch.float32)
        somc = torch.as_tensor(self.sampler.sqrt_one_minus_alphas_cumprod, device=self.device, dtype=torch.float32)

        def _x0(x_t_, eps_, t_):
            a  = sac[t_].view(-1, 1, 1, 1)
            b_ = somc[t_].view(-1, 1, 1, 1)
            return ((x_t_ - b_ * eps_) / a).clamp(-1, 1)

        def _vis(delta):
            lo = delta.flatten(1).min(1).values.view(-1, 1, 1, 1)
            hi = delta.flatten(1).max(1).values.view(-1, 1, 1, 1)
            return (2 * (delta - lo) / (hi - lo + 1e-8) - 1).clamp(-1, 1)

        # EMA model: unconditional vs conditioned
        eps_null = self.ema_model.forward(x=x_t_vis, t=t_sc, x_start=imgs_s, cond=zeros).pred.float()
        eps_cond = self.ema_model.forward(x=x_t_vis, t=t_sc, x_start=imgs_s, cond=feats_s).pred.float()

        rows = [
            imgs_s.clamp(-1, 1),              # original tiles
            x_t_vis.clamp(-1, 1),             # noised at t=750
            _x0(x_t_vis, eps_null, t_vis),    # unconditional x0
            _x0(x_t_vis, eps_cond, t_vis),    # conditioned x0
            _vis(eps_cond - eps_null),         # conditioning delta (where they differ)
        ]
        grid = make_grid(torch.cat(rows), nrow=b, normalize=True, value_range=(-1, 1), padding=2)
        self.logger.experiment.add_image("samples/ema", grid, self.num_samples)

        samples_dir = Path(self.conf.logdir) / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        save_image(grid, samples_dir / f"samples_{self.num_samples:010d}.png")

        # DDIM generation: unconditional vs CFG-guided
        self._log_ddim_samples(feats_s[:2], samples_dir)

    @torch.no_grad()
    def _log_ddim_samples(self, feats_ref, samples_dir: Path) -> None:
        from mopadi.diffusion.base import DummyReturn

        n = feats_ref.shape[0]
        zeros = torch.zeros(n, self.conf.feat_dim, device=self.device)
        backbone = self.ema_model
        sampler = self.sampler

        def _make_model(scale: float):
            _s, _f, _z = scale, feats_ref, zeros
            class _M(torch.nn.Module):
                def __init__(self_):
                    super().__init__()
                    self_._bb = backbone
                def forward(self_, x, t, **kw):
                    t_sc = sampler._scale_timesteps(t)
                    eps_null = backbone.forward(x=x, t=t_sc, x_start=None, cond=_z).pred
                    if _s == 0.0:
                        return DummyReturn(pred=eps_null)
                    eps_cond = backbone.forward(x=x, t=t_sc, x_start=None, cond=_f).pred
                    return DummyReturn(pred=eps_null + _s * (eps_cond - eps_null))
            return _M()

        try:
            sampler = self.conf._make_diffusion_conf(self.conf.T_eval).make_sampler()
            noise = torch.randn(n, 3, self.conf.img_size, self.conf.img_size, device=self.device)
            uncond = sampler.sample(model=cast(Any, _make_model(0.0)), shape=noise.shape, noise=noise, model_kwargs={}, progress=False)
            guided = sampler.sample(model=cast(Any, _make_model(5.0)), shape=noise.shape, noise=noise, model_kwargs={}, progress=False)
        except Exception:
            log.warning("_log_ddim_samples failed", exc_info=True)
            return

        rows = torch.cat([uncond.clamp(-1, 1), guided.clamp(-1, 1)])
        grid = make_grid(rows, nrow=n, normalize=True, value_range=(-1, 1), padding=2)
        self.logger.experiment.add_image("samples/ddim", grid, self.num_samples)
        save_image(grid, samples_dir / f"ddim_{self.num_samples:010d}.png")
