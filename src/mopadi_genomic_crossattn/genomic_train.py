"""
GenomicLitModel — PyTorch Lightning module for training MoPaDi from scratch
with gene-expression conditioning (no image feature extractor).

Overrides from MoPaDi's ``LitModel``:

``setup()``
    Creates ``ZipTilesWithGenomicFeatures`` datasets for train and val.
    Does NOT initialise any image feature extractor.

``on_fit_start()``
    Replaces MoPaDi's WebDataset sanity check (which expects tar shards)
    with a lightweight check on the ZIP-based genomic dataset.

``training_step()``
    Calls the parent implementation and additionally logs ``loss/train``
    so TensorBoard groups it on the same chart as ``loss/val``.

``validation_step()``
    Computes the diffusion loss on a held-out val batch (no gradients,
    EMA model) and logs ``loss/val`` at the same ``num_samples`` x-axis
    as the training scalars.  Validation is run every ``val_every_steps``
    training steps (configured in ``run_genomic_training.py``), capped at
    ``limit_val_batches`` batches so it does not slow training significantly.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch
import torch.nn.functional as F

from mopadi.train_diff_autoenc import LitModel
from mopadi.utils.dist_utils import get_world_size

from .genomic_config import GenomicTrainConfig
from .genomic_dataset import ZipTilesWithGenomicFeatures

log = logging.getLogger(__name__)


class GenomicLitModel(LitModel):
    """MoPaDi diffusion autoencoder conditioned on patient gene-expression.

    Parameters
    ----------
    conf:
        A ``GenomicTrainConfig`` instance (subclass of MoPaDi's ``TrainConfig``).
    """

    def __init__(self, conf: GenomicTrainConfig):
        super().__init__(conf)
        # Overwrite to the typed subclass for IDE support; the parent __init__
        # already stored conf as self.conf — this just re-annotates the type.
        self.conf: GenomicTrainConfig = conf

    @staticmethod
    def _non_identity_permutation(n: int, device: torch.device) -> torch.Tensor:
        """Return a permutation where no element maps to itself (if n > 1)."""
        perm = torch.randperm(n, device=device)
        if n > 1 and torch.equal(perm, torch.arange(n, device=device)):
            perm = torch.roll(perm, shifts=1)
        return perm

    # ------------------------------------------------------------------
    # Dataset creation (overrides parent)
    # ------------------------------------------------------------------

    def setup(self, stage=None) -> None:
        """Create genomic datasets; skip image feature extractor entirely."""
        # ── Seed ────────────────────────────────────────────────────────
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            log.info("Local seed: %d", seed)

        # ── Training dataset ─────────────────────────────────────────────
        self.train_data = ZipTilesWithGenomicFeatures(
            zip_dir=self.conf.zip_dir,
            genomic_h5_dir=self.conf.genomic_feature_dir,
            patient_splits_path=self.conf.patient_splits_path,
            split="train",
            max_tiles_by_subtype=self.conf.max_tiles_by_subtype,
            tile_sampling_seed=self.conf.tile_sampling_seed,
            img_size=self.conf.img_size,
            do_resize=self.conf.do_resize,
            do_normalize=self.conf.do_normalize,
            cache_pickle_tiles_path=getattr(self.conf, "cache_pickle_tiles_path", None),
        )

        # ── Validation dataset ────────────────────────────────────────────
        self.val_data = ZipTilesWithGenomicFeatures(
            zip_dir=self.conf.zip_dir,
            genomic_h5_dir=self.conf.genomic_feature_dir,
            patient_splits_path=self.conf.patient_splits_path,
            split="val",
            max_tiles_by_subtype=self.conf.max_tiles_by_subtype,
            tile_sampling_seed=self.conf.tile_sampling_seed,
            img_size=self.conf.img_size,
            do_resize=self.conf.do_resize,
            do_normalize=self.conf.do_normalize,
        )

        # ── NO image feature extractor ───────────────────────────────────
        # self.feat_extractor remains None (set by parent __init__).
        # self.model.feat_extractor is also None — the model's forward() will
        # raise ValueError if cond is None, but our dataset always provides
        # feat, so this path is never reached during normal training.
        if self.global_rank == 0:
            log.info(
                "train tiles: %d  |  val tiles: %d  |  genomic features: %d-dim",
                len(self.train_data),
                len(self.val_data),
                next(iter(self.train_data._genomic_cache.values())).shape[0]
                if self.train_data._genomic_cache else 0,
            )

    # ------------------------------------------------------------------
    # Val dataloader (parent only defines train_dataloader)
    # ------------------------------------------------------------------

    def val_dataloader(self):
        """Return a DataLoader for the validation split, capped at val_limit_batches.

        MoPaDi's parent ``LitModel`` only defines ``train_dataloader``;
        without this method Lightning silently skips the entire val loop.

        Lightning's ``limit_val_batches`` is unreliable with integer
        ``val_check_interval`` in 2.5.x, so we cap the dataset directly.
        
        Uses stratified sampling across subtypes when possible: picks roughly equal
        tiles per subtype to ensure cond/gap is representative across all classes.
        Randomizes which tiles are selected each epoch (seed = base + epoch).
        """
        import torch.utils.data as tud
        from collections import defaultdict

        # self.conf.batch_size is global batch size in MoPaDi's TrainConfig.
        # Using local self.batch_size here would shrink validation by world_size
        # under DDP (e.g. 4x fewer val batches on 4 GPUs).
        limit = self.conf.val_limit_batches * self.conf.batch_size
        
        # Stratified sampling: group tiles by subtype and pick roughly equal from each
        try:
            subtype_map = self.val_data._subtype_map if hasattr(self.val_data, "_subtype_map") else {}
            tile_paths = self.val_data.tile_paths if hasattr(self.val_data, "tile_paths") else []
            
            if subtype_map and tile_paths and len(subtype_map) > 0:
                # group by subtype
                subtype_indices = defaultdict(list)
                for idx, path in enumerate(tile_paths):
                    from .genomic_dataset import patient_id_from_tile_path
                    pid = patient_id_from_tile_path(path)
                    subtype = subtype_map.get(pid, "unknown")
                    subtype_indices[subtype].append(idx)
                
                # stratified sampling: pick equal tiles from each subtype
                rng = random.Random(self.current_epoch + 42)  # randomize per epoch
                selected_indices = []
                tiles_per_subtype = max(1, limit // max(1, len(subtype_indices)))
                for subtype, indices in subtype_indices.items():
                    sampled = rng.sample(indices, min(tiles_per_subtype, len(indices)))
                    selected_indices.extend(sampled)
                
                selected_indices = selected_indices[:limit]
                dataset = tud.Subset(self.val_data, selected_indices)
                
                if self.global_rank == 0:
                    log.info(
                        f"Val sampling: stratified across {len(subtype_indices)} subtypes, "
                        f"using {len(selected_indices)} tiles (epoch {self.current_epoch})"
                    )
            else:
                # fallback: simple limit
                dataset = (
                    tud.Subset(self.val_data, list(range(limit)))
                    if len(self.val_data) > limit
                    else self.val_data
                )
        except Exception as e:
            # fallback on any error — still randomize per epoch so different
            # tiles are evaluated across epochs for data augmentation purposes
            if self.global_rank == 0:
                log.warning(f"Stratified val sampling failed: {e}, using shuffled limit")
            if len(self.val_data) > limit:
                rng = random.Random(self.current_epoch + 42)
                indices = list(range(len(self.val_data)))
                rng.shuffle(indices)
                dataset = tud.Subset(self.val_data, indices[:limit])
            else:
                dataset = self.val_data
        
        conf = self.conf.clone()
        conf.batch_size = self.batch_size
        return conf.make_loader(dataset, shuffle=False, drop_last=False)

    # ------------------------------------------------------------------
    # TensorBoard logging — grouped train / val loss curves
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """Forward to parent, then add a grouped ``loss/train`` scalar.

        MoPaDi's parent logs the scalar ``"loss"`` directly via
        ``add_scalar``.  We additionally write ``"loss/train"`` so that
        TensorBoard places the training curve on the same chart as
        ``"loss/val"`` (TensorBoard groups series by the prefix before
        the first ``/``).
        """
        out = super().training_step(batch, batch_idx)

        # --- Counterfactual conditioning objective ---
        # Maximises the gap (loss_shuffled − loss_cond) using a softplus penalty
        # instead of a hard hinge.  Softplus(-gap/T) always provides gradient:
        # near log(2) ≈ 0.69 when gap ≈ 0, decays to ≈0 when gap >> T.
        # This avoids the hinge's saturation problem where gradient drops to zero
        # once gap > margin, letting the model drift back toward ignoring genomics.
        cf_weight = float(getattr(self.conf, "counterfactual_loss_weight", 0.0))
        cf_temperature = max(1e-6, float(getattr(self.conf, "counterfactual_temperature", 0.05)))
        cf_every = max(1, int(getattr(self.conf, "counterfactual_every_n_steps", 1)))
        cf_warmup = max(0, int(getattr(self.conf, "counterfactual_warmup_steps", 0)))

        loss_val = out["loss"] if isinstance(out, dict) else out
        if (
            cf_weight > 0.0
            and self.global_step >= cf_warmup
            and (self.global_step % cf_every == 0)
            and batch["feat"].shape[0] > 1
        ):
            imgs = batch["img"].to(self.device)
            # Use the real genomic vectors for the CF loss so the gap measures
            # correct vs. shuffled patient conditioning only.
            feats = batch["feat"].to(self.device, dtype=torch.float32)

            t, _ = self.T_sampler.sample(len(imgs), imgs.device)
            shared_noise = torch.randn_like(imgs)

            losses_cond = self.sampler.training_losses(
                model=self.model,
                x_start=imgs,
                cond=feats,
                t=t,
                noise=shared_noise,
                model_kwargs={"cond": feats},
            )
            loss_cond = losses_cond["loss"].mean()

            perm = self._non_identity_permutation(feats.size(0), feats.device)
            feats_shuffled = feats[perm]
            losses_shuffled = self.sampler.training_losses(
                model=self.model,
                x_start=imgs,
                cond=feats_shuffled,
                t=t,
                noise=shared_noise,
                model_kwargs={"cond": feats_shuffled},
            )
            loss_shuffled = losses_shuffled["loss"].mean()

            gap = loss_shuffled - loss_cond
            # Softplus: -log σ(gap/T) = log(1 + exp(-gap/T))
            # Gradient is always non-zero; effective "soft margin" ≈ T.
            cf_penalty = F.softplus(-gap / cf_temperature)
            cf_term = cf_weight * cf_penalty
            loss_val = loss_val + cf_term

            if isinstance(out, dict):
                out["loss"] = loss_val
            else:
                out = loss_val

            self.log(
                "cond/gap_train",
                gap,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )
            self.log(
                "loss/counterfactual",
                cf_term,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )

        # Expose a Lightning-native metric key for ModelCheckpoint(monitor="loss").
        # add_scalar writes only to TensorBoard and is invisible to callbacks.
        self.log("loss", loss_val, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log("loss/train", loss_val, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        if self.global_rank == 0:
            if self.global_step % 500 == 0 and self.global_step != getattr(self, "_last_train_log_step", -1):
                self._last_train_log_step = self.global_step
                log.info(
                    "step %6d | samples %10d | loss/train %.4f",
                    self.global_step, self.num_samples, loss_val.item(),
                )
        return out

    def validation_step(self, batch, batch_idx):
        """Compute diffusion loss and conditioning gap on a val batch.

        Logs three scalars — all under the same ``num_samples`` x-axis as
        the training loss, so the curves align in TensorBoard:

        ``loss/val``
            Standard val loss with the *correct* genomic conditioning vector.
            Tracks model quality; goes down as training progresses.

        ``loss/val_shuffled``
            Val loss with a *randomly permuted* conditioning vector
            (same images, wrong patient genomic features).

        ``cond/gap``
            ``loss/val_shuffled − loss/val``.  Near zero early in training
            (model ignores conditioning); grows positive once the model
            learns to exploit the genomic signal.  If this stays flat at
            zero after tens of thousands of steps, the genomic conditioning
            is not being learned — a reliable signal to cancel the run.
        """
        # Lightning runs a brief sanity check (2 batches) before training to
        # validate the val loop.  We skip it here because on_fit_start() already
        # does a dedicated sanity check, and logging during the sanity phase
        # can cause issues with partially initialised trainer state.
        if self.trainer.sanity_checking:
            return None

        imgs = batch["img"].to(self.device)
        feats = batch["feat"].to(self.device, dtype=torch.float32)

        with torch.no_grad():
            t, _ = self.T_sampler.sample(len(imgs), imgs.device)

            # ── Loss with correct conditioning ───────────────────────────
            losses_cond = self.sampler.training_losses(
                model=self.ema_model,
                x_start=imgs,
                cond=feats,
                t=t,
                model_kwargs={"cond": feats},
            )
            loss_cond = losses_cond["loss"].mean()

            # ── Loss with shuffled conditioning ──────────────────────────
            # Permute genomic vectors within the batch so each image is
            # paired with a random (likely wrong) patient's features.
            perm = self._non_identity_permutation(feats.size(0), feats.device)
            feats_shuffled = feats[perm]
            losses_shuffled = self.sampler.training_losses(
                model=self.ema_model,
                x_start=imgs,
                cond=feats_shuffled,
                t=t,                    # same noise timesteps for fair comparison
                model_kwargs={"cond": feats_shuffled},
            )
            loss_shuffled = losses_shuffled["loss"].mean()

        if self.global_rank == 0:
            gap = (loss_shuffled - loss_cond).item()
            self.logger.experiment.add_scalar("loss/val",          loss_cond.item(),     self.num_samples)
            self.logger.experiment.add_scalar("loss/val_shuffled", loss_shuffled.item(), self.num_samples)
            self.logger.experiment.add_scalar("cond/gap",          gap,                  self.num_samples)

        # Lightning-native logs for callbacks (e.g. ModelCheckpoint monitor).
        self.log("loss/val", loss_cond, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("loss/val_shuffled", loss_shuffled, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("cond/gap", loss_shuffled - loss_cond, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        # Accumulate for on_validation_epoch_end summary log line.
        if not hasattr(self, "_val_losses"):
            self._val_losses: list = []
        self._val_losses.append(loss_cond.item())
        if not hasattr(self, "_val_gaps"):
            self._val_gaps: list = []
        self._val_gaps.append((loss_shuffled - loss_cond).item())

        return loss_cond

    def on_validation_epoch_end(self) -> None:
        """Log a one-line summary of the completed validation pass."""
        if self.trainer.sanity_checking:
            return
        if not getattr(self, "_val_losses", None):
            return
        if self.global_rank == 0:
            mean_loss = sum(self._val_losses) / len(self._val_losses)
            mean_gap  = sum(self._val_gaps)   / len(self._val_gaps)
            log.info(
                "step %6d | samples %10d | loss/val %.4f | cond/gap %.6e  (%d batches)",
                self.global_step, self.num_samples,
                mean_loss, mean_gap, len(self._val_losses),
            )
        self._val_losses = []
        self._val_gaps   = []

    # ------------------------------------------------------------------
    # Disable LPIPS/FID evaluation (requires feat_extractor which is None)
    # ------------------------------------------------------------------

    def evaluate_scores(self) -> None:
        """No-op: parent's LPIPS/FID evaluation calls feat_extractor.extract_feats()
        which is None in genomic mode.  We use loss/val and cond/gap instead."""
        pass

    def log_sample(self, x_start, cond):
        """Skip reconstruction logging at sample 0 to avoid startup OOM.

        MoPaDi's timing helper treats ``num_samples == 0`` as an active logging
        window. In large 512x512 runs, that triggers an expensive model+EMA
        reconstruction pass immediately at startup, which can exceed GPU memory.
        """
        if self.num_samples <= 0:
            return
        return super().log_sample(x_start, cond)

    # ------------------------------------------------------------------
    # Sanity check (replaces parent's WebDataset-specific check)
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        """Lightweight sanity check on the ZIP genomic dataset.

        The parent implementation calls ``expand_shards()`` which expects tar
        shards and would raise an error with our ZIP-based dataset.  We
        replace it with a single-batch check that verifies shape and dtype.
        """
        actual_world_size = get_world_size()
        expected_world_size = int(getattr(self, "expected_world_size", 1))
        if expected_world_size > 1 and actual_world_size != expected_world_size:
            raise RuntimeError(
                "Distributed world-size mismatch: requested "
                f"{expected_world_size} GPU ranks but got world_size={actual_world_size}. "
                "This usually means Lightning fell back to single-GPU execution. "
                "Check SLURM env and trainer strategy before launching a long run."
            )

        if self.global_rank != 0:
            return

        log.info(
            "Distributed setup: world_size=%d, global_rank=%d, local_rank=%d",
            actual_world_size,
            self.global_rank,
            self.local_rank,
        )

        loader = self.conf.make_loader(
            self.train_data,
            shuffle=False,
            batch_size=min(4, len(self.train_data)),
            num_worker=1,
        )
        try:
            batch = next(iter(loader))
        except StopIteration:
            raise RuntimeError(
                "Training dataset is empty — check zip_dir, genomic_feature_dir "
                "and patient_splits_path."
            )

        img = batch["img"]
        feat = batch["feat"]

        expected_img_shape = (None, 3, self.conf.img_size, self.conf.img_size)
        expected_feat_shape = (None, self.conf.feat_dim)

        def _shape_ok(t: torch.Tensor, expected) -> bool:
            return all(
                e is None or s == e
                for s, e in zip(t.shape, expected)
            )

        if not _shape_ok(img, expected_img_shape):
            raise RuntimeError(
                f"Unexpected image shape: {tuple(img.shape)}, "
                f"expected (B, 3, {self.conf.img_size}, {self.conf.img_size})"
            )
        if not _shape_ok(feat, expected_feat_shape):
            raise RuntimeError(
                f"Unexpected feat shape: {tuple(feat.shape)}, "
                f"expected (B, {self.conf.feat_dim})"
            )

        log.info(
            "[sanity] img %s %s  |  feat %s %s  ✓",
            tuple(img.shape), img.dtype,
            tuple(feat.shape), feat.dtype,
        )
