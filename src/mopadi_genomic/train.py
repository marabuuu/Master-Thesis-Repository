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

import numpy as np
import torch

from mopadi.train_diff_autoenc import LitModel
from mopadi.utils.dist_utils import get_world_size

from .config import GenomicTrainConfig
from .dataset import ZipTilesWithGenomicFeatures

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

        # ── Validation dataset (no tile capping) ─────────────────────────
        self.val_data = ZipTilesWithGenomicFeatures(
            zip_dir=self.conf.zip_dir,
            genomic_h5_dir=self.conf.genomic_feature_dir,
            patient_splits_path=self.conf.patient_splits_path,
            split="val",
            max_tiles_by_subtype=None,   # no cap on validation
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
        if self.global_rank == 0:
            loss_val = out["loss"] if isinstance(out, dict) else out
            self.logger.experiment.add_scalar(
                "loss/train", loss_val.item(), self.num_samples
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
            loss_cond = self.all_gather(losses_cond["loss"]).mean()

            # ── Loss with shuffled conditioning ──────────────────────────
            # Permute genomic vectors within the batch so each image is
            # paired with a random (likely wrong) patient's features.
            perm = torch.randperm(feats.size(0), device=feats.device)
            feats_shuffled = feats[perm]
            losses_shuffled = self.sampler.training_losses(
                model=self.ema_model,
                x_start=imgs,
                cond=feats_shuffled,
                t=t,                    # same noise timesteps for fair comparison
                model_kwargs={"cond": feats_shuffled},
            )
            loss_shuffled = self.all_gather(losses_shuffled["loss"]).mean()

        if self.global_rank == 0:
            gap = (loss_shuffled - loss_cond).item()
            self.logger.experiment.add_scalar("loss/val",          loss_cond.item(),     self.num_samples)
            self.logger.experiment.add_scalar("loss/val_shuffled", loss_shuffled.item(), self.num_samples)
            self.logger.experiment.add_scalar("cond/gap",          gap,                  self.num_samples)

        return loss_cond

    # ------------------------------------------------------------------
    # Sanity check (replaces parent's WebDataset-specific check)
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        """Lightweight sanity check on the ZIP genomic dataset.

        The parent implementation calls ``expand_shards()`` which expects tar
        shards and would raise an error with our ZIP-based dataset.  We
        replace it with a single-batch check that verifies shape and dtype.
        """
        if self.global_rank != 0:
            return

        loader = self.conf.make_loader(
            self.train_data,
            shuffle=False,
            batch_size=min(4, len(self.train_data)),
            num_worker=0,
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
