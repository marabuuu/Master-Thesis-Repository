"""
GenomicLitModel — MoPaDi training with clean genomic conditioning.

Philosophy: Use MoPaDi's built-in dual conditioning (resnet_two_cond=True).
No wrappers, no auxiliary losses during training. Clean, minimal integration.

The training objective is the standard MoPaDi L1 diffusion loss only.
Genomic signal flows through the existing AdaGN conditioning pathway.
Validation metrics show how much the genomic conditioning helps.
"""

from __future__ import annotations

import logging

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Optional

from mopadi.train_diff_autoenc import LitModel
from mopadi.utils.dist_utils import get_world_size

from .genomic_config import GenomicTrainConfig
from .genomic_dataset import ZipTilesWithGenomicFeatures

log = logging.getLogger(__name__)


class GenomicLitModel(LitModel):
    """MoPaDi diffusion model conditioned on patient gene-expression.

    Inherits all training, validation, and sampling logic from MoPaDi's LitModel.
    Only overrides dataset creation (setup) and validation (val_dataloader, validation_step).

    The model uses MoPaDi's dual-conditioning mechanism:
      - Time embedding: standard timestep -> (B, 512)
      - Feature embedding: genomic vector (512-dim) -> (B, 512) via AdaGN at every ResBlock

    This is controlled by the resnet_two_cond=True flag set in GenomicTrainConfig.__post_init__.

    Parameters
    ----------
    conf:
        A GenomicTrainConfig instance (subclass of MoPaDi's TrainConfig).
    """

    def __init__(self, conf: GenomicTrainConfig):
        super().__init__(conf)
        # Re-annotate for IDE support; parent already stored conf
        self.conf: GenomicTrainConfig = conf

    # ------------------------------------------------------------------
    # Dataset creation (override parent)
    # ------------------------------------------------------------------

    def setup(self, stage=None) -> None:
        """Create genomic datasets; skip image feature extractor entirely."""
        if self.conf.seed is not None:
            pl.seed_everything(self.conf.seed + self.global_rank)

        # Training dataset
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
        )

        # Validation dataset
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

        # NO image feature extractor — we use genomic features directly
        if self.global_rank == 0:
            log.info(
                "GenomicLitModel setup: train_data=%d, val_data=%d, "
                "resnet_two_cond=%s, no feat_extractor",
                len(self.train_data),
                len(self.val_data),
                self.conf.net_beatgans_resnet_two_cond,
            )

    # ------------------------------------------------------------------
    # Val dataloader (parent only defines train_dataloader)
    # ------------------------------------------------------------------

    def val_dataloader(self):
        """Validation dataloader with configurable batch limit."""
        from torch.utils.data import DataLoader

        sampler = None
        if self.trainer is not None and self.trainer.world_size > 1:
            sampler = torch.utils.data.DistributedSampler(
                self.val_data,
                num_replicas=self.trainer.world_size,
                rank=self.global_rank,
                shuffle=False,
                seed=self.conf.seed or 42,
            )

        loader = DataLoader(
            self.val_data,
            batch_size=self.conf.batch_size,
            sampler=sampler,
            num_workers=self.conf.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        # Wrap with limit_batches if configured
        if self.conf.val_limit_batches > 0:
            class LimitedDataLoader:
                def __init__(self, loader, max_batches):
                    self.loader = loader
                    self.max_batches = max_batches

                def __iter__(self):
                    for i, batch in enumerate(self.loader):
                        if i >= self.max_batches:
                            break
                        yield batch

                def __len__(self):
                    return min(len(self.loader), self.max_batches)

            return LimitedDataLoader(loader, self.conf.val_limit_batches)

        return loader

    # ------------------------------------------------------------------
    # Validation step (extend parent to add shuffled baseline)
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        """Compute loss on validation set with and without conditioning.

        Logs:
          - loss/val: diffusion loss with genomic conditioning (ordered)
          - loss/val_shuffled: diffusion loss with random conditioning (baseline)
          - cond/gap: gap = loss/val_shuffled - loss/val (how much genomic helps)
        """
        with torch.autocast(device_type='cuda', enabled=self.conf.fp16):
            imgs = batch['img'].to(self.device)
            feats = batch['feat'].to(self.device, dtype=torch.float32)

            # Loss with correct conditioning (ordered)
            model_kwargs = {'cond': feats}
            t, _ = self.T_sampler.sample(len(imgs), imgs.device)
            
            losses_ordered = self.sampler.training_losses(
                model=self.ema_model,
                x_start=imgs,
                cond=feats,
                t=t,
                model_kwargs=model_kwargs,
            )
            loss_ordered = losses_ordered['loss'].mean()

            # Loss with shuffled conditioning (baseline — how much does genomic info help?)
            # Shuffle features across the batch so each image gets a different patient's genes
            perm = torch.randperm(len(feats), device=feats.device)
            feats_shuffled = feats[perm]
            model_kwargs_shuffled = {'cond': feats_shuffled}

            losses_shuffled = self.sampler.training_losses(
                model=self.ema_model,
                x_start=imgs,
                cond=feats_shuffled,
                t=t,
                model_kwargs=model_kwargs_shuffled,
            )
            loss_shuffled = losses_shuffled['loss'].mean()

            # Gap: how much does genomic conditioning improve the loss?
            # Positive gap means correct conditioning is better (model is using genomic signal)
            gap = loss_shuffled - loss_ordered

        # Log metrics
        self.log("loss/val", loss_ordered, on_step=False, on_epoch=True, sync_dist=True)
        self.log("loss/val_shuffled", loss_shuffled, on_step=False, on_epoch=True, sync_dist=True)
        self.log("cond/gap", gap, on_step=False, on_epoch=True, sync_dist=True)

        if self.global_rank == 0 and batch_idx % 10 == 0:
            log.info(
                f"Val batch {batch_idx}: loss/val={loss_ordered:.6f}, "
                f"loss/val_shuffled={loss_shuffled:.6f}, cond/gap={gap:.6f}"
            )

        return {'loss': loss_ordered}

    # ------------------------------------------------------------------
    # Disable LPIPS/FID (no feat_extractor)
    # ------------------------------------------------------------------

    def evaluate_scores(self) -> None:
        """Skip evaluation; feat_extractor is None."""
        pass

    def log_sample(self, x_start, cond):
        """Skip sample logging if num_samples is 0."""
        if getattr(self.conf, 'num_samples', 0) == 0:
            return
        # Otherwise call parent
        return super().log_sample(x_start, cond)

    # ------------------------------------------------------------------
    # Sanity check (override parent's WebDataset check)
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        """Sanity check on genomic dataset instead of WebDataset."""
        if not hasattr(self, 'train_data'):
            return
        
        # Quick check: can we load a batch?
        try:
            batch = next(iter(self.train_dataloader()))
            assert 'img' in batch and 'feat' in batch
            assert batch['img'].shape[0] > 0
            assert batch['feat'].shape == (batch['img'].shape[0], self.conf.feat_dim)
            log.info(
                f"Sanity check passed: img={batch['img'].shape}, feat={batch['feat'].shape}"
            )
        except Exception as e:
            log.warning(f"Sanity check failed: {e}")
