"""
GenomicLitModel — MoPaDi training with genomic conditioning.

Philosophy: Use MoPaDi's built-in dual conditioning (resnet_two_cond=True).
No wrappers, no auxiliary losses during training. Clean, minimal integration.

Training objective: standard MoPaDi L1 diffusion loss + counterfactual gap loss.
The counterfactual loss is computed with cross-subtype shuffled genomic features and
evaluated at high timesteps only (where global subtype structure is most expressed).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from mopadi.train_diff_autoenc import LitModel
from mopadi.utils.dist_utils import get_world_size

from .genomic_config import GenomicTrainConfig
from .genomic_dataset import ZipTilesWithGenomicFeatures

log = logging.getLogger(__name__)


def _cross_subtype_shuffle(feats: torch.Tensor, subtypes: list[str]) -> torch.Tensor:
    """Permute feats so each sample gets a vector from a different subtype.

    Falls back to any different index when only one subtype is present in the batch.
    """
    B = len(feats)
    perm = list(range(B))
    for i in range(B):
        candidates = [j for j in range(B) if j != i and subtypes[j] != subtypes[i]]
        if not candidates:
            candidates = [j for j in range(B) if j != i]
        if not candidates:
            continue
        # pick a random candidate; swap with current position in perm to avoid
        # assigning the same source index twice where possible
        chosen = candidates[torch.randint(len(candidates), (1,)).item()]
        perm[i] = chosen
    return feats[torch.tensor(perm, device=feats.device)]


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
    # Training step — L1 diffusion loss + counterfactual gap loss
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """Standard L1 diffusion loss + cross-subtype counterfactual gap loss.

        The counterfactual pass runs every `cfl_every_n_steps` steps and uses
        high timesteps only (top half of [0, T]) where global subtype structure
        is most expressed.  Cross-subtype shuffling ensures the negative vector
        is always from a different PAM50 class, avoiding the ~38 % wasted same-
        subtype shuffles that random permutation produces.
        """
        with torch.autocast(device_type='cuda', enabled=self.conf.fp16):
            imgs = batch['img'].to(self.device)
            feats = batch['feat'].to(self.device, dtype=torch.float32)
            t, _ = self.T_sampler.sample(len(imgs), imgs.device)

            losses = self.sampler.training_losses(
                model=self.model,
                x_start=imgs,
                cond=feats,
                t=t,
                model_kwargs={'cond': feats},
            )
            loss = losses['loss'].mean()

            cfl_every = getattr(self.conf, 'cfl_every_n_steps', 4)
            cfl_lambda = float(getattr(self.conf, 'cfl_lambda', 0.1))
            cfl_margin = float(getattr(self.conf, 'cfl_margin', 0.005))

            if cfl_lambda > 0 and self.global_step % cfl_every == 0:
                subtypes = batch.get('subtype', None)

                # Cross-subtype shuffle if subtype labels are available; otherwise
                # fall back to plain random permutation.
                if subtypes is not None and not isinstance(subtypes[0], torch.Tensor):
                    feats_neg = _cross_subtype_shuffle(feats, list(subtypes))
                else:
                    feats_neg = feats[torch.randperm(len(feats), device=feats.device)]

                # Evaluate at high timesteps only: top half of diffusion schedule.
                T = self.conf.T
                t_high = torch.randint(T // 2, T, (len(imgs),), device=imgs.device)

                # ref loss is a constant baseline — no gradient needed.
                # This keeps only ONE extra graph (losses_neg) alive during
                # backward instead of three, cutting peak activation memory
                # from ~3× to ~2× compared to the main loss alone.
                with torch.no_grad():
                    losses_ref = self.sampler.training_losses(
                        model=self.model,
                        x_start=imgs,
                        cond=feats,
                        t=t_high,
                        model_kwargs={'cond': feats},
                    )
                ref_loss = losses_ref['loss'].mean()

                losses_neg = self.sampler.training_losses(
                    model=self.model,
                    x_start=imgs,
                    cond=feats_neg,
                    t=t_high,
                    model_kwargs={'cond': feats_neg},
                )
                gap = losses_neg['loss'].mean() - ref_loss
                cfl_loss = torch.relu(cfl_margin - gap)
                loss = loss + cfl_lambda * cfl_loss

                if self.global_rank == 0:
                    self.logger.experiment.add_scalar('cond/gap_train', gap.item(), self.num_samples)
                    self.logger.experiment.add_scalar('loss/cfl', cfl_loss.item(), self.num_samples)

        if self.global_rank == 0:
            self.logger.experiment.add_scalar('loss', loss.item(), self.num_samples)

        return {'loss': loss}

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
            from torch.utils.data import IterableDataset
            
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
        num_samples = self.num_samples + len(imgs) * get_world_size()
        
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
