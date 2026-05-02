"""
Custom PyTorch Lightning callback for composite checkpoint selection.

Tracks both loss/val and cond/gap metrics and saves checkpoints based on a
combined score: score = loss/val - alpha * cond_gap_normalized.
This avoids selecting checkpoints that ignore conditioning entirely.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT

log = logging.getLogger(__name__)


class CompositeMetricCheckpoint(Callback):
    """
    Save best checkpoint using a composite metric: loss/val - alpha * cond/gap_norm.
    
    Maintains a rolling window of recent val_loss and cond/gap values, normalizes
    cond/gap, and selects checkpoints with the best combined score.
    
    Parameters
    ----------
    monitor_loss: str
        Key for validation loss metric (e.g., "loss/val").
    monitor_gap: str
        Key for conditioning gap metric (e.g., "cond/gap").
    alpha: float
        Weight for cond/gap in composite score; higher = prioritize conditioning.
    dirpath: str
        Directory to save checkpoints.
    window_size: int
        Number of recent evaluations to use for normalization.
    save_top_k: int
        Keep top k checkpoints by composite score.
    """

    def __init__(
        self,
        monitor_loss: str = "loss/val",
        monitor_gap: str = "cond/gap",
        alpha: float = 1.0,
        dirpath: str = "checkpoints",
        window_size: int = 20,
        save_top_k: int = 3,
    ):
        self.monitor_loss = monitor_loss
        self.monitor_gap = monitor_gap
        self.alpha = alpha
        self.dirpath = Path(dirpath)
        self.window_size = window_size
        self.save_top_k = save_top_k

        self.loss_values = deque(maxlen=window_size)
        self.gap_values = deque(maxlen=window_size)
        self.best_composite_score = float("inf")
        self.saved_ckpts = []  # list of (score, ckpt_path)

    def on_validation_end(self, trainer: Trainer, pl_module) -> None:
        """Called at end of validation epoch; compute composite score and save if best."""
        metrics = trainer.callback_metrics
        loss_val = metrics.get(self.monitor_loss, None)
        gap_val = metrics.get(self.monitor_gap, None)

        if loss_val is None or gap_val is None:
            return

        loss_val = float(loss_val.item() if hasattr(loss_val, "item") else loss_val)
        gap_val = float(gap_val.item() if hasattr(gap_val, "item") else gap_val)

        self.loss_values.append(loss_val)
        self.gap_values.append(gap_val)

        # normalize cond/gap using rolling window (min-max or z-score)
        if len(self.gap_values) > 1:
            gap_min, gap_max = min(self.gap_values), max(self.gap_values)
            if gap_max > gap_min:
                gap_norm = (gap_val - gap_min) / (gap_max - gap_min)
            else:
                gap_norm = 0.0
        else:
            gap_norm = gap_val

        # composite score: lower is better
        # loss/val wants to be small, cond/gap wants to be large
        # so we compute: loss/val - alpha * (normalized_cond/gap)
        # when gap_norm is high, score decreases (better checkpoint)
        composite_score = loss_val - self.alpha * gap_norm

        if pl_module.global_rank == 0:
            log.info(
                f"Validation @ step {trainer.global_step}: "
                f"loss/val={loss_val:.6f}, cond/gap={gap_val:.6f}, "
                f"gap_norm={gap_norm:.4f}, composite_score={composite_score:.6f}"
            )

        # Always save last.ckpt from ALL ranks for safe resumption.
        # trainer.save_checkpoint calls ddp.barrier internally — must be called
        # from all ranks, not just rank 0, or other ranks will time out.
        if pl_module.global_rank == 0:
            self.dirpath.mkdir(parents=True, exist_ok=True)
        last_ckpt_path = self.dirpath / "last.ckpt"
        trainer.save_checkpoint(str(last_ckpt_path))

        # save best composite checkpoint if score improved
        if composite_score < self.best_composite_score:
            self.best_composite_score = composite_score
            ckpt_path = (
                self.dirpath / f"best-composite-step{trainer.global_step}.ckpt"
            )
            # Rank 0 copies last.ckpt → named composite checkpoint (no extra barrier).
            if pl_module.global_rank == 0:
                import shutil
                shutil.copy2(str(last_ckpt_path), str(ckpt_path))
                log.info(f"Saved composite checkpoint: {ckpt_path}")

            # keep only top_k checkpoints (all ranks maintain consistent state
            # since callback_metrics are synced; only rank 0 touches the filesystem)
            self.saved_ckpts.append((composite_score, ckpt_path))
            self.saved_ckpts.sort()
            if len(self.saved_ckpts) > self.save_top_k:
                _, old_ckpt = self.saved_ckpts.pop()
                if pl_module.global_rank == 0 and old_ckpt.exists():
                    old_ckpt.unlink()
                    log.info(f"Removed old checkpoint: {old_ckpt}")
