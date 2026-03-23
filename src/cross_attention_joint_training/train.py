#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-attention joint training entry point.
Reuses joint_training pipeline, swaps model for CrossAttentionJointLitModel.
"""

from __future__ import annotations

import argparse
import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

try:
    from joint_training.train import _count_genes
except ImportError:  # pragma: no cover
    from src.joint_training.train import _count_genes  # type: ignore[import-not-found]

from .model import CrossAttentionJointLitModel, build_cross_conf


def run_cross_attention_training(joint_cfg: dict, verbose: bool = True) -> None:
    seed = joint_cfg.get("seed", 42)
    pl.seed_everything(seed)

    conf = build_cross_conf(joint_cfg)
    n_genes = _count_genes(joint_cfg)
    if verbose:
        print(f"[CrossJoint] n_genes = {n_genes}")

    model = CrossAttentionJointLitModel(conf, joint_cfg, n_genes)

    gpus = joint_cfg.get("gpus", [0])

    if not os.path.exists(conf.logdir):
        os.makedirs(conf.logdir)

    checkpoint = ModelCheckpoint(
        dirpath=conf.logdir,
        save_last=True,
        save_top_k=1,
        every_n_train_steps=int(conf.save_every_samples // conf.batch_size_effective),
    )

    ckpt_path = os.path.join(conf.logdir, 'last.ckpt')
    if os.path.exists(ckpt_path):
        if verbose:
            print(f"[CrossJoint] Resuming from {ckpt_path}")
    else:
        ckpt_path = None

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=conf.logdir, name=None, version='',
    )

    if len(gpus) > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = 'auto'

    val_check_interval = int(joint_cfg.get("val_check_interval", 1))
    limit_val_batches = float(joint_cfg.get("limit_val_batches", 1.0))
    if verbose:
        if val_check_interval > 1:
            print(f"[CrossJoint] Validation every {val_check_interval} epochs")
        if limit_val_batches < 1.0:
            print(f"[CrossJoint] Using {limit_val_batches*100:.0f}% of val set per check")

    trainer = pl.Trainer(
        max_epochs=int(joint_cfg.get("epochs", 100)),
        limit_train_batches=int(conf.steps_per_epoch),
        devices=gpus,
        accelerator='gpu' if gpus else 'cpu',
        strategy=strategy,
        precision="16-mixed" if conf.fp16 else 32,
        callbacks=[checkpoint, LearningRateMonitor()],
        logger=tb_logger,
        accumulate_grad_batches=int(conf.accum_batches),
        val_check_interval=val_check_interval,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=2,
    )

    trainer.fit(model, ckpt_path=ckpt_path)

    if verbose:
        print(f"[CrossJoint] Training complete. Checkpoints in {conf.logdir}")


def main():
    parser = argparse.ArgumentParser(description="Cross-attention joint training")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)
    joint_cfg = full_cfg.get("joint_training", full_cfg)

    run_cross_attention_training(joint_cfg)


if __name__ == "__main__":
    main()
