#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gene-token transformer joint training entrypoint (Phase 1 baseline)."""

from __future__ import annotations

import argparse
import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from pathlib import Path

try:
    from joint_training.train import _count_genes
except ImportError:  # pragma: no cover
    from src.joint_training.train import _count_genes  # type: ignore[import-not-found]

try:
    from utils.config_utils import resolve_config_paths as _resolve_config_paths
except ImportError:  # pragma: no cover
    from src.utils.config_utils import resolve_config_paths as _resolve_config_paths  # type: ignore[import-not-found]

try:
    from gene_token_transformer_joint_training.model import (
        GeneTokenTransformerJointLitModel,
        build_gene_token_transformer_conf,
    )
except ImportError:  # pragma: no cover
    from src.gene_token_transformer_joint_training.model import (  # type: ignore[import-not-found]
        GeneTokenTransformerJointLitModel,
        build_gene_token_transformer_conf,
    )


def run_gene_token_transformer_training(joint_cfg: dict, verbose: bool = True) -> None:
    seed = joint_cfg.get("seed", 42)
    pl.seed_everything(seed)

    conf = build_gene_token_transformer_conf(joint_cfg)
    n_genes = _count_genes(joint_cfg)
    if verbose:
        print(f"[GeneTokenJoint] n_genes = {n_genes}")

    model = GeneTokenTransformerJointLitModel(conf, joint_cfg, n_genes)

    gpus = joint_cfg.get("gpus", [0])

    # When launched via torchrun each OS process already owns one GPU.
    # Passing devices=[0,1] in that context causes PL to re-spawn subprocesses
    # → only 1/1 DDP processes. With LOCAL_RANK set, use devices=1 instead.
    import os as _os
    if _os.environ.get("LOCAL_RANK") is not None:
        # Launched via torchrun: PL is in TorchElastic mode and does NOT
        # re-spawn processes. It validates: devices * num_nodes == WORLD_SIZE,
        # so we must pass the total GPU count, not 1.
        devices = int(_os.environ.get("WORLD_SIZE", len(gpus)))
    else:
        devices = gpus

    if not os.path.exists(conf.logdir):
        os.makedirs(conf.logdir)

    check_val_every_n_epoch = int(joint_cfg.get("val_check_interval", 1))
    limit_val_batches = float(joint_cfg.get("limit_val_batches", 1.0))

    save_top_k = int(joint_cfg.get("save_top_k", 3))
    every_n_train_steps = max(1, int(conf.save_every_samples // conf.batch_size_effective))
    if save_top_k > 0:
        checkpoint = ModelCheckpoint(
            dirpath=conf.logdir,
            save_last=True,
            save_top_k=save_top_k,
            monitor="val_loss",
            mode="min",
            filename="epoch{epoch:03d}-step{step:08d}",
            every_n_epochs=max(1, check_val_every_n_epoch),
        )
    else:
        checkpoint = ModelCheckpoint(
            dirpath=conf.logdir,
            save_last=True,
            save_top_k=save_top_k,
            filename="epoch{epoch:03d}-step{step:08d}",
            every_n_train_steps=every_n_train_steps,
        )

    ckpt_path = os.path.join(conf.logdir, "last.ckpt")
    if os.path.exists(ckpt_path):
        if verbose:
            print(f"[GeneTokenJoint] Resuming from {ckpt_path}")
    else:
        ckpt_path = None

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=conf.logdir, name=None, version="",
    )

    from pytorch_lightning.strategies import DDPStrategy
    if isinstance(devices, list) and len(devices) > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    elif _os.environ.get("LOCAL_RANK") is not None:
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = "auto"

    if verbose:
        if check_val_every_n_epoch > 1:
            print(f"[GeneTokenJoint] Validation every {check_val_every_n_epoch} epochs")
        if limit_val_batches < 1.0:
            print(f"[GeneTokenJoint] Using {limit_val_batches*100:.0f}% of val set per check")

    trainer = pl.Trainer(
        max_epochs=int(joint_cfg.get("epochs", 100)),
        limit_train_batches=int(conf.steps_per_epoch),
        devices=devices,
        accelerator="gpu" if gpus else "cpu",
        strategy=strategy,
        precision="16-mixed" if conf.fp16 else 32,
        callbacks=[checkpoint, LearningRateMonitor()],
        logger=tb_logger,
        accumulate_grad_batches=int(conf.accum_batches),
        check_val_every_n_epoch=check_val_every_n_epoch,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, ckpt_path=ckpt_path)

    if verbose:
        print(f"[GeneTokenJoint] Training complete. Checkpoints in {conf.logdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gene-token transformer joint training")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parents[2]
    full_cfg = _resolve_config_paths(full_cfg, repo_root)

    joint_cfg = full_cfg.get(
        "gene_token_transformer_joint_training",
        full_cfg.get("joint_training", full_cfg),
    )

    run_gene_token_transformer_training(joint_cfg)


if __name__ == "__main__":
    main()
