#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gene-token + cross-attention joint training entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor

try:
    from joint_training.train import _count_genes
except ImportError:  # pragma: no cover
    from src.joint_training.train import _count_genes  # type: ignore[import-not-found]

try:
    from gene_token_cross_attention_joint_training.model import (
        GeneTokenCrossAttentionJointLitModel,
        build_gene_token_cross_attention_conf,
    )
except ImportError:  # pragma: no cover
    from src.gene_token_cross_attention_joint_training.model import (  # type: ignore[import-not-found]
        GeneTokenCrossAttentionJointLitModel,
        build_gene_token_cross_attention_conf,
    )

try:
    from utils.config_utils import resolve_config_paths as _resolve_config_paths, deep_update as _deep_update
except ImportError:  # pragma: no cover
    from src.utils.config_utils import resolve_config_paths as _resolve_config_paths, deep_update as _deep_update  # type: ignore[import-not-found]

try:
    from utils.training_utils import (
        build_checkpoint_callback,
        choose_ddp_strategy,
        ensure_logdir,
        find_resume_checkpoint,
        resolve_devices_for_launch,
    )
except ImportError:  # pragma: no cover
    from src.utils.training_utils import (  # type: ignore[import-not-found]
        build_checkpoint_callback,
        choose_ddp_strategy,
        ensure_logdir,
        find_resume_checkpoint,
        resolve_devices_for_launch,
    )


def run_gene_token_cross_attention_training(joint_cfg: dict, verbose: bool = True) -> None:
    seed = joint_cfg.get("seed", 42)
    pl.seed_everything(seed)

    conf = build_gene_token_cross_attention_conf(joint_cfg)
    n_genes = _count_genes(joint_cfg)
    if verbose:
        print(f"[GeneTokenCrossJoint] n_genes = {n_genes}")

    model = GeneTokenCrossAttentionJointLitModel(conf, joint_cfg, n_genes)

    gpus = joint_cfg.get("gpus", [0])

    devices = resolve_devices_for_launch(gpus)

    ensure_logdir(conf.logdir)

    check_val_every_n_epoch = int(joint_cfg.get("val_check_interval", 1))
    limit_val_batches = float(joint_cfg.get("limit_val_batches", 1.0))

    checkpoint = build_checkpoint_callback(
        conf=conf,
        joint_cfg=joint_cfg,
        check_val_every_n_epoch=check_val_every_n_epoch,
        filename="{epoch:03d}-{step:08d}",
        auto_insert_metric_name=False,
    )

    ckpt_path = find_resume_checkpoint(conf.logdir, verbose=verbose, prefix="[GeneTokenCrossJoint]")

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=conf.logdir, name=None, version="",
    )

    strategy = choose_ddp_strategy(
        gpus=gpus,
        devices=devices,
        find_unused_parameters=False,
        use_local_rank=True,
    )

    if verbose:
        if check_val_every_n_epoch > 1:
            print(f"[GeneTokenCrossJoint] Validation every {check_val_every_n_epoch} epochs")
        if limit_val_batches < 1.0:
            print(f"[GeneTokenCrossJoint] Using {limit_val_batches*100:.0f}% of val set per check")

    trainer = pl.Trainer(
        max_epochs=int(joint_cfg.get("epochs", 100)),
        limit_train_batches=int(conf.steps_per_epoch),
        devices=devices,
        accelerator="gpu" if gpus else "cpu",
        strategy=strategy,
        precision="16-mixed" if conf.fp16 else ("bf16-mixed" if joint_cfg.get("bf16") else 32),
        callbacks=[checkpoint, LearningRateMonitor()],
        logger=tb_logger,
        accumulate_grad_batches=int(conf.accum_batches),
        check_val_every_n_epoch=check_val_every_n_epoch,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, ckpt_path=ckpt_path)

    if verbose:
        print(f"[GeneTokenCrossJoint] Training complete. Checkpoints in {conf.logdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gene-token + cross-attention joint training")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument(
        "--section",
        type=str,
        default="gene_token_cross_attention_joint_training",
        help=(
            "Config section name whose values override gene_token_transformer_joint_training. "
            "Default: gene_token_cross_attention_joint_training."
        ),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parents[2]
    full_cfg = _resolve_config_paths(full_cfg, repo_root)

    base_cfg = full_cfg.get("gene_token_transformer_joint_training", full_cfg.get("joint_training", {}))
    overrides = full_cfg.get(args.section, {})
    joint_cfg = _deep_update(base_cfg, overrides)

    if "out_dir" not in overrides:
        joint_cfg["out_dir"] = f"{base_cfg.get('out_dir', 'experiments')}_cross"

    run_gene_token_cross_attention_training(joint_cfg)


if __name__ == "__main__":
    main()
