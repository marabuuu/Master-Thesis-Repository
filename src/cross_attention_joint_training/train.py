#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-attention joint training entry point.
Reuses joint_training pipeline, swaps model for CrossAttentionJointLitModel.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import yaml
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

try:
    from joint_training.train import _count_genes
except ImportError:  # pragma: no cover
    from src.joint_training.train import _count_genes  # type: ignore[import-not-found]

try:
    from cross_attention_joint_training.model import CrossAttentionJointLitModel, build_cross_conf
except ImportError:  # pragma: no cover
    from src.cross_attention_joint_training.model import CrossAttentionJointLitModel, build_cross_conf  # type: ignore[import-not-found]


def _deep_update(base: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_config_paths(config_dict: dict, repo_root: Path) -> dict:
    """Recursively resolve relative paths in config against repo_root."""
    def _resolve_path(value: str) -> str:
        repo_candidate = (repo_root / value).resolve()
        normalized = value[2:] if value.startswith("./") else value

        # In this workspace layout, `data/`, `dataframes/`, and `experiments/`
        # live next to
        # the repo root. Use parent fallback when repo-local target is absent.
        if normalized.startswith(("data/", "dataframes/", "experiments/")):
            parent_candidate = (repo_root.parent / normalized).resolve()
            if parent_candidate.exists() or not repo_candidate.exists():
                return str(parent_candidate)

        return str(repo_candidate)

    if isinstance(config_dict, dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                _resolve_config_paths(value, repo_root)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _resolve_config_paths(item, repo_root)
            elif isinstance(value, str):
                if not value.startswith('/') and (
                    value.startswith('./')
                    or value.startswith('../')
                    or any(part in value for part in ['data/', 'experiments/', 'dataframes/', 'slurm/', 'src/'])
                ):
                    config_dict[key] = _resolve_path(value)
    return config_dict


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

    if verbose:
        if check_val_every_n_epoch > 1:
            print(f"[CrossJoint] Validation every {check_val_every_n_epoch} epochs")
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
        check_val_every_n_epoch=check_val_every_n_epoch,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=0,
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

    repo_root = Path(__file__).resolve().parents[2]
    full_cfg = _resolve_config_paths(full_cfg, repo_root)

    base_cfg = full_cfg.get("joint_training", {})
    overrides = full_cfg.get("cross_attention_joint_training", {})
    if isinstance(base_cfg, dict) and isinstance(overrides, dict):
        joint_cfg = _deep_update(base_cfg, overrides)
    elif isinstance(overrides, dict) and overrides:
        joint_cfg = overrides
    else:
        joint_cfg = full_cfg.get("joint_training", full_cfg)

    if "out_dir" not in overrides:
        joint_cfg["out_dir"] = f"{base_cfg.get('out_dir', 'experiments')}_cross_attention"

    run_cross_attention_training(joint_cfg)


if __name__ == "__main__":
    main()
