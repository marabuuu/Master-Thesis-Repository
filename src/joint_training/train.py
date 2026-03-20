#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Joint Genomic VAE + Diffusion Training — entry point.

Follows mopadi's training pattern (sample-count scheduling, TensorBoard,
ModelCheckpoint, DDP) but uses our GenomicTileDataset and JointLitModel.

Usage:
    python -m src.joint_training.train --config src/config.yaml
    python run_pipeline.py --config src/config.yaml --stage joint_training
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

try:
    from joint_training.model import JointLitModel, build_conf
except ImportError:
    from src.joint_training.model import JointLitModel, build_conf  # type: ignore[import-not-found]


def _count_genes(joint_cfg: dict) -> int:
    """Count number of gene columns from the CSV header (fast, no full read)."""
    df = pd.read_csv(joint_cfg["csv_path"], nrows=0)
    drop = {joint_cfg.get("patient_col", "Patient_ID"), joint_cfg.get("label_col")} - {None}

    gene_list_path = joint_cfg.get("gene_list_path")
    if gene_list_path and os.path.exists(gene_list_path):
        with open(gene_list_path) as f:
            gene_list = [line.strip() for line in f if line.strip()]
        available = [g for g in gene_list if g in df.columns]
        if available:
            return len(available)

    return len([c for c in df.columns if c not in drop])


def run_joint_training(joint_cfg: dict, verbose: bool = True) -> None:
    """Run joint genomic encoder + diffusion training (called from pipeline or CLI)."""
    seed = joint_cfg.get("seed", 42)
    pl.seed_everything(seed)

    # Build mopadi TrainConfig
    conf = build_conf(joint_cfg)

    # Determine n_genes from CSV header
    n_genes = _count_genes(joint_cfg)
    if verbose:
        print(f"[Joint] n_genes = {n_genes}")

    # Create model (subclass of mopadi's LitModel)
    model = JointLitModel(conf, joint_cfg, n_genes)

    # ── Trainer setup (follows mopadi's train() pattern) ──────────────
    gpus = joint_cfg.get("gpus", [0])

    if not os.path.exists(conf.logdir):
        os.makedirs(conf.logdir)

    checkpoint = ModelCheckpoint(
        dirpath=conf.logdir,
        save_last=True,
        save_top_k=1,
        every_n_train_steps=int(conf.save_every_samples // conf.batch_size_effective),
    )

    # Resume from existing checkpoint
    ckpt_path = os.path.join(conf.logdir, 'last.ckpt')
    if os.path.exists(ckpt_path):
        if verbose:
            print(f"[Joint] Resuming from {ckpt_path}")
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
    )

    trainer.fit(model, ckpt_path=ckpt_path)

    if verbose:
        print(f"[Joint] Training complete. Checkpoints in {conf.logdir}")


def extract_latents(joint_cfg: dict, ckpt_path: str | None = None,
                    split: str = "all", verbose: bool = True) -> str:
    """Load a trained JointLitModel and save per-patient h5 latent features.

    Parameters
    ----------
    joint_cfg : dict
        The ``joint_training`` section from config.yaml.
    ckpt_path : str or None
        Checkpoint to load. Defaults to ``<out_dir>/joint/last.ckpt``.
    split : str
        Which split to extract: "train", "val", "test", or "all".
    verbose : bool
        Print progress.

    Returns
    -------
    str
        Path to the directory containing h5 files.
    """
    conf = build_conf(joint_cfg)
    n_genes = _count_genes(joint_cfg)

    if ckpt_path is None:
        ckpt_path = os.path.join(conf.logdir, "last.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if verbose:
        print(f"[Joint] Loading checkpoint: {ckpt_path}")

    model = JointLitModel.load_from_checkpoint(
        ckpt_path, conf=conf, joint_cfg=joint_cfg, n_genes=n_genes,
    )
    model.eval()
    model.setup()  # build datasets so we know which patients are in each split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    latent_dir = joint_cfg.get("latent_dir") or os.path.join(conf.base_dir, "latents")
    return model.save_latent_features(out_dir=latent_dir, split=split)


def main():
    parser = argparse.ArgumentParser(description="Joint Genomic VAE + Diffusion Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--extract-latents", action="store_true",
                        help="Extract VAE latent features instead of training")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path (for --extract-latents)")
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "val", "test", "all"],
                        help="Which split to extract latents for")
    args = parser.parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)
    joint_cfg = full_cfg.get("joint_training", full_cfg)

    if args.extract_latents:
        extract_latents(joint_cfg, ckpt_path=args.ckpt, split=args.split)
    else:
        run_joint_training(joint_cfg)


if __name__ == "__main__":
    main()
