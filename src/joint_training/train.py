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
from pathlib import Path

import pandas as pd
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

try:
    from joint_training.model import JointLitModel, build_conf
except ImportError:
    from src.joint_training.model import JointLitModel, build_conf  # type: ignore[import-not-found]

try:
    from utils.logging_utils import build_robust_loggers
except ImportError:
    from src.utils.logging_utils import build_robust_loggers  # type: ignore[import-not-found]

try:
    from utils.config_utils import resolve_config_paths as _resolve_config_paths
except ImportError:
    from src.utils.config_utils import resolve_config_paths as _resolve_config_paths  # type: ignore[import-not-found]

try:
    from utils.training_utils import (
        build_checkpoint_callback,
        choose_ddp_strategy,
        ensure_logdir,
        find_resume_checkpoint,
    )
except ImportError:
    from src.utils.training_utils import (  # type: ignore[import-not-found]
        build_checkpoint_callback,
        choose_ddp_strategy,
        ensure_logdir,
        find_resume_checkpoint,
    )


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

    ensure_logdir(conf.logdir)

    check_val_every_n_epoch = int(joint_cfg.get("val_check_interval", 1))  # epochs between validation
    limit_val_batches = float(joint_cfg.get("limit_val_batches", 1.0))  # fraction of val set to use

    checkpoint = build_checkpoint_callback(
        conf=conf,
        joint_cfg=joint_cfg,
        check_val_every_n_epoch=check_val_every_n_epoch,
        filename="epoch{epoch:03d}-step{step:08d}",
    )

    ckpt_path = find_resume_checkpoint(conf.logdir, verbose=verbose, prefix="[Joint]")

    active_loggers = build_robust_loggers(conf.logdir, joint_cfg, verbose=verbose)

    strategy = choose_ddp_strategy(
        gpus=gpus,
        devices=gpus,
        find_unused_parameters=True,
        use_local_rank=False,
    )

    # Validation frequency tuning for faster training

    if verbose:
        if check_val_every_n_epoch > 1:
            print(f"[Joint] Validation every {check_val_every_n_epoch} epochs (reduced frequency)")
        if limit_val_batches < 1.0:
            print(f"[Joint] Using {limit_val_batches*100:.0f}% of validation set per check")

    trainer = pl.Trainer(
        max_epochs=int(joint_cfg.get("epochs", 100)),
        limit_train_batches=int(conf.steps_per_epoch),
        devices=gpus,
        accelerator='gpu' if gpus else 'cpu',
        strategy=strategy,
        precision="16-mixed" if conf.fp16 else 32,
        callbacks=[checkpoint, LearningRateMonitor()],
        logger=active_loggers,
        accumulate_grad_batches=int(conf.accum_batches),
        check_val_every_n_epoch=check_val_every_n_epoch,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, ckpt_path=ckpt_path)

    if verbose:
        print(f"[Joint] Training complete. Checkpoints in {conf.logdir}")


def extract_latents(
    joint_cfg: dict,
    ckpt_path: str | None = None,
    split: str = "all",
    verbose: bool = True,
    expected_variant: str | None = None,
) -> str:
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

    if ckpt_path is None:
        ckpt_path = os.path.join(conf.logdir, "last.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Prefer the checkpoint's own n_genes to avoid config/checkpoint mismatch
    # (e.g., config currently pointing to full-gene CSV while checkpoint was
    # trained on a 512-gene subset).
    n_genes = None
    try:
        ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp = ckpt_meta.get("hyper_parameters", {}) if isinstance(ckpt_meta, dict) else {}
        ckpt_n_genes = hp.get("n_genes") if isinstance(hp, dict) else None
        if ckpt_n_genes is not None:
            n_genes = int(ckpt_n_genes)
    except Exception:
        # Keep backward-compatible fallback below.
        n_genes = None

    if n_genes is None:
        n_genes = _count_genes(joint_cfg)

    if verbose:
        print(f"[Joint] Loading checkpoint: {ckpt_path}")
        print(f"[Joint] Using n_genes={n_genes} for model construction")

    model = None
    loaded_variant: str | None = None
    load_errors: list[str] = []

    # Try GTCA first for gene-token + cross-attention checkpoints.
    try:
        from gene_token_cross_attention_joint_training.model import (  # type: ignore[import-not-found]
            GeneTokenCrossAttentionJointLitModel,
            build_gene_token_cross_attention_conf,
        )
    except ImportError:
        try:
            from src.gene_token_cross_attention_joint_training.model import (  # type: ignore[import-not-found]
                GeneTokenCrossAttentionJointLitModel,
                build_gene_token_cross_attention_conf,
            )
        except ImportError:
            GeneTokenCrossAttentionJointLitModel = None  # type: ignore[assignment]
            build_gene_token_cross_attention_conf = None  # type: ignore[assignment]

    if (
        model is None
        and GeneTokenCrossAttentionJointLitModel is not None
        and build_gene_token_cross_attention_conf is not None
    ):
        try:
            gtca_conf = build_gene_token_cross_attention_conf(joint_cfg)
            model = GeneTokenCrossAttentionJointLitModel.load_from_checkpoint(
                ckpt_path, conf=gtca_conf, joint_cfg=joint_cfg, n_genes=n_genes, strict=False
            )
            loaded_variant = "gene_token_cross_attention_joint_training"
            if verbose:
                print("[Joint] Loaded as GeneTokenCrossAttentionJointLitModel")
        except Exception as ex:
            load_errors.append(f"GeneTokenCrossAttentionJointLitModel: {ex}")
            if verbose:
                print(f"[Joint] GTCA load failed, trying next loader: {ex}")

    # Try gene-token transformer checkpoints.
    try:
        from gene_token_transformer_joint_training.model import (  # type: ignore[import-not-found]
            GeneTokenTransformerJointLitModel,
            build_gene_token_transformer_conf,
        )
    except ImportError:
        try:
            from src.gene_token_transformer_joint_training.model import (  # type: ignore[import-not-found]
                GeneTokenTransformerJointLitModel,
                build_gene_token_transformer_conf,
            )
        except ImportError:
            GeneTokenTransformerJointLitModel = None  # type: ignore[assignment]
            build_gene_token_transformer_conf = None  # type: ignore[assignment]

    if (
        model is None
        and GeneTokenTransformerJointLitModel is not None
        and build_gene_token_transformer_conf is not None
    ):
        try:
            gtt_conf = build_gene_token_transformer_conf(joint_cfg)
            model = GeneTokenTransformerJointLitModel.load_from_checkpoint(
                ckpt_path, conf=gtt_conf, joint_cfg=joint_cfg, n_genes=n_genes, strict=False
            )
            loaded_variant = "gene_token_transformer_joint_training"
            if verbose:
                print("[Joint] Loaded as GeneTokenTransformerJointLitModel")
        except Exception as ex:
            load_errors.append(f"GeneTokenTransformerJointLitModel: {ex}")
            if verbose:
                print(f"[Joint] GTT load failed, trying next loader: {ex}")

    # Cross-attention checkpoints may include extra parameters.
    try:
        from cross_attention_joint_training.model import CrossAttentionJointLitModel  # type: ignore[import-not-found]
    except ImportError:
        try:
            from src.cross_attention_joint_training.model import CrossAttentionJointLitModel  # type: ignore[import-not-found]
        except ImportError:
            CrossAttentionJointLitModel = None  # type: ignore[assignment]

    if model is None and CrossAttentionJointLitModel is not None:
        try:
            model = CrossAttentionJointLitModel.load_from_checkpoint(
                ckpt_path, conf=conf, joint_cfg=joint_cfg, n_genes=n_genes, strict=False
            )
            loaded_variant = "cross_attention_joint_training"
            if verbose:
                print("[Joint] Loaded as CrossAttentionJointLitModel")
        except Exception as ex:
            load_errors.append(f"CrossAttentionJointLitModel: {ex}")
            if verbose:
                print(f"[Joint] CrossAttention load failed, falling back to JointLitModel: {ex}")
            model = None

    if model is None:
        try:
            model = JointLitModel.load_from_checkpoint(
                ckpt_path, conf=conf, joint_cfg=joint_cfg, n_genes=n_genes, strict=False
            )
            loaded_variant = "joint_training"
            if verbose:
                print("[Joint] Loaded as JointLitModel (strict=False)")
        except Exception as ex:
            load_errors.append(f"JointLitModel: {ex}")
            details = "\n  - ".join(load_errors) if load_errors else "(no loader details)"
            raise RuntimeError(
                "Failed to load checkpoint with available model loaders.\n"
                f"  - {details}"
            ) from ex

    if expected_variant is not None and loaded_variant != expected_variant:
        raise RuntimeError(
            "Checkpoint/model variant mismatch during latent extraction: "
            f"expected '{expected_variant}', loaded '{loaded_variant}'. "
            "Aborting to avoid exporting wrong latent representations."
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

    repo_root = Path(__file__).resolve().parents[2]
    full_cfg = _resolve_config_paths(full_cfg, repo_root)

    joint_cfg = full_cfg.get("joint_training", full_cfg)

    if args.extract_latents:
        extract_latents(joint_cfg, ckpt_path=args.ckpt, split=args.split)
    else:
        run_joint_training(joint_cfg)


if __name__ == "__main__":
    main()
