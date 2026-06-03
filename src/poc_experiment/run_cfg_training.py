"""
Entry point for Backbone-CFG training (CfgBackboneLitModel).

The backbone directly receives genomic features with CFG dropout — no adapter.

Usage (via run_pipeline.py):
    python run_pipeline.py --config src/config.yaml --stage poc_brca_lihc_cfg_v2

Usage (standalone, for debugging):
    python -m src.poc_experiment.run_cfg_training --config src/config.yaml
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from .cfg_model import CfgBackboneLitModel
from ._train_utils import _build_config, _validate_cohort_coverage

log = logging.getLogger(__name__)


def run_cfg_training(cfg: Dict[str, Any], verbose: bool = True) -> None:
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )

    conf = _build_config(cfg)
    log.info(
        "CfgBackboneLitModel: img_size=%d  feat_dim=%d  cfg_dropout=%.2f  "
        "backbone_lr=%.1e  ckpt=%s",
        conf.img_size, conf.feat_dim, conf.cfg_dropout, conf.backbone_lr,
        Path(conf.backbone_ckpt_path).name if conf.backbone_ckpt_path else "none",
    )

    drop_threshold = float(cfg.get("drop_threshold", 0.30))
    for cohort in ("TCGA-BRCA", "TCGA-LIHC"):
        _validate_cohort_coverage(
            patient_splits_path=conf.patient_splits_path,
            zip_dir=conf.zip_dir,
            drop_threshold=drop_threshold,
            check_cohort=cohort,
        )

    conf.make_model_conf()
    model = CfgBackboneLitModel(conf)

    logdir = Path(conf.logdir)
    autoenc_dir = logdir / "autoenc"
    autoenc_dir.mkdir(parents=True, exist_ok=True)

    accum = conf.accum_batches
    val_every_steps = cfg.get("val_every_steps", 2_000) * accum

    ckpt_every_steps = max(
        max(1, conf.save_every_samples // conf.batch_size),
        int(val_every_steps),
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(autoenc_dir),
        filename="{epoch}-{step}",
        save_last=True,
        save_top_k=cfg.get("save_top_k", 3),
        monitor=cfg.get("monitor_metric", "loss/val_ckpt"),
        mode="min",
        every_n_train_steps=ckpt_every_steps,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(logdir), name="", version="")

    import os
    slurm_ntasks = int(os.environ.get("SLURM_NTASKS", 1))
    torchrun_world = int(os.environ.get("WORLD_SIZE", 1))
    already_launched = slurm_ntasks > 1 or int(os.environ.get("LOCAL_RANK", -1)) >= 0

    if already_launched:
        n_devices = max(slurm_ntasks, torchrun_world)
        devices = n_devices
        strategy = "ddp"
        log.info("Pre-launched DDP worker: rank=%s world=%d",
                 os.environ.get("SLURM_PROCID", os.environ.get("RANK", "?")), n_devices)
    else:
        gpus = cfg.get("gpus", [0])
        devices = gpus if isinstance(gpus, list) else [gpus]
        strategy = "ddp" if len(devices) > 1 else "auto"
        n_devices = len(devices) if isinstance(devices, list) else 1

    if n_devices > 1 and conf.batch_size % n_devices != 0:
        raise ValueError(
            f"batch_size={conf.batch_size} must be divisible by n_devices={n_devices}."
        )

    max_steps = conf.total_samples // conf.batch_size
    log.info(
        "Training: max_steps=%d  global_batch=%d (eff %d)  accum=%d  devices=%s",
        max_steps, conf.batch_size, conf.batch_size_effective, accum, devices,
    )

    trainer = pl.Trainer(
        max_steps=max_steps,
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        precision="bf16-mixed" if cfg.get("bf16", False) else (
            "16-mixed" if cfg.get("fp16", False) else "32-true"
        ),
        callbacks=[checkpoint_cb, lr_monitor],
        logger=tb_logger,
        log_every_n_steps=50,
        gradient_clip_val=None,
        accumulate_grad_batches=1,
        val_check_interval=val_every_steps,
        check_val_every_n_epoch=None,
        limit_val_batches=cfg.get("limit_val_batches", 100),
        enable_progress_bar=True,
    )

    last_ckpt = autoenc_dir / "last.ckpt"
    resume_ckpt = str(last_ckpt) if last_ckpt.exists() else None
    if resume_ckpt:
        log.info("Resuming CFG run from: %s", resume_ckpt)
    else:
        log.info("Fresh start — backbone warm from backbone_ckpt_path if set")

    model.expected_world_size = max(1, n_devices)

    try:
        trainer.fit(model, ckpt_path=resume_ckpt)
    except Exception as e:
        log.error("trainer.fit raised %s: %r", type(e).__name__, str(e))
        traceback.print_exc()
        raise

    log.info("Training complete. Last checkpoint: %s", checkpoint_cb.last_model_path)


if __name__ == "__main__":
    import argparse
    import sys
    import yaml
    from pathlib import Path as _Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--section", default="poc_brca_lihc_cfg_v2",
                        help="Config section name (default: poc_brca_lihc_cfg_v2)")
    args = parser.parse_args()

    config_path = _Path(args.config).resolve()
    with open(config_path) as fh:
        full_cfg = yaml.safe_load(fh)

    section = full_cfg.get(args.section)
    if not section:
        sys.exit(f"No '{args.section}' section found in {args.config}")

    run_cfg_training(section, verbose=True)
