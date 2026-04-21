"""
Entry point for genomic-conditioned MoPaDi training from scratch.

Called from ``run_pipeline.py`` via:
    python run_pipeline.py --config src/config.yaml --stage mopadi_genomic_training

Or standalone:
    python -m src.mopadi_genomic.run_genomic_training --config src/config.yaml
"""

from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from mopadi.configs.choices import ModelName
from mopadi.configs.config import PretrainConfig

from .config import GenomicTrainConfig
from .train import GenomicLitModel

log = logging.getLogger(__name__)


def run_genomic_training(cfg: Dict[str, Any], verbose: bool = True) -> None:
    """Launch a from-scratch MoPaDi training run with genomic conditioning.

    Parameters
    ----------
    cfg:
        The ``mopadi_genomic_training`` section from ``config.yaml``,
        with all paths already resolved to absolute strings.
    verbose:
        Whether to configure INFO-level logging.
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )

    # ── Build config ────────────────────────────────────────────────────
    conf = _build_train_config(cfg)
    log.info("GenomicTrainConfig built: img_size=%d, feat_dim=%d, style_ch=%d",
             conf.img_size, conf.feat_dim, conf.style_ch)

    # ── Model ───────────────────────────────────────────────────────────
    conf.make_model_conf()   # populates conf.model_conf
    model = GenomicLitModel(conf)

    # ── Output directories ───────────────────────────────────────────────
    logdir = Path(conf.logdir)
    autoenc_dir = logdir / "autoenc"
    autoenc_dir.mkdir(parents=True, exist_ok=True)

    # ── Callbacks ───────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(autoenc_dir),
        filename="{epoch}-{step}",
        save_last=True,
        save_top_k=cfg.get("save_top_k", 3),
        monitor="loss",
        mode="min",
        every_n_train_steps=max(
            1,
            conf.save_every_samples // conf.batch_size_effective,
        ),
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ── Logger ──────────────────────────────────────────────────────────
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=str(logdir),
        name="",
        version="",
    )

    # ── GPU config ──────────────────────────────────────────────────────
    gpus = cfg.get("gpus", [0])
    accelerator = "gpu" if isinstance(gpus, (list, int)) else "cpu"
    devices = gpus if isinstance(gpus, list) else [gpus]
    strategy = "ddp_find_unused_parameters_false" if len(devices) > 1 else "auto"

    # ── Max steps from total_samples ────────────────────────────────────
    max_steps = conf.total_samples // conf.batch_size_effective

    # ── Validation cadence ───────────────────────────────────────────────
    # Validate every N *training steps* (not epochs) so the interval is
    # predictable regardless of dataset size.  A small batch cap keeps
    # each validation pass fast without hiding meaningful signal.
    val_every_steps = cfg.get("val_every_steps", 5_000)
    # 100 batches × batch_size gives a stable loss estimate while staying fast.
    limit_val_batches = cfg.get("limit_val_batches", 100)

    # ── Trainer ─────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_steps=max_steps,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision="bf16-mixed" if cfg.get("bf16", False) else (
            "16-mixed" if cfg.get("fp16", False) else "32-true"
        ),
        callbacks=[checkpoint_cb, lr_monitor],
        logger=tb_logger,
        log_every_n_steps=50,
        gradient_clip_val=conf.grad_clip if conf.grad_clip > 0 else None,
        accumulate_grad_batches=conf.accum_batches,
        # Step-based validation: val_check_interval (int) = every N steps.
        # check_val_every_n_epoch=None disables the epoch-based fallback.
        val_check_interval=val_every_steps,
        check_val_every_n_epoch=None,
        limit_val_batches=limit_val_batches,
        enable_progress_bar=True,
    )

    # ── Resume from checkpoint ───────────────────────────────────────────
    # "last" (default) auto-discovers last.ckpt in the output directory.
    # Set to null in config to start fresh, or provide an explicit path.
    resume_ckpt = cfg.get("resume_from", "last")
    if resume_ckpt == "last":
        last_ckpt = Path(autoenc_dir) / "last.ckpt"
        resume_ckpt = str(last_ckpt) if last_ckpt.exists() else None
    if resume_ckpt:
        log.info("Resuming from checkpoint: %s", resume_ckpt)

    log.info(
        "Starting training: max_steps=%d, batch_size=%d (effective=%d), gpus=%s",
        max_steps, conf.batch_size, conf.batch_size_effective, devices,
    )

    try:
        trainer.fit(model, ckpt_path=resume_ckpt)
    except Exception as e:
        log.error("trainer.fit raised %s: %r", type(e).__name__, str(e))
        traceback.print_exc()
        raise
    log.info("Training complete.  Last checkpoint: %s", checkpoint_cb.last_model_path)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _build_train_config(cfg: Dict[str, Any]) -> GenomicTrainConfig:
    """Convert a YAML config dict into a fully-populated GenomicTrainConfig.

    Applies sensible defaults and validates that required paths exist.
    """
    # ── Required paths ──────────────────────────────────────────────────
    for key in ("zip_dir", "genomic_feature_dir", "patient_splits_path"):
        if not cfg.get(key):
            raise ValueError(
                f"Missing required key '{key}' in mopadi_genomic_training config."
            )
    if not Path(cfg["patient_splits_path"]).exists():
        raise FileNotFoundError(
            f"patient_splits_path does not exist: {cfg['patient_splits_path']}\n"
            "Run the 'build_genomic_features' stage first."
        )
    if not Path(cfg["genomic_feature_dir"]).is_dir():
        raise FileNotFoundError(
            f"genomic_feature_dir does not exist: {cfg['genomic_feature_dir']}\n"
            "Run the 'build_genomic_features' stage first."
        )

    # ── Assemble flat dict of TrainConfig fields ─────────────────────────
    model_cfg = cfg.get("model", {})

    # Pull model architecture params from either the top-level or a nested
    # "model:" block, so both layouts work in config.yaml.
    def _get(key, default=None):
        return cfg.get(key, model_cfg.get(key, default))

    img_size   = int(_get("img_size", 256))
    net_ch     = int(_get("net_ch", 128))
    style_ch   = int(_get("style_ch", 512))
    feat_dim   = int(_get("feat_dim", style_ch))   # must equal style_ch

    # net_ch_mult: accept list or null
    raw_mult = _get("net_ch_mult")
    net_ch_mult = tuple(raw_mult) if raw_mult else (1, 1, 2, 2, 4, 4)

    # net_attn: attention resolutions (list of img_size // 2**k values)
    raw_attn = _get("net_attn")
    net_attn = tuple(raw_attn) if raw_attn else (16,)

    base_dir = cfg.get("base_dir", cfg.get("output_dir", "checkpoints/genomic_scratch"))

    fields = dict(
        # ── Identity ─────────────────────────────────────────────────
        name="genomic_scratch",
        base_dir=base_dir,
        seed=int(_get("seed", 42)),
        # ── Architecture ─────────────────────────────────────────────
        model_name=ModelName.beatgans_autoenc,
        diffusion_type="beatgans",
        img_size=img_size,
        net_ch=net_ch,
        net_ch_mult=net_ch_mult,
        net_attn=net_attn,
        net_num_res_blocks=int(_get("net_num_res_blocks", 2)),
        net_beatgans_gradient_checkpoint=bool(_get("net_beatgans_gradient_checkpoint", False)),
        style_ch=style_ch,
        feat_dim=feat_dim,
        net_beatgans_embed_channels=style_ch,
        net_beatgans_resnet_two_cond=True,   # required: autoenc forward path raises NotImplementedError() otherwise
        # No projection layer needed: the 512-dim gene vector is used as-is.
        # enc_transform_dim left at default (1024) but is not used in the
        # genomic path since no image feature extractor is active.
        # ── Diffusion ────────────────────────────────────────────────
        T=int(_get("T", 1000)),
        T_eval=int(_get("T_eval", 20)),
        # ── Training ─────────────────────────────────────────────────
        batch_size=int(_get("batch_size", 4)),
        lr=float(_get("lr", 1e-4)),
        ema_decay=float(_get("ema_decay", 0.9999)),
        grad_clip=float(_get("grad_clip", 1.0)),
        fp16=bool(_get("fp16", False)),
        accum_batches=int(_get("accumulate_grad_batches", 1)),
        num_workers=int(_get("num_workers", 4)),
        total_samples=int(_get("total_samples", 200_000_000)),
        steps_per_epoch=int(_get("steps_per_epoch", 5000)),
        save_every_samples=int(_get("save_every_samples", 200_000)),
        reconstruct_every_samples=int(_get("reconstruct_every_samples", 50_000)),
        sample_size=int(_get("sample_size", 16)),
        # ── Genomic dataset ───────────────────────────────────────────
        zip_dir=cfg["zip_dir"],
        genomic_feature_dir=cfg["genomic_feature_dir"],
        patient_splits_path=cfg["patient_splits_path"],
        max_tiles_by_subtype=cfg.get("max_tiles_by_subtype"),
        tile_sampling_seed=int(cfg.get("tile_sampling_seed", 42)),
        do_normalize=bool(cfg.get("do_normalize", True)),
        do_resize=bool(cfg.get("do_resize", False)),
        val_limit_batches=int(cfg.get("limit_val_batches", 100)),
    )

    return GenomicTrainConfig(**fields)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Train MoPaDi from scratch with genomic conditioning."
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config) as fh:
        full_cfg = yaml.safe_load(fh)

    section = full_cfg.get("mopadi_genomic_training", {})
    if not section:
        raise SystemExit("No 'mopadi_genomic_training' section found in config.")

    run_genomic_training(section, verbose=True)
