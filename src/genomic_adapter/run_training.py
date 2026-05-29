"""
Entry point for Genomic Diffusion Adapter (GDA) training.

Usage (standalone):
    python -m src.genomic_adapter.run_training --config src/config.yaml

Usage (via run_pipeline.py):
    python run_pipeline.py --config src/config.yaml --stage genomic_adapter_training

The config.yaml section should look like:

    genomic_adapter_training:
      zip_dir: /data/BRCA-tumor-tiles-corrected
      genomic_feature_dir: /data/genomic_features
      patient_splits_path: /data/patient_splits.json
      output_dir: experiments/20260518_gda_v1
      total_samples: 200_000_000

      # Adapter architecture
      adapter_base_ch: 64
      adapter_n_tokens: 8
      adapter_token_dim: 256
      adapter_t_dim: 256
      adapter_n_heads: 4

      # CFG
      cfg_dropout: 0.30

      # Optimisation
      backbone_lr: 1.0e-4
      adapter_lr:  3.0e-4
      batch_size: 16
      accumulate_grad_batches: 2    # effective batch = 32
      grad_clip: 1.0
      ema_decay: 0.9999
      fp16: false

      # Logging / checkpointing
      val_every_steps: 5000
      limit_val_batches: 100
      save_every_samples: 200_000
      save_top_k: 3
      monitor_metric: loss/val
"""

from __future__ import annotations

import json
import logging
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from mopadi.configs.choices import ModelName

from .config import GDAConfig
from .model import GDALitModel

log = logging.getLogger(__name__)


def _validate_cohort_coverage(
    patient_splits_path: str,
    zip_dir: str,
    drop_threshold: float = 0.30,
    check_cohort: str = "TCGA-LIHC",
) -> None:
    """Raise RuntimeError if too many patients from *check_cohort* lack valid zips.

    Reads the central directory of each zip (fast — no decompression) to detect
    both missing and corrupt archives before any GPU memory is allocated.

    Parameters
    ----------
    patient_splits_path:
        Path to ``patient_splits.json`` produced by ``build_genomic_features``.
    zip_dir:
        Directory containing per-patient ``*.zip`` tile archives.
    drop_threshold:
        Maximum acceptable fraction of dropped patients (0–1). Default 0.30.
    check_cohort:
        Subtype label to check. No-op if this label is absent from splits.
    """
    with open(patient_splits_path) as f:
        splits_data = json.load(f)

    expected_pids: set = set()
    for fold, entries in splits_data.items():
        if fold.startswith("_") or not isinstance(entries, dict):
            continue
        for pid, meta in entries.items():
            if pid.startswith("_") or not isinstance(meta, dict):
                continue
            if meta.get("subtype") == check_cohort:
                expected_pids.add(pid)

    if not expected_pids:
        log.debug("_validate_cohort_coverage: no '%s' patients found — skipping.", check_cohort)
        return

    zip_dir_path = Path(zip_dir)
    valid_pids: set = set()
    invalid: list = []

    for pid in expected_pids:
        matches = list(zip_dir_path.glob(f"{pid}*.zip"))
        if not matches:
            invalid.append((pid, "no zip file found"))
            continue
        zip_path = matches[0]
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                if names:
                    with zf.open(names[0]) as _f:
                        _f.read(256)  # partial read — enough to catch zlib.error
            valid_pids.add(pid)
        except Exception as exc:
            invalid.append((pid, str(exc)))

    n_expected = len(expected_pids)
    n_valid = len(valid_pids)
    n_dropped = n_expected - n_valid
    drop_frac = n_dropped / n_expected if n_expected > 0 else 0.0

    if invalid:
        sample = invalid[:10]
        log.warning(
            "%d %s patients lack valid zips (showing first 10): %s",
            len(invalid), check_cohort,
            [(pid, err) for pid, err in sample],
        )

    log.info(
        "Cohort coverage check [%s]: %d/%d patients have valid zips (%.1f%% dropped)",
        check_cohort, n_valid, n_expected, drop_frac * 100,
    )

    if drop_frac > drop_threshold:
        raise RuntimeError(
            f"\n[GDA Startup] Too many {check_cohort} patients are missing or have corrupt zip archives!\n"
            f"  Expected : {n_expected} patients (from patient_splits.json)\n"
            f"  Valid    : {n_valid}\n"
            f"  Dropped  : {n_dropped} ({drop_frac:.1%})\n"
            f"  Threshold: {drop_threshold:.0%}\n"
            f"  First bad entries: {invalid[:5]}\n"
            f"  → Check {zip_dir} for missing or corrupt zip archives.\n"
            f"  → Re-run the PoC symlink setup and verify LIHC zip integrity."
        )


def run_gda_training(cfg: Dict[str, Any], verbose: bool = True) -> None:
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )

    conf = _build_config(cfg)
    log.info(
        "GDAConfig: img_size=%d  feat_dim=%d  adapter_base_ch=%d  "
        "backbone_lr=%.1e  adapter_lr=%.1e  cfg_dropout=%.2f  "
        "freeze_backbone=%s  backbone_ckpt=%s  reinit_adapter=%s",
        conf.img_size, conf.feat_dim, conf.adapter_base_ch,
        conf.backbone_lr, conf.adapter_lr, conf.cfg_dropout,
        conf.freeze_backbone,
        Path(conf.backbone_ckpt_path).name if conf.backbone_ckpt_path else "none",
        conf.reinit_adapter,
    )

    # Validate zip archives for both cohorts before GPU memory is allocated.
    # The function silently skips any cohort absent from patient_splits.json,
    # so this is safe for both PoC (BRCA+LIHC) and BRCA-only PAM50 runs.
    drop_threshold = float(cfg.get("drop_threshold", 0.30))
    for cohort in ("TCGA-BRCA", "TCGA-LIHC"):
        _validate_cohort_coverage(
            patient_splits_path=conf.patient_splits_path,
            zip_dir=conf.zip_dir,
            drop_threshold=drop_threshold,
            check_cohort=cohort,
        )

    conf.make_model_conf()
    model = GDALitModel(conf)

    logdir = Path(conf.logdir)
    autoenc_dir = logdir / "autoenc"
    autoenc_dir.mkdir(parents=True, exist_ok=True)

    # With automatic_optimization=False, Lightning forbids accumulate_grad_batches != 1.
    # We accumulate manually via is_last_accum(batch_idx), so global_step counts
    # every micro-batch (not every effective optimizer step).
    # All step-based intervals must therefore be scaled by accum_batches.
    accum = conf.accum_batches
    val_every_steps = cfg.get("val_every_steps", 5_000) * accum
    limit_val_batches = cfg.get("limit_val_batches", 100)

    ckpt_every_steps = max(
        max(1, conf.save_every_samples // conf.batch_size),  # micro-batch units
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

    gpus = cfg.get("gpus", [0])
    accelerator = "gpu" if isinstance(gpus, (list, int)) else "cpu"
    devices = gpus if isinstance(gpus, list) else [gpus]
    # freeze_backbone=True leaves backbone params unused in the backward graph;
    # ddp_find_unused_parameters_false would hang waiting for their gradients.
    if len(devices) > 1:
        strategy = "ddp" if conf.freeze_backbone else "ddp_find_unused_parameters_false"
    else:
        strategy = "auto"

    if len(devices) > 1 and conf.batch_size % len(devices) != 0:
        raise ValueError(
            f"batch_size={conf.batch_size} must be divisible by n_devices={len(devices)}."
        )

    # micro-batch steps = total_samples / per-step global batch = total_samples / batch_size
    max_steps = conf.total_samples // conf.batch_size
    log.info("Training: max_steps=%d  global_batch=%d  accum=%d  devices=%s",
             max_steps, conf.batch_size_effective, accum, devices)

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
        gradient_clip_val=None,              # GDA does manual clipping per component
        accumulate_grad_batches=1,           # must be 1 with automatic_optimization=False
        val_check_interval=val_every_steps,
        check_val_every_n_epoch=None,
        limit_val_batches=limit_val_batches,
        enable_progress_bar=True,
    )

    last_ckpt = Path(autoenc_dir) / "last.ckpt"
    resume_ckpt = cfg.get("resume_from", "last")
    if resume_ckpt == "last":
        resume_ckpt = str(last_ckpt) if last_ckpt.exists() else None
    if resume_ckpt:
        log.info("Resuming from checkpoint: %s", resume_ckpt)

    model.expected_world_size = max(1, len(devices))

    try:
        trainer.fit(model, ckpt_path=resume_ckpt)
    except Exception as e:
        log.error("trainer.fit raised %s: %r", type(e).__name__, str(e))
        traceback.print_exc()
        raise

    log.info("Training complete. Last checkpoint: %s", checkpoint_cb.last_model_path)


def _build_config(cfg: Dict[str, Any]) -> GDAConfig:
    for key in ("zip_dir", "genomic_feature_dir", "patient_splits_path"):
        if not cfg.get(key):
            raise ValueError(f"Missing required key '{key}' in genomic_adapter_training config.")
    if not Path(cfg["patient_splits_path"]).exists():
        raise FileNotFoundError(f"patient_splits_path not found: {cfg['patient_splits_path']}")
    if not Path(cfg["genomic_feature_dir"]).is_dir():
        raise FileNotFoundError(f"genomic_feature_dir not found: {cfg['genomic_feature_dir']}")

    model_cfg = cfg.get("model", {})

    def _get(key, default=None):
        return cfg.get(key, model_cfg.get(key, default))

    img_size = int(_get("img_size", 256))
    net_ch = int(_get("net_ch", 128))
    style_ch = int(_get("style_ch", 512))
    feat_dim = style_ch  # backbone cond dimension = style_ch (always zeros in GDA)

    raw_mult = _get("net_ch_mult")
    net_ch_mult = tuple(raw_mult) if raw_mult else (1, 1, 2, 2, 4, 4)
    raw_attn = _get("net_attn")
    net_attn = tuple(raw_attn) if raw_attn else (16,)

    base_dir = cfg.get("output_dir", cfg.get("base_dir", "experiments/gda_v1"))

    fields = dict(
        name="gda",
        base_dir=base_dir,
        seed=int(_get("seed", 42)),
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
        net_beatgans_resnet_two_cond=True,
        T=int(_get("T", 1000)),
        T_eval=int(_get("T_eval", 20)),
        batch_size=int(_get("batch_size", 16)),
        lr=float(_get("backbone_lr", _get("lr", 1e-4))),
        ema_decay=float(_get("ema_decay", 0.9999)),
        grad_clip=float(_get("grad_clip", 1.0)),
        fp16=bool(_get("fp16", False)),
        accum_batches=int(_get("accumulate_grad_batches", 1)),
        num_workers=int(_get("num_workers", 4)),
        total_samples=int(_get("total_samples", 200_000_000)),
        steps_per_epoch=int(_get("steps_per_epoch", 5_000)),
        save_every_samples=int(_get("save_every_samples", 200_000)),
        reconstruct_every_samples=int(_get("reconstruct_every_samples", 50_000)),
        sample_size=int(_get("sample_size", 16)),
        zip_dir=cfg["zip_dir"],
        genomic_feature_dir=cfg["genomic_feature_dir"],
        patient_splits_path=cfg["patient_splits_path"],
        max_tiles_by_subtype=cfg.get("max_tiles_by_subtype"),
        tile_sampling_seed=int(cfg.get("tile_sampling_seed", 42)),
        do_normalize=bool(cfg.get("do_normalize", True)),
        do_resize=bool(cfg.get("do_resize", False)),
        val_limit_batches=int(cfg.get("limit_val_batches", 100)),
        # GDA-specific
        adapter_base_ch=int(_get("adapter_base_ch", 64)),
        adapter_n_tokens=int(_get("adapter_n_tokens", 8)),
        adapter_token_dim=int(_get("adapter_token_dim", 256)),
        adapter_t_dim=int(_get("adapter_t_dim", 256)),
        adapter_n_heads=int(_get("adapter_n_heads", 4)),
        cfg_dropout=float(_get("cfg_dropout", 0.30)),
        backbone_lr=float(_get("backbone_lr", 1e-4)),
        adapter_lr=float(_get("adapter_lr", 3e-4)),
        freeze_backbone=bool(_get("freeze_backbone", False)),
        backbone_ckpt_path=str(_get("backbone_ckpt_path", "")),
        reinit_adapter=bool(_get("reinit_adapter", False)),
        delta_encouragement_weight=float(_get("delta_encouragement_weight", 0.0)),
        # Unused parent fields set to neutral values
        counterfactual_loss_weight=0.0,
        cond_dropout_prob=0.0,
    )

    return GDAConfig(**fields)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Train Genomic Diffusion Adapter from scratch.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config) as fh:
        full_cfg = yaml.safe_load(fh)

    section = full_cfg.get("genomic_adapter_training", {})
    if not section:
        raise SystemExit("No 'genomic_adapter_training' section in config.")

    run_gda_training(section, verbose=True)
