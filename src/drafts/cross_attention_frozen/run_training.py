"""
Entry point for the frozen-backbone CA training (v12).

Called from run_pipeline.py via:
    python run_pipeline.py --config src/config.yaml --stage mopadi_genomic_ca_frozen
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

from src.cross_attention.run_training import run_training_ca, _build_ca_config
from .genomic_config import FrozenBackboneCaConfig
from .model import FrozenBackboneCaLitModel

log = logging.getLogger(__name__)


def run_training_frozen(cfg: Dict[str, Any], verbose: bool = True) -> None:
    """Launch frozen-backbone CA training.

    Reuses the full run_training_ca infrastructure; only the model class and
    config type differ.
    """
    import pytorch_lightning as pl
    from datetime import timedelta
    import traceback
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.strategies import DDPStrategy
    from torch.nn.parallel import DistributedDataParallel

    from src.mopadi_genomic_crossattn.checkpoint_callback import CompositeMetricCheckpoint

    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )

    conf = _build_frozen_config(cfg)
    log.info(
        "FrozenBackboneCaConfig: img_size=%d, feat_dim=%d, cfg_dropout=%.2f, "
        "n_tokens=%d, pretrained_ckpt=%s",
        conf.img_size, conf.feat_dim, conf.cfg_dropout,
        conf.genomic_ca_n_tokens,
        Path(conf.pretrained_backbone_ckpt).name if conf.pretrained_backbone_ckpt else "none",
    )

    conf.make_model_conf()
    model = FrozenBackboneCaLitModel(conf)

    logdir = Path(conf.logdir)
    autoenc_dir = logdir / "autoenc"
    autoenc_dir.mkdir(parents=True, exist_ok=True)

    val_every_steps = cfg.get("val_every_steps", 5_000)
    limit_val_batches = cfg.get("limit_val_batches", 100)
    ckpt_every_steps = max(
        max(1, conf.save_every_samples // conf.batch_size_effective),
        int(val_every_steps),
    )

    use_composite_ckpt = cfg.get("use_composite_metric_checkpoint", False)
    if use_composite_ckpt:
        checkpoint_cb = CompositeMetricCheckpoint(
            monitor_loss=cfg.get("monitor_loss", "loss/val"),
            monitor_gap=cfg.get("monitor_gap", "cond/gap"),
            alpha=cfg.get("composite_alpha", 1.0),
            dirpath=str(autoenc_dir),
            window_size=cfg.get("composite_window_size", 20),
            save_top_k=cfg.get("save_top_k", 3),
        )
    else:
        checkpoint_cb = ModelCheckpoint(
            dirpath=str(autoenc_dir),
            filename="{epoch}-{step}",
            save_last=True,
            save_top_k=cfg.get("save_top_k", 3),
            monitor=cfg.get("monitor_metric", "loss/val"),
            mode="min",
            every_n_train_steps=ckpt_every_steps,
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(logdir), name="", version="")

    gpus = cfg.get("gpus", [0])
    env_gpus = os.environ.get("MOPADI_GPUS", "").strip()
    if env_gpus:
        try:
            gpus = [int(x.strip()) for x in env_gpus.split(",") if x.strip()]
        except ValueError as exc:
            raise ValueError(f"Invalid MOPADI_GPUS='{env_gpus}'.") from exc
    devices = gpus if isinstance(gpus, list) else [gpus]

    class _DefaultStreamDDPStrategy(DDPStrategy):
        def _setup_model(self, model):
            device_ids = self.determine_ddp_device_ids()
            return DistributedDataParallel(module=model, device_ids=device_ids, **self._ddp_kwargs)

        def teardown(self) -> None:
            import gc, torch
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            gc.collect()
            super().teardown()

    strategy = (
        _DefaultStreamDDPStrategy(
            find_unused_parameters=True,
            init_sync=False,
            broadcast_buffers=False,
            static_graph=False,
            timeout=timedelta(hours=2),
        )
        if len(devices) > 1
        else "auto"
    )

    max_steps = conf.total_samples // conf.batch_size_effective

    trainer = pl.Trainer(
        max_steps=max_steps,
        accelerator="gpu" if devices else "cpu",
        devices=devices,
        strategy=strategy,
        precision="bf16-mixed" if cfg.get("bf16", False) else (
            "16-mixed" if cfg.get("fp16", False) else "32-true"
        ),
        callbacks=[checkpoint_cb, lr_monitor],
        logger=tb_logger,
        log_every_n_steps=50,
        val_check_interval=val_every_steps,
        check_val_every_n_epoch=None,
        limit_val_batches=limit_val_batches,
        enable_progress_bar=True,
    )

    resume_ckpt = cfg.get("resume_from", "last")
    if resume_ckpt == "last":
        last_ckpt = autoenc_dir / "last.ckpt"
        resume_ckpt = str(last_ckpt) if last_ckpt.exists() else None
    if resume_ckpt:
        log.info("Resuming from: %s", resume_ckpt)

    model.__dict__["expected_world_size"] = max(1, len(devices))
    log.info(
        "Starting frozen-backbone CA: max_steps=%d, global_batch=%d, gpus=%s",
        max_steps, conf.batch_size_effective, devices,
    )

    try:
        trainer.fit(model, ckpt_path=resume_ckpt)
    except Exception as e:
        log.error("trainer.fit raised %s: %r", type(e).__name__, str(e))
        traceback.print_exc()
        raise


def _build_frozen_config(cfg: Dict[str, Any]) -> FrozenBackboneCaConfig:
    """Build FrozenBackboneCaConfig from the mopadi_genomic_ca_frozen YAML section."""
    # Reuse the base config builder, then layer frozen-specific fields on top.
    base = _build_ca_config(cfg)

    def _get(key, default):
        val = cfg.get(key, default)
        return default if val is None else val

    # Convert base GenomicCaConfig → FrozenBackboneCaConfig by rebuilding
    # with the additional frozen-specific fields.
    import dataclasses
    base_dict = dataclasses.asdict(base)
    base_dict["pretrained_backbone_ckpt"] = str(_get("pretrained_backbone_ckpt", ""))
    base_dict["frozen_backbone"] = bool(_get("frozen_backbone", True))
    base_dict["cfg_dropout"] = float(_get("cfg_dropout", 0.30))
    base_dict["genomic_ca_n_tokens"] = int(_get("genomic_ca_n_tokens", 8))

    known = {f.name for f in dataclasses.fields(FrozenBackboneCaConfig)}
    return FrozenBackboneCaConfig(**{k: v for k, v in base_dict.items() if k in known})
