"""
Entry point for genomic cross-attention MoPaDi training.

Called from run_pipeline.py via:
    python run_pipeline.py --config src/config.yaml --stage mopadi_genomic_ca
"""
from __future__ import annotations

import logging
import os
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.nn.parallel import DistributedDataParallel

from mopadi.configs.choices import ModelName

from src.mopadi_genomic_crossattn.checkpoint_callback import CompositeMetricCheckpoint
from .genomic_config import GenomicCaConfig
from .model import GenomicCaLitModel

log = logging.getLogger(__name__)


class _DefaultStreamDDPStrategy(DDPStrategy):
    """DDP on the default CUDA stream — avoids cudaEventDestroy in PyTorch 2.7+."""

    def _setup_model(self, model):
        device_ids = self.determine_ddp_device_ids()
        return DistributedDataParallel(module=model, device_ids=device_ids, **self._ddp_kwargs)

    def teardown(self) -> None:
        import gc
        import torch
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        gc.collect()
        super().teardown()


def run_training_ca(cfg: Dict[str, Any], verbose: bool = True) -> None:
    """Launch genomic cross-attention MoPaDi training run.

    Parameters
    ----------
    cfg:
        The ``mopadi_genomic_ca`` section from ``config.yaml``.
    verbose:
        Whether to configure INFO-level logging.
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )

    conf = _build_ca_config(cfg)
    log.info(
        "GenomicCaConfig: img_size=%d, feat_dim=%d, ema_decay=%.4f, "
        "use_ca=%s, pred_gap_lambda=%.2f, cfl_lambda=%.2f",
        conf.img_size, conf.feat_dim, conf.ema_decay,
        conf.use_genomic_cross_attn, conf.pred_gap_lambda, conf.cfl_lambda,
    )

    conf.make_model_conf()
    model = GenomicCaLitModel(conf)

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

    if len(devices) > 1 and conf.batch_size % len(devices) != 0:
        raise ValueError(
            f"batch_size ({conf.batch_size}) must be divisible by "
            f"number of devices ({len(devices)})."
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
        # gradient_clip_val and accumulate_grad_batches are handled manually in
        # GenomicCaLitModel.training_step (automatic_optimization=False).
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
        "Starting: max_steps=%d, global_batch=%d (effective=%d), gpus=%s",
        max_steps, conf.batch_size, conf.batch_size_effective, devices,
    )

    try:
        trainer.fit(model, ckpt_path=resume_ckpt)
    except Exception as e:
        log.error("trainer.fit raised %s: %r", type(e).__name__, str(e))
        traceback.print_exc()
        raise


def _build_ca_config(cfg: Dict[str, Any]) -> GenomicCaConfig:
    """Build GenomicCaConfig from the mopadi_genomic_ca YAML section."""
    for key in ("zip_dir", "genomic_feature_dir", "patient_splits_path"):
        if not cfg.get(key):
            raise ValueError(f"Missing required key '{key}' in mopadi_genomic_ca config.")
    if not Path(cfg["patient_splits_path"]).exists():
        raise FileNotFoundError(
            f"patient_splits_path not found: {cfg['patient_splits_path']}"
        )
    if not Path(cfg["genomic_feature_dir"]).is_dir():
        raise FileNotFoundError(
            f"genomic_feature_dir not found: {cfg['genomic_feature_dir']}"
        )

    def _get(key: str, default: Any) -> Any:
        val = cfg.get(key, default)
        return default if val is None else val

    def _get_bool_env_or_cfg(env_name: str, cfg_key: str, default: bool = False) -> bool:
        env_val = os.environ.get(env_name)
        if env_val is not None and env_val.strip() != "":
            return env_val.strip().lower() in ("1", "true", "yes", "on")
        return bool(_get(cfg_key, default))

    img_size = int(_get("img_size", 512))
    style_ch = int(_get("style_ch", 512))
    feat_dim = int(_get("feat_dim", style_ch))
    net_ch_mult = tuple(_get("net_ch_mult", None) or (1, 1, 2, 2, 4, 4))

    fields: Dict[str, Any] = dict(
        name="genomic_ca_training",
        base_dir=cfg.get("base_dir", cfg.get("output_dir", "checkpoints/genomic_ca")),
        seed=int(_get("seed", 42)),
        model_name=ModelName.beatgans_autoenc,
        diffusion_type="beatgans",
        img_size=img_size,
        net_ch=int(_get("net_ch", 128)),
        net_ch_mult=net_ch_mult,
        net_attn=tuple(_get("net_attn", None) or (16,)),
        net_num_res_blocks=int(_get("net_num_res_blocks", 2)),
        net_beatgans_gradient_checkpoint=_get_bool_env_or_cfg(
            "MOPADI_GRADIENT_CHECKPOINT",
            "net_beatgans_gradient_checkpoint",
            False,
        ),
        style_ch=style_ch,
        feat_dim=feat_dim,
        net_beatgans_embed_channels=style_ch,
        net_beatgans_resnet_two_cond=True,
        T=int(_get("T", 1000)),
        T_eval=int(_get("T_eval", 20)),
        batch_size=int(_get("batch_size", 4)),
        lr=float(_get("lr", 1e-4)),
        ema_decay=float(_get("ema_decay", 0.999)),
        grad_clip=float(_get("grad_clip", 1.0)),
        fp16=bool(_get("fp16", False)),
        accum_batches=int(_get("accumulate_grad_batches", 1)),
        num_workers=int(os.environ.get("MOPADI_NUM_WORKERS", "") or _get("num_workers", 4)),
        total_samples=int(_get("total_samples", 2_500_000)),
        steps_per_epoch=int(_get("steps_per_epoch", 5000)),
        save_every_samples=int(_get("save_every_samples", 200_000)),
        reconstruct_every_samples=int(_get("reconstruct_every_samples", 100_000)),
        sample_size=int(_get("sample_size", 8)),
        zip_dir=cfg["zip_dir"],
        genomic_feature_dir=cfg["genomic_feature_dir"],
        patient_splits_path=cfg["patient_splits_path"],
        max_tiles_by_subtype=_get("max_tiles_by_subtype", None),
        tile_sampling_seed=int(_get("tile_sampling_seed", 42)),
        do_normalize=bool(_get("do_normalize", True)),
        do_resize=bool(_get("do_resize", False)),
        val_limit_batches=int(_get("limit_val_batches", 100)),
        # Inherited CFL fields
        cfl_lambda=float(_get("cfl_lambda", 0.3)),
        cfl_margin=float(_get("cfl_margin", 0.005)),
        cfl_every_n_steps=int(_get("cfl_every_n_steps", 4)),
        # New fields
        pred_gap_lambda=float(_get("pred_gap_lambda", 1.0)),
        pred_gap_margin=float(_get("pred_gap_margin", 0.01)),
        use_genomic_cross_attn=bool(_get("use_genomic_cross_attn", True)),
        genomic_ca_heads=int(_get("genomic_ca_heads", 8)),
        genomic_ca_n_tokens=int(_get("genomic_ca_n_tokens", 4)),
    )

    return GenomicCaConfig(**fields)
