"""
Entry point for genomic-conditioned MoPaDi training with patchified cross-attention.

Called from run_pipeline.py via:
    python run_pipeline.py --config src/config.yaml --stage mopadi_genomic_crossattn

Or standalone:
    python -m src.mopadi_genomic_crossattn.run_training --config src/config.yaml
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
from pytorch_lightning.strategies import DDPStrategy
from torch.nn.parallel import DistributedDataParallel

from mopadi.configs.choices import ModelName

try:
    from mopadi_genomic_crossattn.config import GenomicCrossAttnConfig
    from mopadi_genomic_crossattn.model import GenomicCrossAttnLitModel
except ImportError:
    from src.mopadi_genomic_crossattn.config import GenomicCrossAttnConfig
    from src.mopadi_genomic_crossattn.model import GenomicCrossAttnLitModel

log = logging.getLogger(__name__)


class _DefaultStreamDDPStrategy(DDPStrategy):
    """DDPStrategy that initialises DDP on the default CUDA stream.

    PyTorch Lightning wraps DDP.__init__ inside torch.cuda.stream(new_stream).
    In PyTorch 2.7+cu128 this causes the Reducer and _broadcast_coalesced to
    allocate CUDA events on the new stream, which is released before the
    Reducer is destroyed, leading to `cudaEventDestroy` → illegal memory
    access in TensorImpl::~TensorImpl and Reducer::~Reducer.  Running on the
    default stream (which lives for the entire process) avoids this entirely.

    teardown() synchronises CUDA before PL unwraps DDP so that the Reducer
    (and any in-flight allreduce CUDA events) are fully retired before GC.
    """

    def _setup_model(self, model):
        device_ids = self.determine_ddp_device_ids()
        log.debug(
            "setting up DDP on default stream (device_ids=%s, kwargs=%s)",
            device_ids, self._ddp_kwargs,
        )
        return DistributedDataParallel(module=model, device_ids=device_ids, **self._ddp_kwargs)

    def teardown(self) -> None:
        import gc
        import torch
        if torch.cuda.is_available():
            try:
                # Ensure all in-flight CUDA work (allreduce, etc.) is retired
                # before PL unwraps DDP and drops the Reducer reference.  In
                # PyTorch 2.7+cu128 the Reducer's CUDA events crash on
                # destruction if any GPU work is still pending.  If CUDA is
                # already in an error state (training failed mid-step), the
                # synchronize itself raises; swallow it so super().teardown()
                # still runs and doesn't mask the original exception.
                torch.cuda.synchronize()
            except Exception:
                pass
        # Flush reference cycles (backward graph → NCCL work objects → CUDA
        # events) while CUDA is idle so the events are destroyed safely here
        # rather than in the Reducer destructor after DDP is unwrapped.
        gc.collect()
        super().teardown()


def run_genomic_crossattn_training(cfg: Dict[str, Any], verbose: bool = True) -> None:
    """Launch a genomic cross-attention MoPaDi training run.

    Parameters
    ----------
    cfg:
        The ``mopadi_genomic_crossattn`` section from ``config.yaml``.
    verbose:
        Whether to configure INFO-level logging.
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )

    conf = _build_train_config(cfg)
    log.info(
        "GenomicCrossAttnConfig: img_size=%d, feat_dim=%d, style_ch=%d, "
        "patch_size=%d, heads=%d, dim_per_head=%d, genomic_weight=%.2f",
        conf.img_size, conf.feat_dim, conf.style_ch,
        conf.cross_attn_patch_size, conf.cross_attn_heads, conf.cross_attn_dim_per_head,
        conf.genomic_guided_loss_weight,
    )

    conf.make_model_conf()
    model = GenomicCrossAttnLitModel(conf)

    logdir = Path(conf.logdir)
    autoenc_dir = logdir / "autoenc"
    autoenc_dir.mkdir(parents=True, exist_ok=True)

    val_every_steps = cfg.get("val_every_steps", 5_000)
    limit_val_batches = cfg.get("limit_val_batches", 100)

    ckpt_every_steps = max(
        max(1, conf.save_every_samples // conf.batch_size_effective),
        int(val_every_steps),
    )
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

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=str(logdir),
        name="",
        version="",
    )

    gpus = cfg.get("gpus", [0])
    # Optional runtime override from launcher, e.g. MOPADI_GPUS=0,1
    # Useful for fast fallback to 2-GPU runs without editing config.yaml.
    env_gpus = os.environ.get("MOPADI_GPUS", "").strip()
    if env_gpus:
        try:
            gpus = [int(x.strip()) for x in env_gpus.split(",") if x.strip() != ""]
        except ValueError as exc:
            raise ValueError(
                f"Invalid MOPADI_GPUS='{env_gpus}'. Expected comma-separated integers."
            ) from exc
        if not gpus:
            raise ValueError("MOPADI_GPUS was set but no valid GPU ids were provided.")
        log.info("Overriding config GPUs with MOPADI_GPUS=%s", gpus)

    accelerator = "gpu" if isinstance(gpus, (list, int)) else "cpu"
    devices = gpus if isinstance(gpus, list) else [gpus]
    # All ranks are launched with identical code and seed, so the initial
    # parameter broadcast inside DDP.__init__ (_sync_module_states) is
    # structurally redundant.  In PyTorch 2.7+cu128, that broadcast runs
    # inside a non-default CUDA stream (added by PL's _setup_model), and
    # the temporary flat buffers it allocates crash on destruction with
    # "CUDA illegal memory access" in TensorImpl::~TensorImpl/destroyEvent.
    # Disabling the sync eliminates the crash with no correctness impact.
    strategy = (
        _DefaultStreamDDPStrategy(
            find_unused_parameters=False,
            init_sync=False,
            broadcast_buffers=False,
        )
        if len(devices) > 1 else "auto"
    )

    if len(devices) > 1 and conf.batch_size % len(devices) != 0:
        raise ValueError(
            f"conf.batch_size ({conf.batch_size}) must be divisible by number of "
            f"devices ({len(devices)}). Set batch_size to GLOBAL batch size."
        )

    max_steps = conf.total_samples // conf.batch_size_effective

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
        val_check_interval=val_every_steps,
        check_val_every_n_epoch=None,
        limit_val_batches=limit_val_batches,
        enable_progress_bar=True,
    )

    resume_ckpt = cfg.get("resume_from", "last")
    if resume_ckpt == "last":
        last_ckpt = Path(autoenc_dir) / "last.ckpt"
        resume_ckpt = str(last_ckpt) if last_ckpt.exists() else None
    if resume_ckpt:
        log.info("Resuming from checkpoint: %s", resume_ckpt)

    model.expected_world_size = max(1, len(devices))

    log.info(
        "Starting training: max_steps=%d, global_batch=%d (effective=%d), "
        "local_batch_per_rank=%d, gpus=%s",
        max_steps,
        conf.batch_size,
        conf.batch_size_effective,
        conf.batch_size // max(1, len(devices)),
        devices,
    )

    try:
        trainer.fit(model, ckpt_path=resume_ckpt)
    except Exception as e:
        log.error("trainer.fit raised %s: %r", type(e).__name__, str(e))
        traceback.print_exc()
        raise
    log.info("Training complete. Last checkpoint: %s", checkpoint_cb.last_model_path)


def _build_train_config(cfg: Dict[str, Any]) -> GenomicCrossAttnConfig:
    for key in ("zip_dir", "genomic_feature_dir", "patient_splits_path"):
        if not cfg.get(key):
            raise ValueError(
                f"Missing required key '{key}' in mopadi_genomic_crossattn config."
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

    model_cfg = cfg.get("model", {})

    def _get(key, default=None):
        return cfg.get(key, model_cfg.get(key, default))

    img_size = int(_get("img_size", 512))
    net_ch = int(_get("net_ch", 128))
    style_ch = int(_get("style_ch", 512))
    feat_dim = int(_get("feat_dim", style_ch))

    raw_mult = _get("net_ch_mult")
    net_ch_mult = tuple(raw_mult) if raw_mult else (1, 1, 2, 2, 4, 4)

    raw_attn = _get("net_attn")
    net_attn = tuple(raw_attn) if raw_attn else (16,)

    base_dir = cfg.get("base_dir", cfg.get("output_dir", "checkpoints/genomic_crossattn"))

    cross_attn_cfg = cfg.get("cross_attention", {})

    fields = dict(
        name="genomic_crossattn",
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
        batch_size=int(_get("batch_size", 4)),
        lr=float(_get("lr", 1e-4)),
        ema_decay=float(_get("ema_decay", 0.9999)),
        grad_clip=float(_get("grad_clip", 1.0)),
        fp16=bool(_get("fp16", False)),
        accum_batches=int(_get("accumulate_grad_batches", 1)),
        num_workers=int(_get("num_workers", 4)),
        total_samples=int(_get("total_samples", 2_500_000)),
        steps_per_epoch=int(_get("steps_per_epoch", 5000)),
        save_every_samples=int(_get("save_every_samples", 200_000)),
        reconstruct_every_samples=int(_get("reconstruct_every_samples", 100_000)),
        sample_size=int(_get("sample_size", 8)),
        zip_dir=cfg["zip_dir"],
        genomic_feature_dir=cfg["genomic_feature_dir"],
        patient_splits_path=cfg["patient_splits_path"],
        max_tiles_by_subtype=cfg.get("max_tiles_by_subtype"),
        tile_sampling_seed=int(cfg.get("tile_sampling_seed", 42)),
        do_normalize=bool(cfg.get("do_normalize", True)),
        do_resize=bool(cfg.get("do_resize", False)),
        val_limit_batches=int(cfg.get("limit_val_batches", 100)),
        # Cross-attention
        cross_attn_heads=int(cross_attn_cfg.get("heads", 4)),
        cross_attn_dim_per_head=int(cross_attn_cfg.get("dim_per_head", 64)),
        cross_attn_patch_size=int(cross_attn_cfg.get("patch_size", 16)),
        cross_attn_lr=float(cfg.get("cross_attn_lr", _get("lr", 1e-4))),
        unet_lr=float(cfg.get("unet_lr", _get("lr", 1e-4))),
        # Genomic-guided loss
        genomic_guided_loss_weight=float(cfg.get("genomic_guided_loss_weight", 0.3)),
        genomic_guided_high_t_frac=float(cfg.get("genomic_guided_high_t_frac", 0.8)),
    )

    return GenomicCrossAttnConfig(**fields)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Train MoPaDi with genomic cross-attention and dual loss."
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config) as fh:
        full_cfg = yaml.safe_load(fh)

    section = full_cfg.get("mopadi_genomic_crossattn", {})
    if not section:
        raise SystemExit("No 'mopadi_genomic_crossattn' section found in config.")

    run_genomic_crossattn_training(section, verbose=True)
