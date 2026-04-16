#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared utilities for training entrypoints."""

from __future__ import annotations

import os
from typing import Any

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy


def _num_gpus(gpus: Any) -> int:
    if gpus is None:
        return 0
    if isinstance(gpus, (list, tuple)):
        return len(gpus)
    if isinstance(gpus, int):
        return max(0, gpus)
    return 0


def ensure_logdir(logdir: str) -> None:
    os.makedirs(logdir, exist_ok=True)


def build_checkpoint_callback(
    conf: Any,
    joint_cfg: dict,
    check_val_every_n_epoch: int,
    filename: str,
    auto_insert_metric_name: bool | None = None,
) -> ModelCheckpoint:
    save_top_k = int(joint_cfg.get("save_top_k", 3))
    every_n_train_steps = max(1, int(conf.save_every_samples // conf.batch_size_effective))

    kwargs: dict[str, Any] = {
        "dirpath": conf.logdir,
        "save_last": True,
        "save_top_k": save_top_k,
        "filename": filename,
    }
    if auto_insert_metric_name is not None:
        kwargs["auto_insert_metric_name"] = auto_insert_metric_name

    if save_top_k > 0:
        kwargs.update(
            {
                "monitor": "val_loss",
                "mode": "min",
                "every_n_epochs": max(1, check_val_every_n_epoch),
            }
        )
    else:
        kwargs.update({"every_n_train_steps": every_n_train_steps})

    return ModelCheckpoint(**kwargs)


def find_resume_checkpoint(logdir: str, verbose: bool = False, prefix: str = "") -> str | None:
    ckpt_path = os.path.join(logdir, "last.ckpt")
    if os.path.exists(ckpt_path):
        if verbose:
            if prefix:
                print(f"{prefix} Resuming from {ckpt_path}")
            else:
                print(f"Resuming from {ckpt_path}")
        return ckpt_path
    return None


def resolve_devices_for_launch(gpus: Any) -> Any:
    if os.environ.get("LOCAL_RANK") is not None:
        world_size = os.environ.get("WORLD_SIZE")
        if world_size is not None:
            try:
                return int(world_size)
            except ValueError:
                pass
        return max(1, _num_gpus(gpus))
    return gpus


def choose_ddp_strategy(
    gpus: Any,
    devices: Any,
    find_unused_parameters: bool,
    use_local_rank: bool,
) -> DDPStrategy | str:
    if isinstance(devices, (list, tuple)) and len(devices) > 1:
        return DDPStrategy(find_unused_parameters=find_unused_parameters)

    if _num_gpus(gpus) > 1:
        return DDPStrategy(find_unused_parameters=find_unused_parameters)

    if use_local_rank and os.environ.get("LOCAL_RANK") is not None:
        return DDPStrategy(find_unused_parameters=find_unused_parameters)

    return "auto"
