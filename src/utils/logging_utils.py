#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities for robust PyTorch Lightning logging."""

from __future__ import annotations

from typing import Any, List, Mapping

from pytorch_lightning import loggers as pl_loggers


class SafeTensorBoardLogger(pl_loggers.TensorBoardLogger):
    """
    TensorBoard logger that disables itself on write errors.

    This prevents training from crashing due to event-file I/O issues.
    """

    def __init__(self, *args: Any, verbose: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._disabled = False
        self._verbose = verbose

    def _disable(self, where: str, exc: Exception) -> None:
        if self._disabled:
            return
        self._disabled = True
        if self._verbose:
            print(
                "[WARN] TensorBoard logging disabled after error in "
                f"{where}: {exc}. Training will continue with remaining loggers."
            )

    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        if self._disabled:
            return
        try:
            super().log_hyperparams(params, *args, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on runtime I/O failures
            self._disable("log_hyperparams", exc)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        if self._disabled:
            return
        try:
            super().log_metrics(metrics, step)
        except Exception as exc:  # pragma: no cover - depends on runtime I/O failures
            self._disable("log_metrics", exc)

    def save(self) -> None:
        if self._disabled:
            return
        try:
            super().save()
        except Exception as exc:  # pragma: no cover - depends on runtime I/O failures
            self._disable("save", exc)

    def finalize(self, status: str) -> None:
        if self._disabled:
            return
        try:
            super().finalize(status)
        except Exception as exc:  # pragma: no cover - depends on runtime I/O failures
            self._disable("finalize", exc)


def build_robust_loggers(logdir: str, cfg: dict[str, Any], verbose: bool = True):
    """Build logger list with safe TensorBoard and CSV fallback."""
    tb_enabled = bool(cfg.get("enable_tensorboard", True))
    csv_enabled = bool(cfg.get("enable_csv_logger", True))

    tb_flush_secs = int(cfg.get("tb_flush_secs", 30))
    tb_max_queue = int(cfg.get("tb_max_queue", 10))

    loggers: List[Any] = []

    if tb_enabled:
        loggers.append(
            SafeTensorBoardLogger(
                save_dir=logdir,
                name=None,
                version="",
                flush_secs=tb_flush_secs,
                max_queue=tb_max_queue,
                verbose=verbose,
            )
        )

    if csv_enabled:
        loggers.append(pl_loggers.CSVLogger(save_dir=logdir, name="csv_logs", version=""))

    if verbose:
        active = []
        if tb_enabled:
            active.append(f"TensorBoard(flush_secs={tb_flush_secs}, max_queue={tb_max_queue})")
        if csv_enabled:
            active.append("CSV")
        if active:
            print(f"[INFO] Active loggers: {', '.join(active)}")
        else:
            print("[WARN] All loggers disabled by config; no metrics will be persisted.")

    return loggers if loggers else False
