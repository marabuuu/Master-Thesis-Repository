#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Curves — Data Extraction
==================================

Parse training metrics from diffusion fine-tuning runs.  Three data
sources are supported:

1. **TensorBoard event files**  (``events.out.tfevents.*``)
   — most reliable; contains scalar metrics logged via ``self.log()``.

2. **Slurm / stdout log** (``*.out``)
   — tqdm progress bars with ``loss_step``, ``loss_epoch``, ``val_loss``.

3. **Slurm / stderr log** (``*.err``)
   — early-stopping messages, model summary, val-loss improvement lines.

All parsers return a unified ``TrainingRun`` dict (or populate one in
place) so that :mod:`src.visualization.training_plots` can visualise
the data without caring about the source.

Usage
-----
.. code-block:: python

    from src.statistics.training_curves import (
        parse_tensorboard_events,
        parse_lightning_log,
        parse_experiment_dir,
    )
    from src.visualization.training_plots import plot_training_summary

    run = parse_experiment_dir("/path/to/experiment/logdir")
    plot_training_summary(run, save_path="summary.png", show=False)

CLI
---
.. code-block:: bash

    python -m src.statistics.training_curves \\
        --logdir /path/to/experiment/logdir \\
        --output training_summary.png
"""

from __future__ import annotations

import argparse
import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from utils.config_utils import resolve_config_paths
except ImportError:
    from src.utils.config_utils import resolve_config_paths  # type: ignore[import-not-found]


# ===================================================================
#  TrainingRun type alias (documented in visualization.training_plots)
# ===================================================================
TrainingRun = Dict[str, Any]


def _empty_run() -> TrainingRun:
    """Return an empty ``TrainingRun`` template."""
    return {
        "epochs": [],
        "loss_epoch": [],
        "val_loss": [],
        "val_loss_shuffled": [],
        # Actual epoch indices where each val_loss entry was logged.
        # Populated by parse_tensorboard_events when val_check_interval > 1.
        # Avoids mapping val_loss[i] → epoch i (wrong) instead of epoch 5i (right).
        "val_epochs": [],
        "loss_step": [],
        "step_numbers": [],
        "epoch_step_boundaries": [],
        "lr": [],
        "lr_steps": [],
        "genomic_guided_loss": [],
        "genomic_guided_steps": [],
        "genomic_train_loss": [],
        "genomic_train_steps": [],
        "genomic_val_loss": [],
        "genomic_val_steps": [],
        "counterfactual_loss": [],
        "counterfactual_steps": [],
        "cond_gap": [],
        "cond_gap_steps": [],
        "cond_gap_train": [],
        "cond_gap_train_steps": [],
        "val_loss_shuffled": [],
        "val_loss_shuffled_steps": [],
        "improved_epochs": [],
        "best_val_epoch": None,
        "best_val_loss": None,
        "series": {},
        "meta": {},
    }


# ===================================================================
#  1.  TensorBoard event-file parsing
# ===================================================================


def parse_tensorboard_events(
    logdir: str | Path,
    tags: Optional[List[str]] = None,
) -> TrainingRun:
    """Parse TensorBoard ``events.out.tfevents.*`` files.

    Parameters
    ----------
    logdir : str | Path
        Directory containing the event file(s).
    tags : list[str], optional
        Scalar tags to extract.  ``None`` = auto-detect common ones.

    Returns
    -------
    TrainingRun
        Populated dict with whatever tags were found.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        print(
            "[WARN] tensorboard package not installed — "
            "skipping event-file parsing.  "
            "Install with: pip install tensorboard"
        )
        return _empty_run()

    logdir = Path(logdir)
    event_files = sorted(logdir.glob("events.out.tfevents.*"))
    if not event_files:
        # Also search one level down (common layout: logdir/version_0/)
        event_files = sorted(logdir.rglob("events.out.tfevents.*"))
    if not event_files:
        print(f"[WARN] No TensorBoard event files in {logdir}")
        return _empty_run()

    # Use the directory containing the event file for the accumulator
    event_dir = str(event_files[0].parent)
    ea = EventAccumulator(event_dir)
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])
    if not available_tags:
        print("[WARN] No scalar tags in event file")
        return _empty_run()

    if tags is None:
        # If not specified, examine all available tags
        tags = available_tags

    run = _empty_run()

    def _store_series(tag_name: str, steps: List[int], values: List[float]) -> None:
        series = run.setdefault("series", {})
        series[tag_name] = {"tag": tag_name, "steps": steps, "values": values}

    def _set_if_empty(key: str, values: List[float], steps: List[int]) -> None:
        if not run[key]:
            run[key] = values
            step_key = f"{key}_steps"
            if step_key in run:
                run[step_key] = steps

    # Keep track of epoch numbers if available
    epoch_values = []
    epoch_steps = []
    has_epoch_tag = False

    for tag in tags:
        if tag not in available_tags:
            continue
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        # Map tag → TrainingRun key
        tag_lower = tag.lower().replace("/", "_").replace("-", "_")
        _store_series(tag_lower, steps, values)

        # Special: the "epoch" tag tells us the epoch counter at each step
        if tag == "epoch":
            has_epoch_tag = True
            epoch_values = values
            epoch_steps = steps
            if values:
                unique_epochs = sorted(set(int(e) for e in values))
                run["epochs"] = unique_epochs

        if tag_lower in ("loss_epoch", "train_loss_epoch"):
            run["loss_epoch"] = values
            run["epoch_step_boundaries"] = steps
            if not run["epochs"]:
                run["epochs"] = list(range(len(values)))
        elif tag_lower in ("val_loss", "val_loss_epoch", "valid_loss", "loss_val"):
            run["val_loss"] = values
            run["_val_loss_steps"] = steps
            if not run["epochs"]:
                run["epochs"] = list(range(len(values)))
        elif tag_lower in ("loss", "loss_step", "train_loss", "loss_train", "train_loss_step", "loss_train"):
            if not run["loss_step"] or len(values) < len(run["loss_step"]):
                # Prefer loss/train over generic "loss" if both exist
                if tag in ("loss/train", "loss_train") or "train" in tag:
                    run["loss_step"] = values
                    run["step_numbers"] = steps
                elif not run["loss_step"]:
                    run["loss_step"] = values
                    run["step_numbers"] = steps
        elif tag_lower in ("loss_genomic_guided", "genomic_guided_loss"):
            _set_if_empty("genomic_guided_loss", values, steps)
        elif tag_lower in ("loss_counterfactual", "counterfactual_loss"):
            _set_if_empty("counterfactual_loss", values, steps)
        elif tag_lower in ("cond_gap",):
            _set_if_empty("cond_gap", values, steps)
        elif tag_lower in ("cond_gap_train",):
            _set_if_empty("cond_gap_train", values, steps)
        elif tag_lower in ("loss_val_shuffled", "val_loss_shuffled"):
            _set_if_empty("val_loss_shuffled", values, steps)
        elif "lr" in tag_lower:
            run["lr"] = values
            run["lr_steps"] = steps

    # Ensure epochs list length matches
    max_len = max(len(run["loss_epoch"]), len(run["val_loss"]))
    if not has_epoch_tag:
        # Only auto-generate epochs if we don't have the explicit epoch tag
        if max_len and not run["epochs"]:
            run["epochs"] = list(range(max_len))
        elif len(run["epochs"]) < max_len:
            run["epochs"] = list(range(max_len))

    # If we have the epoch tag, use it to bucket step-level metrics accurately.
    if epoch_values and epoch_steps and len(run["epochs"]) > 1:
        # Create a mapping from step to epoch number
        step_to_epoch = dict(zip(epoch_steps, [int(e) for e in epoch_values]))

        # Bucket cond_gap by epoch
        if run.get("cond_gap") and run.get("cond_gap_steps"):
            buckets: Dict[int, List[float]] = {}
            for step, loss in zip(run.get("cond_gap_steps", []), run.get("cond_gap", [])):
                epoch_id = step_to_epoch.get(step)
                if epoch_id is not None:
                    buckets.setdefault(epoch_id, []).append(loss)
            if buckets:
                epochs_sorted = sorted(buckets.keys())
                cond_gap_aggregated = [float(np.mean(buckets[e])) for e in epochs_sorted]
                run["cond_gap"] = cond_gap_aggregated
                # Don't set cond_gap_steps anymore since we aggregated to epochs

        # Bucket cond_gap_train by epoch  
        if run.get("cond_gap_train") and run.get("cond_gap_train_steps"):
            buckets = {}
            for step, loss in zip(run.get("cond_gap_train_steps", []), run.get("cond_gap_train", [])):
                epoch_id = step_to_epoch.get(step)
                if epoch_id is not None:
                    buckets.setdefault(epoch_id, []).append(loss)
            if buckets:
                epochs_sorted = sorted(buckets.keys())
                run["cond_gap_train"] = [float(np.mean(buckets[e])) for e in epochs_sorted]

        # Bucket loss_step by epoch
        if run["loss_step"] and run["step_numbers"]:
            buckets: Dict[int, List[float]] = {}
            for step, loss in zip(run["step_numbers"], run["loss_step"]):
                epoch_id = step_to_epoch.get(step)
                if epoch_id is not None:
                    buckets.setdefault(epoch_id, []).append(loss)
            if buckets:
                epochs_sorted = sorted(buckets.keys())
                run["loss_epoch"] = [float(np.mean(buckets[e])) for e in epochs_sorted]
                # Ensure epochs list covers all epochs we have data for
                if max(epochs_sorted) >= len(run["epochs"]):
                    run["epochs"] = list(range(max(epochs_sorted) + 1))

        # Bucket val_loss by epoch
        val_steps = run.pop("_val_loss_steps", [])
        if run["val_loss"] and val_steps:
            buckets = {}
            for step, loss in zip(val_steps, run["val_loss"]):
                epoch_id = step_to_epoch.get(step)
                if epoch_id is not None:
                    buckets.setdefault(epoch_id, []).append(loss)
            if buckets:
                epochs_sorted = sorted(buckets.keys())
                run["val_loss"] = [float(np.mean(buckets[e])) for e in epochs_sorted]
                run["val_epochs"] = epochs_sorted
                if max(epochs_sorted) >= len(run["epochs"]):
                    run["epochs"] = list(range(max(epochs_sorted) + 1))
    else:
        # Fallback: epoch-alignment without the epoch tag
        val_steps = run.pop("_val_loss_steps", [])
        if val_steps and run["epochs"]:
            if len(run["val_loss"]) < len(run["epochs"]):
                n_val = len(run["val_loss"])
                n_epochs = len(run["epochs"])
                if n_val == 1:
                    run["val_epochs"] = [run["epochs"][-1]]
                else:
                    step = max(1.0, (n_epochs - 1) / float(n_val - 1))
                    run["val_epochs"] = [run["epochs"][min(int(round(i * step)), n_epochs - 1)] for i in range(n_val)]
            else:
                run["val_epochs"] = list(run["epochs"][: len(run["val_loss"])])
        elif run["epochs"]:
            run["val_epochs"] = list(run["epochs"][: len(run["val_loss"])])

    run["meta"]["source"] = "tensorboard"
    run["meta"]["event_file"] = str(event_files[0])
    run["meta"]["available_tags"] = available_tags
    return run


# ===================================================================
#  2.  Lightning tqdm stdout log parsing
# ===================================================================


def parse_lightning_log(
    log_path: str | Path,
) -> TrainingRun:
    """Parse a PyTorch Lightning log stream.

    This parser is intentionally tolerant because the current training run
    writes useful information in both ``.out`` and ``.err``:

    - ``.out`` contains epoch/progress-bar lines such as ``Epoch 0: 0/24661``
      from which we infer epoch boundaries and steps-per-epoch.
    - ``.err`` contains the human-readable metric summaries we care about,
      e.g. ``step 500 | samples 4000 | loss/train 0.0030``.

    Rather than assume Lightning is printing ``loss_epoch=...`` and
    ``val_loss=...`` in the tqdm suffix, we recover the actual step metrics
    that are present and aggregate them into epoch-level series only when we
    have enough information to do so.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    text = log_path.read_text(errors="replace")
    run = _empty_run()
    lines = text.splitlines()

    epoch_re = re.compile(r"Epoch\s+(\d+):")
    epoch_progress_re = re.compile(r"Epoch\s+\d+:.*?\|\s*0/(\d+)")
    step_re = re.compile(r"\bstep\s+(\d+)\b")
    metric_re = re.compile(
        r"(loss/train|loss/val|loss/val_shuffled|loss/genomic_guided|loss/genomic_train|loss/genomic_val|"
        r"loss/counterfactual|cond/gap_train|cond/gap|val_loss|"
        r"val_loss_shuffled|loss_step|train_loss_step|train_loss|loss)"
        r"\s*[:=]?\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)"
    )

    # Keep raw samples so we can aggregate them into epoch-level metrics later.
    step_samples: Dict[str, List[Tuple[int, float]]] = {
        "loss_step": [],
        "val_loss": [],
        "val_loss_shuffled": [],
        "genomic_guided_loss": [],
        "counterfactual_loss": [],
        "cond_gap": [],
        "cond_gap_train": [],
    }
    epoch_seen: List[int] = []
    current_step: Optional[int] = None
    steps_per_epoch: Optional[int] = None

    def _record(metric_key: str, step_num: int, value: float) -> None:
        if metric_key in step_samples:
            step_samples[metric_key].append((step_num, value))
        series = run.setdefault("series", {})
        series.setdefault(metric_key, {"tag": metric_key, "steps": [], "values": []})
        series[metric_key]["steps"].append(step_num)
        series[metric_key]["values"].append(value)

    metric_aliases = {
        "loss": "loss_step",
        "train_loss": "loss_step",
        "loss_step": "loss_step",
        "train_loss_step": "loss_step",
        "loss/train": "loss_step",
        "loss/val": "val_loss",
        "val_loss": "val_loss",
        "loss/val_shuffled": "val_loss_shuffled",
        "val_loss_shuffled": "val_loss_shuffled",
        "loss/genomic_guided": "genomic_guided_loss",
        "loss/genomic_train": "genomic_train_loss",
        "loss/genomic_val": "genomic_val_loss",
        "loss/counterfactual": "counterfactual_loss",
        "cond/gap": "cond_gap",
        "cond/gap_train": "cond_gap_train",
    }

    for line in lines:
        m_epoch = epoch_re.search(line)
        if m_epoch:
            epoch_seen.append(int(m_epoch.group(1)))

        m_steps = epoch_progress_re.search(line)
        if m_steps:
            steps_per_epoch = int(m_steps.group(1))

        m_step = step_re.search(line)
        if m_step:
            current_step = int(m_step.group(1))

        for m in metric_re.finditer(line):
            raw_metric = m.group(1)
            metric_key = metric_aliases.get(raw_metric, raw_metric)
            value = float(m.group(2))
            step_num = current_step if current_step is not None else len(run.get(metric_key, []))
            _record(metric_key, step_num, value)

    # Derive epoch count from actual progress information if we have it.
    max_epoch_seen = max(epoch_seen) if epoch_seen else None

    def _aggregate_by_epoch(samples: List[Tuple[int, float]]) -> tuple[List[int], List[float]]:
        if not samples:
            return [], []

        buckets: Dict[int, List[float]] = {}
        if steps_per_epoch and steps_per_epoch > 0:
            for step_num, value in samples:
                buckets.setdefault(step_num // steps_per_epoch, []).append(value)
        elif max_epoch_seen is not None:
            # Best effort when we know how many epochs exist but do not know the
            # per-epoch step count.  Distribute samples evenly across epochs.
            n_epochs = max_epoch_seen + 1
            ordered = [value for _step, value in samples]
            chunk_size = max(1, len(ordered) // n_epochs)
            for idx in range(n_epochs):
                start = idx * chunk_size
                end = (idx + 1) * chunk_size if idx < n_epochs - 1 else len(ordered)
                chunk = ordered[start:end]
                if chunk:
                    buckets[idx] = chunk
        else:
            buckets[0] = [value for _step, value in samples]

        epochs_sorted = sorted(buckets)
        return epochs_sorted, [float(np.mean(buckets[e])) for e in epochs_sorted]

    # If the log printed explicit epoch-level values, keep them. Otherwise,
    # derive them from the step-level series rather than inventing 50 epochs.
    for key, samples in step_samples.items():
        epochs, values = _aggregate_by_epoch(samples)
        if key == "loss_step":
            run["loss_step"] = [v for _s, v in step_samples[key]]
            run["step_numbers"] = [s for s, _v in step_samples[key]]
        elif key == "val_loss":
            run["val_loss"] = values
            run["val_epochs"] = epochs
        elif key == "val_loss_shuffled":
            run["val_loss_shuffled"] = values
            run["val_loss_shuffled_steps"] = epochs
        elif key == "genomic_guided_loss":
            run["genomic_guided_loss"] = values
            run["genomic_guided_steps"] = epochs
        elif key == "genomic_train_loss":
            run["genomic_train_loss"] = values
            run["genomic_train_steps"] = epochs
        elif key == "genomic_val_loss":
            run["genomic_val_loss"] = values
            run["genomic_val_steps"] = epochs
        elif key == "counterfactual_loss":
            run["counterfactual_loss"] = values
            run["counterfactual_steps"] = epochs
        elif key == "cond_gap":
            run["cond_gap"] = values
            run["cond_gap_steps"] = epochs
        elif key == "cond_gap_train":
            run["cond_gap_train"] = values
            run["cond_gap_train_steps"] = epochs

    if run["loss_step"]:
        run["epoch_step_boundaries"] = []
        if steps_per_epoch and steps_per_epoch > 0:
            max_step = max(run["step_numbers"] or [0])
            n_epochs = max(1, (max_step // steps_per_epoch) + 1)
        elif max_epoch_seen is not None:
            n_epochs = max_epoch_seen + 1
        else:
            n_epochs = max(1, len(run["loss_step"]))

        run["epochs"] = list(range(n_epochs))

        # If we did not manage to construct epoch losses from explicit series,
        # fall back to evenly chunking the logged step losses.  This keeps the
        # x-axis honest and anchored to the observed run length instead of an
        # arbitrary hard-coded 50-epoch assumption.
        if not run["loss_epoch"]:
            step_losses = run["loss_step"]
            chunk_size = max(1, len(step_losses) // n_epochs)
            inferred = []
            for idx in range(n_epochs):
                start = idx * chunk_size
                end = (idx + 1) * chunk_size if idx < n_epochs - 1 else len(step_losses)
                chunk = step_losses[start:end]
                if chunk:
                    inferred.append(float(np.mean(chunk)))
            run["loss_epoch"] = inferred

    if not run["val_loss"] and "val_loss" in run.get("series", {}):
        run["val_loss"] = run["series"]["val_loss"]["values"]

    if run["val_loss"] and not run.get("val_epochs"):
        run["val_epochs"] = list(range(len(run["val_loss"])))

    run["meta"]["source"] = "lightning_log"
    run["meta"]["log_path"] = str(log_path)
    if steps_per_epoch is not None:
        run["meta"]["steps_per_epoch"] = steps_per_epoch
    if max_epoch_seen is not None:
        run["meta"]["max_epoch_seen"] = max_epoch_seen

    # --- metadata from header ---
    lr_match = re.search(r"UNet LR\s*:\s*([\d.eE+-]+)", text)
    proj_lr_match = re.search(r"Proj LR\s*:\s*([\d.eE+-]+)", text)
    patients_match = re.search(r"(\d+)\s+in common", text)
    train_split_match = re.search(
        r"Patient split:\s*(\d+)\s*train,\s*(\d+)\s*val", text
    )

    if lr_match:
        run["meta"]["unet_lr"] = float(lr_match.group(1))
    if proj_lr_match:
        run["meta"]["proj_lr"] = float(proj_lr_match.group(1))
    if patients_match:
        run["meta"]["common_patients"] = int(patients_match.group(1))
    if train_split_match:
        run["meta"]["n_train"] = int(train_split_match.group(1))
        run["meta"]["n_val"] = int(train_split_match.group(2))

    return run


# ===================================================================
#  3.  Stderr log parsing (early stopping, model summary)
# ===================================================================


def parse_stderr_log(
    err_path: str | Path,
    run: Optional[TrainingRun] = None,
) -> TrainingRun:
    """Parse the ``.err`` log for early-stopping and model-summary info.

    Parameters
    ----------
    err_path : str | Path
        Path to the ``.err`` log file.
    run : TrainingRun, optional
        Existing run dict to augment *in-place*.  If ``None``, a fresh
        one is created.

    Returns
    -------
    TrainingRun
    """
    err_path = Path(err_path)
    if not err_path.exists():
        raise FileNotFoundError(f"Stderr log not found: {err_path}")

    if run is None:
        run = _empty_run()

    text = err_path.read_text(errors="replace")

    # val_loss improvement lines from ModelCheckpoint
    # "Metric val_loss improved by 0.003 >= min_delta = 0.0001.
    #  New best score: 0.030"
    # or: "Metric val_loss improved. New best score: 0.033"
    improved_re = re.compile(
        r"val_loss improved.*?New best score:\s*([0-9]*\.?[0-9]+)"
    )
    improved_scores: List[float] = []
    for m in improved_re.finditer(text):
        improved_scores.append(float(m.group(1)))
    run["meta"]["improved_val_scores"] = improved_scores

    # The improvement messages come epoch-sequentially, so map them
    # to epoch indices (0-based)
    run["improved_epochs"] = list(range(len(improved_scores)))

    # Best score
    if improved_scores:
        run["best_val_loss"] = improved_scores[-1]
        run["best_val_epoch"] = len(improved_scores) - 1

    # Early-stopping trigger
    stop_re = re.compile(
        r"did not improve in the last (\d+) records.*"
        r"Best score:\s*([0-9.]+)"
    )
    m_stop = stop_re.search(text)
    if m_stop:
        run["meta"]["early_stopping_patience"] = int(m_stop.group(1))
        run["meta"]["early_stopping_best_score"] = float(m_stop.group(2))
        run["meta"]["early_stopped"] = True
    else:
        run["meta"]["early_stopped"] = False

    # Model parameters
    params_re = re.compile(r"([\d,.]+)\s+Total params")
    m_params = params_re.search(text)
    if m_params:
        run["meta"]["total_params_str"] = m_params.group(1)

    trainable_re = re.compile(r"([\d,.]+)\s+Trainable params")
    m_trainable = trainable_re.search(text)
    if m_trainable:
        run["meta"]["trainable_params_str"] = m_trainable.group(1)

    run["meta"]["err_path"] = str(err_path)
    return run


# ===================================================================
#  4.  Convenience: parse an entire experiment directory
# ===================================================================


def parse_experiment_dir(
    logdir: str | Path,
    prefer_tensorboard: bool = True,
) -> TrainingRun:
    """Parse all available data sources in an experiment output directory.

    The function looks for TensorBoard event files, ``.out`` logs, and
    ``.err`` logs automatically.  Results are merged into a single
    :pydata:`TrainingRun` dict.

    Parameters
    ----------
    logdir : str | Path
        Top-level experiment directory (the one containing ``last.ckpt``,
        ``events.out.tfevents.*``, etc.).
    prefer_tensorboard : bool
        If both a TensorBoard event file **and** a parseable ``.out`` log
        are found, prefer TensorBoard for epoch-level metrics (they are
        more precise).

    Returns
    -------
    TrainingRun
    """
    logdir = Path(logdir)

    run = _empty_run()
    run["meta"]["logdir"] = str(logdir)

    def _newest(paths: List[Path]) -> Optional[Path]:
        if not paths:
            return None
        return max(paths, key=lambda p: p.stat().st_mtime)

    def _merge(dst: TrainingRun, src: TrainingRun) -> None:
        for key in ("epochs", "loss_epoch", "val_loss", "val_epochs", "loss_step", "step_numbers", "epoch_step_boundaries", "lr", "lr_steps", "genomic_guided_loss", "genomic_guided_steps", "counterfactual_loss", "counterfactual_steps", "cond_gap", "cond_gap_steps", "cond_gap_train", "cond_gap_train_steps", "val_loss_shuffled", "val_loss_shuffled_steps", "improved_epochs"):
            if src.get(key) and not dst.get(key):
                dst[key] = src[key]

        for key, value in src.get("series", {}).items():
            dst.setdefault("series", {})[key] = value

        for key, value in src.get("meta", {}).items():
            if key not in dst["meta"]:
                dst["meta"][key] = value

    # --- TensorBoard ---
    tb_run = parse_tensorboard_events(logdir)

    # --- stdout (.out) log ---
    out_files = sorted(logdir.rglob("*.out"))
    log_run = _empty_run()
    out_file = _newest(out_files)
    if out_file is not None:
        try:
            log_run = parse_lightning_log(out_file)
        except Exception as e:
            print(f"[WARN] Could not parse {out_file}: {e}")

    # --- stderr (.err) log ---
    err_files = sorted(logdir.rglob("*.err"))
    err_file = _newest(err_files)
    err_run = _empty_run()
    if err_file is not None:
        try:
            err_run = parse_lightning_log(err_file)
            parse_stderr_log(err_file, run=err_run)
        except Exception as e:
            print(f"[WARN] Could not parse {err_file}: {e}")

    # --- Merge strategy ---
    # TensorBoard epoch-level metrics are more reliable (precise floats).
    # The stdout log gives us per-step losses and metadata.
    # The stderr log gives us early-stopping info.

    if prefer_tensorboard and tb_run.get("loss_epoch"):
        run["loss_epoch"] = tb_run["loss_epoch"]
        run["epochs"] = tb_run["epochs"]
    elif log_run.get("loss_epoch"):
        run["loss_epoch"] = log_run["loss_epoch"]
        run["epochs"] = log_run["epochs"]

    if prefer_tensorboard and tb_run.get("val_loss"):
        run["val_loss"] = tb_run["val_loss"]
        if tb_run.get("val_epochs"):
            run["val_epochs"] = tb_run["val_epochs"]
        if not run["epochs"]:
            run["epochs"] = tb_run["epochs"]
    elif log_run.get("val_loss"):
        run["val_loss"] = log_run["val_loss"]
        if log_run.get("val_epochs"):
            run["val_epochs"] = log_run["val_epochs"]
        if not run["epochs"]:
            run["epochs"] = log_run["epochs"]

    # If val_loss exists but val_epochs was not carried from source, infer a
    # consistent fallback mapping across the full epoch range.
    if run.get("val_loss") and not run.get("val_epochs") and run.get("epochs"):
        n_val = len(run["val_loss"])
        n_epochs = len(run["epochs"])
        if n_val == 1:
            run["val_epochs"] = [run["epochs"][-1]]
        else:
            step = max(1.0, (n_epochs - 1) / float(max(1, n_val - 1)))
            run["val_epochs"] = [
                run["epochs"][min(int(round(i * step)), n_epochs - 1)]
                for i in range(n_val)
            ]

    # Per-step loss always comes from the log (TensorBoard may have it
    # but with sample-count steps that are harder to interpret).
    if log_run.get("loss_step"):
        run["loss_step"] = log_run["loss_step"]
        run["step_numbers"] = log_run["step_numbers"]
        run["epoch_step_boundaries"] = log_run.get(
            "epoch_step_boundaries", []
        )
    elif tb_run.get("loss_step"):
        run["loss_step"] = tb_run["loss_step"]
        run["step_numbers"] = tb_run["step_numbers"]

    # LR from TensorBoard if available
    if tb_run.get("lr"):
        run["lr"] = tb_run["lr"]
        run["lr_steps"] = tb_run["lr_steps"]

    # Merge the richer scalar series from TensorBoard and the Lightning logs.
    _merge(run, tb_run)
    _merge(run, log_run)
    _merge(run, err_run)

    # Rebuild epoch-level curves from the merged step-level series.
    steps_per_epoch = run.get("meta", {}).get("steps_per_epoch")

    def _aggregate_from_series(series_name: str) -> tuple[List[int], List[float]]:
        payload = run.get("series", {}).get(series_name)
        if not payload:
            return [], []
        steps = payload.get("steps", [])
        values = payload.get("values", [])
        if not steps or not values or len(steps) != len(values):
            return [], []
        if not steps_per_epoch or steps_per_epoch <= 0:
            return list(range(len(values))), [float(v) for v in values]
        buckets: Dict[int, List[float]] = {}
        for step_num, value in zip(steps, values):
            buckets.setdefault(int(step_num) // int(steps_per_epoch), []).append(float(value))
        epochs = sorted(buckets)
        return epochs, [float(np.mean(buckets[e])) for e in epochs]

    if run.get("series", {}).get("loss_step"):
        run["epochs"], run["loss_epoch"] = _aggregate_from_series("loss_step")

    if run.get("series", {}).get("val_loss"):
        run["val_epochs"], run["val_loss"] = _aggregate_from_series("val_loss")

    if run.get("series", {}).get("val_loss_shuffled"):
        run["val_loss_shuffled_steps"], run["val_loss_shuffled"] = _aggregate_from_series("val_loss_shuffled")

    if run.get("series", {}).get("genomic_guided_loss"):
        run["genomic_guided_steps"], run["genomic_guided_loss"] = _aggregate_from_series("genomic_guided_loss")

    if run.get("series", {}).get("counterfactual_loss"):
        run["counterfactual_steps"], run["counterfactual_loss"] = _aggregate_from_series("counterfactual_loss")

    if run.get("series", {}).get("cond_gap"):
        run["cond_gap_steps"], run["cond_gap"] = _aggregate_from_series("cond_gap")

    if run.get("series", {}).get("cond_gap_train"):
        run["cond_gap_train_steps"], run["cond_gap_train"] = _aggregate_from_series("cond_gap_train")

    return run


# ===================================================================
#  5.  Summary printing
# ===================================================================


def print_training_summary(run: TrainingRun) -> None:
    """Print a concise textual summary of a training run."""
    meta = run.get("meta", {})
    epochs = run.get("epochs", [])
    train_loss = run.get("loss_epoch", [])
    val_loss = run.get("val_loss", [])

    print("\n" + "=" * 60)
    print("DIFFUSION FINE-TUNING — TRAINING SUMMARY")
    print("=" * 60)

    if meta.get("logdir"):
        print(f"  Directory : {meta['logdir']}")
    if meta.get("source"):
        print(f"  Source    : {meta['source']}")

    if epochs:
        print(f"  Epochs    : {len(epochs)}")
    if train_loss:
        best_idx = int(np.argmin(train_loss))
        print(
            f"  Train loss: final={train_loss[-1]:.6f}  "
            f"best={train_loss[best_idx]:.6f} (ep {epochs[best_idx]})"
        )
    if val_loss:
        best_idx = int(np.argmin(val_loss))
        print(
            f"  Val loss  : final={val_loss[-1]:.6f}  "
            f"best={val_loss[best_idx]:.6f} (ep {epochs[best_idx]})"
        )

    if meta.get("unet_lr"):
        print(f"  UNet LR   : {meta['unet_lr']}")
    if meta.get("proj_lr"):
        print(f"  Proj LR   : {meta['proj_lr']}")
    if run.get("genomic_guided_loss"):
        print(f"  Genomic   : loss/genomic_guided={run['genomic_guided_loss'][-1]:.6f}")
    if run.get("counterfactual_loss"):
        print(f"  Counterf. : loss/counterfactual={run['counterfactual_loss'][-1]:.6f}")
    if run.get("cond_gap"):
        print(f"  cond/gap  : final={run['cond_gap'][-1]:.6e}")
    if meta.get("n_train"):
        print(
            f"  Patients  : {meta['n_train']} train, "
            f"{meta.get('n_val', '?')} val"
        )
    if meta.get("early_stopped"):
        print(
            f"  Early stop: after "
            f"{meta.get('early_stopping_patience', '?')} epochs patience  "
            f"(best={meta.get('early_stopping_best_score', '?')})"
        )

    # Convergence assessment
    if val_loss and len(val_loss) >= 4:
        half = len(val_loss) // 2
        first_half = np.mean(val_loss[:half])
        second_half = np.mean(val_loss[half:])
        rel_drop = (first_half - second_half) / (first_half + 1e-10)
        std_last = np.std(val_loss[half:])
        print(f"\n  Convergence:")
        print(
            f"    Relative improvement (1st vs 2nd half): {rel_drop:.1%}"
        )
        print(
            f"    Std of 2nd-half val loss:               {std_last:.6f}"
        )

    n_step = len(run.get("loss_step", []))
    if n_step:
        print(f"\n  Batch-level: {n_step} step-loss data points")

    print("=" * 60 + "\n")


# ===================================================================
#  CLI
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse & plot training statistics for diffusion "
                    "fine-tuning runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML config file with training_stats section "
             "(overrides individual arguments)",
    )
    parser.add_argument(
        "--logdir", type=str, nargs="+", default=None,
        help="Experiment log directory(ies) containing event files "
             "and/or .out/.err logs",
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", default=None,
        help="Labels for each run (for legend)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file path for the summary plot",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Don't display the plot (only save)",
    )
    parser.add_argument(
        "--prefer-log", action="store_true",
        help="Prefer .out log over TensorBoard for epoch-level metrics",
    )
    args = parser.parse_args()

    # --- Load config if provided ---
    stats_cfg = {}
    repo_root = None  # Will be inferred from config location
    
    if args.config:
        try:
            config_path = Path(args.config).resolve()
            # Infer repo root from config path location
            if config_path.name == "config.yaml" and (config_path.parent / "run_pipeline.py").exists():
                repo_root = config_path.parent
            elif (config_path.parent / "run_pipeline.py").exists():
                repo_root = config_path.parent
            elif (config_path.parent.parent / "run_pipeline.py").exists():
                repo_root = config_path.parent.parent
            
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Resolve paths in config relative to repo root
            if repo_root:
                resolve_config_paths(config, repo_root)
            
            stats_cfg = config.get("training_stats", {})
            
            # Use config values if CLI args not explicitly set
            if not args.logdir and stats_cfg.get("logdir"):
                args.logdir = [stats_cfg["logdir"]]
            if not args.output and stats_cfg.get("output_dir"):
                args.output = str(Path(stats_cfg["output_dir"]) / "training_summary.png")
        except Exception as e:
            print(f"[WARN] Could not load config {args.config}: {e}")

    # --- Validate required arguments ---
    if not args.logdir:
        parser.error(
            "Either --logdir or --config with training_stats.logdir is required"
        )

    runs: List[TrainingRun] = []
    source_names: List[str] = []

    for logdir in args.logdir:
        logdir_path = Path(logdir)
        if not logdir_path.exists():
            print(f"[ERROR] Directory not found: {logdir}")
            continue
        try:
            run = parse_experiment_dir(
                logdir_path,
                prefer_tensorboard=not args.prefer_log,
            )
            runs.append(run)
            source_names.append(logdir_path.name)
            print_training_summary(run)
        except Exception as e:
            print(f"[ERROR] Failed to parse {logdir}: {e}")

    if not runs:
        print("No training data found!")
        return

    labels = args.labels if args.labels else source_names

    # Import plotting from visualization module
    from ..visualization.training_plots import (
        plot_loss_curves,
        plot_training_summary,
        plot_batch_loss_trajectory,
        plot_train_val_comparison,
        plot_genomic_diagnostics,
        plot_early_stopping,
        plot_run_comparison,
    )

    # Determine what plots to generate
    requested_plots = stats_cfg.get("plots", ["summary"]) if stats_cfg else ["summary"]
    output_dir = Path(stats_cfg.get("output_dir", ".")) if stats_cfg else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Shorthand for plot type mapping
    plot_fn_map = {
        "loss_curves": plot_loss_curves,
        "batch_trajectory": plot_batch_loss_trajectory,
        "train_val_comparison": plot_train_val_comparison,
        "genomic_diagnostics": plot_genomic_diagnostics,
        "early_stopping": plot_early_stopping,
        "summary": plot_training_summary,
        "comparison": plot_run_comparison,
    }
    
    # Common plotting kwargs from config
    cmap_cat = stats_cfg.get("cmap_categorical", "batlowS")
    cmap_seq = stats_cfg.get("cmap_sequential", "batlow")
    figsize = tuple(stats_cfg.get("figsize", (16, 12)))
    
    # Generate each requested plot
    for plot_type in requested_plots:
        plot_type = plot_type.strip().lower()
        if plot_type not in plot_fn_map:
            print(f"[WARN] Unknown plot type '{plot_type}', skipping")
            continue
        
        fn = plot_fn_map[plot_type]
        fname = output_dir / f"training_{plot_type}.png"
        
        try:
            if plot_type == "summary":
                plot_training_summary(
                    runs[0],
                    figsize=figsize,
                    cmap_name=cmap_cat,
                    save_path=str(fname),
                    show=not args.no_show,
                )
            elif plot_type == "loss_curves":
                plot_loss_curves(
                    runs,
                    labels=labels,
                    cmap_name=cmap_cat,
                    save_path=str(fname),
                    show=not args.no_show,
                )
            elif plot_type == "batch_trajectory":
                plot_batch_loss_trajectory(
                    runs[0],
                    cmap_name=cmap_seq,
                    save_path=str(fname),
                    show=not args.no_show,
                )
            elif plot_type == "train_val_comparison":
                plot_train_val_comparison(
                    runs[0],
                    cmap_name=cmap_cat,
                    save_path=str(fname),
                    show=not args.no_show,
                )
            elif plot_type == "genomic_diagnostics":
                plot_genomic_diagnostics(
                    runs[0],
                    cmap_name=cmap_cat,
                    save_path=str(fname),
                    show=not args.no_show,
                )
            elif plot_type == "early_stopping":
                plot_early_stopping(
                    runs[0],
                    cmap_name=cmap_cat,
                    save_path=str(fname),
                    show=not args.no_show,
                )
            elif plot_type == "comparison" and len(runs) > 1:
                plot_run_comparison(
                    runs,
                    labels=labels,
                    cmap_name=cmap_cat,
                    save_path=str(fname),
                    show=not args.no_show,
                )
            else:
                continue
            
            print(f"Saved {plot_type} plot to {fname}")
                
        except Exception as e:
            print(f"[ERROR] Failed to generate {plot_type} plot: {e}")


if __name__ == "__main__":
    main()
