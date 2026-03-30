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


def resolve_config_paths(config_dict: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    """
    Recursively resolve relative paths in config relative to repo_root.
    
    Converts paths like "./data/..." or "experiments/..." to absolute paths
    based on the repository root. Leaves absolute paths unchanged.
    
    Parameters
    ----------
    config_dict : Dict[str, Any]
        Configuration dictionary (may contain nested dicts and lists)
    repo_root : Path
        Repository root directory to use as base for relative paths
    
    Returns
    -------
    Dict[str, Any]
        Configuration with resolved paths
    """
    def _resolve_path(value: str) -> str:
        repo_candidate = (repo_root / value).resolve()
        normalized = value[2:] if value.startswith("./") else value

        # In this workspace layout, `data/`, `dataframes/`, and `experiments/`
        # are siblings of the repository root.
        if normalized.startswith(("data/", "dataframes/", "experiments/")):
            parent_candidate = (repo_root.parent / normalized).resolve()
            if parent_candidate.exists() or not repo_candidate.exists():
                return str(parent_candidate)

        return str(repo_candidate)

    if isinstance(config_dict, dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                resolve_config_paths(value, repo_root)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        resolve_config_paths(item, repo_root)
            elif isinstance(value, str):
                # Detect if this looks like a path:
                # - starts with ./ or ../
                # - contains path separators and common dir names, or is a relative path
                # - is NOT already absolute
                if not value.startswith('/') and (
                    value.startswith('./') or 
                    value.startswith('../') or
                    any(part in value for part in ['data/', 'experiments/', 'dataframes/', 'slurm/', 'src/'])
                ):
                    config_dict[key] = _resolve_path(value)
    
    return config_dict


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
        "loss_step": [],
        "step_numbers": [],
        "epoch_step_boundaries": [],
        "lr": [],
        "lr_steps": [],
        "improved_epochs": [],
        "best_val_epoch": None,
        "best_val_loss": None,
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

    for tag in tags:
        if tag not in available_tags:
            continue
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        # Map tag → TrainingRun key
        tag_lower = tag.lower().replace("/", "_").replace("-", "_")

        if tag_lower in ("loss_epoch", "train_loss", "train_loss_epoch", "train_loss_epoch"):
            run["loss_epoch"] = values
            # Derive epoch indices (Lightning logs once per epoch for
            # on_epoch=True metrics)
            if not run["epochs"]:
                run["epochs"] = list(range(len(values)))
        elif tag_lower in ("val_loss", "val_loss_epoch", "valid_loss", "val_loss_epoch"):
            run["val_loss"] = values
            if not run["epochs"]:
                run["epochs"] = list(range(len(values)))
        elif tag_lower in ("loss", "loss_step", "train_loss_step"):
            run["loss_step"] = values
            run["step_numbers"] = steps
        elif "lr" in tag_lower:
            run["lr"] = values
            run["lr_steps"] = steps

    # Ensure epochs list length matches
    max_len = max(len(run["loss_epoch"]), len(run["val_loss"]))
    if max_len and not run["epochs"]:
        run["epochs"] = list(range(max_len))
    elif len(run["epochs"]) < max_len:
        run["epochs"] = list(range(max_len))

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
    """Parse a PyTorch Lightning stdout log (tqdm progress bars).

    Extracts ``loss_step``, ``loss_epoch``, and ``val_loss`` from lines
    such as::

        Epoch 0: 100%|██| 4210/4210 [..., loss_step=0.047, val_loss=0.033, loss_epoch=0.037]

    The parser picks the **last** occurrence of each metric per epoch
    (the authoritative 100% tqdm update).

    Parameters
    ----------
    log_path : str | Path
        Path to the ``.out`` log file.

    Returns
    -------
    TrainingRun
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    text = log_path.read_text(errors="replace")
    run = _empty_run()

    # --- epoch-level metrics (last report per epoch) ---
    # We collect all values per epoch and keep the last one.
    epoch_loss: Dict[int, float] = {}
    epoch_val: Dict[int, float] = {}

    # Patterns for the tqdm suffix
    epoch_re = re.compile(r"Epoch\s+(\d+)")
    loss_epoch_re = re.compile(r"loss_epoch\s*=\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)")
    val_loss_re = re.compile(r"val_loss\s*=\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)")

    current_epoch: Optional[int] = None
    for line in text.splitlines():
        m_ep = epoch_re.search(line)
        if m_ep:
            current_epoch = int(m_ep.group(1))

        if current_epoch is None:
            continue

        m_le = loss_epoch_re.search(line)
        if m_le:
            epoch_loss[current_epoch] = float(m_le.group(1))

        m_vl = val_loss_re.search(line)
        if m_vl:
            epoch_val[current_epoch] = float(m_vl.group(1))

    # Build sorted epoch-level lists
    all_epochs = sorted(set(epoch_loss.keys()) | set(epoch_val.keys()))
    run["epochs"] = all_epochs
    run["loss_epoch"] = [epoch_loss.get(e, float("nan")) for e in all_epochs]
    run["val_loss"] = [epoch_val.get(e, float("nan")) for e in all_epochs]

    # Remove trailing NaNs
    for key in ("loss_epoch", "val_loss"):
        while run[key] and (run[key][-1] != run[key][-1]):  # NaN check
            run[key].pop()

    # --- per-step loss ---
    step_loss_re = re.compile(r"loss_step\s*=\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)")
    step_losses: List[float] = []
    step_epoch_boundaries: List[int] = []
    prev_epoch: Optional[int] = None

    for line in text.splitlines():
        m_ep = epoch_re.search(line)
        if m_ep:
            ep = int(m_ep.group(1))
            if ep != prev_epoch:
                step_epoch_boundaries.append(len(step_losses))
                prev_epoch = ep

        m_sl = step_loss_re.search(line)
        if m_sl:
            step_losses.append(float(m_sl.group(1)))

    # Deduplicate consecutive identical values (tqdm re-renders)
    deduped: List[float] = []
    for v in step_losses:
        if not deduped or v != deduped[-1]:
            deduped.append(v)
    run["loss_step"] = deduped
    run["step_numbers"] = list(range(len(deduped)))
    run["epoch_step_boundaries"] = step_epoch_boundaries

    # --- metadata from header ---
    lr_match = re.search(r"UNet LR\s*:\s*([\d.eE+-]+)", text)
    proj_lr_match = re.search(r"Proj LR\s*:\s*([\d.eE+-]+)", text)
    patients_match = re.search(r"(\d+)\s+in common", text)
    train_split_match = re.search(
        r"Patient split:\s*(\d+)\s*train,\s*(\d+)\s*val", text
    )

    run["meta"]["source"] = "lightning_log"
    run["meta"]["log_path"] = str(log_path)
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

    # --- TensorBoard ---
    tb_run = parse_tensorboard_events(logdir)

    # --- stdout (.out) log ---
    out_files = sorted(logdir.glob("*.out"))
    log_run = _empty_run()
    if out_files:
        try:
            log_run = parse_lightning_log(out_files[0])
        except Exception as e:
            print(f"[WARN] Could not parse {out_files[0]}: {e}")

    # --- stderr (.err) log ---
    err_files = sorted(logdir.glob("*.err"))
    if err_files:
        try:
            parse_stderr_log(err_files[0], run=run)
        except Exception as e:
            print(f"[WARN] Could not parse {err_files[0]}: {e}")

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
        if not run["epochs"]:
            run["epochs"] = tb_run["epochs"]
    elif log_run.get("val_loss"):
        run["val_loss"] = log_run["val_loss"]
        if not run["epochs"]:
            run["epochs"] = log_run["epochs"]

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

    # --- INFER MISSING EPOCH METRICS FROM STEP METRICS ---
    if not run.get("loss_epoch") and run.get("loss_step"):
        step_losses = run["loss_step"]
        boundaries = run.get("epoch_step_boundaries", [])
        inferred_epochs = []
        
        if boundaries and len(boundaries) > 1:
            for i in range(len(boundaries) - 1):
                chunk = step_losses[boundaries[i]:boundaries[i+1]]
                if chunk: inferred_epochs.append(sum(chunk) / len(chunk))
            last_chunk = step_losses[boundaries[-1]:]
            if last_chunk: inferred_epochs.append(sum(last_chunk) / len(last_chunk))
        else:
            # Try to infer the number of epochs from checkpoint filenames (if any).
            # If we have a checkpoint named `epoch=99-step=500000.ckpt`, we can use
            # that to set a realistic epoch count instead of an arbitrary 50.
            n_epochs = None
            for ckpt in Path(logdir).glob("epoch=*.ckpt"):
                m = re.search(r"epoch=(\d+)", ckpt.name)
                if m:
                    epoch_id = int(m.group(1))
                    if n_epochs is None or epoch_id + 1 > n_epochs:
                        n_epochs = epoch_id + 1

            if n_epochs is None:
                n_epochs = min(50, len(step_losses))

            # Ensure we have at most one loss value per epoch
            n_epochs = min(n_epochs, len(step_losses))

            chunk_size = len(step_losses) // n_epochs
            for i in range(n_epochs):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < n_epochs - 1 else len(step_losses)
                chunk = step_losses[start:end]
                if chunk:
                    inferred_epochs.append(sum(chunk) / len(chunk))

        run["loss_epoch"] = inferred_epochs
        if not run.get("epochs") or len(run["epochs"]) != len(inferred_epochs):
            run["epochs"] = list(range(len(inferred_epochs)))
        run["meta"]["inferred_epoch_loss"] = True

    # Merge metadata
    for src in (tb_run, log_run):
        for k, v in src.get("meta", {}).items():
            if k not in run["meta"]:
                run["meta"][k] = v

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
