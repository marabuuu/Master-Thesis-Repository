#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Plots
==============

Crameri-coloured plotting functions for diffusion model fine-tuning runs.

All data **parsing** lives in :mod:`src.statistics.training_curves`;
this module only handles visualisation.

Plots available
---------------
- :func:`plot_loss_curves` – train & validation epoch-level loss
- :func:`plot_batch_loss_trajectory` – per-step loss across all epochs
- :func:`plot_train_val_comparison` – overlaid train vs val with gap shading
- :func:`plot_genomic_diagnostics` – auxiliary genomic losses and cond/gap
- :func:`plot_lr_schedule` – learning-rate over training steps / epochs
- :func:`plot_early_stopping` – val loss with patience / best markers
- :func:`plot_training_summary` – multi-panel figure combining all above

Colour maps
-----------
Every function defaults to `Fabio Crameri`_ scientific colour maps
(``cmcrameri`` package).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np

try:
    from .core import (
        CATEGORICAL_CMAP,
        SEQUENTIAL_CMAP,
        _check_matplotlib,
        get_categorical_colors,
        get_crameri_cmap,
        save_figure,
        setup_style,
        show_or_save,
    )
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.visualization.core import (  # type: ignore[no-redef]
        CATEGORICAL_CMAP,
        SEQUENTIAL_CMAP,
        _check_matplotlib,
        get_categorical_colors,
        get_crameri_cmap,
        save_figure,
        setup_style,
        show_or_save,
    )

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator
except ImportError:  # pragma: no cover
    pass

# Pylance/type checkers: names are guaranteed to exist when matplotlib is
# installed (the import above is guarded so static analysis loses track).
if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator



# ===================================================================
#  Typing alias for parsed training data
# ===================================================================
#  A ``TrainingRun`` is any mapping that may contain the following keys
#  (all optional — only present if the source provides them):
#
#      epochs         – list[int]
#      loss_epoch     – list[float]   per-epoch training loss
#      val_loss       – list[float]   per-epoch validation loss
#      loss_step      – list[float]   per-step training loss
#      step_numbers   – list[int]     global step for each loss_step entry
#      lr             – list[float]   learning-rate values
#      lr_steps       – list[int]     step indices for lr entries
#      best_val_epoch – int | None
#      best_val_loss  – float | None
#      patience_stops – list[int]     epochs where patience was exhausted
#      meta           – dict[str, Any]
#
TrainingRun = Dict[str, Any]


def _aggregate_by_step(
    steps: List[float],
    values: List[float],
) -> Tuple[List[float], List[float]]:
    """Average duplicate entries at the same step and return sorted unique steps."""
    if not steps:
        return [], []
    from collections import defaultdict
    agg: dict = defaultdict(list)
    for s, v in zip(steps, values):
        agg[s].append(v)
    sorted_steps = sorted(agg)
    return sorted_steps, [float(np.mean(agg[s])) for s in sorted_steps]


def _bin_series(
    steps: List[float],
    values: List[float],
    n_bins: int = 50,
) -> Tuple[List[float], List[float]]:
    """Bin a dense (steps, values) series into n_bins equal-width step windows.

    Returns the bin-centre x and the mean y per bin.  Useful for turning a
    noisy step-level loss curve into a smooth epoch-level trend line without
    throwing away data in the faded raw plot.
    """
    if not steps:
        return [], []
    arr_s = np.asarray(steps, dtype=float)
    arr_v = np.asarray(values, dtype=float)
    edges = np.linspace(arr_s.min(), arr_s.max(), n_bins + 1)
    centers, means = [], []
    for i in range(n_bins):
        mask = (arr_s >= edges[i]) & (arr_s < edges[i + 1])
        if mask.any():
            centers.append(float(0.5 * (edges[i] + edges[i + 1])))
            means.append(float(arr_v[mask].mean()))
    return centers, means


def _ema_series(
    steps: List[float],
    values: List[float],
    alpha: float = 0.005,
    max_points: int = 2000,
) -> Tuple[List[float], List[float]]:
    """Exponential moving average smoothing of a noisy step-level series.

    Unlike bin-averaging, EMA inherits the very first value and then
    exponentially tracks the true mean — so a training-loss curve that starts
    at 1.0 and drops quickly will actually *show* 1.0 at the left edge, which
    bin-averaging over wide windows cannot do.

    Steps are sorted before smoothing so multi-file data merges correctly.
    The result is thinned to at most *max_points* for fast rendering.
    """
    if not steps:
        return [], []
    pairs = sorted(zip(steps, values))
    s = [p[0] for p in pairs]
    v = [p[1] for p in pairs]
    ema = [v[0]]
    for i in range(1, len(v)):
        ema.append(alpha * v[i] + (1.0 - alpha) * ema[-1])
    # Thin evenly so we don't plot 40 000 points
    stride = max(1, len(s) // max_points)
    return s[::stride], ema[::stride]


def _series_payload(run: TrainingRun, *candidate_names: str) -> Optional[Dict[str, Any]]:
    series = run.get("series", {})
    for name in candidate_names:
        key = name.lower().replace("/", "_").replace("-", "_")
        payload = series.get(key)
        if payload:
            return payload
    return None


def _series_xy(run: TrainingRun, *candidate_names: str) -> Tuple[List[float], List[float]]:
    payload = _series_payload(run, *candidate_names)
    if not payload:
        return [], []
    return payload.get("steps", []), payload.get("values", [])


def _plot_smoothed_series(
    ax: Any,
    x: Sequence[float],
    y: Sequence[float],
    color: str,
    label: str,
    linewidth: float = 0.4,
    alpha: float = 0.35,
    smooth_color: Optional[str] = None,
) -> None:
    if not y:
        return
    ax.plot(x, y, linewidth=linewidth, alpha=alpha, color=color, label=label)
    if len(y) > 50:
        w = max(10, len(y) // 80)
        kernel = np.ones(w) / w
        smoothed = np.convolve(np.asarray(y, dtype=float), kernel, mode="valid")
        offset = w // 2
        ax.plot(
            x[offset: offset + len(smoothed)],
            smoothed,
            linewidth=1.6,
            color=smooth_color or color,
            alpha=0.95,
        )


# ===================================================================
#  Loss curves (train + val, epoch-level)
# ===================================================================


def plot_loss_curves(
    runs: Union[TrainingRun, List[TrainingRun]],
    labels: Optional[List[str]] = None,
    title: str = "Diffusion Fine-Tuning Loss",
    cmap_name: str = CATEGORICAL_CMAP,
    figsize: Tuple[float, float] = (10, 5),
    log_scale: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Figure:
    """Plot epoch-level training and validation loss curves.

    Parameters
    ----------
    runs : dict or list[dict]
        One or several :pydata:`TrainingRun` dicts (from
        :mod:`statistics.training_curves` parsers).
    labels : list[str], optional
        Legend labels for each run.
    title : str
    cmap_name : str
        Crameri colourmap for distinguishing runs.
    log_scale : bool
        Use log scale on the y-axis.
    save_path : str | Path | None
    show : bool
    """
    _check_matplotlib()
    setup_style()

    if isinstance(runs, dict):
        runs = [runs]
    if labels is None:
        labels = [f"Run {i + 1}" for i in range(len(runs))]

    colors = get_categorical_colors(max(2 * len(runs), 4), cmap_name=cmap_name)

    fig, ax = plt.subplots(figsize=figsize)

    for i, (run, label) in enumerate(zip(runs, labels)):
        epochs = run.get("epochs", [])
        train = run.get("loss_epoch", [])
        val = run.get("val_loss", [])

        c_train = colors[2 * i]
        c_val = colors[2 * i + 1]

        if train and epochs:
            ep = epochs[: len(train)]
            ax.plot(ep, train, "-o", markersize=3, linewidth=1.5,
                    color=c_train, label=f"{label} — train", alpha=0.85)
            _annotate_min(ax, ep, train, color=c_train)

        if val and epochs:
            # Use val_epochs (actual epoch positions) when available so that
            # val_loss entries are plotted at their true epoch rather than at
            # sequential indices 0…N-1 (which would make val look like it
            # stopped early when val_check_interval > 1).
            val_ep = run.get("val_epochs")
            ep = val_ep if val_ep and len(val_ep) == len(val) else epochs[: len(val)]
            ax.plot(ep, val, "-s", markersize=3, linewidth=1.5,
                    color=c_val, label=f"{label} — val", alpha=0.85)
            _annotate_min(ax, ep, val, color=c_val)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontweight="bold")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    if log_scale:
        ax.set_yscale("log")

    fig.tight_layout()
    show_or_save(fig, save_path=save_path, show=show)
    return fig


# ===================================================================
#  Batch-level (step-level) loss trajectory
# ===================================================================


def plot_batch_loss_trajectory(
    run: TrainingRun,
    title: str = "Batch-Level Loss Trajectory",
    cmap_name: str = SEQUENTIAL_CMAP,
    smoothing_window: Optional[int] = None,
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Figure:
    """Plot the per-step ``loss_step`` across the entire training run.

    A moving-average smoothing line is overlaid when the trajectory has
    more than 50 data points.

    Parameters
    ----------
    run : dict
        A single :pydata:`TrainingRun`.
    smoothing_window : int, optional
        Window size for moving average.  ``None`` = auto.
    """
    _check_matplotlib()
    setup_style()

    loss_step = run.get("loss_step", [])
    if not loss_step:
        raise ValueError("No per-step loss data (loss_step) in run")

    step_nums = run.get("step_numbers", list(range(len(loss_step))))
    cmap = get_crameri_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=figsize)

    # Raw trajectory (faded)
    ax.plot(step_nums, loss_step, linewidth=0.4, alpha=0.4,
            color=cmap(0.3), label="loss (raw)")

    # Smoothed line
    n = len(loss_step)
    if n > 50:
        w = smoothing_window or max(10, n // 80)
        kernel = np.ones(w) / w
        smoothed = np.convolve(loss_step, kernel, mode="valid")
        offset = w // 2
        ax.plot(step_nums[offset: offset + len(smoothed)], smoothed,
                linewidth=1.8, color=cmap(0.7),
                label=f"moving avg (w={w})")

    # Mark epoch boundaries if available
    epoch_boundaries = run.get("epoch_step_boundaries", [])
    for eb in epoch_boundaries:
        ax.axvline(eb, color="grey", alpha=0.15, linewidth=0.8)

    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    show_or_save(fig, save_path=save_path, show=show)
    return fig


# ===================================================================
#  Train vs Val comparison with gap shading
# ===================================================================


def plot_train_val_comparison(
    run: TrainingRun,
    title: str = "Train vs Validation Loss",
    cmap_name: str = CATEGORICAL_CMAP,
    figsize: Tuple[float, float] = (10, 5),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Figure:
    """Overlay training and validation loss with shaded gap region.

    The shaded region between the two curves makes over-fitting
    or under-fitting immediately visible.
    """
    _check_matplotlib()
    setup_style()

    epochs = run.get("epochs", [])
    train = run.get("loss_epoch", [])
    val = run.get("val_loss", [])

    if not train or not val or not epochs:
        raise ValueError("Both loss_epoch and val_loss are required")

    # Use val_epochs (actual epoch positions) when available so that
    # val_loss entries land at the right x position (e.g. every 5th epoch)
    # rather than being truncated to len(val) epochs.
    val_epochs = run.get("val_epochs")
    val_ep = np.array(val_epochs) if val_epochs and len(val_epochs) == len(val) else np.array(epochs[: len(val)])

    n_train = min(len(epochs), len(train))
    train_ep = np.array(epochs[:n_train])
    tr = np.array(train[:n_train])
    vl = np.array(val)

    colors = get_categorical_colors(4, cmap_name=cmap_name)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(train_ep, tr, "-o", markersize=3, linewidth=1.5, color=colors[0],
            label="Train loss", alpha=0.85)
    ax.plot(val_ep, vl, "-s", markersize=3, linewidth=1.5, color=colors[1],
            label="Val loss", alpha=0.85)

    # Shade the train-val gap only over the shared x range
    if len(train_ep) > 0 and len(val_ep) > 0:
        x_min = max(train_ep[0], val_ep[0])
        x_max = min(train_ep[-1], val_ep[-1])
        if x_min < x_max:
            import numpy as _np
            x_shared = _np.linspace(x_min, x_max, 300)
            tr_interp = _np.interp(x_shared, train_ep, tr)
            vl_interp = _np.interp(x_shared, val_ep, vl)
            ax.fill_between(x_shared, tr_interp, vl_interp, alpha=0.12,
                            color=colors[2], label="Train-Val gap")

    _annotate_min(ax, val_ep.tolist(), vl.tolist(), color=colors[1], label="best val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontweight="bold")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    show_or_save(fig, save_path=save_path, show=show)
    return fig


# ===================================================================
#  Genomic diagnostics summary
# ===================================================================


def plot_genomic_diagnostics(
    run: TrainingRun,
    title: str = "Genomic Conditioning Diagnostics",
    cmap_name: str = CATEGORICAL_CMAP,
    figsize: Tuple[float, float] = (16, 12),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Figure:
    """Visualise diffusion loss, auxiliary genomic losses, and cond/gap.

    Panels:
      1. Epoch-level image loss (train vs val)
      2. Validation image loss vs shuffled-conditioning loss
      3. Step-level auxiliary genomic losses
      4. Validation cond/gap trajectory
    """
    _check_matplotlib()
    setup_style()

    epochs = run.get("epochs", [])
    train = run.get("loss_epoch", [])
    val = run.get("val_loss", [])
    val_shuffled = run.get("val_loss_shuffled", [])
    val_epochs = run.get("val_epochs", [])
    cond_gap = run.get("cond_gap", [])

    if not (epochs or train or val or val_shuffled or cond_gap):
        raise ValueError("No genomic diagnostics available in run")

    colors = get_categorical_colors(8, cmap_name=cmap_name)
    seq_cmap = get_crameri_cmap(SEQUENTIAL_CMAP)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes_flat = axes.flatten()

    # Panel 1: epoch-level image loss
    ax = axes_flat[0]
    if train and epochs:
        train_ep = epochs[: len(train)]
        ax.plot(train_ep, train, "-o", markersize=3, linewidth=1.5,
                color=colors[0], label="Train loss", alpha=0.9)
        _annotate_min(ax, train_ep, train, color=colors[0])
    if val and epochs:
        val_ep = val_epochs if val_epochs and len(val_epochs) == len(val) else epochs[: len(val)]
        ax.plot(val_ep, val, "-s", markersize=3, linewidth=1.5,
                color=colors[1], label="Val loss", alpha=0.9)
        _annotate_min(ax, val_ep, val, color=colors[1])
    ax.set_title("Epoch-Level Image Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25)
    ax.legend(framealpha=0.9)

    # Panel 2: validation correct vs shuffled conditioning
    # Aggregate to one point per validation run (TensorBoard logs ~100 per-batch
    # entries at the same global step, which causes vertical stacking in plots).
    ax = axes_flat[1]
    val_x, val_y = _aggregate_by_step(*_series_xy(run, "loss/val", "val_loss"))
    shuffled_x, shuffled_y = _aggregate_by_step(*_series_xy(run, "loss/val_shuffled", "val_loss_shuffled"))
    gap_x, gap_y = _aggregate_by_step(*_series_xy(run, "cond/gap", "cond_gap"))
    if val_y and shuffled_y:
        ax.plot(val_x, val_y, "-o", markersize=3, linewidth=1.5,
                color=colors[2], label="Val loss (correct)", alpha=0.9)
        ax.plot(shuffled_x, shuffled_y, "-s", markersize=3, linewidth=1.5,
                color=colors[3], label="Val loss (shuffled)", alpha=0.9)
        shared_n = min(len(val_x), len(shuffled_x), len(val_y), len(shuffled_y))
        if shared_n:
            gap = np.asarray(shuffled_y[:shared_n]) - np.asarray(val_y[:shared_n])
            ax2 = ax.twinx()
            ax2.plot(val_x[:shared_n], gap, color=seq_cmap(0.7), linewidth=1.2,
                     alpha=0.85, label="cond/gap")
            ax2.axhline(0.0, color="grey", linewidth=0.8, alpha=0.35)
            ax2.set_ylabel("cond/gap")
            ax2.tick_params(axis="y", labelcolor=seq_cmap(0.7))
            handles_left, labels_left = ax.get_legend_handles_labels()
            handles_right, labels_right = ax2.get_legend_handles_labels()
            ax2.legend(handles_left + handles_right, labels_left + labels_right,
                       framealpha=0.9, loc="upper right")
    elif gap_y and gap_x:
        ax.plot(gap_x, gap_y, "-o", markersize=3, linewidth=1.5,
                color=seq_cmap(0.7), label="cond/gap", alpha=0.9)
        ax.axhline(0.0, color="grey", linewidth=0.8, alpha=0.35)
    ax.set_title("Validation Conditioning Gap")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(framealpha=0.9, loc="upper right")

    # Panel 3: guidance_delta + auxiliary losses
    # guidance_delta = MSE(eps_cond, eps_null) — the primary CFG diagnostic.
    # Aggregated because TensorBoard logs it per-batch during validation.
    ax = axes_flat[2]
    guidance_x, guidance_y = _aggregate_by_step(*_series_xy(run, "cond/guidance_delta"))
    genomic_train_x, genomic_train_y = _series_xy(run, "loss/genomic_train", "genomic_train_loss")
    genomic_val_x, genomic_val_y = _series_xy(run, "loss/genomic_val", "genomic_val_loss")
    genomic_x, genomic_y = _series_xy(run, "loss/genomic_guided", "genomic_guided_loss")
    cf_x, cf_y = _series_xy(run, "loss/counterfactual", "counterfactual_loss")
    gap_train_x, gap_train_y = _series_xy(run, "cond/gap_train", "cond_gap_train")
    _has_aux = bool(genomic_train_y or genomic_val_y or genomic_y or cf_y or gap_train_y)
    if guidance_y:
        ax.plot(guidance_x, guidance_y, "-o", markersize=3, linewidth=1.5,
                color=colors[0], label="guidance_delta", alpha=0.9)
        _annotate_min(ax, guidance_x, guidance_y, color=colors[0])
        # Log scale helps when delta spans several orders of magnitude
        positive = [v for v in guidance_y if v > 0]
        if positive and max(positive) / max(min(positive), 1e-12) > 20:
            ax.set_yscale("log")
    if genomic_train_y:
        _plot_smoothed_series(ax, genomic_train_x, genomic_train_y, colors[4], "loss/genomic_train", smooth_color=seq_cmap(0.75))
    elif genomic_y:
        _plot_smoothed_series(ax, genomic_x, genomic_y, colors[4], "loss/genomic_guided", smooth_color=seq_cmap(0.75))
    if genomic_val_y:
        _plot_smoothed_series(ax, genomic_val_x, genomic_val_y, colors[1], "loss/genomic_val", smooth_color=seq_cmap(0.55))
    if cf_y:
        _plot_smoothed_series(ax, cf_x, cf_y, colors[5], "loss/counterfactual", smooth_color=seq_cmap(0.55))
    if gap_train_y:
        _plot_smoothed_series(ax, gap_train_x, gap_train_y, colors[6], "cond/gap_train", smooth_color=seq_cmap(0.35))
    if not guidance_y and not _has_aux:
        ax.text(0.5, 0.5, "No guidance_delta or auxiliary losses found",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Guidance Delta & Auxiliary Losses")
    ax.set_xlabel("Step")
    ax.set_ylabel("guidance_delta / Loss")
    ax.grid(True, alpha=0.25)
    if guidance_y or _has_aux:
        ax.legend(framealpha=0.9)

    # Panel 4: validation cond/gap over time (aggregated — same step de-duplication)
    # gap_x / gap_y already aggregated above in Panel 2 block.
    ax = axes_flat[3]
    if gap_y:
        ax.plot(gap_x, gap_y, "-o", markersize=3, linewidth=1.5,
                color=seq_cmap(0.8), label="cond/gap", alpha=0.9)
        ax.axhline(0.0, color="grey", linewidth=0.8, alpha=0.35)
        _annotate_min(ax, gap_x, gap_y, color=seq_cmap(0.8), label="min gap")
    else:
        ax.text(0.5, 0.5, "No validation cond/gap found", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Validation cond/gap")
    ax.set_xlabel("Step")
    ax.set_ylabel("cond/gap")
    ax.grid(True, alpha=0.25)
    ax.legend(framealpha=0.9)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    show_or_save(fig, save_path=save_path, show=show)
    return fig


# ===================================================================
#  Learning-rate schedule
# ===================================================================


def plot_lr_schedule(
    run: TrainingRun,
    title: str = "Learning Rate Schedule",
    cmap_name: str = SEQUENTIAL_CMAP,
    figsize: Tuple[float, float] = (10, 4),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Figure:
    """Plot the learning-rate schedule over training steps or epochs."""
    _check_matplotlib()
    setup_style()

    lr = run.get("lr", [])
    if not lr:
        raise ValueError("No learning-rate data (lr) in run")

    x = run.get("lr_steps", list(range(len(lr))))
    xlabel = "Step" if run.get("lr_steps") else "Index"

    cmap = get_crameri_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, lr, linewidth=1.5, color=cmap(0.6))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Learning Rate")
    ax.set_title(title, fontweight="bold")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    show_or_save(fig, save_path=save_path, show=show)
    return fig


# ===================================================================
#  Early-stopping visualisation
# ===================================================================


def plot_early_stopping(
    run: TrainingRun,
    title: str = "Early Stopping — Validation Loss",
    cmap_name: str = CATEGORICAL_CMAP,
    figsize: Tuple[float, float] = (10, 5),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Figure:
    """Val loss curve with best-epoch marker and patience annotations."""
    _check_matplotlib()
    setup_style()

    epochs = run.get("epochs", [])
    val = run.get("val_loss", [])
    if not val or not epochs:
        raise ValueError("val_loss and epochs are required")

    val_ep = run.get("val_epochs")
    if val_ep and len(val_ep) == len(val):
        ep, vl = val_ep, val
    else:
        n = min(len(epochs), len(val))
        ep, vl = epochs[:n], val[:n]
    n = len(ep)

    colors = get_categorical_colors(4, cmap_name=cmap_name)
    best_idx = int(np.argmin(vl))
    best_epoch = ep[best_idx]
    best_val = vl[best_idx]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(ep, vl, "-o", markersize=4, linewidth=1.5,
            color=colors[0], label="Val loss")

    # Best epoch
    ax.scatter([best_epoch], [best_val], s=160, marker="*",
               color=colors[1], edgecolors="black", linewidth=0.6,
               zorder=5, label=f"Best: {best_val:.5f} (ep {best_epoch})")

    # Patience region shading (from best epoch to end)
    patience = run.get("meta", {}).get("early_stopping_patience")
    if patience and best_idx + patience < n:
        stop_ep = ep[min(best_idx + patience, n - 1)]
        ax.axvspan(best_epoch, stop_ep, alpha=0.08, color=colors[2],
                    label=f"Patience window ({patience} ep)")

    # Improved-epoch markers from stderr parsing
    improved_epochs = run.get("improved_epochs", [])
    for ie in improved_epochs:
        if ie in ep:
            idx = ep.index(ie)
            ax.annotate("", xy=(ie, vl[idx]),
                        xytext=(ie, vl[idx] + 0.001),
                        arrowprops=dict(arrowstyle="->", color="green",
                                        lw=1.5))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title(title, fontweight="bold")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    show_or_save(fig, save_path=save_path, show=show)
    return fig


# ===================================================================
#  Multi-panel summary figure
# ===================================================================


def plot_training_summary(
    run: TrainingRun,
    title: str = "Diffusion Fine-Tuning — Training Summary",
    cmap_name: str = CATEGORICAL_CMAP,
    figsize: Tuple[float, float] = (16, 12),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Figure:
    """Generate a multi-panel summary of a diffusion fine-tuning run.

    Panels (depending on data availability):
      1. Train & val epoch-level loss
      2. Batch-level loss trajectory
      3. Train vs val gap
      4. Early stopping / val loss detail
    """
    _check_matplotlib()
    setup_style()

    has_train = bool(run.get("loss_epoch"))
    has_val = bool(run.get("val_loss"))
    has_batch = bool(run.get("loss_step"))
    has_both = has_train and has_val

    n_panels = sum([has_train or has_val, has_batch, has_both, has_val])
    n_panels = max(n_panels, 1)

    ncols = min(2, n_panels)
    nrows = (n_panels + 1) // 2

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize[0], figsize[1] * nrows / 2))
    axes_flat = np.atleast_1d(axes).flatten()

    panel = 0
    colors = get_categorical_colors(6, cmap_name=cmap_name)
    seq_cmap = get_crameri_cmap(SEQUENTIAL_CMAP)
    epochs = run.get("epochs", [])

    # Panel 1: epoch-level loss curves
    if has_train or has_val:
        ax = axes_flat[panel]
        panel += 1
        if has_train and epochs:
            tr = run["loss_epoch"]
            ep = epochs[: len(tr)]
            ax.plot(ep, tr, "-o", markersize=3, lw=1.5,
                    color=colors[0], label="Train", alpha=0.85)
            _annotate_min(ax, ep, tr, color=colors[0])
        if has_val and epochs:
            vl = run["val_loss"]
            val_ep = run.get("val_epochs")
            ep = val_ep if val_ep and len(val_ep) == len(vl) else epochs[: len(vl)]
            ax.plot(ep, vl, "-s", markersize=3, lw=1.5,
                    color=colors[1], label="Val", alpha=0.85)
            _annotate_min(ax, ep, vl, color=colors[1])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Epoch-Level Loss")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.25)

    # Panel 2: batch-level trajectory
    if has_batch:
        ax = axes_flat[panel]
        panel += 1
        ls = run["loss_step"]
        sn = run.get("step_numbers", list(range(len(ls))))
        ax.plot(sn, ls, lw=0.4, alpha=0.4, color=seq_cmap(0.3))
        n = len(ls)
        if n > 50:
            w = max(10, n // 80)
            smoothed = np.convolve(ls, np.ones(w) / w, mode="valid")
            off = w // 2
            ax.plot(sn[off: off + len(smoothed)], smoothed,
                    lw=1.8, color=seq_cmap(0.7), label=f"MA (w={w})")
            ax.legend(framealpha=0.9)
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Loss")
        ax.set_title("Batch-Level Loss")
        ax.grid(True, alpha=0.25)

    # Panel 3: train vs val gap
    if has_both:
        ax = axes_flat[panel]
        panel += 1
        tr = np.array(run["loss_epoch"])
        vl = np.array(run["val_loss"])
        n_train = min(len(epochs), len(tr))
        train_ep = np.array(epochs[:n_train])
        val_epochs = run.get("val_epochs")
        if val_epochs and len(val_epochs) == len(vl):
            val_ep = np.array(val_epochs)
        else:
            val_ep = np.array(epochs[: len(vl)])

        ax.plot(train_ep, tr[:n_train], "-o", markersize=3, lw=1.5,
                color=colors[0], label="Train", alpha=0.85)
        ax.plot(val_ep, vl, "-s", markersize=3, lw=1.5,
                color=colors[1], label="Val", alpha=0.85)

        # Shade train-val gap only over shared x-range via interpolation.
        if len(train_ep) > 0 and len(val_ep) > 0:
            x_min = max(train_ep[0], val_ep[0])
            x_max = min(train_ep[-1], val_ep[-1])
            if x_min < x_max:
                x_shared = np.linspace(x_min, x_max, 300)
                tr_interp = np.interp(x_shared, train_ep, tr[:n_train])
                vl_interp = np.interp(x_shared, val_ep, vl)
                ax.fill_between(x_shared, tr_interp, vl_interp, alpha=0.12, color=colors[2])

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Train–Val Gap")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.25)

    # Panel 4: early-stopping detail
    if has_val:
        ax = axes_flat[panel]
        panel += 1
        vl = run["val_loss"]
        val_ep = run.get("val_epochs")
        ep = val_ep if val_ep and len(val_ep) == len(vl) else epochs[: len(vl)]
        ax.plot(ep, vl, "-o", markersize=4, lw=1.5, color=colors[0])
        best_idx = int(np.argmin(vl))
        ax.scatter([ep[best_idx]], [vl[best_idx]], s=160, marker="*",
                   color=colors[1], edgecolors="black", lw=0.6, zorder=5,
                   label=f"Best val={vl[best_idx]:.5f}")
        patience = run.get("meta", {}).get("early_stopping_patience")
        if patience and best_idx + patience < len(ep):
            ax.axvspan(ep[best_idx], ep[min(best_idx + patience, len(ep) - 1)],
                        alpha=0.08, color=colors[2],
                        label=f"Patience ({patience})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Loss")
        ax.set_title("Early Stopping Detail")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.25)

    # Hide unused axes
    for idx in range(panel, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    show_or_save(fig, save_path=save_path, show=show)
    return fig


# ===================================================================
#  Multi-run comparison
# ===================================================================


def plot_run_comparison(
    runs: List[TrainingRun],
    labels: Optional[List[str]] = None,
    metric: str = "val_loss",
    title: str = "Run Comparison",
    cmap_name: str = CATEGORICAL_CMAP,
    figsize: Tuple[float, float] = (10, 5),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Figure:
    """Compare a specific metric across several training runs.

    Parameters
    ----------
    metric : str
        Key into *runs* dicts, e.g. ``"val_loss"``, ``"loss_epoch"``.
    """
    _check_matplotlib()
    setup_style()

    if labels is None:
        labels = [f"Run {i + 1}" for i in range(len(runs))]

    colors = get_categorical_colors(len(runs), cmap_name=cmap_name)

    fig, ax = plt.subplots(figsize=figsize)

    for run, label, color in zip(runs, labels, colors):
        epochs = run.get("epochs", [])
        values = run.get(metric, [])
        if not values or not epochs:
            continue
        n = min(len(epochs), len(values))
        ax.plot(epochs[:n], values[:n], "-o", markersize=3,
                linewidth=1.5, color=color, label=label, alpha=0.85)
        _annotate_min(ax, epochs[:n], values[:n], color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title, fontweight="bold")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    show_or_save(fig, save_path=save_path, show=show)
    return fig


# ===================================================================
#  Internal helpers
# ===================================================================


def _annotate_min(
    ax: Any,
    x: Sequence,
    y: Sequence,
    color: str = "black",
    label: str = "min",
) -> None:
    """Mark the minimum value with a star annotation."""
    if not y:
        return
    idx = int(np.argmin(y))
    ax.scatter([x[idx]], [y[idx]], s=100, marker="*",
               color=color, edgecolors="black", linewidth=0.5, zorder=5)
    ax.annotate(
        f"{y[idx]:.5f}",
        xy=(x[idx], y[idx]),
        textcoords="offset points",
        xytext=(5, 6),
        fontsize=7,
        color=color,
    )


# ===================================================================
#  GDA v13 — direct TFEvents loader + diagnostic plotter
# ===================================================================


def load_gda_tfevents(
    logdir: Union[str, Path],
    large_file_budget: int = 4000,
    small_file_threshold_mb: float = 10.0,
) -> Dict[str, Tuple[List[float], List[float]]]:
    """Load scalar metrics from a TFEvents logdir into (steps, values) pairs.

    Each event file is loaded **separately** so that small early files (which
    contain the initial high-loss phase) get proportionally more representation
    than the large later files.  Without this, global reservoir sampling across
    all files starves the early file of its ~8 slots out of 8 000, causing the
    plot to miss the loss=1.0 starting point entirely.

    Strategy
    --------
    - Files  < *small_file_threshold_mb* : read **all** events (no cap).
    - Files >= *small_file_threshold_mb* : cap at *large_file_budget* per tag.

    Parameters
    ----------
    large_file_budget : int
        Max scalar events per tag for large files (reservoir-sampled).
    small_file_threshold_mb : float
        Files below this size (MB) are read in full.
    """
    import glob
    import os as _os

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise ImportError(
            "tensorboard is required: pip install tensorboard"
        ) from exc

    files = sorted(glob.glob(_os.path.join(str(logdir), "events.out.tfevents.*")))
    result: Dict[str, Tuple[List[float], List[float]]] = {}

    for fpath in files:
        size_mb = _os.path.getsize(fpath) / (1024 ** 2)
        budget = 0 if size_mb < small_file_threshold_mb else large_file_budget
        ea = EventAccumulator(fpath, size_guidance={"scalars": budget})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            if tag not in result:
                result[tag] = ([], [])
            result[tag][0].extend(float(e.step) for e in events)
            result[tag][1].extend(float(e.value) for e in events)

    return result


def plot_gda_diagnostics(
    logdir: Union[str, Path],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[float, float] = (16, 18),
    cmap_name: str = CATEGORICAL_CMAP,
    title: str = "GDA — Training Diagnostics",
) -> "Figure":
    """Seven-panel diagnostic figure for any GDA run from TFEvents.

    Panels
    ------
    1. Training loss  (step-level, smoothed)
    2. Validation loss  (aggregated — one point per val run)
    3. guidance_delta  = E[‖Δε_own − Δε_null‖²]  (primary CFG signal)
    4. g_token_diversity  (variance of genomic token embeddings)
    5. g_vs_null_dist  (L2 distance between cond and null tokens)
    6. Learning-rate schedule  (backbone AdamW + adapter AdamW-1)
    7. contrastive_loss  (auxiliary genomic conditioning loss, EMA-smoothed)

    Parameters
    ----------
    logdir : str | Path
        Directory containing the TFEvents files, e.g.
        ``experiments/20260526_poc_brca_lihc_gda_v2/gda/``.
    save_path : str | Path | None
        If given, the figure is written to this path (PNG/PDF/SVG).
    show : bool
        Call ``plt.show()`` after rendering.
    """
    _check_matplotlib()
    setup_style()

    data = load_gda_tfevents(logdir)

    def _get(tag: str) -> Tuple[List[float], List[float]]:
        return data.get(tag, ([], []))

    train_steps, train_vals = _get("loss/train")
    # Prefer loss/val (sample-step scale, more points) over loss/val_ckpt
    # (optimizer-step scale, only 4 clean points from latest job).
    # _aggregate_by_step collapses duplicates from the old per-batch logging bug.
    val_steps_raw, val_vals_raw = _get("loss/val")
    if not val_vals_raw:
        val_steps_raw, val_vals_raw = _get("loss/val_ckpt")
    val_steps, val_vals = _aggregate_by_step(val_steps_raw, val_vals_raw)

    delta_steps_raw, delta_vals_raw = _get("cond/guidance_delta")
    delta_steps, delta_vals = _aggregate_by_step(delta_steps_raw, delta_vals_raw)

    div_steps_raw, div_vals_raw = _get("cond/g_token_diversity")
    div_steps, div_vals = _aggregate_by_step(div_steps_raw, div_vals_raw)

    dist_steps_raw, dist_vals_raw = _get("cond/g_vs_null_dist")
    dist_steps, dist_vals = _aggregate_by_step(dist_steps_raw, dist_vals_raw)

    lr0_steps, lr0_vals = _get("lr-AdamW")
    lr1_steps, lr1_vals = _get("lr-AdamW-1")

    contrastive_steps_raw, contrastive_vals_raw = _get("cond/contrastive_loss")

    colors = get_categorical_colors(8, cmap_name=cmap_name)
    seq_cmap = get_crameri_cmap(SEQUENTIAL_CMAP)

    fig, axes = plt.subplots(4, 2, figsize=figsize)
    axs = axes.flatten()

    # ------------------------------------------------------------------
    # Panel 1 — Combined train + val loss (epoch-binned, no scatter)
    # x-axis in millions of samples so it reads like a normal loss curve.
    # ------------------------------------------------------------------
    ax = axs[0]
    has_loss = False
    # Pre-compute EMA-smoothed train loss (sorts multi-file data by step)
    ex_M, ey = [], []
    if train_vals:
        raw_x_M = [s / 1e6 for s in train_steps]
        ex_M, ey = _ema_series(raw_x_M, list(train_vals), alpha=0.005)
        ax.plot(ex_M, ey, linewidth=2.2, color=colors[0], label="train loss (EMA)")
        has_loss = True
    # Pre-compute binned val loss
    exv_M, eyv = [], []
    best_idx = 0
    n_vbins = 50
    if val_vals:
        raw_xv_M = [s / 1e6 for s in val_steps]
        yv_arr = np.asarray(val_vals, dtype=float)
        n_vbins = min(50, max(6, len(yv_arr) // 3))
        exv_M, eyv = _bin_series(raw_xv_M, list(yv_arr), n_bins=n_vbins)
        if exv_M:
            ax.plot(exv_M, eyv, linewidth=2.2, color=colors[1], label="val loss")
            has_loss = True
        best_idx = int(np.argmin(yv_arr))
        ax.scatter([raw_xv_M[best_idx]], [yv_arr[best_idx]], s=130, marker="*",
                   color=colors[2], edgecolors="black", linewidth=0.6, zorder=5,
                   label=f"best val  {yv_arr[best_idx]:.4f}")
    if not has_loss:
        ax.text(0.5, 0.5, "No loss data", ha="center", va="center",
                transform=ax.transAxes, color="grey")
    ax.set_title("Training and Validation Loss", fontweight="bold")
    ax.set_xlabel("Samples Seen (millions)")
    ax.set_ylabel("MSE Loss")
    ax.legend(framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.22)

    # ------------------------------------------------------------------
    # Panel 2 — Same EMA/binned curves on log scale (fine convergence)
    # ------------------------------------------------------------------
    ax = axs[1]
    has_loss2 = False
    if ex_M:
        ax.plot(ex_M, ey, linewidth=2.2, color=colors[0], label="train loss (EMA)")
        has_loss2 = True
    if exv_M:
        ax.plot(exv_M, eyv, linewidth=2.2, color=colors[1], label="val loss")
        has_loss2 = True
    if val_vals:
        ax.scatter([raw_xv_M[best_idx]], [yv_arr[best_idx]], s=130, marker="*",
                   color=colors[2], edgecolors="black", linewidth=0.6, zorder=5,
                   label=f"best val  {yv_arr[best_idx]:.4f}")
    if has_loss2:
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "No loss data", ha="center", va="center",
                transform=ax.transAxes, color="grey")
    ax.set_title("Training and Validation Loss (log scale)", fontweight="bold")
    ax.set_xlabel("Samples Seen (millions)")
    ax.set_ylabel("MSE Loss (log)")
    ax.legend(framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.22, which="both")

    # ------------------------------------------------------------------
    # Panel 3 — guidance_delta (primary CFG health metric)
    # ------------------------------------------------------------------
    ax = axs[2]
    if delta_vals:
        ax.plot(delta_steps, delta_vals, "-o", markersize=4, linewidth=1.4,
                color=seq_cmap(0.75), label="guidance_delta")
        ax.axhline(0.0, color="grey", linewidth=0.8, alpha=0.4, linestyle="--")
        positive = [v for v in delta_vals if v > 0]
        if positive and max(positive) / max(min(positive), 1e-12) > 20:
            ax.set_yscale("log")
        ax.legend(framealpha=0.9, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No cond/guidance_delta data yet", ha="center", va="center",
                transform=ax.transAxes, color="grey")
    ax.set_title("Guidance Delta  E[‖Δε_own − Δε_null‖²]", fontweight="bold")
    ax.set_xlabel("Global Step (≈ samples seen)")
    ax.set_ylabel("guidance_delta")
    ax.grid(True, alpha=0.22)

    # ------------------------------------------------------------------
    # Panel 4 — g_token_diversity
    # ------------------------------------------------------------------
    ax = axs[3]
    if div_vals:
        ax.plot(div_steps, div_vals, "-o", markersize=4, linewidth=1.4,
                color=colors[3], label="g_token_diversity")
        ax.legend(framealpha=0.9, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No cond/g_token_diversity data yet", ha="center", va="center",
                transform=ax.transAxes, color="grey")
    ax.set_title("Genomic Token Diversity", fontweight="bold")
    ax.set_xlabel("Global Step (≈ samples seen)")
    ax.set_ylabel("g_token_diversity")
    ax.grid(True, alpha=0.22)

    # ------------------------------------------------------------------
    # Panel 5 — g_vs_null_dist
    # ------------------------------------------------------------------
    ax = axs[4]
    if dist_vals:
        ax.plot(dist_steps, dist_vals, "-o", markersize=4, linewidth=1.4,
                color=colors[4], label="g_vs_null_dist")
        ax.axhline(0.0, color="grey", linewidth=0.8, alpha=0.4, linestyle="--")
        ax.legend(framealpha=0.9, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No cond/g_vs_null_dist data yet", ha="center", va="center",
                transform=ax.transAxes, color="grey")
    ax.set_title("Genomic vs Null Token Distance", fontweight="bold")
    ax.set_xlabel("Global Step (≈ samples seen)")
    ax.set_ylabel("g_vs_null_dist (L2)")
    ax.grid(True, alpha=0.22)

    # ------------------------------------------------------------------
    # Panel 6 — Learning rate (both optimisers)
    # ------------------------------------------------------------------
    ax = axs[5]
    has_lr = False
    if lr0_vals:
        ax.plot(lr0_steps, lr0_vals, linewidth=1.4, color=colors[0],
                label="backbone (AdamW, 1e-4)")
        has_lr = True
    if lr1_vals:
        ax.plot(lr1_steps, lr1_vals, linewidth=1.4, color=colors[1],
                label="adapter+enc (AdamW-1, 3e-4)")
        has_lr = True
    if has_lr:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
        ax.legend(framealpha=0.9, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No LR data", ha="center", va="center",
                transform=ax.transAxes, color="grey")
    ax.set_title("Learning Rate Schedule", fontweight="bold")
    ax.set_xlabel("Optimizer Step (current job)")
    ax.set_ylabel("Learning Rate")
    ax.grid(True, alpha=0.22)

    # ------------------------------------------------------------------
    # Panel 7 — Contrastive loss  (dense per-step → EMA smoothed)
    # Only available from the job that introduced the auxiliary loss.
    # x-axis matches panels 1-2 (millions of samples / steps).
    # ------------------------------------------------------------------
    ax = axs[6]
    if contrastive_vals_raw:
        ct_x_M = [s / 1e6 for s in contrastive_steps_raw]
        # Bin-average avoids EMA initialization artifacts (first sampled value may be a
        # near-zero outlier that makes EMA appear to rise even when the true mean falls).
        bct_M, bct_y = _bin_series(ct_x_M, list(contrastive_vals_raw), n_bins=80)
        ax.plot(bct_M, bct_y, linewidth=2.2, color=colors[5],
                label="contrastive loss (bin avg)")
        if bct_y:
            ax.axhline(bct_y[0], color=colors[5], linewidth=0.7, alpha=0.35, linestyle="--")
            ax.annotate(f"start {bct_y[0]:.4f}", xy=(bct_M[0], bct_y[0]),
                        xytext=(6, 4), textcoords="offset points", fontsize=7, color=colors[5])
            ax.annotate(f"now {bct_y[-1]:.4f}", xy=(bct_M[-1], bct_y[-1]),
                        xytext=(-60, -12), textcoords="offset points", fontsize=7, color=colors[5])
        ax.legend(framealpha=0.9, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No cond/contrastive_loss data yet\n(added in later job restarts)",
                ha="center", va="center", transform=ax.transAxes, color="grey")
    ax.set_title("Contrastive Loss — Genomic Conditioning", fontweight="bold")
    ax.set_xlabel("Samples Seen (millions)")
    ax.set_ylabel("Contrastive Loss")
    ax.grid(True, alpha=0.22)

    # Hide unused 8th panel
    axs[7].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    show_or_save(fig, save_path=save_path, show=show)
    return fig


# Backward-compat alias
plot_gda_v13_diagnostics = plot_gda_diagnostics


# ===================================================================
#  CFG backbone — combined warm-start + main run diagnostics
# ===================================================================


def plot_cfg_diagnostics(
    main_logdir: Union[str, Path],
    warm_logdir: Optional[Union[str, Path]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[float, float] = (16, 14),
    cmap_name: str = CATEGORICAL_CMAP,
    title: str = "CFG Backbone — Training Diagnostics",
) -> "Figure":
    """Four-panel diagnostic figure for a CFG backbone run, optionally stitched
    to a warm-start predecessor run on a continuous x-axis (samples seen).

    If *warm_logdir* is given, its events are placed first on the x-axis and the
    main run's steps are offset by the warm run's final step so the combined curve
    reads as a single uninterrupted training history.  A vertical dashed line marks
    the stitch point.

    Panels
    ------
    1. loss/train (EMA-smoothed) + loss/val (aggregated) — linear scale
    2. loss/val (correct) vs loss/val_shuffled — same scale, shows conditioning gap
    3. cond/gap  = loss(cross-cohort shuffle) − loss(correct)  — must be > 0
    4. cond/signal = E[‖ε_cond − ε_null‖²]  — must grow monotonically

    Parameters
    ----------
    main_logdir : str | Path
        TFEvents directory for the primary / most recent run.
    warm_logdir : str | Path | None
        TFEvents directory for the warm-start predecessor.  ``None`` = single run.
    save_path : str | Path | None
    show : bool
    """
    _check_matplotlib()
    setup_style()

    def _load(logdir: Union[str, Path]) -> Dict[str, Tuple[List[float], List[float]]]:
        return load_gda_tfevents(logdir)

    def _get(data: Dict, tag: str) -> Tuple[List[float], List[float]]:
        return data.get(tag, ([], []))

    main_data = _load(main_logdir)

    stitch_step: float = 0.0
    warm_data: Dict[str, Tuple[List[float], List[float]]] = {}
    if warm_logdir is not None:
        warm_data = _load(warm_logdir)
        # Determine the last step recorded in the warm run (use loss/train as reference).
        w_steps, _ = _get(warm_data, "loss/train")
        stitch_step = float(max(w_steps)) if w_steps else 0.0

    def _combine(tag: str) -> Tuple[List[float], List[float]]:
        """Return (steps_M, values) combining warm + main, x in millions of samples."""
        ws, wv = _get(warm_data, tag) if warm_data else ([], [])
        ms, mv = _get(main_data, tag)
        # Offset main steps so they continue after warm
        ms_off = [s + stitch_step for s in ms]
        all_s = list(ws) + ms_off
        all_v = list(wv) + list(mv)
        pairs = sorted(zip(all_s, all_v))
        if not pairs:
            return [], []
        s_sorted = [p[0] / 1e6 for p in pairs]
        v_sorted = [p[1] for p in pairs]
        return s_sorted, v_sorted

    def _combine_agg(tag: str) -> Tuple[List[float], List[float]]:
        """Combine and de-duplicate (aggregate) sparse val series."""
        s, v = _combine(tag)
        if not s:
            return [], []
        # _aggregate_by_step expects int-keyed steps; round M-scale back to steps
        raw_steps = [round(x * 1e6) for x in s]
        agg_s, agg_v = _aggregate_by_step(raw_steps, v)
        return [x / 1e6 for x in agg_s], agg_v

    stitch_M = (stitch_step / 1e6) if stitch_step > 0 else None

    # --- load all series ------------------------------------------------
    train_s, train_v         = _combine("loss/train")
    val_s, val_v             = _combine_agg("loss/val")
    shuffled_s, shuffled_v   = _combine_agg("loss/val_shuffled")
    gap_s, gap_v             = _combine_agg("cond/gap")
    signal_s, signal_v       = _combine("cond/signal")

    # EMA-smooth train
    ema_s, ema_v = _ema_series(train_s, train_v, alpha=0.005) if train_v else ([], [])

    colors = get_categorical_colors(8, cmap_name=cmap_name)
    seq_cmap = get_crameri_cmap(SEQUENTIAL_CMAP)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axs = axes.flatten()

    def _stitch_line(ax: Any) -> None:
        if stitch_M is not None:
            ax.axvline(stitch_M, color="grey", linewidth=1.2, linestyle="--", alpha=0.6,
                       label=f"warm→main ({stitch_M:.1f}M)")

    # ------------------------------------------------------------------
    # Panel 1 — Training loss + validation loss (linear)
    # ------------------------------------------------------------------
    ax = axs[0]
    if ema_s:
        ax.plot(ema_s, ema_v, linewidth=2.2, color=colors[0], label="train loss (EMA)")
    if val_v:
        ax.plot(val_s, val_v, "-o", markersize=5, linewidth=1.8,
                color=colors[1], label="val loss")
        best_idx = int(np.argmin(val_v))
        ax.scatter([val_s[best_idx]], [val_v[best_idx]], s=130, marker="*",
                   color=colors[2], edgecolors="black", linewidth=0.6, zorder=5,
                   label=f"best val  {val_v[best_idx]:.4f}")
    _stitch_line(ax)
    ax.set_title("Training & Validation Loss", fontweight="bold")
    ax.set_xlabel("Samples Seen (millions)")
    ax.set_ylabel("MSE Loss")
    ax.legend(framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.22)

    # ------------------------------------------------------------------
    # Panel 2 — Val correct vs val shuffled
    # ------------------------------------------------------------------
    ax = axs[1]
    if val_v:
        ax.plot(val_s, val_v, "-o", markersize=5, linewidth=1.8,
                color=colors[1], label="val loss (correct g)")
    if shuffled_v:
        ax.plot(shuffled_s, shuffled_v, "-s", markersize=5, linewidth=1.8,
                color=colors[3], label="val loss (shuffled g)")
    _stitch_line(ax)
    ax.set_title("Val Loss: Correct vs Cross-Cohort Shuffled Conditioning", fontweight="bold")
    ax.set_xlabel("Samples Seen (millions)")
    ax.set_ylabel("MSE Loss")
    ax.legend(framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.22)

    # ------------------------------------------------------------------
    # Panel 3 — cond/gap
    # ------------------------------------------------------------------
    ax = axs[2]
    if gap_v:
        ax.plot(gap_s, gap_v, "-o", markersize=5, linewidth=1.8,
                color=seq_cmap(0.8), label="cond/gap")
        ax.fill_between(gap_s, 0, gap_v,
                        where=[v >= 0 for v in gap_v],
                        alpha=0.12, color=seq_cmap(0.8))
    ax.axhline(0.0, color="grey", linewidth=1.0, alpha=0.5, linestyle="--")
    _stitch_line(ax)
    ax.set_title("cond/gap  =  loss(shuffled) − loss(correct)", fontweight="bold")
    ax.set_xlabel("Samples Seen (millions)")
    ax.set_ylabel("cond/gap  [> 0 = conditioning works]")
    ax.legend(framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.22)
    if not gap_v:
        ax.text(0.5, 0.5, "No cond/gap data", ha="center", va="center",
                transform=ax.transAxes, color="grey")

    # ------------------------------------------------------------------
    # Panel 4 — cond/signal
    # ------------------------------------------------------------------
    ax = axs[3]
    if signal_v:
        ax.plot(signal_s, signal_v, "-o", markersize=4, linewidth=1.6,
                color=colors[4], label="cond/signal")
        ax.axhline(0.0, color="grey", linewidth=0.8, alpha=0.4, linestyle="--")
        positive = [v for v in signal_v if v > 0]
        if positive and max(positive) / max(min(positive), 1e-12) > 20:
            ax.set_yscale("log")
    _stitch_line(ax)
    ax.set_title("cond/signal  =  E[‖ε_cond − ε_null‖²]", fontweight="bold")
    ax.set_xlabel("Samples Seen (millions)")
    ax.set_ylabel("cond/signal  [must grow ↑]")
    ax.legend(framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.22)
    if not signal_v:
        ax.text(0.5, 0.5, "No cond/signal data", ha="center", va="center",
                transform=ax.transAxes, color="grey")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    show_or_save(fig, save_path=save_path, show=show)
    return fig


# ===================================================================
#  PoC conditioning-mode ablation — four runs overlaid
# ===================================================================


def plot_poc_ablation(
    logdirs: Dict[str, Union[str, Path]],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[float, float] = (14, 14),
    cmap_name: str = CATEGORICAL_CMAP,
    title: str = "Investigation of Conditioning",
) -> "Figure":
    """4 × 3 grid comparing the four PoC conditioning modes.

    Rows (one per conditioning mode) × columns:
      Col 0 — Train loss (EMA-smoothed, solid) + Validation loss (dashed)
      Col 1 — Conditioning Signal  E[‖ε_cond − ε_null‖²]  (must grow)
      Col 2 — Conditioning Gap  L(shuffled) − L(correct)  (must be > 0)

    Axes within each column share their x-axis scale.  Each row is labelled
    with its conditioning mode name on the left.

    Parameters
    ----------
    logdirs : dict[label → path]
        Ordered mapping of human-readable run label to the TFEvents directory,
        e.g. ``{"Zero": "…/zero/gda", "1-hot": "…/orthogonal/gda", …}``.
    """
    _check_matplotlib()
    setup_style()

    n_runs = len(logdirs)
    colors = get_categorical_colors(max(n_runs, 4), cmap_name=cmap_name)

    # Load all runs up front
    all_data: List[Tuple[str, Dict]] = []
    for label, logdir in logdirs.items():
        data = load_gda_tfevents(logdir)
        all_data.append((label, data))

    # Determine column-level y-range helpers for shared axes
    # (sharex per column; sharey is intentionally NOT shared — each row may
    # differ in scale for signal/gap, but loss rows benefit from a common y)
    fig, axes = plt.subplots(
        n_runs, 3,
        figsize=figsize,
        sharex="col",
        sharey="col",
    )
    # Ensure axes is always 2-D
    if n_runs == 1:
        axes = axes[None, :]

    col_titles = [
        "Train & Validation Loss",
        "Conditioning Signal\n"
        r"$\mathbb{E}[\|\hat{\varepsilon}_\mathrm{cond} - \hat{\varepsilon}_\mathrm{null}\|^2]$",
        "Conditioning Gap\n"
        r"$\mathcal{L}(\hat{\varepsilon}_\mathrm{shuffled}) - \mathcal{L}(\hat{\varepsilon}_\mathrm{cond})$",
    ]

    # Collect signal values across runs to decide log-scale
    all_sig_vals = [
        v for _, d in all_data
        for v in d.get("cond/signal", ([], []))[1]
        if v > 0
    ]
    signal_log = bool(all_sig_vals and max(all_sig_vals) / max(min(all_sig_vals), 1e-12) > 20)

    for row, (label, data) in enumerate(all_data):
        color = colors[row]
        ax_loss, ax_signal, ax_gap = axes[row]

        # ── Col 0: train + val loss ──────────────────────────────────────
        train_steps, train_vals = data.get("loss/train", ([], []))
        val_steps_raw, val_vals_raw = data.get("loss/val", ([], []))
        if not val_vals_raw:
            val_steps_raw, val_vals_raw = data.get("loss/val_ckpt", ([], []))
        val_steps, val_vals = _aggregate_by_step(val_steps_raw, val_vals_raw)

        if train_vals:
            s_M, v = _ema_series(
                [s / 1e6 for s in train_steps], list(train_vals), alpha=0.005
            )
            ax_loss.plot(s_M, v, linewidth=2.0, color=color, label="Train (smoothed)")
        if val_vals:
            ax_loss.plot(
                [s / 1e6 for s in val_steps], val_vals,
                linewidth=1.8, linestyle="--", color=color,
                label="Validation", alpha=0.85,
            )
        ax_loss.legend(framealpha=0.9, fontsize=8)
        ax_loss.set_ylabel("MSE Loss")
        ax_loss.grid(True, alpha=0.22)

        # ── Col 1: conditioning signal ───────────────────────────────────
        sig_steps, sig_vals = data.get("cond/signal", ([], []))
        if sig_vals:
            ax_signal.plot(
                [s / 1e6 for s in sig_steps], sig_vals,
                "-o", markersize=3, linewidth=1.8, color=color,
            )
        ax_signal.axhline(0.0, color="grey", linewidth=0.8, alpha=0.4, linestyle="--")
        ax_signal.set_ylabel("Conditioning Signal")
        if signal_log:
            ax_signal.set_yscale("log")
        ax_signal.grid(True, alpha=0.22)

        # ── Col 2: conditioning gap ──────────────────────────────────────
        gap_steps_raw, gap_vals_raw = data.get("cond/gap", ([], []))
        gap_steps, gap_vals = _aggregate_by_step(gap_steps_raw, gap_vals_raw)
        if gap_vals:
            ax_gap.plot(
                [s / 1e6 for s in gap_steps], gap_vals,
                "-o", markersize=3, linewidth=1.8, color=color,
            )
            ax_gap.fill_between(
                [s / 1e6 for s in gap_steps], 0, gap_vals,
                where=[v >= 0 for v in gap_vals],
                alpha=0.10, color=color,
            )
        ax_gap.axhline(0.0, color="grey", linewidth=1.0, alpha=0.5, linestyle="--")
        ax_gap.set_ylabel("Conditioning Gap")
        ax_gap.grid(True, alpha=0.22)

        # Row label on the left side of the loss panel
        ax_loss.set_title(label, loc="left", fontsize=10, fontweight="bold", color=color)

    # Column titles on top row only
    for col, col_title in enumerate(col_titles):
        axes[0, col].set_title(col_title, fontsize=10, fontweight="bold")

    # x-axis labels only on the bottom row
    for col in range(3):
        axes[-1, col].set_xlabel("Samples Seen (millions)")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    show_or_save(fig, save_path=save_path, show=show)
    return fig


# ===================================================================
#  Paper-style per-run figures  (two standalone PNGs)
# ===================================================================

_PAPER_RC: Dict[str, Any] = {
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "0.75",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def _paper_style() -> None:
    import matplotlib.pyplot as _plt
    _plt.rcParams.update(_PAPER_RC)


def _strip_spines(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.45, color="0.65")
    ax.set_axisbelow(True)


def _add_panel_label(ax: Any, label: str) -> None:
    ax.text(
        0.03, 0.97, label,
        transform=ax.transAxes,
        fontsize=10, fontweight="bold",
        va="top", ha="left",
    )


def _batlow_colors(positions: List[float]) -> List[str]:
    try:
        from cmcrameri import cm as cmc
        cmap = cmc.batlow
    except (ImportError, AttributeError):
        import matplotlib as _mpl
        cmap = _mpl.colormaps["viridis"]
    import matplotlib as _mpl
    return [_mpl.colors.to_hex(cmap(p)) for p in positions]


def _subsample_series(
    steps: List[float],
    values: List[float],
    n: int,
) -> Tuple[List[float], List[float]]:
    """Uniform subsample to at most *n* points, preserving first and last."""
    if len(steps) <= n:
        return steps, values
    idx = np.round(np.linspace(0, len(steps) - 1, n)).astype(int)
    return [steps[i] for i in idx], [values[i] for i in idx]


def plot_paper_losses(
    run_dir: Union[str, Path],
    out_path: Union[str, Path],
    ylim_top: Optional[float] = None,
) -> None:
    """Single-panel paper figure: train loss (noisy + EMA) + val + val shuffled.

    The raw training loss is shown as a thin, semi-transparent trace so the
    step-to-step noise is visible.  A heavily EMA-smoothed overlay provides the
    readable trend line.  No panel title — only axis labels and panel label (a).

    Args:
        ylim_top: If given, fix the y-axis upper limit to this value (useful for
            comparing plots across runs on a shared scale).
    """
    import matplotlib.pyplot as _plt
    import matplotlib.ticker as _ticker

    _paper_style()

    data = load_gda_tfevents(run_dir)

    tr_steps, tr_vals   = data.get("loss/train", ([], []))
    va_steps_raw, va_vals_raw = data.get("loss/val", ([], []))
    vs_steps_raw, vs_vals_raw = data.get("loss/val_shuffled", ([], []))
    va_steps, va_vals = _aggregate_by_step(va_steps_raw, va_vals_raw)
    vs_steps, vs_vals = _aggregate_by_step(vs_steps_raw, vs_vals_raw)

    if not tr_vals:
        raise RuntimeError(f"loss/train not found in {run_dir}")

    # Raw: subsample to ~2 500 pts → noise is visible but not overwhelming
    tr_s_raw, tr_v_raw = _subsample_series(
        [s / 1e6 for s in tr_steps], list(tr_vals), 2_500
    )
    # Smooth: EMA with α=0.005 (slow-tracking → very smooth trend)
    tr_s_ema, tr_v_ema = _ema_series(
        [s / 1e6 for s in tr_steps], list(tr_vals), alpha=0.005
    )

    c_train, c_val, c_shuf = _batlow_colors([0.15, 0.55, 0.80])

    fig, ax = _plt.subplots(figsize=(5.2, 3.4))

    # Noisy raw trace
    ax.plot(tr_s_raw, tr_v_raw, lw=0.4, alpha=0.20, color=c_train, rasterized=True)
    # Smooth overlay
    ax.plot(tr_s_ema, tr_v_ema, lw=1.8, alpha=0.95, color=c_train, label="Train loss")

    if va_vals:
        ax.plot(
            [s / 1e6 for s in va_steps], va_vals,
            lw=1.6, color=c_val, label="Val loss",
        )

    ax.set_xlabel("Samples (×10⁶)")
    ax.set_ylabel("MSE loss")
    ax.legend(loc="upper right")
    _strip_spines(ax)
    _add_panel_label(ax, "(a)")

    # Focus on converged region (skip the opening loss spike)
    if ylim_top is not None:
        ax.set_ylim(bottom=0, top=ylim_top)
    elif tr_v_ema:
        post_spike = [v for s, v in zip(tr_s_ema, tr_v_ema) if s > 0.5]
        if post_spike:
            ax.set_ylim(bottom=0, top=min(max(post_spike) * 2.5, 0.6))

    fig.tight_layout()
    fig.savefig(out_path)
    _plt.close(fig)
    print(f"Saved: {out_path}")


def plot_paper_conditioning(
    run_dir: Union[str, Path],
    out_path: Union[str, Path],
    ylims: Optional[Dict[str, float]] = None,
) -> None:
    """Three-panel paper figure: conditioning signal, gap, and BRCA–LIHC separation.

    Each panel shows the raw noisy trace (thin, low alpha) plus a smoothed overlay.
    No panel titles — y-axis labels carry the metric names.

    Args:
        ylims: Optional dict with keys "signal", "gap", "sep" setting the y-axis
            upper limit for each panel (useful for shared axes across runs).
    """
    import matplotlib.pyplot as _plt
    import matplotlib.ticker as _ticker

    _paper_style()

    if ylims is None:
        ylims = {}

    data = load_gda_tfevents(run_dir)

    sig_steps, sig_vals   = data.get("cond/signal", ([], []))
    gap_steps_r, gap_v_r  = data.get("cond/gap", ([], []))
    sep_steps, sep_vals   = data.get("cond/brca_lihc_sep", ([], []))
    gap_steps, gap_vals   = _aggregate_by_step(gap_steps_r, gap_v_r)

    c_sig, c_gap, c_sep = _batlow_colors([0.20, 0.55, 0.78])

    fig, axes = _plt.subplots(1, 3, figsize=(10.5, 3.2))

    panels = [
        (axes[0], sig_steps, sig_vals, c_sig, "Conditioning signal", "signal"),
        (
            axes[1],
            [float(s) for s in gap_steps],
            gap_vals,
            c_gap,
            "Conditioning gap",
            "gap",
        ),
        (axes[2], sep_steps, sep_vals, c_sep, "BRCA–LIHC separation", "sep"),
    ]

    for ax, steps, vals, color, ylabel, ylim_key in panels:
        if not vals:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="0.5", fontsize=9)
            ax.set_xlabel("Samples (×10⁶)")
            ax.set_ylabel(ylabel)
            _strip_spines(ax)
            ax.yaxis.set_major_formatter(
                _ticker.ScalarFormatter(useMathText=True)
            )
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            continue

        s_M = [s / 1e6 for s in steps]

        # Raw noisy trace (subsampled to ~1 500 pts)
        s_raw, v_raw = _subsample_series(s_M, list(vals), 1_500)
        ax.plot(s_raw, v_raw, lw=0.4, alpha=0.20, color=color, rasterized=True)

        # EMA smooth overlay
        s_ema, v_ema = _ema_series(s_M, list(vals), alpha=0.05)
        ax.plot(s_ema, v_ema, lw=1.8, alpha=0.95, color=color)

        ax.set_xlabel("Samples (×10⁶)")
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(
            _ticker.ScalarFormatter(useMathText=True)
        )
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        top = ylims.get(ylim_key)
        # Only apply the shared ylim when the series has meaningful values on
        # that scale (> 1 % of top). Otherwise let matplotlib auto-scale so
        # near-zero runs (e.g. zeros conditioning) still show their flat line.
        data_max = max(abs(v) for v in vals) if vals else 0.0
        if top is not None and data_max >= 0.01 * top:
            ax.set_ylim(bottom=0, top=top)
        else:
            ax.set_ylim(bottom=0)
        _strip_spines(ax)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(out_path)
    _plt.close(fig)
    print(f"Saved: {out_path}")


def compute_shared_ylims(run_dirs: List[Union[str, Path]]) -> Dict[str, float]:
    """Compute y-axis upper limits that span all supplied run directories.

    Returns a dict with keys:
        "loss_top"   — for plot_paper_losses  (ylim_top)
        "signal"     — for plot_paper_conditioning  ylims["signal"]
        "gap"        — for plot_paper_conditioning  ylims["gap"]
        "sep"        — for plot_paper_conditioning  ylims["sep"]
    """
    margin = 1.2

    loss_post_max = 0.0
    sig_max = 0.0
    gap_max = 0.0
    sep_max = 0.0

    for rd in run_dirs:
        data = load_gda_tfevents(rd)

        # Val loss — only look at the converged region (> 0.5 M samples)
        va_s_r, va_v_r = data.get("loss/val", ([], []))
        va_steps, va_vals = _aggregate_by_step(va_s_r, va_v_r)
        post = [v for s, v in zip(va_steps, va_vals) if s > 5e5]
        if post:
            loss_post_max = max(loss_post_max, max(post))

        sig_s, sig_v = data.get("cond/signal", ([], []))
        if sig_v:
            sig_max = max(sig_max, max(sig_v))

        gap_s_r, gap_v_r = data.get("cond/gap", ([], []))
        _, gap_v = _aggregate_by_step(gap_s_r, gap_v_r)
        if gap_v:
            gap_max = max(gap_max, max(gap_v))

        sep_s, sep_v = data.get("cond/brca_lihc_sep", ([], []))
        if sep_v:
            sep_max = max(sep_max, max(sep_v))

    return {
        "loss_top": loss_post_max * 3.0 if loss_post_max > 0 else None,
        "signal":   sig_max * margin if sig_max > 0 else None,
        "gap":      gap_max * margin if gap_max > 0 else None,
        "sep":      sep_max * margin if sep_max > 0 else None,
    }


# ===================================================================
#  CLI entry-point
# ===================================================================

if __name__ == "__main__":
    import argparse
    import sys as _sys
    # Allow `python src/visualization/training_plots.py` from any cwd
    _repo_root = str(Path(__file__).resolve().parents[2])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)
    # Re-import with absolute paths so the relative imports above resolve
    from src.visualization.training_plots import (  # noqa: E402
        plot_cfg_diagnostics,
        plot_gda_diagnostics,
        plot_poc_ablation,
    )

    _BASE = "/mnt/bulk-saturn/maralampert/genhist/experiments"
    _CFG_MAIN = f"{_BASE}/20260601_poc_brca_lihc_cfg_v2_dgx/gda"
    _CFG_WARM = f"{_BASE}/20260529_poc_brca_lihc_cfg_v1/gda"
    _CFG_OUT  = f"{_BASE}/20260601_poc_brca_lihc_cfg_v2_dgx/cfg_diagnostics.png"
    _GDA_LOGDIR = f"{_BASE}/20260526_poc_brca_lihc_gda_v2/gda"
    _GDA_OUT    = f"{_BASE}/20260526_poc_brca_lihc_gda_v2/gda_diagnostics.png"
    _POC_OUT    = f"{_BASE}/poc_ablation_conditioning.png"
    _POC_LOGDIRS: "dict[str, str]" = {
        "Zero":    f"{_BASE}/20260603_poc_128_zero/gda",
        "Noise":   f"{_BASE}/20260603_poc_128_noise/gda",
        "1-hot":   f"{_BASE}/20260603_poc_128_orthogonal/gda",
        "Genomic": f"{_BASE}/20260603_poc_128_rna/gda",
    }

    parser = argparse.ArgumentParser(
        description="Plot training diagnostics from TFEvents."
    )
    parser.add_argument(
        "--mode", choices=["cfg", "gda", "poc_ablation", "per_run"], default="cfg",
        help=(
            "'cfg' = CFG backbone 4-panel; 'gda' = GDA v13 7-panel; "
            "'poc_ablation' = 4-mode PoC grid; "
            "'per_run' = two paper-quality PNGs for one run directory."
        ),
    )
    # per_run mode
    parser.add_argument(
        "--run-dir", default=None,
        help="[per_run] Single run directory. Use --run-dirs for multiple.",
    )
    parser.add_argument(
        "--run-dirs", nargs="+", default=None,
        help=(
            "[per_run] One or more run directories. When more than one is given "
            "the y-axis limits are shared across all plots so they are comparable."
        ),
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="[per_run] Output directory (default: same as each run dir).",
    )
    # CFG-mode args
    parser.add_argument(
        "--main-logdir", default=_CFG_MAIN,
        help="[cfg] Main run TFEvents directory.",
    )
    parser.add_argument(
        "--warm-logdir", default=_CFG_WARM,
        help="[cfg] Warm-start predecessor TFEvents directory (optional).",
    )
    # GDA-mode arg
    parser.add_argument(
        "--logdir", default=_GDA_LOGDIR,
        help="[gda] Directory containing TFEvents files.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output figure path (PNG/PDF/SVG). Defaults per mode.",
    )
    parser.add_argument(
        "--title", default=None,
        help="Figure title override.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not call plt.show() — just save to --out.",
    )
    args = parser.parse_args()

    kwargs: dict = {}
    if args.title:
        kwargs["title"] = args.title

    if args.mode == "per_run":
        # Collect the list of run dirs: --run-dirs wins over --run-dir
        if args.run_dirs:
            run_dirs = [Path(d).resolve() for d in args.run_dirs]
        elif args.run_dir:
            run_dirs = [Path(args.run_dir).resolve()]
        else:
            parser.error("--run-dir or --run-dirs is required for per_run mode")

        # Compute shared y-axis limits when comparing multiple runs
        shared: Dict[str, Any] = {}
        if len(run_dirs) > 1:
            print(f"Computing shared y-limits across {len(run_dirs)} run(s)...")
            shared = compute_shared_ylims(run_dirs)
            fmt = lambda v: f"{v:.4g}" if v is not None else "None"
            print(f"  loss_top={fmt(shared['loss_top'])}  "
                  f"signal={fmt(shared['signal'])}  "
                  f"gap={fmt(shared['gap'])}  "
                  f"sep={fmt(shared['sep'])}")

        shared_out_dir = Path(args.out_dir).resolve() if args.out_dir else None
        use_prefix = shared_out_dir is not None and len(run_dirs) > 1

        for run_dir in run_dirs:
            out_dir = shared_out_dir if shared_out_dir else run_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            # Use the experiment directory name (parent of /gda) for uniqueness
            prefix = f"{run_dir.parent.name}_" if use_prefix else ""
            print(f"per_run | run_dir={run_dir} | out_dir={out_dir} | prefix={prefix!r}")
            plot_paper_losses(
                run_dir,
                out_dir / f"{prefix}training_losses.png",
                ylim_top=shared.get("loss_top"),
            )
            plot_paper_conditioning(
                run_dir,
                out_dir / f"{prefix}conditioning_metrics.png",
                ylims={k: shared[k] for k in ("signal", "gap", "sep") if shared.get(k)},
            )
        _sys.exit(0)

    if args.mode == "cfg":
        out_path = Path(args.out or _CFG_OUT)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        warm = args.warm_logdir if args.warm_logdir else None
        print(f"CFG mode | warm={warm} | main={args.main_logdir}")
        fig = plot_cfg_diagnostics(
            main_logdir=args.main_logdir,
            warm_logdir=warm,
            save_path=out_path,
            show=not args.no_show,
            **kwargs,
        )
    elif args.mode == "gda":
        out_path = Path(args.out or _GDA_OUT)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"GDA mode | logdir={args.logdir}")
        fig = plot_gda_diagnostics(
            logdir=args.logdir,
            save_path=out_path,
            show=not args.no_show,
            **kwargs,
        )
    else:  # poc_ablation
        out_path = Path(args.out or _POC_OUT)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"PoC ablation mode | runs: {list(_POC_LOGDIRS.keys())}")
        fig = plot_poc_ablation(
            logdirs=_POC_LOGDIRS,
            save_path=out_path,
            show=not args.no_show,
            **kwargs,
        )

    print(f"Saved: {out_path}")
