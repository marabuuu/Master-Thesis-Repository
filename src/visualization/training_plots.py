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

.. _Fabio Crameri: https://doi.org/10.5281/zenodo.1243862
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np

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
