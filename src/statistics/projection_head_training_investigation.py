#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Projection Head Training Investigation
=======================================

Comprehensive visualization and analysis of projection-head training runs
(distribution-matching mode).  Works with **three** data sources that are
produced by ``projection_head_genomic.py``:

1. **Slurm / stdout log** (``*.out``)
   – epoch-level total, mean, var, diversity losses
   – learning-rate schedule, per-epoch wall-clock time, GPU memory

2. **Slurm / stderr log** (``*.err``)
   – per-batch loss values from tqdm progress bars

3. **Saved checkpoints** (``*.pt``)
   – model weights  → weight-norm evolution, weight histograms
   – ``target_mean`` / ``target_std`` in the best checkpoint
   – project real genomic H5s through each checkpoint
     → projected-feature statistics, distribution overlap,
       pairwise-cosine-similarity heat-maps, UMAP embeddings

Figures are written to ``<out-dir>/figures/`` by default.

Usage
-----
# Minimal (log only)
python projection_head_training_investigation.py \\
    --log-file slurm/projection_head_train-30506.out

# Full analysis including checkpoint + genomic feature inspection
python projection_head_training_investigation.py \\
    --log-file  slurm/projection_head_train-30506.out \\
    --err-file  slurm/projection_head_train-30506.err \\
    --ckpt-dir  experiments/20260218_projection_head \\
    --genomic-h5-dir experiments/.../mopadi_features/train \\
    --out-dir   experiments/20260218_projection_head/investigation

# Show plots interactively (default: save only)
python projection_head_training_investigation.py --log-file ... --show
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

# Use non-interactive backend when not showing plots
matplotlib.use("Agg")

# Optional heavy imports guarded at runtime
try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


# ======================================================================
#  1.  LOG-FILE PARSING
# ======================================================================

def parse_stdout_log(log_path: Path) -> Dict[str, Any]:
    """
    Parse the ``*.out`` log produced by distribution-matching training.

    Returns a dict with keys:
        epochs, total, mean, var, diversity   – lists of floats
        lr                                    – list of floats
        epoch_time                            – list of floats (seconds)
        gpu_alloc, gpu_reserved               – lists of floats (GB)
        best_epochs                           – list of ints (epochs marked ★)
        meta                                  – dict of scalar metadata
    """
    text = Path(log_path).read_text()

    data: Dict[str, Any] = {
        "epochs": [],
        "total": [],
        "mean": [],
        "var": [],
        "diversity": [],
        "lr": [],
        "epoch_time": [],
        "gpu_alloc": [],
        "gpu_reserved": [],
        "best_epochs": [],
        "meta": {},
    }

    # Epoch header:  "Epoch 5/50 | Time: 0.3s | LR: 9.76e-05"
    header_re = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s*\|\s*Time:\s*([\d.]+)s\s*\|\s*LR:\s*([\d.eE+-]+)"
    )
    # Losses line:  "Losses: total=..., mean=..., var=..., diversity=..."
    loss_re = re.compile(
        r"total=([\d.]+).*?mean=([\d.]+).*?var=([\d.]+).*?diversity=([\d.]+)"
    )
    # GPU memory:  "GPU Memory: 0.54 GB allocated, 0.55 GB reserved"
    gpu_re = re.compile(
        r"GPU Memory:\s*([\d.]+)\s*GB allocated,\s*([\d.]+)\s*GB reserved"
    )
    # Best marker:  "★ NEW BEST!"
    best_re = re.compile(r"NEW BEST")

    current_epoch: Optional[int] = None
    for line in text.splitlines():
        m = header_re.search(line)
        if m:
            current_epoch = int(m.group(1))
            data["epochs"].append(current_epoch)
            data["epoch_time"].append(float(m.group(3)))
            data["lr"].append(float(m.group(4)))
            if data["meta"].get("total_epochs") is None:
                data["meta"]["total_epochs"] = int(m.group(2))
            continue

        m = loss_re.search(line)
        if m:
            data["total"].append(float(m.group(1)))
            data["mean"].append(float(m.group(2)))
            data["var"].append(float(m.group(3)))
            data["diversity"].append(float(m.group(4)))
            continue

        m = gpu_re.search(line)
        if m:
            data["gpu_alloc"].append(float(m.group(1)))
            data["gpu_reserved"].append(float(m.group(2)))
            continue

        if best_re.search(line) and current_epoch is not None:
            data["best_epochs"].append(current_epoch)

    # Training summary
    best_loss_m = re.search(r"Best loss:\s*([\d.]+)", text)
    if best_loss_m:
        data["meta"]["best_loss"] = float(best_loss_m.group(1))
    total_time_m = re.search(r"Total time:\s*([\d.]+)\s*minutes", text)
    if total_time_m:
        data["meta"]["total_time_min"] = float(total_time_m.group(1))

    return data


def parse_stderr_log(err_path: Path) -> Dict[str, Any]:
    """
    Parse the ``*.err`` log (tqdm output) for per-batch loss values.

    Returns dict with keys:
        batch_losses  – dict mapping epoch (int) → list of per-batch total losses
    """
    text = Path(err_path).read_text()

    batch_losses: Dict[int, List[float]] = {}

    # tqdm lines look like:
    #   Epoch 2/50:  71%|…| 17/24 [..., loss=0.113, mean=0.0129, var=0.201]
    pat = re.compile(
        r"Epoch\s+(\d+)/\d+.*?loss=([\d.]+)"
    )

    for line in text.splitlines():
        m = pat.search(line)
        if m:
            ep = int(m.group(1))
            loss_val = float(m.group(2))
            batch_losses.setdefault(ep, []).append(loss_val)

    # Deduplicate: tqdm re-renders lines, so we may have duplicates.
    # The *last* occurrence at each progress-% is the authoritative one.
    # A simple heuristic: keep only the last N values where N = num_batches
    # (the number of unique batch indices this epoch).
    # Since we can't always know N from .err alone, keep all distinct values
    # in order.
    return {"batch_losses": batch_losses}


# ======================================================================
#  2.  CHECKPOINT ANALYSIS
# ======================================================================

def load_checkpoints(ckpt_dir: Path) -> List[Dict[str, Any]]:
    """Load all ``*.pt`` checkpoint files, sorted by epoch."""
    assert torch is not None, "PyTorch is required for checkpoint analysis"
    ckpts: List[Dict[str, Any]] = []
    for p in sorted(ckpt_dir.glob("projection_head*.pt")):
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            ckpt["_path"] = p
            ckpts.append(ckpt)
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}")
    ckpts.sort(key=lambda c: c.get("epoch", 0))
    return ckpts


def compute_weight_norms(ckpts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    For each checkpoint, compute the L2 norm of every parameter tensor
    as well as the total model L2 norm.

    Returns dict with:
        epochs       – list of ints
        total_norm   – list of floats
        layer_norms  – dict[layer_name] → list of floats
    """
    result: Dict[str, Any] = {"epochs": [], "total_norm": [], "layer_norms": {}}
    for ckpt in ckpts:
        ep = ckpt.get("epoch", 0)
        sd = ckpt.get("state_dict", {})
        result["epochs"].append(ep)
        total_sq = 0.0
        for name, tensor in sd.items():
            norm = tensor.float().norm().item()
            total_sq += norm ** 2
            result["layer_norms"].setdefault(name, []).append(norm)
        result["total_norm"].append(total_sq ** 0.5)
    return result


# ======================================================================
#  3.  PROJECTION ANALYSIS  (requires genomic H5 dir)
# ======================================================================

def project_genomic_features(
    ckpt: Dict[str, Any],
    h5_paths: List[Path],
    genomic_key: str = "feats",
    max_samples: int = 500,
    device: str = "cpu",
) -> np.ndarray:
    """
    Load genomic H5s, push them through the projection head defined in
    *ckpt*, and return the projected features as a (N, D) numpy array.
    """
    import h5py

    # Rebuild projection head
    from finetune_diffusion.projection_head_genomic import ProjectionHead

    cfg = ckpt.get("config", {})
    ph = ProjectionHead(
        in_dim=cfg.get("in_dim", 512),
        out_dim=cfg.get("out_dim", 512),
        arch=cfg.get("arch", "mlp"),
    )
    ph.load_state_dict(ckpt["state_dict"])
    ph.eval().to(device)

    feats = []
    for p in h5_paths[:max_samples]:
        with h5py.File(p, "r") as f:
            # Use .get and cast to h5py.Dataset so static type-checkers
            # recognize __getitem__ / reading operations
            from typing import cast
            ds = f.get(genomic_key)
            if ds is None:
                raise KeyError(f"Key '{genomic_key}' not found in {p}")
            ds = cast(h5py.Dataset, ds)
            # Read entire dataset safely (equivalent to [:])
            arr = ds[()]
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            feats.append(arr)
    # Ensure torch is available (guard for static type-checkers/runtime)
    assert torch is not None, "PyTorch must be installed to project genomic features"
    feats_t = torch.tensor(np.stack(feats), dtype=torch.float32, device=device)

    with torch.no_grad():
        projected = ph(feats_t).cpu().numpy()
    return projected


# ======================================================================
#  4.  PLOTTING HELPERS
# ======================================================================

_COLORS = {
    "total": "#2c7bb6",
    "mean": "#d7191c",
    "var": "#fdae61",
    "diversity": "#abd9e9",
    "lr": "#636363",
}


def _save_fig(fig: Figure, out_dir: Path, name: str, show: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------- plot functions ----------

def plot_epoch_losses(data: Dict[str, Any], out_dir: Path, show: bool = False) -> None:
    """4-panel figure: total + component losses."""
    epochs = data["epochs"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Projection Head Training — Epoch Losses", fontsize=14, fontweight="bold")

    for ax, key, title in zip(
        axes.flat,
        ["total", "mean", "var", "diversity"],
        ["Total Loss", "Mean-Matching Loss", "Variance Loss", "Diversity Loss"],
    ):
        vals = data[key]
        ax.plot(epochs, vals, "-o", markersize=3, color=_COLORS[key], linewidth=1.4, label=title)

        # Mark best
        if vals:
            best_idx = int(np.argmin(vals))
            ax.scatter(
                [epochs[best_idx]], [vals[best_idx]],
                s=120, marker="*", color=_COLORS[key], edgecolors="black",
                linewidth=0.6, zorder=5,
            )
            ax.annotate(
                f"  min={vals[best_idx]:.5f} (ep {epochs[best_idx]})",
                xy=(epochs[best_idx], vals[best_idx]),
                fontsize=8, color=_COLORS[key],
            )

        # Mark "new best" epochs
        for be in data.get("best_epochs", []):
            if be in epochs:
                idx = epochs.index(be)
                ax.axvline(be, color="green", alpha=0.15, linewidth=1)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_fig(fig, out_dir, "epoch_losses", show)


def plot_loss_components_stacked(data: Dict[str, Any], out_dir: Path, show: bool = False) -> None:
    """Stacked area chart showing relative contribution of each loss component."""
    epochs = np.array(data["epochs"])
    mean_arr = np.array(data["mean"])
    var_arr = np.array(data["var"])
    div_arr = np.array(data["diversity"])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        epochs, mean_arr, var_arr, div_arr,
        labels=["Mean-Matching", "Variance", "Diversity"],
        colors=[_COLORS["mean"], _COLORS["var"], _COLORS["diversity"]],
        alpha=0.75,
    )
    ax.plot(epochs, np.array(data["total"]), "k-", linewidth=1.5, label="Total")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Composition over Epochs (stacked)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save_fig(fig, out_dir, "loss_components_stacked", show)


def plot_lr_schedule(data: Dict[str, Any], out_dir: Path, show: bool = False) -> None:
    """Learning-rate schedule."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data["epochs"], data["lr"], "-o", markersize=3, color=_COLORS["lr"], linewidth=1.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule (Cosine Annealing)")
    ax.grid(True, alpha=0.25)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
    fig.tight_layout()
    _save_fig(fig, out_dir, "lr_schedule", show)


def plot_epoch_time(data: Dict[str, Any], out_dir: Path, show: bool = False) -> None:
    """Per-epoch wall-clock time."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(data["epochs"], data["epoch_time"], color="#7fbf7b", edgecolor="white", width=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (s)")
    ax.set_title("Training Time per Epoch")
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    _save_fig(fig, out_dir, "epoch_time", show)


def plot_batch_losses(batch_data: Dict[str, Any], out_dir: Path, show: bool = False) -> None:
    """
    Plot per-batch loss trajectories for selected epochs overlaid,
    plus a full batch-level loss trajectory across all epochs.
    """
    batch_losses = batch_data["batch_losses"]
    if not batch_losses:
        print("  [SKIP] No batch-level data available.")
        return

    # --- Panel 1: overlaid per-epoch batch curves for first, middle, last epochs ---
    all_epochs = sorted(batch_losses.keys())
    selected = []
    if len(all_epochs) >= 3:
        selected = [all_epochs[0], all_epochs[len(all_epochs) // 2], all_epochs[-1]]
    else:
        selected = all_epochs

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Batch-Level Loss Dynamics", fontsize=14, fontweight="bold")

    cmap = plt.get_cmap("viridis")
    ax = axes[0]
    for i, ep in enumerate(selected):
        vals = batch_losses[ep]
        color = cmap(i / max(len(selected) - 1, 1))
        ax.plot(range(len(vals)), vals, alpha=0.7, label=f"Epoch {ep}", color=color, linewidth=1.2)
    ax.set_xlabel("Batch (tqdm update)")
    ax.set_ylabel("Loss")
    ax.set_title("Within-Epoch Loss (selected epochs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # --- Panel 2: continuous batch-level loss ---
    ax2 = axes[1]
    x_vals, y_vals = [], []
    offset = 0
    for ep in all_epochs:
        vals = batch_losses[ep]
        for j, v in enumerate(vals):
            x_vals.append(offset + j)
            y_vals.append(v)
        offset += len(vals)
    ax2.plot(x_vals, y_vals, linewidth=0.5, alpha=0.6, color=_COLORS["total"])
    # Smoothed line
    if len(y_vals) > 20:
        window = max(5, len(y_vals) // 50)
        smoothed = np.convolve(y_vals, np.ones(window) / window, mode="valid")
        ax2.plot(
            x_vals[window // 2 : window // 2 + len(smoothed)],
            smoothed,
            linewidth=1.5,
            color="red",
            label=f"Moving avg (w={window})",
        )
        ax2.legend(fontsize=8)
    ax2.set_xlabel("Global Batch Index")
    ax2.set_ylabel("Loss")
    ax2.set_title("Continuous Batch-Level Loss")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_fig(fig, out_dir, "batch_losses", show)


def plot_weight_norms(ckpts: List[Dict[str, Any]], out_dir: Path, show: bool = False) -> None:
    """Layer-wise and total weight L2 norms across checkpoints."""
    wn = compute_weight_norms(ckpts)
    if not wn["epochs"]:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Weight Norm Evolution", fontsize=14, fontweight="bold")

    # Total norm
    ax = axes[0]
    ax.plot(wn["epochs"], wn["total_norm"], "-o", markersize=5, color="#2c7bb6", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total L2 Norm")
    ax.set_title("Total Model Weight Norm")
    ax.grid(True, alpha=0.25)

    # Per-layer norms
    ax2 = axes[1]
    cmap = plt.get_cmap("tab10")
    for i, (name, norms) in enumerate(wn["layer_norms"].items()):
        short = name.replace("net.", "L")
        ax2.plot(wn["epochs"], norms, "-o", markersize=4, color=cmap(i / 10),
                 linewidth=1.2, label=short)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("L2 Norm")
    ax2.set_title("Per-Layer Weight Norms")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_fig(fig, out_dir, "weight_norms", show)


def plot_weight_histograms(ckpts: List[Dict[str, Any]], out_dir: Path, show: bool = False) -> None:
    """Histogram of weight values for each checkpoint (overlaid)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("viridis")
    n = len(ckpts)
    for i, ckpt in enumerate(ckpts):
        ep = ckpt.get("epoch", i)
        sd = ckpt.get("state_dict", {})
        all_vals = np.concatenate([v.float().numpy().ravel() for v in sd.values()])
        color = cmap(i / max(n - 1, 1))
        ax.hist(all_vals, bins=120, alpha=0.35, color=color, label=f"Epoch {ep}",
                density=True, histtype="stepfilled", linewidth=0.8)
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Density")
    ax.set_title("Weight Distribution across Checkpoints")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save_fig(fig, out_dir, "weight_histograms", show)


def plot_projected_vs_target(
    ckpts: List[Dict[str, Any]],
    h5_paths: List[Path],
    out_dir: Path,
    genomic_key: str = "feats",
    device: str = "cpu",
    show: bool = False,
) -> None:
    """
    For the best and last checkpoint, project genomic features and compare
    their per-dimension mean/std to the target distribution.
    """
    # Identify the best checkpoint (has target_mean / target_std)
    best_ckpt = None
    for c in ckpts:
        if "target_mean" in c:
            best_ckpt = c
            break
    if best_ckpt is None:
        print("  [SKIP] No checkpoint with target_mean/target_std found.")
        return

    target_mean = best_ckpt["target_mean"].numpy()
    target_std = best_ckpt["target_std"].numpy()
    dim = len(target_mean)

    # Select a few checkpoints to compare
    selected = _select_representative_ckpts(ckpts)

    fig, axes = plt.subplots(len(selected), 2, figsize=(16, 5 * len(selected)))
    if len(selected) == 1:
        axes = axes[np.newaxis, :]  # ensure 2D
    fig.suptitle("Projected Feature Statistics vs Target", fontsize=14, fontweight="bold")

    for row, ckpt in enumerate(selected):
        ep = ckpt.get("epoch", "?")
        projected = project_genomic_features(ckpt, h5_paths, genomic_key, device=device)
        proj_mean = projected.mean(axis=0)
        proj_std = projected.std(axis=0)

        # Mean comparison
        ax = axes[row, 0]
        x = np.arange(dim)
        ax.bar(x - 0.2, target_mean, width=0.4, alpha=0.6, label="Target mean", color="#2c7bb6")
        ax.bar(x + 0.2, proj_mean, width=0.4, alpha=0.6, label=f"Projected mean (ep {ep})", color="#d7191c")
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Mean")
        ax.set_title(f"Epoch {ep} — Per-Dim Mean")
        ax.legend(fontsize=8)
        # Only show every Nth tick
        step = max(1, dim // 20)
        ax.set_xticks(x[::step])
        ax.grid(True, alpha=0.2, axis="y")

        # Std comparison
        ax2 = axes[row, 1]
        ax2.bar(x - 0.2, target_std, width=0.4, alpha=0.6, label="Target std", color="#2c7bb6")
        ax2.bar(x + 0.2, proj_std, width=0.4, alpha=0.6, label=f"Projected std (ep {ep})", color="#fdae61")
        ax2.set_xlabel("Dimension")
        ax2.set_ylabel("Std")
        ax2.set_title(f"Epoch {ep} — Per-Dim Std")
        ax2.legend(fontsize=8)
        ax2.set_xticks(x[::step])
        ax2.grid(True, alpha=0.2, axis="y")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_fig(fig, out_dir, "projected_vs_target", show)


def plot_projected_distribution_overlap(
    ckpts: List[Dict[str, Any]],
    h5_paths: List[Path],
    out_dir: Path,
    genomic_key: str = "feats",
    n_dims_to_show: int = 6,
    device: str = "cpu",
    show: bool = False,
) -> None:
    """
    Overlay histograms of projected feature values vs target Gaussian for
    a random subset of dimensions, using the best checkpoint.
    """
    best_ckpt = None
    for c in ckpts:
        if "target_mean" in c:
            best_ckpt = c
            break
    if best_ckpt is None:
        print("  [SKIP] No checkpoint with target stats.")
        return

    target_mean = best_ckpt["target_mean"].numpy()
    target_std = best_ckpt["target_std"].numpy()
    dim = len(target_mean)
    projected = project_genomic_features(best_ckpt, h5_paths, genomic_key, device=device)
    ep = best_ckpt.get("epoch", "?")

    rng = np.random.default_rng(42)
    dims = rng.choice(dim, size=min(n_dims_to_show, dim), replace=False)
    dims.sort()

    ncols = min(3, len(dims))
    nrows = (len(dims) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)
    fig.suptitle(f"Projected vs Target Distribution (best ckpt, ep {ep})", fontsize=13, fontweight="bold")

    for idx, d in enumerate(dims):
        ax = axes.flat[idx]
        proj_vals = projected[:, d]
        ax.hist(proj_vals, bins=30, density=True, alpha=0.6, color="#d7191c", label="Projected")

        # Overlay target Gaussian
        x_range = np.linspace(
            min(proj_vals.min(), target_mean[d] - 3 * target_std[d]),
            max(proj_vals.max(), target_mean[d] + 3 * target_std[d]),
            200,
        )
        from scipy.stats import norm as sp_norm
        pdf = sp_norm.pdf(x_range, loc=target_mean[d], scale=target_std[d])
        ax.plot(x_range, pdf, "b-", linewidth=1.5, label="Target N(μ,σ)")
        ax.set_title(f"Dim {d}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    # Hide unused
    for idx in range(len(dims), nrows * ncols):
        axes.flat[idx].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_fig(fig, out_dir, "distribution_overlap", show)


def plot_pairwise_cosine_sim(
    ckpts: List[Dict[str, Any]],
    h5_paths: List[Path],
    out_dir: Path,
    genomic_key: str = "feats",
    max_samples: int = 100,
    device: str = "cpu",
    show: bool = False,
) -> None:
    """
    Pairwise cosine-similarity heat-map of projected features
    (checks for mode collapse).
    """
    selected = _select_representative_ckpts(ckpts, max_n=3)
    n = len(selected)
    if n == 0:
        print("  [SKIP] No checkpoints selected for pairwise cosine similarity.")
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Pairwise Cosine Similarity of Projected Features", fontsize=13, fontweight="bold")

    for i, ckpt in enumerate(selected):
        ep = ckpt.get("epoch", "?")
        projected = project_genomic_features(ckpt, h5_paths, genomic_key, max_samples=max_samples, device=device)
        # Normalize
        norms = np.linalg.norm(projected, axis=1, keepdims=True) + 1e-8
        proj_normed = projected / norms
        cos_sim = proj_normed @ proj_normed.T

        ax = axes[i]
        im = ax.imshow(cos_sim, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"Epoch {ep}\nmean cos-sim={cos_sim[np.triu_indices_from(cos_sim, k=1)].mean():.3f}")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Sample")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_fig(fig, out_dir, "pairwise_cosine_similarity", show)


def plot_umap_projections(
    ckpts: List[Dict[str, Any]],
    h5_paths: List[Path],
    out_dir: Path,
    genomic_key: str = "feats",
    max_samples: int = 500,
    device: str = "cpu",
    show: bool = False,
) -> None:
    """UMAP of projected features at different training stages."""
    try:
        from umap import UMAP
    except ImportError:
        print("  [SKIP] umap-learn not installed. pip install umap-learn")
        return

    selected = _select_representative_ckpts(ckpts)
    n = len(selected)
    if n == 0:
        print("  [SKIP] No checkpoints selected for UMAP projections.")
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("UMAP of Projected Genomic Features", fontsize=13, fontweight="bold")

    for i, ckpt in enumerate(selected):
        ep = ckpt.get("epoch", "?")
        projected = project_genomic_features(ckpt, h5_paths, genomic_key, max_samples=max_samples, device=device)

        reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        emb = reducer.fit_transform(projected)

        # If a tuple was returned (embedding, meta), unwrap the first element.
        if isinstance(emb, tuple) and len(emb) > 0:
            emb = emb[0]

        # Cast to Any so static type-checkers don't complain about
        # attribute access on union / unknown types (tuple, sparse, etc.).
        emb_any = cast(Any, emb)

        # Ensure embedding is a dense numpy array: some UMAP backends
        # or SciPy transforms may return sparse matrices (e.g. coo_matrix)
        # which do not support 2D slicing. Convert if necessary.
        if hasattr(emb_any, "toarray"):
            emb = emb_any.toarray()
        elif hasattr(emb_any, "A"):
            emb = emb_any.A
        else:
            emb = np.asarray(emb_any)

        ax = axes[i]
        sc = ax.scatter(emb[:, 0], emb[:, 1], s=8, alpha=0.6, c=np.arange(len(emb)), cmap="Spectral")
        ax.set_title(f"Epoch {ep} (N={len(emb)})")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(True, alpha=0.15)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_fig(fig, out_dir, "umap_projections", show)


# ======================================================================
#  5.  SUMMARY REPORT
# ======================================================================

def print_summary(data: Dict[str, Any]) -> None:
    """Print a concise textual summary of the training run."""
    print("\n" + "=" * 60)
    print("PROJECTION HEAD TRAINING — SUMMARY")
    print("=" * 60)

    if data["epochs"]:
        print(f"  Epochs trained:   {len(data['epochs'])}")
        print(f"  Final total loss: {data['total'][-1]:.6f}")
        best_idx = int(np.argmin(data["total"]))
        print(f"  Best total loss:  {data['total'][best_idx]:.6f}  (epoch {data['epochs'][best_idx]})")
        print(f"  Best mean loss:   {min(data['mean']):.6f}")
        print(f"  Best var loss:    {min(data['var']):.6f}")
        print(f"  Diversity always: {data['diversity'][0]:.6f}  (constant)")
        if data["lr"]:
            print(f"  LR range:         {min(data['lr']):.2e} → {max(data['lr']):.2e}")
        if data["epoch_time"]:
            print(f"  Epoch time range: {min(data['epoch_time']):.1f}s – {max(data['epoch_time']):.1f}s")

    # Convergence assessment
    if len(data["total"]) >= 10:
        last_10 = data["total"][-10:]
        first_10 = data["total"][:10]
        rel_drop = (np.mean(first_10) - np.mean(last_10)) / (np.mean(first_10) + 1e-10)
        std_last10 = np.std(last_10)
        print(f"\n  Convergence:")
        print(f"    Relative drop (first 10 vs last 10): {rel_drop:.1%}")
        print(f"    Std of last 10 epochs:               {std_last10:.6f}")
        if std_last10 < 0.005 and rel_drop > 0.3:
            print("    → Training appears well-converged.")
        elif std_last10 < 0.01:
            print("    → Training appears to have plateaued.")
        else:
            print("    → Loss still fluctuating — consider more epochs or lower LR.")

    # Warnings
    if all(d == 0.0 for d in data["diversity"]):
        print("\n  ⚠ Diversity loss is always 0.0.")
        print("    This means the diversity hinge target is already satisfied")
        print("    (projected features are spread out enough). Good sign — no mode collapse.")

    print("=" * 60 + "\n")


# ======================================================================
#  HELPERS
# ======================================================================

def _select_representative_ckpts(
    ckpts: List[Dict[str, Any]], max_n: int = 4
) -> List[Dict[str, Any]]:
    """Pick a few representative checkpoints (first, middle, last, best)."""
    if len(ckpts) <= max_n:
        return ckpts

    # Always include first and last
    selected_indices = {0, len(ckpts) - 1}

    # Include best
    for i, c in enumerate(ckpts):
        if "target_mean" in c:  # best checkpoint has target stats
            selected_indices.add(i)
            break

    # Fill remaining with evenly spaced
    while len(selected_indices) < max_n:
        gaps = sorted(selected_indices)
        # Find the largest gap
        max_gap, insert_idx = 0, 0
        for a, b in zip(gaps, gaps[1:]):
            if b - a > max_gap:
                max_gap = b - a
                insert_idx = (a + b) // 2
        if max_gap <= 1:
            break
        selected_indices.add(insert_idx)

    return [ckpts[i] for i in sorted(selected_indices)]


def _find_h5_files(genomic_h5_dir: Path) -> List[Path]:
    """Find all genomic H5 files in the given directory (recursive)."""
    h5s = sorted(genomic_h5_dir.rglob("*.h5"))
    if not h5s:
        # Try looking in train/ subdirectory
        h5s = sorted((genomic_h5_dir / "train").rglob("*.h5")) if (genomic_h5_dir / "train").exists() else []
    return h5s


# ======================================================================
#  MAIN
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Investigate projection-head training results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log-file", type=str, required=True,
        help="Path to the Slurm .out log file",
    )
    parser.add_argument(
        "--err-file", type=str, default=None,
        help="Path to the Slurm .err log file (for batch-level losses)",
    )
    parser.add_argument(
        "--ckpt-dir", type=str, default=None,
        help="Directory containing projection_head_*.pt checkpoints",
    )
    parser.add_argument(
        "--genomic-h5-dir", type=str, default=None,
        help="Directory with genomic .h5 feature files (enables projection analysis)",
    )
    parser.add_argument(
        "--genomic-key", type=str, default="feats",
        help="HDF5 key for genomic feature vectors",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory for figures (default: alongside log file)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for projection analysis (e.g. cpu, cuda:0)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display plots interactively",
    )
    args = parser.parse_args()

    # --- Resolve output directory ---
    if args.out_dir:
        out_dir = Path(args.out_dir) / "figures"
    elif args.ckpt_dir:
        out_dir = Path(args.ckpt_dir) / "investigation" / "figures"
    else:
        out_dir = Path(args.log_file).parent / "investigation" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Figures will be saved to: {out_dir}\n")

    if args.show:
        matplotlib.use("TkAgg")

    # ---- Phase 1: Log-based analysis ----
    print("[Phase 1] Parsing stdout log …")
    data = parse_stdout_log(Path(args.log_file))
    print_summary(data)

    print("[Phase 1] Generating epoch-level plots …")
    plot_epoch_losses(data, out_dir, show=args.show)
    plot_loss_components_stacked(data, out_dir, show=args.show)
    if data["lr"]:
        plot_lr_schedule(data, out_dir, show=args.show)
    if data["epoch_time"]:
        plot_epoch_time(data, out_dir, show=args.show)

    # ---- Phase 2: Batch-level analysis (.err) ----
    if args.err_file:
        print("\n[Phase 2] Parsing stderr log for batch-level losses …")
        batch_data = parse_stderr_log(Path(args.err_file))
        n_epochs_with_batches = len(batch_data["batch_losses"])
        total_batches = sum(len(v) for v in batch_data["batch_losses"].values())
        print(f"  Found batch data for {n_epochs_with_batches} epochs ({total_batches} total tqdm updates)")
        plot_batch_losses(batch_data, out_dir, show=args.show)
    else:
        print("\n[Phase 2] Skipped (no --err-file provided)")

    # ---- Phase 3: Checkpoint analysis ----
    if args.ckpt_dir:
        print("\n[Phase 3] Loading checkpoints …")
        ckpts = load_checkpoints(Path(args.ckpt_dir))
        print(f"  Loaded {len(ckpts)} checkpoints: "
              + ", ".join(f"ep{c.get('epoch','?')}" for c in ckpts))
        plot_weight_norms(ckpts, out_dir, show=args.show)
        plot_weight_histograms(ckpts, out_dir, show=args.show)

        # ---- Phase 4: Projection analysis (needs genomic H5s) ----
        if args.genomic_h5_dir:
            print("\n[Phase 4] Projection analysis …")
            h5_paths = _find_h5_files(Path(args.genomic_h5_dir))
            print(f"  Found {len(h5_paths)} genomic H5 files")
            if h5_paths:
                plot_projected_vs_target(
                    ckpts, h5_paths, out_dir,
                    genomic_key=args.genomic_key, device=args.device, show=args.show,
                )
                plot_projected_distribution_overlap(
                    ckpts, h5_paths, out_dir,
                    genomic_key=args.genomic_key, device=args.device, show=args.show,
                )
                plot_pairwise_cosine_sim(
                    ckpts, h5_paths, out_dir,
                    genomic_key=args.genomic_key, device=args.device, show=args.show,
                )
                plot_umap_projections(
                    ckpts, h5_paths, out_dir,
                    genomic_key=args.genomic_key, device=args.device, show=args.show,
                )
            else:
                print("  [SKIP] No H5 files found.")
        else:
            print("\n[Phase 4] Skipped (no --genomic-h5-dir provided)")
    else:
        print("\n[Phase 3–4] Skipped (no --ckpt-dir provided)")

    print(f"\n✓ All figures saved to {out_dir}")


if __name__ == "__main__":
    main()
