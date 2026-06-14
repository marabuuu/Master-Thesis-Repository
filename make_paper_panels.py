#!/usr/bin/env python
"""Generate two combined panel PNGs for the PoC conditioning ablation.

Run from Master-Thesis-Repository/:
    source .venv/bin/activate
    python make_paper_panels.py

Outputs (written to experiments/paper_plots/):
    combined_conditioning_metrics.png  — 4-row × 3-col conditioning panel
    combined_loss_curves.png           — 4-row × 2-col loss curves panel
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.visualization.training_plots import (
    load_gda_tfevents,
    _aggregate_by_step,
    _ema_series,
    _subsample_series,
    _batlow_colors,
    _strip_spines,
    _PAPER_RC,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE = Path("/mnt/bulk-saturn/maralampert/genhist/experiments")
OUT = _BASE / "20260612_training_metrics_poc"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = {
    "Zero":       _BASE / "20260607_poc_128_zero_30M/gda",
    "Noise":      _BASE / "20260607_poc_128_noise_30M/gda",
    "Orthogonal": _BASE / "20260605_poc_128_orthogonal_nonorm_30M/gda",
    "Genomic":    _BASE / "20260607_poc_128_rna_norm_30M/gda",
}

RUN_NAMES = list(RUNS.keys())

# ---------------------------------------------------------------------------
# Shared colours (match existing per-run paper plots)
# ---------------------------------------------------------------------------

C_SIG, C_GAP, C_SEP = _batlow_colors([0.20, 0.55, 0.78])
C_TRAIN, C_VAL = _batlow_colors([0.15, 0.55])

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

# Bump all font sizes from the base _PAPER_RC
_COND_RC = {
    **_PAPER_RC,
    "font.size": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
}


def _sci_fmt(ax) -> None:
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))


def _row_label(ax, name: str) -> None:
    ax.text(
        -0.30, 0.5, name,
        transform=ax.transAxes,
        rotation=90, va="center", ha="right",
        fontsize=15, fontweight="bold", color="black",
    )


def _add_inset(ax, steps, vals, color) -> None:
    """Zoom inset in top-right corner showing this run's own data range.

    For truly-zero data (signal/gap of Zero run): ±10⁻²⁰.
    For near-zero data (Noise, or Zero sep): run's actual value range.
    Y-axis uses the same ×10^N convention as the main axes (ScalarFormatter).
    """
    if not vals:
        return

    max_abs = max(abs(v) for v in vals)

    if max_abs < 1e-20:
        y_in_bot, y_in_top = -1e-20, 1e-20
    else:
        v_min = min(vals)
        v_max = max(vals)
        span = v_max - v_min if v_max != v_min else abs(v_max)
        margin = span * 0.18
        y_in_bot = v_min - margin if v_min < 0 else -span * 0.08
        y_in_top = v_max + margin

    ax_in = ax.inset_axes([0.50, 0.42, 0.46, 0.50])
    s_M = [s / 1e6 for s in steps]
    s_in, v_in = _subsample_series(s_M, list(vals), 800)
    ax_in.plot(s_in, v_in, lw=0.7, color=color, alpha=0.9, rasterized=True)

    ax_in.set_ylim(y_in_bot, y_in_top)
    # Three clean reference ticks: bottom / zero / top
    ax_in.set_yticks([y_in_bot, 0.0, y_in_top])
    ax_in.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax_in.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    # Make the offset-text (×10^N) visible and readable inside the inset
    ot = ax_in.yaxis.get_offset_text()
    ot.set_visible(True)
    ot.set_fontsize(6.5)

    ax_in.set_xticks([])
    ax_in.tick_params(labelsize=6.5, pad=1)
    for spine in ax_in.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("0.4")
    ax_in.patch.set_facecolor("0.97")
    ax_in.grid(True, linestyle="--", linewidth=0.3, alpha=0.5, color="0.65")
    ax_in.axhline(0.0, color="0.6", lw=0.5, ls="--", alpha=0.6)


# Rows that get insets in every conditioning column
_INSET_ROWS = {"Zero", "Noise"}


# ---------------------------------------------------------------------------
# Figure 1 – Conditioning metrics
# ---------------------------------------------------------------------------

def make_conditioning_panel() -> None:
    """4 rows × 3 cols: rows = runs, cols = signal / gap / separation."""
    plt.rcParams.update(_COND_RC)

    # --- load & cache all data -------------------------------------------
    all_data = {name: load_gda_tfevents(logdir) for name, logdir in RUNS.items()}

    tags   = ["cond/signal", "cond/gap", "cond/brca_lihc_sep"]
    colors = [C_SIG, C_GAP, C_SEP]
    ylabels = [
        "Conditioning signal",
        "Conditioning gap",
        "Cond. separation",
    ]

    # --- compute shared y-limits per column --------------------------------
    # Use EMA-smoothed max (not raw max) for signal and sep so single spikes
    # do not inflate the y-axis.  Gap keeps raw min/max (already aggregated).
    # y_bot is slightly negative for signal/sep so near-zero oscillations show.
    y_tops: list[float | None] = []
    y_bots: list[float] = []

    for tag in tags:
        col_max = 0.0
        col_min = 0.0
        for name, data in all_data.items():
            raw_s, raw_v = data.get(tag, ([], []))
            if tag == "cond/gap":
                _, vals = _aggregate_by_step(raw_s, raw_v)
                steps_f: list[float] = list(range(len(vals)))
            else:
                steps_f = list(raw_s)
                vals = list(raw_v)
            fin = [v for v in vals if np.isfinite(v)]
            if not fin:
                continue
            if tag in ("cond/signal", "cond/brca_lihc_sep") and len(fin) > 10:
                s_M = [s / 1e6 for s in steps_f]
                _, v_ema = _ema_series(s_M, fin, alpha=0.05)
                effective_max = max(v_ema) if v_ema else max(fin)
            else:
                effective_max = max(fin)
            col_max = max(col_max, effective_max)
            col_min = min(col_min, min(fin))

        top = col_max * 1.25 if col_max > 0 else None
        y_tops.append(top)
        if col_min < 0:
            y_bots.append(col_min * 1.20)   # gap: preserve negatives
        elif top is not None:
            y_bots.append(-top * 0.08)       # signal/sep: slight dip below 0
        else:
            y_bots.append(0.0)

    # --- create figure ----------------------------------------------------
    fig, axes = plt.subplots(
        4, 3,
        figsize=(13, 10),
        gridspec_kw={"hspace": 0.40, "wspace": 0.50},
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.07)

    for row, name in enumerate(RUN_NAMES):
        data = all_data[name]

        for col, (tag, color, ylabel, y_top, y_bot) in enumerate(
            zip(tags, colors, ylabels, y_tops, y_bots)
        ):
            ax = axes[row, col]

            raw_s, raw_v = data.get(tag, ([], []))
            if tag == "cond/gap":
                agg_s, vals = _aggregate_by_step(raw_s, raw_v)
                steps = [float(s) for s in agg_s]
            else:
                steps, vals = list(raw_s), list(raw_v)

            if vals:
                s_M = [s / 1e6 for s in steps]
                s_sub, v_sub = _subsample_series(s_M, vals, 1_500)
                ax.plot(s_sub, v_sub, lw=0.4, alpha=0.20, color=color, rasterized=True)
                s_ema, v_ema = _ema_series(s_M, vals, alpha=0.05)
                ax.plot(s_ema, v_ema, lw=1.8, color=color)

            ax.axhline(0.0, color="0.65", lw=0.7, ls="--", alpha=0.5)
            ax.set_ylim(bottom=y_bot, top=y_top)
            _sci_fmt(ax)
            _strip_spines(ax)

            ax.set_ylabel(ylabel, fontsize=15)
            if row < 3:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Samples (×10⁶)", fontsize=15)

            # --- Inset zoom for Zero and Noise rows -----------------------
            if name in _INSET_ROWS and vals:
                _add_inset(ax, steps, vals, color)

        _row_label(axes[row, 0], name)

    out = OUT / "combined_conditioning_metrics.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2 – Loss curves
# ---------------------------------------------------------------------------

def make_loss_panel() -> None:
    """4 rows × 2 cols: rows = runs, cols = [linear loss, log loss]."""
    plt.rcParams.update(_COND_RC)

    all_data = {name: load_gda_tfevents(logdir) for name, logdir in RUNS.items()}

    # --- compute shared y-limits across all runs --------------------------
    # Linear scale: use post-convergence maximum (skip first 0.5 M samples)
    lin_top = 0.0
    for data in all_data.values():
        va_s_r, va_v_r = data.get("loss/val", ([], []))
        va_steps, va_vals = _aggregate_by_step(va_s_r, va_v_r)
        post = [v for s, v in zip(va_steps, va_vals) if s > 5e5 and np.isfinite(v)]
        if post:
            lin_top = max(lin_top, max(post))
    lin_top = lin_top * 3.0 if lin_top > 0 else None

    # Log scale: use actual max of val loss (skip initial spike, > 0.5 M)
    log_top = lin_top  # same upper bound works for log
    log_bot: float | None = None
    for data in all_data.values():
        va_s_r, va_v_r = data.get("loss/val", ([], []))
        va_steps, va_vals = _aggregate_by_step(va_s_r, va_v_r)
        post_pos = [v for s, v in zip(va_steps, va_vals) if s > 5e5 and v > 0]
        if post_pos:
            candidate = min(post_pos) * 0.7
            log_bot = min(log_bot, candidate) if log_bot is not None else candidate

    # --- create figure ----------------------------------------------------
    fig, axes = plt.subplots(
        4, 2,
        figsize=(10, 10),
        gridspec_kw={"hspace": 0.35, "wspace": 0.45},
    )
    fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.07)

    for row, name in enumerate(RUN_NAMES):
        data = all_data[name]

        tr_steps, tr_vals = data.get("loss/train", ([], []))
        va_s_r, va_v_r   = data.get("loss/val",   ([], []))
        va_steps, va_vals = _aggregate_by_step(va_s_r, va_v_r)

        tr_s_ema = tr_v_ema = tr_s_raw = tr_v_raw = []
        if tr_vals:
            tr_s_M = [s / 1e6 for s in tr_steps]
            tr_s_raw, tr_v_raw = _subsample_series(tr_s_M, list(tr_vals), 2_500)
            tr_s_ema, tr_v_ema = _ema_series(tr_s_M, list(tr_vals), alpha=0.005)

        va_s_M = [s / 1e6 for s in va_steps]

        for col in range(2):
            ax = axes[row, col]
            log = (col == 1)

            if tr_v_raw:
                ax.plot(tr_s_raw, tr_v_raw, lw=0.4, alpha=0.20,
                        color=C_TRAIN, rasterized=True)
            if tr_v_ema:
                ax.plot(tr_s_ema, tr_v_ema, lw=1.8, color=C_TRAIN,
                        label="Train loss")
            if va_vals:
                ax.plot(va_s_M, va_vals, lw=1.6, color=C_VAL, label="Val loss")

            if log:
                ax.set_yscale("log")
                if log_bot is not None and log_top is not None:
                    ax.set_ylim(bottom=log_bot, top=log_top)
                ax.set_ylabel("MSE loss (log scale)", fontsize=15)
            else:
                if lin_top is not None:
                    ax.set_ylim(bottom=0, top=lin_top)
                ax.set_ylabel("MSE loss", fontsize=15)

            _strip_spines(ax)

            # Legend on first row only
            if row == 0:
                ax.legend(loc="upper right", fontsize=13)

            if row < 3:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Samples (×10⁶)", fontsize=15)

        _row_label(axes[row, 0], name)

    out = OUT / "combined_loss_curves.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating conditioning metrics panel …")
    make_conditioning_panel()
    print("Generating loss curves panel …")
    make_loss_panel()
    print("Done.")
