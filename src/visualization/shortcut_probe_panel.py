#!/usr/bin/env python3
"""
Shortcut-Probe Panel
====================

Compact companion figure to ``cfg_dose_response_panel.py``. Compares the real
Virchow2 subtype classifier against a trivial "shortcut probe" — a linear
classifier trained on nothing but global color/texture statistics (channel
mean/std, HSV stats, a coarse color histogram, Laplacian variance; see
``extract_color_texture_features.py``) — across Real tiles and generated
tiles at CFG = 1, 3, 5.

Purpose: if the color-only probe tracked the Virchow2 classifier (both rising
similarly with CFG), that would suggest the generated-tile "exaggeration"
seen in classifier_panel.png is at least partly a low-level color/style
shortcut. Instead the probe sits at chance (~0.53) everywhere, while Virchow2
rises from 0.885 (real) to 0.92-0.97 (generated) — evidence against a trivial
color/texture shortcut explanation.

Wide/short aspect ratio so it can sit next to or below cfg_dose_response_panel
on a single slide.

Usage:
    python -m src.visualization.shortcut_probe_panel
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

_REPO = Path(__file__).resolve().parents[2]
_EXP = _REPO.parent / "experiments"
sys.path.insert(0, str(_REPO))

from src.visualization.training_plots import _PAPER_RC, _strip_spines

CV_DIR = _EXP / "20260618_subtype_classifier_cv"
OUT = CV_DIR / "shortcut_probe_panel.png"

try:
    from cmcrameri import cm as cmc
    _cmap = cmc.batlow
except ImportError:
    _cmap = plt.cm.viridis

C_DEEP = to_hex(_cmap(0.45))   # matches C_GEN in classifier_panel.py
C_SHORTCUT = "0.65"
C_CHANCE = "0.75"

GROUPS = ["Real", "CFG = 1", "CFG = 3", "CFG = 5"]


def _load_deep_metrics() -> tuple[list[float], list[Optional[list[float]]]]:
    with open(CV_DIR / "cv_results" / "cv_summary.json") as f:
        real = json.load(f)
    real_auc = real["aggregate_tile_metrics"]["roc_auc"]
    real_ci = real["bootstrap_ci"]["tile_roc_auc"]
    real_err = [real_auc - real_ci["ci_lower"], real_ci["ci_upper"] - real_auc]

    aucs, errs = [real_auc], [real_err]
    for gen_dir in ["generated_eval_10k", "generated_eval_cfg3", "generated_eval_cfg5"]:
        with open(CV_DIR / gen_dir / "evaluation_metrics.json") as f:
            d = json.load(f)
        aucs.append(d["tile_metrics"]["roc_auc"])
        errs.append(None)
    return aucs, errs


def _load_shortcut_metrics() -> tuple[list[float], list[Optional[list[float]]]]:
    with open(CV_DIR / "color_baseline" / "cv_real" / "cv_results" / "cv_summary.json") as f:
        real = json.load(f)
    real_auc = real["aggregate_tile_metrics"]["roc_auc"]
    real_ci = real["bootstrap_ci"]["tile_roc_auc"]
    real_err = [real_auc - real_ci["ci_lower"], real_ci["ci_upper"] - real_auc]

    aucs, errs = [real_auc], [real_err]
    for cfg in ["cfg1", "cfg3", "cfg5"]:
        with open(CV_DIR / "color_baseline" / f"generated_eval_{cfg}" / "evaluation_metrics.json") as f:
            d = json.load(f)
        aucs.append(d["tile_metrics"]["roc_auc"])
        errs.append(None)
    return aucs, errs


def build_panel(output_path: Optional[str | Path] = None) -> None:
    rc_overrides = {
        **_PAPER_RC,
        "font.size": 15,
        "axes.labelsize": 16,
        "xtick.labelsize": 15,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "axes.titlesize": 17,
    }
    plt.rcParams.update(rc_overrides)

    deep_aucs, deep_errs = _load_deep_metrics()
    short_aucs, short_errs = _load_shortcut_metrics()

    x = np.arange(len(GROUPS))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.2))

    deep_yerr = np.array([e if e is not None else [0, 0] for e in deep_errs]).T
    short_yerr = np.array([e if e is not None else [0, 0] for e in short_errs]).T

    ax.bar(x - w / 2, deep_aucs, w, color=C_DEEP, label="Virchow2 (deep features)",
           yerr=deep_yerr, capsize=4, error_kw={"lw": 1.2, "ecolor": "0.2"})
    ax.bar(x + w / 2, short_aucs, w, color=C_SHORTCUT, label="Color/texture (shortcut probe)",
           yerr=short_yerr, capsize=4, error_kw={"lw": 1.2, "ecolor": "0.2"})

    for xi, v in zip(x - w / 2, deep_aucs):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=12, color=C_DEEP)
    for xi, v in zip(x + w / 2, short_aucs):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=12, color="0.35")

    ax.axhline(0.5, color=C_CHANCE, ls="--", lw=1.0, zorder=1)
    ax.text(len(GROUPS) - 0.5, 0.505, "chance", color=C_CHANCE, fontsize=11, ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS)
    ax.set_ylabel("Tile AUROC\n(Basal vs LumA)")
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              fontsize=12, frameon=False)
    _strip_spines(ax)
    fig.tight_layout()

    out = Path(output_path) if output_path else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Panel] Saved -> {out}")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()
    build_panel(output_path=args.output)


if __name__ == "__main__":
    main()
