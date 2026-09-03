#!/usr/bin/env python3
"""
CFG Dose-Response Panel
========================

3-row x 2-column panel comparing generated-tile subtype separability across
classifier-free-guidance scales (CFG = 1, 3, 5). Reuses the exact ROC and
per-patient dot-plot panel functions from ``classifier_panel.py`` (its C/D
panels) so each row is styled identically and directly comparable:

  Row 1: CFG = 1  (experiments/20260618_subtype_classifier_cv/generated_eval_10k)
  Row 2: CFG = 3  (experiments/20260618_subtype_classifier_cv/generated_eval_cfg3)
  Row 3: CFG = 5  (experiments/20260618_subtype_classifier_cv/generated_eval_cfg5)
  Col 1: Generated ROC vs Real (CV) reference
  Col 2: Per-patient mean P(Basal)

Usage:
    python -m src.visualization.cfg_dose_response_panel
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── path setup ───────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]
_EXP = _REPO.parent / "experiments"
sys.path.insert(0, str(_REPO))

from src.visualization.training_plots import _PAPER_RC
from src.visualization.classifier_panel import _panel_gen_roc, _panel_patient_dots

# ── fixed paths ──────────────────────────────────────────────────────────────
CV_DIR = _EXP / "20260618_subtype_classifier_cv"
CV_RESULTS = CV_DIR / "cv_results"
OUT = CV_DIR / "cfg_dose_response_panel.png"

ROWS = [
    ("CFG = 1", CV_DIR / "generated_eval_10k"),
    ("CFG = 3", CV_DIR / "generated_eval_cfg3"),
    ("CFG = 5", CV_DIR / "generated_eval_cfg5"),
]


# ── data loading ─────────────────────────────────────────────────────────────

def _load_oof() -> pd.DataFrame:
    return pd.read_parquet(CV_RESULTS / "all_oof_predictions.parquet")


def _load_row_data(gen_dir: Path):
    with open(gen_dir / "evaluation_metrics.json") as f:
        gen_eval = json.load(f)
    gen_tiles = pd.read_parquet(gen_dir / "tile_predictions.parquet")
    gen_patients = pd.read_parquet(gen_dir / "per_patient_predictions.parquet")
    return gen_eval, gen_tiles, gen_patients


# ── assemble ─────────────────────────────────────────────────────────────────

def build_panel(output_path: Optional[str | Path] = None) -> None:
    rc_overrides = {
        **_PAPER_RC,
        "font.size": 17,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "axes.titlesize": 19,
    }
    plt.rcParams.update(rc_overrides)

    oof_df = _load_oof()

    fig = plt.figure(figsize=(12, 15.5))
    gs = gridspec.GridSpec(
        3, 2, figure=fig, hspace=0.45, wspace=0.38,
        left=0.13, right=0.97, top=0.96, bottom=0.05,
    )

    for row_i, (label, gen_dir) in enumerate(ROWS):
        print(f"[Panel] Row {row_i}: {label} ({gen_dir.name}) ...")
        gen_eval, gen_tiles, gen_patients = _load_row_data(gen_dir)
        threshold = gen_eval["threshold"]

        ax_roc = fig.add_subplot(gs[row_i, 0])
        ax_dots = fig.add_subplot(gs[row_i, 1])

        _panel_gen_roc(ax_roc, gen_tiles, oof_df)
        _panel_patient_dots(ax_dots, gen_patients, threshold)

        ax_roc.text(
            -0.34, 0.5, label, transform=ax_roc.transAxes,
            rotation=90, ha="center", va="center",
            fontsize=21, fontweight="bold",
        )

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
