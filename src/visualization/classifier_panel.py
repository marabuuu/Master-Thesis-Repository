#!/usr/bin/env python3
"""
Subtype Classifier Panel
========================

Publication-quality 2×2 panel:
  (A) ROC — per-fold curves + aggregate with bootstrap 95% CI
  (B) Precision-Recall — per-fold curves + aggregate with bootstrap 95% CI
  (C) Generated tiles — ROC vs real CV reference
  (D) Generated tiles — per-patient mean P(Basal) dot plot

Crameri batlow colourmap, subtype colours match dataset_statistics.py
(LumA = light green, Basal = dark red from the 5-class PAM50 palette).

Usage:
    python -m src.visualization.classifier_panel
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_hex

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# ── path setup ───────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]
_EXP = _REPO.parent / "experiments"
sys.path.insert(0, str(_REPO))

from src.visualization.training_plots import (
    _PAPER_RC,
    _strip_spines,
)

# ── fixed paths ──────────────────────────────────────────────────────────────
CV_DIR = _EXP / "20260618_subtype_classifier_cv"
CV_RESULTS = CV_DIR / "cv_results"
GEN_EVAL = CV_DIR / "generated_eval"
OUT = CV_DIR / "classifier_panel.png"

# ── colours ──────────────────────────────────────────────────────────────────
# Match the 5-class PAM50 palette from dataset_statistics / build_label_palette
C_BASAL = "#7c3847"
C_LUMA = "#cbe1b3"

try:
    from cmcrameri import cm as cmc
    _cmap = cmc.batlow
except ImportError:
    _cmap = plt.cm.viridis

FOLD_COLORS = [to_hex(_cmap(p)) for p in np.linspace(0.12, 0.88, 5)]
C_AGG = "0.15"
C_GEN = to_hex(_cmap(0.45))


# ── data loading ─────────────────────────────────────────────────────────────

def _load_oof() -> pd.DataFrame:
    return pd.read_parquet(CV_RESULTS / "all_oof_predictions.parquet")


def _load_cv_summary() -> dict:
    with open(CV_RESULTS / "cv_summary.json") as f:
        return json.load(f)


def _load_gen_eval() -> dict:
    with open(GEN_EVAL / "evaluation_metrics.json") as f:
        return json.load(f)


def _load_gen_tiles() -> pd.DataFrame:
    return pd.read_parquet(GEN_EVAL / "tile_predictions.parquet")


def _load_gen_patients() -> pd.DataFrame:
    return pd.read_parquet(GEN_EVAL / "per_patient_predictions.parquet")


# ── panel A: ROC — per-fold lines + aggregate CI ─────────────────────────────

def _panel_roc_cv(ax: plt.Axes, oof_df: pd.DataFrame, cv_summary: dict,
                  n_bootstrap: int = 1000, seed: int = 42) -> None:
    base_fpr = np.linspace(0, 1, 200)

    # Per-fold curves
    for fold_i in sorted(oof_df["fold"].unique()):
        fdf = oof_df[oof_df["fold"] == fold_i]
        fy = fdf["y_true"].to_numpy(dtype=np.int64)
        fp = fdf["p_pos"].to_numpy(dtype=np.float64)
        if len(np.unique(fy)) < 2:
            continue
        f_fpr, f_tpr, _ = roc_curve(fy, fp)
        f_auc = roc_auc_score(fy, fp)
        ax.plot(f_fpr, f_tpr, color=FOLD_COLORS[fold_i], lw=0.9, alpha=0.6,
                label=f"Fold {fold_i} = {f_auc:.3f}")

    # Bootstrap CI band
    y = oof_df["y_true"].to_numpy(dtype=np.int64)
    p = oof_df["p_pos"].to_numpy(dtype=np.float64)
    fpr, tpr, _ = roc_curve(y, p)
    auc_val = roc_auc_score(y, p)
    ci = cv_summary["bootstrap_ci"]["tile_roc_auc"]

    rng = np.random.RandomState(seed)
    pids = oof_df["patient_id"].unique()
    tpr_samples = []
    for _ in range(n_bootstrap):
        boot_pids = rng.choice(pids, size=len(pids), replace=True)
        bdf = pd.concat([oof_df[oof_df["patient_id"] == pid] for pid in boot_pids],
                        ignore_index=True)
        by = bdf["y_true"].to_numpy(dtype=np.int64)
        bp = bdf["p_pos"].to_numpy(dtype=np.float64)
        if len(np.unique(by)) < 2:
            continue
        b_fpr, b_tpr, _ = roc_curve(by, bp)
        tpr_samples.append(np.interp(base_fpr, b_fpr, b_tpr))

    tpr_stack = np.array(tpr_samples)
    ax.fill_between(base_fpr,
                    np.percentile(tpr_stack, 2.5, axis=0),
                    np.percentile(tpr_stack, 97.5, axis=0),
                    alpha=0.12, color=C_AGG, linewidth=0, zorder=0)

    # Aggregate curve
    ax.plot(fpr, tpr, color=C_AGG, lw=2.0, zorder=5,
            label=f"Aggregate = {auc_val:.3f} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")

    ax.plot([0, 1], [0, 1], ls="--", color="0.65", lw=0.7)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    _strip_spines(ax)


# ── panel B: PR — per-fold lines + aggregate CI ─────────────────────────────

def _panel_pr_cv(ax: plt.Axes, oof_df: pd.DataFrame, cv_summary: dict,
                 n_bootstrap: int = 1000, seed: int = 42) -> None:
    base_rec = np.linspace(0, 1, 200)

    # Per-fold curves
    for fold_i in sorted(oof_df["fold"].unique()):
        fdf = oof_df[oof_df["fold"] == fold_i]
        fy = fdf["y_true"].to_numpy(dtype=np.int64)
        fp = fdf["p_pos"].to_numpy(dtype=np.float64)
        if len(np.unique(fy)) < 2:
            continue
        f_prec, f_rec, _ = precision_recall_curve(fy, fp)
        f_ap = average_precision_score(fy, fp)
        ax.plot(f_rec, f_prec, color=FOLD_COLORS[fold_i], lw=0.9, alpha=0.6,
                label=f"Fold {fold_i} = {f_ap:.3f}")

    # Bootstrap CI band
    y = oof_df["y_true"].to_numpy(dtype=np.int64)
    p = oof_df["p_pos"].to_numpy(dtype=np.float64)
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)

    rng = np.random.RandomState(seed)
    pids = oof_df["patient_id"].unique()
    prec_samples = []
    for _ in range(n_bootstrap):
        boot_pids = rng.choice(pids, size=len(pids), replace=True)
        bdf = pd.concat([oof_df[oof_df["patient_id"] == pid] for pid in boot_pids],
                        ignore_index=True)
        by = bdf["y_true"].to_numpy(dtype=np.int64)
        bp = bdf["p_pos"].to_numpy(dtype=np.float64)
        if len(np.unique(by)) < 2:
            continue
        b_prec, b_rec, _ = precision_recall_curve(by, bp)
        prec_samples.append(np.interp(base_rec, b_rec[::-1], b_prec[::-1]))

    prec_stack = np.array(prec_samples)
    ax.fill_between(base_rec,
                    np.percentile(prec_stack, 2.5, axis=0),
                    np.percentile(prec_stack, 97.5, axis=0),
                    alpha=0.12, color=C_AGG, linewidth=0, zorder=0)

    # Aggregate curve
    ax.plot(rec, prec, color=C_AGG, lw=2.0, zorder=5,
            label=f"Aggregate = {ap:.3f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    _strip_spines(ax)


# ── panel C: generated ROC vs real reference ─────────────────────────────────

def _panel_gen_roc(ax: plt.Axes, gen_tiles: pd.DataFrame,
                   oof_df: pd.DataFrame) -> None:
    # Real CV reference (dashed)
    ry = oof_df["y_true"].to_numpy(dtype=np.int64)
    rp = oof_df["p_pos"].to_numpy(dtype=np.float64)
    r_fpr, r_tpr, _ = roc_curve(ry, rp)
    r_auc = roc_auc_score(ry, rp)
    ax.plot(r_fpr, r_tpr, color=C_AGG, lw=1.3, ls="--", alpha=0.5,
            label=f"Real (CV) = {r_auc:.3f}")

    # Generated ROC
    gy = gen_tiles["y_true"].to_numpy(dtype=np.int64)
    gp = gen_tiles["p_pos"].to_numpy(dtype=np.float64)
    g_fpr, g_tpr, _ = roc_curve(gy, gp)
    g_auc = roc_auc_score(gy, gp)
    ax.plot(g_fpr, g_tpr, color=C_GEN, lw=1.8,
            label=f"Generated = {g_auc:.3f}")

    ax.plot([0, 1], [0, 1], ls="--", color="0.65", lw=0.7)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    _strip_spines(ax)


# ── panel D: per-patient mean P(Basal) dot plot ─────────────────────────────

def _panel_patient_dots(ax: plt.Axes, gen_patients: pd.DataFrame,
                        threshold: float) -> None:
    basal = gen_patients[gen_patients["subtype"] == "Basal"]["mean_p_basal"].to_numpy()
    luma = gen_patients[gen_patients["subtype"] == "LumA"]["mean_p_basal"].to_numpy()

    jitter_b = np.random.RandomState(0).uniform(-0.12, 0.12, size=len(basal))
    jitter_l = np.random.RandomState(1).uniform(-0.12, 0.12, size=len(luma))

    ax.scatter(basal, 1.0 + jitter_b, c=C_BASAL, s=35, alpha=0.8,
               edgecolors="white", linewidths=0.4, zorder=5)
    ax.scatter(luma, 0.0 + jitter_l, c=C_LUMA, s=35, alpha=0.8,
               edgecolors="0.4", linewidths=0.4, zorder=5)

    ax.axvline(threshold, color="0.3", ls="--", lw=0.8, zorder=3)

    # Median markers
    ax.plot(np.median(basal), 1.0, marker="|", color="black", ms=14, mew=2, zorder=6)
    ax.plot(np.median(luma), 0.0, marker="|", color="black", ms=14, mew=2, zorder=6)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["LumA\n(n={})".format(len(luma)),
                        "Basal\n(n={})".format(len(basal))])
    ax.set_xlabel("Mean P(Basal) per patient")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.5, 1.5)

    # Accuracy annotations
    basal_acc = (basal >= threshold).mean()
    luma_acc = (luma < threshold).mean()
    ax.text(0.97, 1.35, f"{basal_acc:.0%}", transform=ax.get_yaxis_transform(),
            ha="right", va="top", fontsize=8, color=C_BASAL, fontweight="bold")
    ax.text(0.97, -0.35, f"{luma_acc:.0%}", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="0.35", fontweight="bold")

    _strip_spines(ax)


# ── assemble ─────────────────────────────────────────────────────────────────

def build_panel(
    output_path: Optional[str | Path] = None,
    n_bootstrap: int = 1000,
) -> None:
    plt.rcParams.update(_PAPER_RC)

    oof_df = _load_oof()
    cv_summary = _load_cv_summary()
    gen_eval = _load_gen_eval()
    gen_tiles = _load_gen_tiles()
    gen_patients = _load_gen_patients()
    threshold = gen_eval["threshold"]

    fig = plt.figure(figsize=(10, 9))

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.32, wspace=0.32,
                           left=0.08, right=0.97,
                           top=0.97, bottom=0.06)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    print("[Panel] A: ROC with per-fold lines + bootstrap CI ...")
    _panel_roc_cv(ax_a, oof_df, cv_summary, n_bootstrap=n_bootstrap)

    print("[Panel] B: Precision-Recall with per-fold lines + bootstrap CI ...")
    _panel_pr_cv(ax_b, oof_df, cv_summary, n_bootstrap=n_bootstrap)

    print("[Panel] C: Generated ROC vs real ...")
    _panel_gen_roc(ax_c, gen_tiles, oof_df)

    print("[Panel] D: Per-patient predictions ...")
    _panel_patient_dots(ax_d, gen_patients, threshold)

    out = Path(output_path) if output_path else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Panel] Saved → {out}")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    args = p.parse_args()
    build_panel(output_path=args.output, n_bootstrap=args.n_bootstrap)


if __name__ == "__main__":
    main()
