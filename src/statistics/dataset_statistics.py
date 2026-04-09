#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset Statistics
==================

Generates publication-quality dataset overview figures from a clinical CSV:

  - AJCC pathologic tumor stage distribution (bar chart)
  - Menopause status distribution (bar chart)
  - PAM50 subtype distribution (pie chart)
  - Kaplan-Meier overall survival curves stratified by PAM50 subtype

All plots use Crameri scientific colour maps.  PAM50 subtype colours are
derived from the same ``build_label_palette`` call used by the latent-space
visualisation pipeline, guaranteeing a consistent colour scheme across figures.

Usage (CLI via run_pipeline.py)
--------------------------------
    python run_pipeline.py --config src/config.yaml --stage dataset_statistics

Usage (programmatic)
---------------------
    from src.statistics.dataset_statistics import run_dataset_statistics
    run_dataset_statistics(config["dataset_statistics"])
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Lazy imports — graceful if heavy deps absent during import-time checks
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:  # pragma: no cover
    plt = None
    mticker = None
    HAS_MPL = False

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    HAS_LIFELINES = True
except ImportError:  # pragma: no cover
    KaplanMeierFitter = None
    logrank_test = None
    HAS_LIFELINES = False

try:
    from visualization.core import (
        CATEGORICAL_CMAP,
        SEQUENTIAL_CMAP,
        build_label_palette,
        get_crameri_cmap,
        save_figure,
        setup_style,
    )
except ImportError:
    from src.visualization.core import (  # type: ignore[import-not-found]
        CATEGORICAL_CMAP,
        SEQUENTIAL_CMAP,
        build_label_palette,
        get_crameri_cmap,
        save_figure,
        setup_style,
    )


# ---------------------------------------------------------------------------
# Canonical PAM50 subtype ordering (determines colour assignment order)
# ---------------------------------------------------------------------------
_PAM50_ORDER = ["Basal", "Basal-like", "Her2", "HER2", "LumA", "LumB", "Normal"]


def _canonical_order(subtypes: List[str]) -> List[str]:
    """Sort subtypes: known PAM50 first in canonical order, then alphabetical."""
    known = [s for s in _PAM50_ORDER if s in subtypes]
    other = sorted(s for s in subtypes if s not in _PAM50_ORDER)
    return known + other


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_clinical_data(csv_path: str, patient_col: str) -> pd.DataFrame:
    """Load clinical CSV, deduplicate on patient column, report shape."""
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[DatasetStats] Loaded {len(df)} rows from {Path(csv_path).name}")

    if patient_col in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=[patient_col])
        if len(df) < before:
            print(f"[DatasetStats] Deduplicated {before - len(df)} duplicate patient rows")

    return df


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_tumor_stage(
    df: pd.DataFrame,
    stage_col: str,
    output_dir: Path,
    cmap_name: str = SEQUENTIAL_CMAP,
    figsize: tuple = (10, 6),
    dpi: int = 200,
) -> None:
    """Horizontal bar chart of AJCC pathologic tumor stage distribution."""
    if not HAS_MPL:
        return

    if stage_col not in df.columns:
        print(f"[DatasetStats][WARN] Column '{stage_col}' not found — skipping tumor stage plot")
        return

    stage_values = (
        df[stage_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    invalid_values = {"[Not Available]", "[Discrepancy]", "nan", ""}
    stage_values = stage_values[~stage_values.isin(invalid_values)]
    counts = stage_values.value_counts().sort_index()

    if counts.empty:
        print("[DatasetStats][WARN] No valid tumor stage values found — skipping")
        return

    assert plt is not None
    assert mticker is not None

    cmap = get_crameri_cmap(cmap_name)
    colours = [cmap(i) for i in np.linspace(0.15, 0.85, len(counts))]
    count_values = counts.to_numpy(dtype=float)
    max_count = float(count_values.max()) if count_values.size else 0.0

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(counts.index.tolist(), count_values, color=colours, edgecolor="white", linewidth=0.5)

    # Annotate counts
    for bar, val in zip(bars, count_values):
        ax.text(
            bar.get_width() + max_count * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(int(val)),
            va="center", ha="left", fontsize=9,
        )

    ax.set_xlabel("Number of patients", fontsize=11)
    ax.set_title("AJCC Pathologic Tumor Stage Distribution", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    fig.tight_layout()
    save_figure(fig, output_dir / "tumor_stage_distribution.png", dpi=dpi)


def plot_menopause_status(
    df: pd.DataFrame,
    menopause_col: str,
    output_dir: Path,
    cmap_name: str = SEQUENTIAL_CMAP,
    figsize: tuple = (9, 5),
    dpi: int = 200,
) -> None:
    """Horizontal bar chart of menopause status distribution."""
    if not HAS_MPL:
        return

    if menopause_col not in df.columns:
        print(f"[DatasetStats][WARN] Column '{menopause_col}' not found — skipping menopause plot")
        return

    menopause_values = (
        df[menopause_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    invalid_values = {"[Not Available]", "[Unknown]", "nan", ""}
    menopause_values = menopause_values[~menopause_values.isin(invalid_values)]
    counts = menopause_values.value_counts()

    if counts.empty:
        print("[DatasetStats][WARN] No valid menopause values found — skipping")
        return

    assert plt is not None
    assert mticker is not None

    # Wrap long labels for readability
    wrapped = [lbl.replace(" or ", "\nor ") for lbl in counts.index]

    cmap = get_crameri_cmap(cmap_name)
    colours = [cmap(i) for i in np.linspace(0.2, 0.8, len(counts))]
    count_values = counts.to_numpy(dtype=float)
    max_count = float(count_values.max()) if count_values.size else 0.0

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(wrapped, count_values, color=colours, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, count_values):
        ax.text(
            bar.get_width() + max_count * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(int(val)),
            va="center", ha="left", fontsize=9,
        )

    ax.set_xlabel("Number of patients", fontsize=11)
    ax.set_title("Menopause Status Distribution", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    fig.tight_layout()
    save_figure(fig, output_dir / "menopause_status_distribution.png", dpi=dpi)


def plot_subtype_distribution(
    df: pd.DataFrame,
    subtype_col: str,
    output_dir: Path,
    cmap_name: str = CATEGORICAL_CMAP,
    figsize: tuple = (8, 8),
    dpi: int = 200,
) -> Dict[str, str]:
    """Pie chart of PAM50 subtype distribution.  Returns the subtype→colour palette
    so it can be reused by the Kaplan-Meier plot for a consistent colour scheme."""
    if subtype_col not in df.columns:
        print(f"[DatasetStats][WARN] Column '{subtype_col}' not found — skipping subtype pie")
        return {}

    counts = df[subtype_col].dropna().astype(str).str.strip()
    counts = counts[counts != ""].value_counts()

    if counts.empty:
        print("[DatasetStats][WARN] No valid subtype values found — skipping")
        return {}

    if not HAS_MPL:
        return {}

    assert plt is not None

    # Canonical ordering for consistent colour assignment (matches latent-space plots)
    ordered = _canonical_order(list(counts.index))
    counts = counts.reindex(ordered).dropna()

    palette = build_label_palette(np.array(ordered), cmap_name=cmap_name)
    colours = [palette[s] for s in counts.index]
    count_values = counts.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    pie_result = ax.pie(
        count_values,
        labels=None,
        colors=colours,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        pctdistance=0.75,
    )
    wedges = pie_result[0]
    autotexts = pie_result[2] if len(pie_result) > 2 else []
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
        at.set_color("white")

    # Legend with absolute counts
    legend_labels = [f"{s}  (n={counts[s]:,})" for s in counts.index]
    ax.legend(
        wedges, legend_labels,
        title="PAM50 Subtype",
        title_fontsize=10,
        fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(len(counts), 3),
        frameon=False,
    )
    ax.set_title("PAM50 Subtype Distribution", fontsize=13, fontweight="bold", pad=20)

    fig.tight_layout()
    save_figure(fig, output_dir / "pam50_subtype_distribution.png", dpi=dpi)

    return palette


def plot_kaplan_meier_by_subtype(
    df: pd.DataFrame,
    subtype_col: str,
    os_time_col: str,
    os_event_col: str,
    output_dir: Path,
    palette: Optional[Dict[str, str]] = None,
    cmap_name: str = CATEGORICAL_CMAP,
    min_patients: int = 10,
    figsize: tuple = (10, 7),
    dpi: int = 200,
) -> None:
    """Kaplan-Meier overall survival curves, one per PAM50 subtype.

    Performs pairwise log-rank tests between subtypes and annotates the plot
    with a summary p-value table if ≥2 subtypes are present.
    """
    if not HAS_LIFELINES:
        print("[DatasetStats][WARN] lifelines not installed — skipping Kaplan-Meier plot")
        return

    if not HAS_MPL:
        return

    assert plt is not None
    assert mticker is not None
    assert KaplanMeierFitter is not None
    assert logrank_test is not None

    required = [subtype_col, os_time_col, os_event_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[DatasetStats][WARN] Missing columns {missing} — skipping KM plot")
        return

    km_df = df[[subtype_col, os_time_col, os_event_col]].copy()
    km_df[os_time_col] = pd.to_numeric(km_df[os_time_col], errors="coerce")
    km_df[os_event_col] = pd.to_numeric(km_df[os_event_col], errors="coerce")
    km_df = km_df.dropna()

    # Convert time from days to years for readability
    km_df["os_years"] = km_df[os_time_col] / 365.25

    subtypes = _canonical_order(list(km_df[subtype_col].unique()))
    subtypes = [s for s in subtypes if (km_df[subtype_col] == s).sum() >= min_patients]

    if len(subtypes) < 2:
        print(f"[DatasetStats][WARN] Fewer than 2 subtypes with ≥{min_patients} patients — skipping KM")
        return

    if palette is None:
        palette = build_label_palette(np.array(subtypes), cmap_name=cmap_name)

    fig, ax = plt.subplots(figsize=figsize)

    kmf_fits = {}
    for subtype in subtypes:
        mask = km_df[subtype_col] == subtype
        sub = km_df[mask]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["os_years"], event_observed=sub[os_event_col], label=subtype)
        kmf_fits[subtype] = kmf

        colour = palette.get(subtype, "#888888")
        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            color=colour,
            linewidth=2.5,
            ci_alpha=0.12,
        )

    # Pairwise log-rank p-values (all pairs, summarised below plot)
    if len(subtypes) >= 2:
        pairs = []
        for i, s1 in enumerate(subtypes):
            for s2 in subtypes[i + 1:]:
                mask1 = km_df[subtype_col] == s1
                mask2 = km_df[subtype_col] == s2
                t1 = km_df.loc[mask1, "os_years"]
                e1 = km_df.loc[mask1, os_event_col]
                t2 = km_df.loc[mask2, "os_years"]
                e2 = km_df.loc[mask2, os_event_col]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = logrank_test(t1, t2, event_observed_A=e1, event_observed_B=e2)
                pval = result.p_value
                sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
                pairs.append(f"{s1} vs {s2}: p={pval:.4f} {sig}")

        pval_text = "Log-rank tests:\n" + "\n".join(pairs)
        ax.text(
            0.98, 0.02, pval_text,
            transform=ax.transAxes,
            fontsize=8, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
        )

    # Patient counts in legend
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for lbl in labels:
        n = (km_df[subtype_col] == lbl).sum()
        new_labels.append(f"{lbl}  (n={n:,})")
    ax.legend(handles, new_labels, title="PAM50 Subtype", fontsize=9,
              title_fontsize=10, frameon=True, framealpha=0.9)

    ax.set_xlabel("Time (years)", fontsize=11)
    ax.set_ylabel("Overall Survival Probability", fontsize=11)
    ax.set_title("Overall Survival by PAM50 Subtype (Kaplan-Meier)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    fig.tight_layout()
    save_figure(fig, output_dir / "kaplan_meier_overall_survival_by_subtype.png", dpi=dpi)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_AVAILABLE_PLOTS = {
    "tumor_stage_distribution",
    "menopause_distribution",
    "subtype_distribution",
    "kaplan_meier_by_subtype",
}


def run_dataset_statistics(cfg: dict, verbose: bool = True) -> None:
    """Run all configured dataset statistics plots.

    Parameters
    ----------
    cfg : dict
        The ``dataset_statistics`` section from ``config.yaml``.
    verbose : bool
        Print progress messages.
    """
    if not HAS_MPL:
        raise ImportError("matplotlib is required. Install it: pip install matplotlib")

    setup_style()

    # ── Config keys ──────────────────────────────────────────────────────────
    clinical_csv  = cfg.get("clinical_csv") or cfg.get("csv_path")
    if not clinical_csv:
        raise ValueError("dataset_statistics: 'clinical_csv' path is required")

    output_dir    = Path(cfg.get("output_dir", "./experiments/dataset_statistics"))
    patient_col   = cfg.get("patient_col", "bcr_patient_barcode")
    subtype_col   = cfg.get("subtype_col", "Majority_Subtype_mRNA")
    stage_col     = cfg.get("stage_col", "ajcc_pathologic_tumor_stage")
    menopause_col = cfg.get("menopause_col", "menopause_status")
    os_time_col   = cfg.get("os_time_col", "OS.time")
    os_event_col  = cfg.get("os_event_col", "OS")
    min_patients  = int(cfg.get("min_patients_km", 10))

    requested     = set(cfg.get("plots", list(_AVAILABLE_PLOTS)))
    cmap_cat      = cfg.get("cmap_categorical", CATEGORICAL_CMAP)
    cmap_seq      = cfg.get("cmap_sequential",  SEQUENTIAL_CMAP)
    figsize_dist  = tuple(cfg.get("figsize_distributions", [10, 6]))
    figsize_pie   = tuple(cfg.get("figsize_pie",  [8, 8]))
    figsize_km    = tuple(cfg.get("figsize_km",   [10, 7]))
    dpi           = int(cfg.get("dpi", 200))

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_clinical_data(str(clinical_csv), patient_col)

    if verbose:
        n_sub = df[subtype_col].nunique() if subtype_col in df.columns else 0
        print(f"[DatasetStats] Patients: {len(df):,}  |  Subtypes: {n_sub}  |  Output: {output_dir}")

    # ── Plot: tumor stage ─────────────────────────────────────────────────────
    if "tumor_stage_distribution" in requested:
        plot_tumor_stage(df, stage_col, output_dir,
                         cmap_name=cmap_seq, figsize=figsize_dist, dpi=dpi)

    # ── Plot: menopause ───────────────────────────────────────────────────────
    if "menopause_distribution" in requested:
        plot_menopause_status(df, menopause_col, output_dir,
                              cmap_name=cmap_seq, figsize=figsize_dist, dpi=dpi)

    # ── Plot: subtype pie (returns palette for KM re-use) ─────────────────────
    palette: Dict[str, str] = {}
    if "subtype_distribution" in requested:
        palette = plot_subtype_distribution(df, subtype_col, output_dir,
                                            cmap_name=cmap_cat, figsize=figsize_pie, dpi=dpi)

    # ── Plot: Kaplan-Meier ────────────────────────────────────────────────────
    if "kaplan_meier_by_subtype" in requested:
        # Use the same palette generated by the pie chart; if the pie wasn't
        # requested, build a fresh palette here so colours still match.
        if not palette and subtype_col in df.columns:
            subtypes = df[subtype_col].dropna().unique().tolist()
            palette = build_label_palette(np.array(_canonical_order(subtypes)), cmap_name=cmap_cat)
        plot_kaplan_meier_by_subtype(
            df, subtype_col, os_time_col, os_event_col,
            output_dir, palette=palette,
            cmap_name=cmap_cat, min_patients=min_patients,
            figsize=figsize_km, dpi=dpi,
        )

    if verbose:
        print(f"[DatasetStats] All plots saved to {output_dir}")


def main() -> None:
    """CLI entry point — reads config.yaml and runs the dataset_statistics stage."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Generate dataset statistics plots")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if "dataset_statistics" not in config:
        raise ValueError("No 'dataset_statistics' section found in config.yaml")

    run_dataset_statistics(config["dataset_statistics"])


if __name__ == "__main__":
    main()
