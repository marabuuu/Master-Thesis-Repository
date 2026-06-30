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
    from src.visualization.dataset_statistics import run_dataset_statistics
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
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    HAS_LIFELINES = True
except ImportError:  # pragma: no cover
    KaplanMeierFitter = None
    logrank_test = None
    multivariate_logrank_test = None
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
    subtype_col: str,
    output_dir: Path,
    palette: Optional[Dict[str, str]] = None,
    cmap_name: str = CATEGORICAL_CMAP,
    figsize: tuple = (10, 6),
    dpi: int = 200,
) -> None:
    """Stacked horizontal bar chart of AJCC stage split by PAM50 subtype."""
    if not HAS_MPL:
        return

    if stage_col not in df.columns:
        print(f"[DatasetStats][WARN] Column '{stage_col}' not found — skipping tumor stage plot")
        return

    if subtype_col not in df.columns:
        print(f"[DatasetStats][WARN] Column '{subtype_col}' not found — skipping tumor stage plot")
        return

    stage_values = df[stage_col].dropna().astype(str).str.strip()
    invalid_values = {"[Not Available]", "[Discrepancy]", "nan", ""}
    stage_values = stage_values[~stage_values.isin(invalid_values)]

    subtype_values = df[subtype_col].fillna("").astype(str).str.strip()
    subtype_values = subtype_values[~subtype_values.isin({"", "nan"})]

    valid_idx = stage_values.index.intersection(subtype_values.index)
    stage_values = stage_values.loc[valid_idx]
    subtype_values = subtype_values.loc[valid_idx]

    if stage_values.empty or subtype_values.empty:
        print("[DatasetStats][WARN] No valid stage/subtype pairs found — skipping")
        return

    stage_by_subtype = pd.crosstab(stage_values, subtype_values)
    stage_by_subtype = stage_by_subtype.sort_index()

    ordered_subtypes = _canonical_order(list(stage_by_subtype.columns))
    stage_by_subtype = stage_by_subtype.reindex(columns=ordered_subtypes, fill_value=0)

    if stage_by_subtype.empty:
        print("[DatasetStats][WARN] No valid tumor stage values found — skipping")
        return

    assert plt is not None
    assert mticker is not None

    if palette is None:
        palette = build_label_palette(np.array(ordered_subtypes), cmap_name=cmap_name)

    totals = stage_by_subtype.sum(axis=1).to_numpy(dtype=float)
    max_count = float(totals.max()) if totals.size else 0.0

    fig, ax = plt.subplots(figsize=figsize)
    left = np.zeros(len(stage_by_subtype), dtype=float)
    for subtype in ordered_subtypes:
        vals = stage_by_subtype[subtype].to_numpy(dtype=float)
        if not np.any(vals):
            continue
        ax.barh(
            stage_by_subtype.index.tolist(),
            vals,
            left=left,
            color=palette.get(subtype, "#888888"),
            edgecolor="white",
            linewidth=0.5,
            label=subtype,
        )
        left += vals

    for i, total in enumerate(totals):
        ax.text(
            total + max_count * 0.01,
            i,
            str(int(total)),
            va="center", ha="left", fontsize=9,
        )

    ax.set_xlabel("Number of patients", fontsize=11)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.invert_yaxis()
    ax.legend(title="PAM50 Subtype", fontsize=9, title_fontsize=10, frameon=False,
              bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    save_figure(fig, output_dir / "tumor_stage_distribution.png", dpi=dpi)


def plot_menopause_status(
    df: pd.DataFrame,
    menopause_col: str,
    subtype_col: str,
    output_dir: Path,
    palette: Optional[Dict[str, str]] = None,
    cmap_name: str = CATEGORICAL_CMAP,
    figsize: tuple = (9, 5),
    dpi: int = 200,
) -> None:
    """Stacked horizontal bar chart of menopause status split by PAM50 subtype."""
    if not HAS_MPL:
        return

    if menopause_col not in df.columns:
        print(f"[DatasetStats][WARN] Column '{menopause_col}' not found — skipping menopause plot")
        return

    if subtype_col not in df.columns:
        print(f"[DatasetStats][WARN] Column '{subtype_col}' not found — skipping menopause plot")
        return

    menopause_values = df[menopause_col].dropna().astype(str).str.strip()
    invalid_values = {"[Not Available]", "[Unknown]", "nan", ""}
    menopause_values = menopause_values[~menopause_values.isin(invalid_values)]

    subtype_values = df[subtype_col].fillna("").astype(str).str.strip()
    subtype_values = subtype_values[~subtype_values.isin({"", "nan"})]

    valid_idx = menopause_values.index.intersection(subtype_values.index)
    menopause_values = menopause_values.loc[valid_idx]
    subtype_values = subtype_values.loc[valid_idx]

    if menopause_values.empty or subtype_values.empty:
        print("[DatasetStats][WARN] No valid menopause/subtype pairs found — skipping")
        return

    meno_by_subtype = pd.crosstab(menopause_values, subtype_values)
    ordered_subtypes = _canonical_order(list(meno_by_subtype.columns))
    meno_by_subtype = meno_by_subtype.reindex(columns=ordered_subtypes, fill_value=0)

    if meno_by_subtype.empty:
        print("[DatasetStats][WARN] No valid menopause values found — skipping")
        return

    assert plt is not None
    assert mticker is not None

    # Wrap long labels for readability
    wrapped_labels = [lbl.replace(" or ", "\nor ") for lbl in meno_by_subtype.index]

    if palette is None:
        palette = build_label_palette(np.array(ordered_subtypes), cmap_name=cmap_name)

    totals = meno_by_subtype.sum(axis=1).to_numpy(dtype=float)
    max_count = float(totals.max()) if totals.size else 0.0

    fig, ax = plt.subplots(figsize=figsize)
    left = np.zeros(len(meno_by_subtype), dtype=float)
    for subtype in ordered_subtypes:
        vals = meno_by_subtype[subtype].to_numpy(dtype=float)
        if not np.any(vals):
            continue
        ax.barh(
            wrapped_labels,
            vals,
            left=left,
            color=palette.get(subtype, "#888888"),
            edgecolor="white",
            linewidth=0.5,
            label=subtype,
        )
        left += vals

    for i, total in enumerate(totals):
        ax.text(
            total + max_count * 0.01,
            i,
            str(int(total)),
            va="center", ha="left", fontsize=9,
        )

    ax.set_xlabel("Number of patients", fontsize=11)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    ax.legend(title="PAM50 Subtype", fontsize=9, title_fontsize=10, frameon=False)

    fig.tight_layout()
    save_figure(fig, output_dir / "menopause_status_distribution.png", dpi=dpi)


def plot_subtype_distribution(
    df: pd.DataFrame,
    subtype_col: str,
    output_dir: Path,
    cmap_name: str = CATEGORICAL_CMAP,
    palette: Optional[Dict[str, str]] = None,
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

    if palette is None:
        palette = build_label_palette(np.array(ordered), cmap_name=cmap_name)
    colours = [palette.get(s, "#888888") for s in counts.index]
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
        at.set_fontsize(14)
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

    # ── Layout: KM curves (top) + pairwise p-value table (bottom) ────────────
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.08)
    ax       = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off")

    # ── Fit and plot KM curves (no CI shading) ────────────────────────────────
    for subtype in subtypes:
        mask   = km_df[subtype_col] == subtype
        sub    = km_df[mask]
        kmf    = KaplanMeierFitter()
        kmf.fit(sub["os_years"], event_observed=sub[os_event_col], label=subtype)
        kmf.plot_survival_function(
            ax=ax,
            ci_show=False,
            color=palette.get(subtype, "#888888"),
            linewidth=2.5,
        )

    # ── Patient counts in legend ──────────────────────────────────────────────
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [f"{lbl}  (n={(km_df[subtype_col] == lbl).sum():,})" for lbl in labels]
    ax.legend(handles, new_labels, title="PAM50 Subtype", fontsize=9,
              title_fontsize=10, frameon=False,
              bbox_to_anchor=(1.02, 1), loc="upper left")

    ax.set_xlabel("Time (years)", fontsize=11)
    ax.set_ylabel("Overall Survival Probability", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # ── Overall multivariate log-rank test ────────────────────────────────────
    assert multivariate_logrank_test is not None
    mlr      = multivariate_logrank_test(km_df["os_years"], km_df[subtype_col], km_df[os_event_col])
    overall_p = mlr.p_value
    p_str    = "p < 0.001" if overall_p < 0.001 else f"p = {overall_p:.3f}"
    ax.text(
        0.98, 0.98, f"Overall log-rank {p_str}",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="#aaaaaa"),
    )

    # ── Pairwise log-rank tests ───────────────────────────────────────────────
    assert logrank_test is not None
    pairwise: dict = {}
    for i, s1 in enumerate(subtypes):
        for s2 in subtypes[i + 1:]:
            m1  = km_df[subtype_col] == s1
            m2  = km_df[subtype_col] == s2
            res = logrank_test(
                km_df.loc[m1, "os_years"], km_df.loc[m2, "os_years"],
                event_observed_A=km_df.loc[m1, os_event_col],
                event_observed_B=km_df.loc[m2, os_event_col],
            )
            pairwise[(s1, s2)] = res.p_value

    def _fmt_p(p: float) -> str:
        if p < 0.001:
            return "<.001"
        if p < 0.01:
            return f"{p:.3f}"
        return f"{p:.3f}"

    # Lower-triangular matrix: rows = subtypes[1:], cols = subtypes[:-1]
    row_labels = subtypes[1:]
    col_labels = subtypes[:-1]
    cell_text  = []
    for s_row in row_labels:
        row = []
        for s_col in col_labels:
            key = (s_col, s_row) if (s_col, s_row) in pairwise else (s_row, s_col)
            if s_col == s_row or key not in pairwise:
                row.append("")
            else:
                ri = subtypes.index(s_row)
                ci = subtypes.index(s_col)
                row.append(_fmt_p(pairwise[key]) if ci < ri else "")
        cell_text.append(row)

    tbl = ax_table.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.25)

    # Style: header row and row-label column
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0 or c == -1:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("white")

    ax_table.set_title("Pairwise log-rank p-values", fontsize=9, pad=4)

    fig.tight_layout()
    save_figure(fig, output_dir / "kaplan_meier_overall_survival_by_subtype.png", dpi=dpi)


def plot_combined_overview(
    df: pd.DataFrame,
    subtype_col: str,
    stage_col: str,
    os_time_col: str,
    os_event_col: str,
    output_dir: Path,
    palette: Optional[Dict[str, str]] = None,
    cmap_name: str = CATEGORICAL_CMAP,
    min_patients: int = 10,
    figsize: tuple = (14, 10),
    dpi: int = 200,
) -> None:
    """Three-panel overview: PAM50 pie (left) | KM survival (right-top) | AJCC stage (right-bottom).

    A single shared PAM50 colour legend is placed below all panels.
    Log-rank test annotations are intentionally omitted.
    Font sizes are tuned for inclusion in a thesis/report.
    """
    if not HAS_MPL:
        return

    assert plt is not None
    assert mticker is not None

    if subtype_col not in df.columns:
        print(f"[DatasetStats][WARN] '{subtype_col}' not found — skipping combined overview")
        return

    # ── Build shared palette ────────────────────────────────────────────────
    subtype_vals = df[subtype_col].dropna().astype(str).str.strip()
    subtype_vals = subtype_vals[subtype_vals != ""]
    ordered_subtypes = _canonical_order(subtype_vals.unique().tolist())
    if palette is None:
        palette = build_label_palette(np.array(ordered_subtypes), cmap_name=cmap_name)

    TITLE_FS  = 14
    LABEL_FS  = 12
    TICK_FS   = 11
    LEGEND_FS = 11

    fig, axes = plt.subplot_mosaic(
        [["pie", "km"], ["pie", "stage"]],
        figsize=figsize,
    )
    ax_pie   = axes["pie"]
    ax_km    = axes["km"]
    ax_stage = axes["stage"]

    # ══ Panel A: PAM50 subtype pie ═══════════════════════════════════════════
    counts = subtype_vals.value_counts()
    ordered_for_pie = _canonical_order(list(counts.index))
    counts = counts.reindex(ordered_for_pie).dropna()
    colours = [palette.get(s, "#888888") for s in counts.index]

    wedges, _, autotexts = ax_pie.pie(
        counts.to_numpy(dtype=float),
        labels=None,
        colors=colours,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(TICK_FS)
        at.set_fontweight("bold")
        at.set_color("white")
    # no title

    # ══ Panel B: Kaplan-Meier (no log-rank annotations) ══════════════════════
    if HAS_LIFELINES and os_time_col in df.columns and os_event_col in df.columns:
        assert KaplanMeierFitter is not None

        km_df = df[[subtype_col, os_time_col, os_event_col]].copy()
        km_df[os_time_col]  = pd.to_numeric(km_df[os_time_col],  errors="coerce")
        km_df[os_event_col] = pd.to_numeric(km_df[os_event_col], errors="coerce")
        km_df = km_df.dropna()
        km_df["os_years"] = km_df[os_time_col] / 365.25

        km_subtypes = _canonical_order(list(km_df[subtype_col].unique()))
        km_subtypes = [s for s in km_subtypes if (km_df[subtype_col] == s).sum() >= min_patients]

        for subtype in km_subtypes:
            mask = km_df[subtype_col] == subtype
            sub  = km_df[mask]
            kmf  = KaplanMeierFitter()
            kmf.fit(sub["os_years"], event_observed=sub[os_event_col], label=subtype)
            kmf.plot_survival_function(
                ax=ax_km,
                ci_show=False,
                color=palette.get(subtype, "#888888"),
                linewidth=2.0,
            )

        if multivariate_logrank_test is not None:
            mlr_ov = multivariate_logrank_test(
                km_df["os_years"], km_df[subtype_col], km_df[os_event_col]
            )
            ov_p   = mlr_ov.p_value
            ov_str = "p < 0.001" if ov_p < 0.001 else f"p = {ov_p:.3f}"
            ax_km.text(
                0.97, 0.97, f"Log-rank\n{ov_str}",
                transform=ax_km.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="#aaaaaa"),
            )
    else:
        ax_km.text(0.5, 0.5, "OS data not available", ha="center", va="center",
                   transform=ax_km.transAxes, fontsize=LABEL_FS, color="gray")

    ax_km.set_xlabel("Time (years)", fontsize=LABEL_FS)
    ax_km.set_ylabel("Overall Survival Probability", fontsize=LABEL_FS)
    ax_km.set_ylim(0, 1.05)
    ax_km.set_xlim(left=0)
    ax_km.tick_params(labelsize=TICK_FS)
    ax_km.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    if ax_km.get_legend():
        ax_km.get_legend().remove()

    # ══ Panel C: AJCC tumor stage horizontal stacked bar ═════════════════════
    if stage_col in df.columns:
        stage_values = df[stage_col].dropna().astype(str).str.strip()
        invalid_values = {"[Not Available]", "[Discrepancy]", "nan", ""}
        stage_values = stage_values[~stage_values.isin(invalid_values)]

        sub_vals_stage = df[subtype_col].fillna("").astype(str).str.strip()
        sub_vals_stage = sub_vals_stage[~sub_vals_stage.isin({"", "nan"})]

        valid_idx      = stage_values.index.intersection(sub_vals_stage.index)
        stage_values   = stage_values.loc[valid_idx]
        sub_vals_stage = sub_vals_stage.loc[valid_idx]

        if not stage_values.empty:
            stage_by_subtype = pd.crosstab(stage_values, sub_vals_stage)
            stage_by_subtype = stage_by_subtype.sort_index()
            ordered_st = _canonical_order(list(stage_by_subtype.columns))
            stage_by_subtype = stage_by_subtype.reindex(columns=ordered_st, fill_value=0)

            totals    = stage_by_subtype.sum(axis=1).to_numpy(dtype=float)
            max_count = float(totals.max()) if totals.size else 0.0
            left      = np.zeros(len(stage_by_subtype), dtype=float)

            for subtype in ordered_st:
                vals = stage_by_subtype[subtype].to_numpy(dtype=float)
                if not np.any(vals):
                    continue
                ax_stage.barh(
                    stage_by_subtype.index.tolist(),
                    vals,
                    left=left,
                    color=palette.get(subtype, "#888888"),
                    edgecolor="white",
                    linewidth=0.5,
                )
                left += vals

            for i, total in enumerate(totals):
                ax_stage.text(
                    total + max_count * 0.01, i,
                    str(int(total)),
                    va="center", ha="left", fontsize=TICK_FS,
                )
        else:
            ax_stage.text(0.5, 0.5, "Stage data not available", ha="center", va="center",
                          transform=ax_stage.transAxes, fontsize=LABEL_FS, color="gray")
    else:
        ax_stage.text(0.5, 0.5, "Stage data not available", ha="center", va="center",
                      transform=ax_stage.transAxes, fontsize=LABEL_FS, color="gray")

    ax_stage.set_xlabel("Number of patients", fontsize=LABEL_FS)
    ax_stage.tick_params(labelsize=TICK_FS)
    ax_stage.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_stage.invert_yaxis()

    # ── Single shared legend below all panels ──────────────────────────────
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=palette.get(s, "#888888"),
              label=f"{s}  (n={int(counts.get(s, 0)):,})")
        for s in ordered_subtypes
        if s in palette
    ]
    fig.legend(
        handles=legend_handles,
        title="PAM50 Subtype",
        title_fontsize=LEGEND_FS,
        fontsize=LEGEND_FS,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(ordered_subtypes), 6),
        frameon=False,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    save_figure(fig, output_dir / "combined_overview.png", dpi=dpi)
    print("[DatasetStats] Saved combined overview → combined_overview.png")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_AVAILABLE_PLOTS = {
    "tumor_stage_distribution",
    "menopause_distribution",
    "subtype_distribution",
    "kaplan_meier_by_subtype",
    "combined_overview",
}


def _fmt_median_iqr(series: pd.Series, decimals: int = 1) -> str:
    """Format numeric series as median [Q1, Q3]."""
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return "NA"
    q1 = float(vals.quantile(0.25))
    med = float(vals.median())
    q3 = float(vals.quantile(0.75))
    return f"{med:.{decimals}f} [{q1:.{decimals}f}, {q3:.{decimals}f}]"


def _fmt_count_pct(mask: pd.Series, denom: int) -> str:
    """Format count and percentage for a boolean mask."""
    count = int(mask.sum())
    if denom <= 0:
        return f"{count} (NA)"
    pct = 100.0 * count / denom
    return f"{count} ({pct:.1f}%)"


def write_patient_characteristics_table(
    df: pd.DataFrame,
    output_dir: Path,
    age_col: str = "age_at_initial_pathologic_diagnosis",
    followup_col: str = "OS.time",
    sex_col: str = "gender",
    status_col: str = "vital_status",
) -> None:
    """Write a patient characteristics table to LaTeX (.tex) and Markdown (.md).

    Continuous variables: median [Q1, Q3].
    Categorical variables: n (%).

    The LaTeX output uses the ``booktabs`` package (\\toprule / \\midrule /
    \\bottomrule) and escapes ``%`` as ``\\%`` so the file compiles without
    errors.  Categorical sub-rows are indented with ``\\quad``.
    BMI is intentionally omitted — it is not present in the TCGA clinical data.
    """
    if df.empty:
        print("[DatasetStats][WARN] Empty dataframe — skipping patient characteristics table")
        return

    n_total = len(df)

    # ------------------------------------------------------------------
    # Build a list of (latex_char, latex_val, md_char, md_val) tuples.
    # latex_char / latex_val go directly into the .tex file (already escaped).
    # md_char / md_val go into the Markdown table.
    # A tuple with latex_char == "" inserts an \addlinespace row separator.
    # ------------------------------------------------------------------
    Entry = Dict[str, str]
    entries: List[Entry] = []

    def _separator() -> Entry:
        return {"latex_char": "", "latex_val": "", "md_char": "", "md_val": ""}

    def _row(char: str, val: str) -> Entry:
        """Plain row — escape % for LaTeX, keep raw for Markdown."""
        return {
            "latex_char": char,
            "latex_val": val.replace("%", r"\%"),
            "md_char": char,
            "md_val": val,
        }

    def _header(char: str) -> Entry:
        """Category header row: bold in LaTeX, bold in Markdown, no value."""
        return {
            "latex_char": r"\textit{" + char + "}",
            "latex_val": "",
            "md_char": f"**{char}**",
            "md_val": "",
        }

    def _subrow(char: str, val: str) -> Entry:
        """Indented sub-row inside a category block."""
        return {
            "latex_char": r"\quad " + char,
            "latex_val": val.replace("%", r"\%"),
            "md_char": f"&nbsp;&nbsp;{char}",
            "md_val": val,
        }

    # ── Age at diagnosis ────────────────────────────────────────────────
    if age_col in df.columns:
        entries.append(_row("Age at diagnosis, years", _fmt_median_iqr(df[age_col], decimals=1)))
    else:
        print(f"[DatasetStats][INFO] Column '{age_col}' not found — age row will show NA")
        entries.append(_row("Age at diagnosis, years", "not available"))

    # ── Follow-up from diagnosis ────────────────────────────────────────
    if followup_col in df.columns:
        fu_years = pd.to_numeric(df[followup_col], errors="coerce") / 365.25
        entries.append(_row("Follow-up from diagnosis, years", _fmt_median_iqr(fu_years, decimals=1)))
    else:
        print(f"[DatasetStats][INFO] Column '{followup_col}' not found — follow-up row will show NA")
        entries.append(_row("Follow-up from diagnosis, years", "not available"))

    entries.append(_separator())

    # ── Sex ─────────────────────────────────────────────────────────────
    if sex_col in df.columns:
        sex_vals = df[sex_col].fillna("").astype(str).str.strip()
        female_mask  = sex_vals.str.upper().isin({"FEMALE", "F"})
        male_mask    = sex_vals.str.upper().isin({"MALE", "M"})
        other_mask   = ~female_mask & ~male_mask & (sex_vals != "") & (sex_vals.str.lower() != "nan")
        missing_mask = (sex_vals == "") | (sex_vals.str.lower() == "nan")

        entries.append(_header("Sex"))
        entries.append(_subrow("Female", _fmt_count_pct(female_mask, n_total)))
        if int(male_mask.sum()) > 0:
            entries.append(_subrow("Male", _fmt_count_pct(male_mask, n_total)))
        if int(other_mask.sum()) > 0:
            entries.append(_subrow("Other / unknown", _fmt_count_pct(other_mask, n_total)))
        if int(missing_mask.sum()) > 0:
            entries.append(_subrow("Missing", _fmt_count_pct(missing_mask, n_total)))
    else:
        print(f"[DatasetStats][INFO] Column '{sex_col}' not found — sex rows will show NA")
        entries.append(_header("Sex"))
        entries.append(_subrow("Female", "not available"))

    entries.append(_separator())

    # ── Vital status ────────────────────────────────────────────────────
    if status_col in df.columns:
        status_vals  = df[status_col].fillna("").astype(str).str.strip()
        alive_mask   = status_vals.str.upper() == "ALIVE"
        dead_mask    = status_vals.str.upper() == "DEAD"
        other_mask   = ~alive_mask & ~dead_mask & (status_vals != "") & (status_vals.str.lower() != "nan")
        missing_mask = (status_vals == "") | (status_vals.str.lower() == "nan")

        entries.append(_header("Vital status"))
        entries.append(_subrow("Alive", _fmt_count_pct(alive_mask, n_total)))
        entries.append(_subrow("Dead",  _fmt_count_pct(dead_mask,  n_total)))
        if int(other_mask.sum()) > 0:
            entries.append(_subrow("Other / unknown", _fmt_count_pct(other_mask, n_total)))
        if int(missing_mask.sum()) > 0:
            entries.append(_subrow("Missing", _fmt_count_pct(missing_mask, n_total)))
    else:
        print(f"[DatasetStats][INFO] Column '{status_col}' not found — vital status rows will show NA")
        entries.append(_header("Vital status"))
        entries.append(_subrow("Alive", "not available"))
        entries.append(_subrow("Dead",  "not available"))

    # ------------------------------------------------------------------
    # Build LaTeX table (booktabs style, compiles without modification)
    # ------------------------------------------------------------------
    latex_rows: List[str] = []
    for e in entries:
        if not e["latex_char"] and not e["latex_val"]:
            latex_rows.append(r"  \addlinespace")
        else:
            latex_rows.append(f"  {e['latex_char']} & {e['latex_val']} \\\\")

    latex_body = "\n".join(latex_rows)

    latex_table = (
        r"\begin{table}[htbp]" "\n"
        r"\centering" "\n"
        r"\caption{Patient characteristics. "
        r"Continuous variables are presented as median [IQR]; "
        r"categorical variables as $n$ (\%). "
        f"N~=~{n_total:,}." + r"}" "\n"
        r"\label{tab:patient_characteristics}" "\n"
        r"\begin{tabular}{ll}" "\n"
        r"\toprule" "\n"
        r"  \textbf{Characteristic} & \textbf{Value} \\" "\n"
        r"\midrule" "\n"
        + latex_body + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )

    latex_path = output_dir / "patient_characteristics_table.tex"
    latex_path.write_text(latex_table, encoding="utf-8")

    # ------------------------------------------------------------------
    # Build Markdown (renders in GitHub / VS Code / Obsidian)
    # ------------------------------------------------------------------
    md_table_rows: List[str] = []
    for e in entries:
        if not e["md_char"] and not e["md_val"]:
            continue  # skip separator rows — Markdown tables have no spacer rows
        md_table_rows.append(f"| {e['md_char']} | {e['md_val']} |")

    md_lines = [
        "# Patient Characteristics",
        "",
        f"**Total patients: {n_total:,}**",
        "",
        "> Continuous variables: median \\[Q1, Q3\\]. Categorical variables: n (%).",
        "",
        "| Characteristic | Value |",
        "|:---|:---|",
        *md_table_rows,
        "",
        "---",
        "",
        "## LaTeX source",
        "",
        "```latex",
        latex_table,
        "```",
    ]

    md_path = output_dir / "patient_characteristics_table.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[DatasetStats] Saved patient characteristics table → {latex_path.name}, {md_path.name}")


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
    figsize_dist     = tuple(cfg.get("figsize_distributions", [10, 6]))
    figsize_pie      = tuple(cfg.get("figsize_pie",      [8, 8]))
    figsize_km       = tuple(cfg.get("figsize_km",       [10, 7]))
    figsize_combined = tuple(cfg.get("figsize_combined", [20, 7]))
    dpi              = int(cfg.get("dpi", 200))

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_clinical_data(str(clinical_csv), patient_col)

    # ── Optional: restrict to patients with computed segmentation masks ────────
    masks_dir_cfg = cfg.get("masks_dir")
    if masks_dir_cfg:
        masks_dir_path = Path(masks_dir_cfg)
        if masks_dir_path.exists():
            available_ids = {p.stem for p in masks_dir_path.glob("*.zip")}
            before = len(df)
            df = df[df[patient_col].isin(available_ids)].copy()
            if verbose:
                print(f"[DatasetStats] Filtered to patients with segmentation masks: "
                      f"{before:,} → {len(df):,}")
        else:
            if verbose:
                print(f"[DatasetStats][WARN] masks_dir not found: {masks_dir_path} — skipping filter")

    shared_palette: Dict[str, str] = {}
    if subtype_col in df.columns:
        subtype_values = df[subtype_col].dropna().astype(str).str.strip()
        subtype_values = subtype_values[subtype_values != ""]
        ordered = _canonical_order(subtype_values.unique().tolist())
        if ordered:
            shared_palette = build_label_palette(np.array(ordered), cmap_name=cmap_cat)

    # Apply per-subtype colour overrides (e.g. Normal → pinkish to distinguish from Basal)
    palette_override = cfg.get("palette_override", {})
    for subtype, colour in palette_override.items():
        shared_palette[str(subtype)] = str(colour)

    if verbose:
        n_sub = df[subtype_col].nunique() if subtype_col in df.columns else 0
        print(f"[DatasetStats] Patients: {len(df):,}  |  Subtypes: {n_sub}  |  Output: {output_dir}")

    # ── Plot: tumor stage ─────────────────────────────────────────────────────
    if "tumor_stage_distribution" in requested:
        plot_tumor_stage(df, stage_col, subtype_col, output_dir,
                         palette=shared_palette, cmap_name=cmap_cat,
                         figsize=figsize_dist, dpi=dpi)

    # ── Plot: menopause ───────────────────────────────────────────────────────
    if "menopause_distribution" in requested:
        plot_menopause_status(df, menopause_col, subtype_col, output_dir,
                              palette=shared_palette, cmap_name=cmap_cat,
                              figsize=figsize_dist, dpi=dpi)

    # ── Plot: subtype pie (returns palette for KM re-use) ─────────────────────
    palette: Dict[str, str] = {}
    if "subtype_distribution" in requested:
        palette = plot_subtype_distribution(df, subtype_col, output_dir,
                                            cmap_name=cmap_cat, palette=shared_palette,
                                            figsize=figsize_pie, dpi=dpi)
    elif shared_palette:
        palette = shared_palette

    # ── Plot: Kaplan-Meier ────────────────────────────────────────────────────
    if "kaplan_meier_by_subtype" in requested:
        if not palette and subtype_col in df.columns:
            subtypes = df[subtype_col].dropna().unique().tolist()
            palette = build_label_palette(np.array(_canonical_order(subtypes)), cmap_name=cmap_cat)
            for subtype, colour in palette_override.items():
                palette[str(subtype)] = str(colour)
        plot_kaplan_meier_by_subtype(
            df, subtype_col, os_time_col, os_event_col,
            output_dir, palette=palette,
            cmap_name=cmap_cat, min_patients=min_patients,
            figsize=figsize_km, dpi=dpi,
        )

    # ── Plot: combined three-panel overview ───────────────────────────────────
    if "combined_overview" in requested:
        plot_combined_overview(
            df, subtype_col, stage_col, os_time_col, os_event_col,
            output_dir, palette=palette or shared_palette,
            cmap_name=cmap_cat, min_patients=min_patients,
            figsize=figsize_combined, dpi=dpi,
        )

    # ── Table: patient characteristics (LaTeX + Markdown) ───────────────────
    write_patient_characteristics_table(
        df,
        output_dir,
        age_col=cfg.get("age_col", "age_at_initial_pathologic_diagnosis"),
        followup_col=cfg.get("followup_col", os_time_col),
        sex_col=cfg.get("sex_col", "gender"),
        status_col=cfg.get("status_col", "vital_status"),
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
