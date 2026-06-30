#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure panel for the PoC conditioning experiment.

A  Virchow2 tile-level UMAP  — histology morphology space
B  Genomic feature UMAP      — RNA-seq feature space (per patient)
C  Cosine-distance clustermap of genomic features
D  Per-cohort silhouette scores (cosine metric, 512-D features)

Run from Master-Thesis-Repository/ with the venv active:
    python -m src.visualization.poc_conditioning_panel
    python -m src.visualization.poc_conditioning_panel --output path/to/panel.pdf
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from sklearn.metrics import pairwise_distances, silhouette_samples
import umap as umap_module
from tqdm import tqdm

# ── Python path ───────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]   # Master-Thesis-Repository/
_WS_ROOT = _REPO_ROOT.parent                       # genhist/
sys.path.insert(0, str(_REPO_ROOT))

from src.visualization.core import setup_style     # noqa: E402

# ── Colors — match palette_override from virchow2_umap_cohort config ─────────
# ColorBrewer PuOr, colorblind-safe
PALETTE: Dict[str, str] = {
    "TCGA-BRCA": "#FDAE61",   # light amber
    "TCGA-LIHC": "#762A83",   # dark violet
}
COHORTS = ["TCGA-BRCA", "TCGA-LIHC"]
SPLIT_MARKERS: Dict[str, str] = {"train": "o", "val": "s", "test": "^"}

# ── Fixed paths ───────────────────────────────────────────────────────────────
_POC_EXP = _WS_ROOT / "experiments" / "20260524_poc_breast_vs_liver_genomic_features"
VIRCHOW_CSV = _WS_ROOT / "experiments" / "virchow2_umap_cohort" / "virchow2_umap_tile_coords.csv"
H5_DIR = _POC_EXP / "genomic_h5"
SPLITS_JSON = _POC_EXP / "patient_splits.json"
OUTPUT_DIR = _WS_ROOT / "experiments" / "20260607_poc_128_rna_norm_30M" / "conditioning_panel"

SEED = 42
N_CLUSTERMAP = 300   # patients sub-sampled for the clustermap


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_splits(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (pid_to_split, pid_to_cohort) from patient_splits.json."""
    with open(path) as f:
        raw = json.load(f)
    pid_to_split: Dict[str, str] = {}
    pid_to_cohort: Dict[str, str] = {}
    for split in ("train", "val", "test"):
        for pid, meta in raw.get(split, {}).items():
            if pid.startswith("_"):
                continue
            pid_to_split[pid] = split
            if isinstance(meta, dict):
                pid_to_cohort[pid] = meta.get("subtype", "unknown")
    return pid_to_split, pid_to_cohort


def _load_genomic_h5(
    h5_dir: Path,
    pid_to_split: Dict[str, str],
    pid_to_cohort: Dict[str, str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load per-patient 512-D genomic features. Returns (X, cohorts, splits)."""
    embeddings: List[np.ndarray] = []
    cohorts: List[str] = []
    splits: List[str] = []
    for h5_path in tqdm(sorted(h5_dir.glob("*.h5")), desc="Loading H5"):
        pid = h5_path.stem
        if pid not in pid_to_split:
            continue
        with h5py.File(h5_path, "r") as f:
            feat = f["feats"][:].flatten().astype(np.float32)
        embeddings.append(feat)
        cohorts.append(pid_to_cohort[pid])
        splits.append(pid_to_split[pid])
    return np.stack(embeddings), np.array(cohorts), np.array(splits)


# ── Panel helpers ─────────────────────────────────────────────────────────────

def _scatter_umap(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    cohorts: np.ndarray,
    splits: np.ndarray,
    *,
    point_size: float,
    alpha: float,
    rasterized: bool,
) -> None:
    """Scatter with cohort color and split marker shape."""
    zorder_map = {"test": 3, "val": 2, "train": 1}
    for split, marker in SPLIT_MARKERS.items():
        for cohort in COHORTS:
            mask = (cohorts == cohort) & (splits == split)
            if not mask.any():
                continue
            ax.scatter(
                x[mask], y[mask],
                c=PALETTE[cohort],
                marker=marker,
                s=point_size,
                alpha=alpha,
                edgecolors="none",
                linewidths=0,
                rasterized=rasterized,
                zorder=zorder_map.get(split, 1),
            )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1", fontsize=10, labelpad=3)
    ax.set_ylabel("UMAP 2", fontsize=10, labelpad=3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _make_clustermap_img(X: np.ndarray, cohorts: np.ndarray) -> np.ndarray:
    """Render cosine-distance clustermap to a (H, W, 3) uint8 array."""
    import warnings
    from scipy.spatial.distance import squareform

    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(X), size=min(N_CLUSTERMAP, len(X)), replace=False)
    dist = pairwise_distances(X[idx], metric="cosine").astype(np.float64)
    # Pass as DataFrame so seaborn treats it as a pre-computed distance matrix
    dist_df = pd.DataFrame(dist)
    row_colors = pd.Series(cohorts[idx]).map(PALETTE).values

    try:
        from cmcrameri import cm as cmc
        cmap_heat = cmc.lajolla
    except ImportError:
        cmap_heat = "YlOrRd"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = sns.clustermap(
            dist_df,
            cmap=cmap_heat,
            figsize=(6, 6),
            row_colors=row_colors,
            col_colors=row_colors,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={"label": "Cosine distance", "shrink": 0.55},
            dendrogram_ratio=0.15,
            colors_ratio=0.03,
        )

    buf = io.BytesIO()
    g.fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(g.fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def _silhouette_panel(
    ax: plt.Axes,
    X: np.ndarray,
    cohorts: np.ndarray,
) -> None:
    """Per-cohort silhouette distribution (cosine, 512-D features)."""
    print("  Computing silhouette samples (cosine, 512-D)…")
    sil = silhouette_samples(X, cohorts, metric="cosine")
    df = pd.DataFrame({"Cohort": cohorts, "sil": sil})

    sns.violinplot(
        data=df, x="Cohort", y="sil",
        hue="Cohort",
        palette=PALETTE, order=COHORTS,
        inner="box",
        cut=0,
        linewidth=0.9,
        legend=False,
        ax=ax,
    )

    mean_sil = float(np.mean(sil))
    ax.axhline(mean_sil, color="0.45", lw=1.0, ls="--", zorder=0, alpha=0.8)
    ax.axhline(0, color="0.75", lw=0.6, ls="-", zorder=0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Silhouette coefficient", fontsize=10)
    ax.set_xlabel("")
    ax.set_xticks(range(len(COHORTS)))
    ax.set_xticklabels([c.replace("TCGA-", "") for c in COHORTS], fontsize=10)
    ax.text(
        0.97, mean_sil + 0.05,
        f"mean = {mean_sil:.2f}",
        transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=8.5, color="0.4",
    )
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.11, 1.07, letter,
        transform=ax.transAxes,
        fontsize=18, fontweight="bold",
        va="top", ha="left",
        clip_on=False,
    )


def _build_legend(fig: plt.Figure) -> None:
    """Two-section figure legend: Cohort colors + Data Split markers."""
    cohort_handles = [
        Patch(facecolor=PALETTE[c], edgecolor="none", label=c)
        for c in COHORTS
    ]
    split_handles = [
        Line2D(
            [0], [0],
            marker=SPLIT_MARKERS[s], color="0.3",
            markerfacecolor="0.3", markersize=7,
            linestyle="", label=lbl,
        )
        for s, lbl in [("train", "Train"), ("val", "Validation"), ("test", "Test")]
    ]

    kw = dict(frameon=False, fontsize=10, borderpad=0.4)

    leg1 = fig.legend(
        handles=cohort_handles,
        title="Cohort",
        title_fontsize=10,
        ncol=len(COHORTS),
        loc="lower left",
        bbox_to_anchor=(0.12, -0.03),
        **kw,
    )
    leg1.get_title().set_fontweight("bold")

    leg2 = fig.legend(
        handles=split_handles,
        title="Data Split",
        title_fontsize=10,
        ncol=3,
        loc="lower right",
        bbox_to_anchor=(0.88, -0.03),
        **kw,
    )
    leg2.get_title().set_fontweight("bold")

    fig.add_artist(leg1)   # keep first legend after adding second


# ── Main ──────────────────────────────────────────────────────────────────────

def run(output_path: Optional[Path] = None) -> None:
    setup_style(style="white", extra_rc={
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.dpi": 300,
    })

    # ─ Load data ─────────────────────────────────────────────────────────────
    print("Loading Virchow2 tile UMAP coordinates…")
    df_tile = pd.read_csv(VIRCHOW_CSV)

    print("Loading patient splits…")
    pid_to_split, pid_to_cohort = _load_splits(SPLITS_JSON)

    print("Loading genomic H5 features…")
    X, cohorts, splits = _load_genomic_h5(H5_DIR, pid_to_split, pid_to_cohort)
    print(f"  {len(X)} patients × {X.shape[1]}D")

    print("Computing genomic UMAP…")
    X_umap = umap_module.UMAP(
        n_neighbors=15, min_dist=0.1, metric="cosine",
        n_components=2, random_state=SEED,
    ).fit_transform(X)

    # ─ Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.42, wspace=0.32,
        left=0.08, right=0.97, top=0.97, bottom=0.07,
    )
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    # A — Virchow2 tile UMAP (69 500 tiles, rasterized)
    print("Rendering A…")
    _scatter_umap(
        ax_A,
        df_tile["umap_1"].values, df_tile["umap_2"].values,
        df_tile["subtype"].values, df_tile["split"].values,
        point_size=3, alpha=0.25, rasterized=True,
    )

    # B — Genomic patient UMAP (~1 400 patients)
    print("Rendering B…")
    _scatter_umap(
        ax_B,
        X_umap[:, 0], X_umap[:, 1],
        cohorts, splits,
        point_size=42, alpha=0.78, rasterized=False,
    )

    # C — Cosine-distance clustermap (300 patients, rendered to image)
    print("Rendering C (clustermap)…")
    cm_img = _make_clustermap_img(X, cohorts)
    ax_C.imshow(cm_img, aspect="auto", interpolation="lanczos")
    ax_C.axis("off")

    # D — Per-cohort silhouette distribution
    print("Rendering D (silhouette)…")
    _silhouette_panel(ax_D, X, cohorts)

    # Panel letters
    for ax, letter in [(ax_A, "A"), (ax_B, "B"), (ax_C, "C"), (ax_D, "D")]:
        _panel_letter(ax, letter)

    # Shared legend (bottom)
    _build_legend(fig)

    # ─ Save ──────────────────────────────────────────────────────────────────
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "conditioning_panel.pdf"

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {output_path}")
    print(f"Saved → {png_path}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate conditioning experiment panel")
    p.add_argument("--output", default=None,
                   help="Output PDF path (PNG saved alongside). Default: experiments/.../conditioning_panel.pdf")
    args = p.parse_args()
    run(Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
