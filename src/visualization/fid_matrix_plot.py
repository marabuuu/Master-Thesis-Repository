#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2×2 Fréchet Distance matrix plot with Crameri scientific colormaps.

Reads the ``fd_results.json`` produced by ``fid_subtype_evaluation.py`` and
renders a heatmap where:

    rows = {real LumA,  real Basal}
    cols = {gen  LumA,  gen  Basal}

The diagonal (same-class, real vs. generated) shows within-class fidelity;
the off-diagonal shows cross-class separation.  Both are important: small
diagonal + large off-diagonal = the genomic conditioning is informative.

Usage
-----
# Standalone (re-plot from saved JSON):
python -m src.visualization.fid_matrix_plot \\
    --results experiments/.../fid_evaluation/fd_results.json \\
    --output  experiments/.../fid_evaluation/fd_matrix.png

# Programmatic (called from fid_subtype_evaluation.run()):
from src.visualization.fid_matrix_plot import plot_fd_matrix
plot_fd_matrix(results_dict, Path("output/fd_matrix.png"))
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def _fd_matrix_array(
    results: Dict,
) -> tuple[np.ndarray, List[str], List[str]]:
    """Extract FD values as a 2×2 numpy array from a results dict.

    Returns:
        matrix: (2, 2) float array where matrix[i, j] = FD(rows[i], cols[j])
        row_labels: human-readable row labels
        col_labels: human-readable column labels
    """
    rows: List[str] = results["rows"]   # e.g. ["real_LumA", "real_Basal"]
    cols: List[str] = results["cols"]   # e.g. ["gen_LumA",  "gen_Basal"]
    fd = results["fd_matrix"]

    n_rows, n_cols = len(rows), len(cols)
    matrix = np.zeros((n_rows, n_cols))
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            key = f"{r}_vs_{c}"
            matrix[i, j] = fd[key]

    subtypes: List[str] = results.get("subtypes", ["LumA", "Basal"])

    row_labels = [f"Real {s}" for s in subtypes]
    col_labels = [f"Gen {s}"  for s in subtypes]
    return matrix, row_labels, col_labels


def plot_fd_matrix(
    results: Union[Dict, str, Path],
    output_path: Optional[Union[str, Path]] = None,
    cmap_name: str = "batlow",
    figsize: tuple[float, float] = (5.5, 4.5),
    dpi: int = 200,
    annotate_tile_counts: bool = True,
) -> matplotlib.figure.Figure:
    """Render the 2×2 FD matrix as a heatmap with Crameri colourmap.

    Args:
        results:   fd_results.json dict or path to it.
        output_path: where to save the PNG; if None the figure is returned
                     without saving.
        cmap_name: Crameri colormap name (``batlow``, ``devon``, ``lajolla``…).
        annotate_tile_counts: show tile counts in subtitle.
    """
    import cmcrameri.cm as cmc  # Crameri scientific colourmaps

    if not isinstance(results, dict):
        with open(results) as f:
            results = json.load(f)

    matrix, row_labels, col_labels = _fd_matrix_array(results)
    test_mode: bool = results.get("test_mode", False)
    tile_counts: Dict[str, int] = results.get("tile_counts", {})

    # ── Crameri colormap ─────────────────────────────────────────────────────
    try:
        cmap = getattr(cmc, cmap_name)
    except AttributeError:
        # fall back gracefully if name is wrong
        cmap = cmc.batlow

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap=cmap, aspect="equal")

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # ── Cell annotations ─────────────────────────────────────────────────────
    vmin, vmax = matrix.min(), matrix.max()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            # pick text colour for contrast against background
            norm_val = (val - vmin) / (vmax - vmin + 1e-9)
            text_color = "white" if norm_val > 0.55 else "black"
            is_diag = (i == j)
            weight = "bold" if is_diag else "normal"
            ax.text(
                j, i,
                f"{val:.1f}",
                ha="center", va="center",
                fontsize=13, fontweight=weight,
                color=text_color,
            )

    # ── Colourbar ────────────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fréchet Distance (InceptionV3)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # ── Title ────────────────────────────────────────────────────────────────
    title = "FD Matrix: Real vs Generated Tiles by PAM50 Subtype"
    if test_mode:
        title += "\n(test mode — tile counts too low for statistical reliability)"
    ax.set_title(title, fontsize=10, pad=16)

    # ── Subtitle with tile counts ─────────────────────────────────────────────
    if annotate_tile_counts and tile_counts:
        parts = [f"{k}: {v:,}" for k, v in sorted(tile_counts.items())]
        subtitle = "  |  ".join(parts)
        fig.text(
            0.5, 0.01, subtitle,
            ha="center", va="bottom",
            fontsize=7, color="#555555",
            transform=fig.transFigure,
        )

    # ── Guidance annotation ───────────────────────────────────────────────────
    guidance = results.get("guidance_scale", 1.0)
    if guidance != 1.0:
        fig.text(
            0.98, 0.01, f"CFG scale={guidance}",
            ha="right", va="bottom", fontsize=7, color="#555555",
        )

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot 2×2 FD matrix from fd_results.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results", required=True, help="Path to fd_results.json")
    p.add_argument("--output", default=None,
                   help="Output PNG path (default: same dir as results, fd_matrix.png)")
    p.add_argument("--cmap", default="batlow",
                   help="Crameri colormap name (batlow, devon, lajolla, oslo, …)")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    results_path = Path(args.results)
    out = Path(args.output) if args.output else results_path.parent / "fd_matrix.png"
    fig = plot_fd_matrix(results_path, output_path=out, cmap_name=args.cmap, dpi=args.dpi)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
