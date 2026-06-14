"""Four-panel FID matrix comparison figure with a shared colour scale.

Rows (top to bottom): zero / noise / orthogonal / genomic conditioning.

Usage (from Master-Thesis-Repository/ with venv active):
    python -m src.visualization.poc_fid_comparison
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm as crameri_cm

_EXPS_ROOT = Path(__file__).resolve().parents[3] / "experiments"

EXPERIMENTS: list[tuple[str, Path]] = [
    (
        "Zero",
        _EXPS_ROOT / "20260607_poc_128_zero_30M/fid_evaluation_last_fixed/fd_results.json",
    ),
    (
        "Noise",
        _EXPS_ROOT / "20260607_poc_128_noise_30M/fid_evaluation_last_fixed/fd_results.json",
    ),
    (
        "Orthogonal",
        _EXPS_ROOT / "20260605_poc_128_orthogonal_nonorm_30M/fid_evaluation_scale1_last_10k/fd_results.json",
    ),
    (
        "Genomic",
        _EXPS_ROOT / "20260607_poc_128_rna_norm_30M/fid_evaluation_last_fixed/fd_results.json",
    ),
]

OUT = _EXPS_ROOT / "fid_panel_comparison.png"


def _load(json_path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    data = json.loads(json_path.read_text())
    fid_map = data.get("fid_matrix") or data.get("fd_matrix")
    rows, cols = data["rows"], data["cols"]
    matrix = np.array(
        [[fid_map.get(f"{r}_vs_{c}", float("nan")) for c in cols] for r in rows]
    )
    return matrix, rows, cols


_COHORT_DISPLAY = {"BRCA": "Breast", "LIHC": "Liver"}


def _strip(labels: list[str]) -> list[str]:
    out = []
    for lb in labels:
        code = lb.replace("real_TCGA-", "").replace("gen_TCGA-", "")
        out.append(_COHORT_DISPLAY.get(code, code))
    return out


def main() -> None:
    all_data = [(_label, *_load(path)) for _label, path in EXPERIMENTS]

    all_vals = np.concatenate(
        [m[~np.isnan(m)].ravel() for _, m, _, _ in all_data]
    )
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    cmap = crameri_cm.oslo
    n = len(all_data)

    # Layout: n heatmap axes in column 0, single colourbar in column 1
    fig = plt.figure(figsize=(3.8, n * 2.0 + 0.3))
    gs = gridspec.GridSpec(
        n, 2,
        width_ratios=[1, 0.06],
        hspace=0.45,
        wspace=0.12,
        left=0.12, right=0.88, top=0.97, bottom=0.06,
    )

    axes = []
    for i, (label, matrix, rows, cols) in enumerate(all_data):
        ax = fig.add_subplot(gs[i, 0])
        axes.append(ax)

        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

        rl = _strip(rows)
        cl = _strip(cols)
        ax.set_xticks(range(len(cl)))
        ax.set_yticks(range(len(rl)))
        ax.set_xticklabels(cl, fontsize=11)
        ax.set_yticklabels(rl, fontsize=11)

        ax.set_ylabel("Real", fontsize=11, labelpad=3)
        if i == n - 1:
            ax.set_xlabel("Generated", fontsize=11, labelpad=3)

        # Value annotations with contrast-aware text colour
        for ii in range(matrix.shape[0]):
            for jj in range(matrix.shape[1]):
                v = matrix[ii, jj]
                if not np.isnan(v):
                    norm_val = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    text_color = "white" if norm_val < 0.5 else "black"
                    ax.text(
                        jj, ii, f"{v:.1f}",
                        ha="center", va="center",
                        fontsize=11, color=text_color,
                    )

    # Shared colourbar spanning all rows
    cbar_ax = fig.add_subplot(gs[:, 1])
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("FID", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
