"""Regenerate the TFD separability heatmap from pre-computed JSON results.

Matches the visual style of tfd_channel_contributions_absolute.png so both
figures can be placed side-by-side in LaTeX without clashing colour schemes:
  - magma colourmap
  - no axis spines
  - make_axes_locatable colorbar (same padding as channel contributions)
  - noise-floor values on the diagonal shown as plain numbers (no "NF" prefix)
  - identical figure height to the channel contributions plot for easy side-by-side

Output: tfd_separability_heatmap_v2.png  (next to the existing heatmap)

Usage
-----
    python -m src.visualization.tfd_heatmap
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from cmcrameri import cm as crameri_cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_JSON = Path(
    "../experiments"
    "/20260509_tfd_separability/results/tfd_separability_results.json"
)
OUTPUT_DIR = RESULTS_JSON.parent

PAM50_ORDER = ["Basal", "Her2", "LumA", "LumB", "Normal"]

# Match the auto-computed figsize of plot_channel_contributions for 10 pairs × 7 channels:
#   figsize = (max(9, 7*1.4), max(5, 10*0.7+1.5)) = (9.8, 8.5)
# We set the heatmap to the same HEIGHT so LaTeX \includegraphics[height=X]{...} aligns both.
FIGSIZE = (6.5, 8.5)
DPI = 200


def _build_matrix(
    class_names: list[str],
    pairwise_tfd: dict[str, float],
    noise_floor: dict[str, float],
) -> np.ndarray:
    n = len(class_names)
    mat = np.full((n, n), np.nan)
    for key, val in pairwise_tfd.items():
        a, b = key.split("__vs__")
        if a in class_names and b in class_names:
            i, j = class_names.index(a), class_names.index(b)
            mat[i, j] = val
            mat[j, i] = val
    for i, name in enumerate(class_names):
        if name in noise_floor and np.isfinite(noise_floor[name]):
            mat[i, i] = noise_floor[name]
    return mat


def main() -> None:
    with open(RESULTS_JSON) as f:
        data = json.load(f)

    raw_names: list[str] = data["class_names"]
    class_names = [c for c in PAM50_ORDER if c in raw_names]
    class_names += [c for c in raw_names if c not in PAM50_ORDER]

    mat = _build_matrix(
        class_names,
        data["pairwise_tfd"],
        data.get("noise_floor", {}),
    )
    n = len(class_names)

    # Mask diagonal and lower triangle
    mask = np.tri(n, dtype=bool)
    mat_masked = np.where(mask, np.nan, mat)

    vmin = np.nanmin(mat_masked)
    vmax = np.nanmax(mat_masked)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    cmap = crameri_cm.oslo
    cmap_copy = cmap.copy()
    cmap_copy.set_bad(color="#e0e0e0")
    im = ax.imshow(mat_masked, cmap=cmap_copy, aspect="equal",
                   interpolation="nearest", vmin=vmin, vmax=vmax)

    # Remove spines (matches channel contributions style)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colorbar via make_axes_locatable (identical to channel contributions)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    cb = fig.colorbar(im, cax=cax)
    cb.update_ticks()
    cb.set_label("TopoFD", fontsize=16)
    cb.ax.tick_params(labelsize=14)

    # Axis labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=16)
    ax.set_yticklabels(class_names, fontsize=16)
    ax.grid(False)

    # Cell annotations — upper triangle only
    for i in range(n):
        for j in range(n):
            if j <= i:
                continue
            val = mat_masked[i, j]
            if not np.isfinite(val):
                continue
            brightness = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            text_color = "white" if brightness < 0.72 else "black"
            ax.text(j, i, f"{val:.1f}",
                    ha="center", va="center", fontsize=14,
                    fontweight="bold", color=text_color)

    fig.tight_layout()
    out = OUTPUT_DIR / "tfd_separability_heatmap_v3.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
