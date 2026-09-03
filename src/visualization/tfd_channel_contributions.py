"""Regenerate the TFD channel contributions heatmap from pre-computed JSON results.

Matches the visual style of tfd_separability_heatmap_v2.png so both figures
can be placed side-by-side in LaTeX without clashing colour schemes:
  - magma colourmap
  - no axis spines
  - make_axes_locatable colorbar with "TFD  (×10⁶)" label
  - cell annotations as "Xk" (×10³), white on dark cells / black on bright cells
  - identical figure height (8.5) to the separability heatmap for easy side-by-side

Output: tfd_channel_contributions_absolute_v2.png  (next to the existing plot)

Usage
-----
    python -m src.visualization.tfd_channel_contributions
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

CHANNEL_NAMES = {
    0: "Background",
    1: "Neutrophils",
    2: "Epithelium",
    3: "Lymphocytes",
    4: "Plasma Cells",
    5: "Eosinophils",
    6: "Connective",
}
N_CHANNELS = len(CHANNEL_NAMES)

PAM50_ORDER = ["Basal", "Her2", "LumA", "LumB", "Normal"]

# Match heatmap height exactly so LaTeX \includegraphics[width=X]{...} aligns both.
FIGSIZE = (8.5, 9.5)
DPI = 200


def _sorted_pairs(ppc: dict[str, dict]) -> list[tuple[str, dict]]:
    """Return pairs ordered by PAM50 subtype index (Basal < Her2 < LumA < LumB < Normal)."""
    def _rank(key: str) -> tuple[int, int]:
        a, b = key.split("__vs__")
        ra = PAM50_ORDER.index(a) if a in PAM50_ORDER else 99
        rb = PAM50_ORDER.index(b) if b in PAM50_ORDER else 99
        return (ra, rb)

    return sorted(ppc.items(), key=lambda kv: _rank(kv[0]))


def main() -> None:
    with open(RESULTS_JSON) as f:
        data = json.load(f)

    ppc: dict[str, dict] = data["pairwise_per_channel"]
    sorted_pairs = _sorted_pairs(ppc)

    col_labels = [CHANNEL_NAMES[c] for c in range(N_CHANNELS)]
    pair_labels = [
        "{} vs {}".format(*key.split("__vs__"))
        for key, _ in sorted_pairs
    ]
    vals = np.array([
        [float(ch_vals.get(str(c), float("nan"))) for c in range(N_CHANNELS)]
        for _, ch_vals in sorted_pairs
    ])

    n_pairs = len(pair_labels)

    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    cmap = crameri_cm.oslo_r
    im = ax.imshow(vals, cmap=cmap, aspect="auto", interpolation="nearest",
                   vmin=vmin, vmax=vmax)

    for spine in ax.spines.values():
        spine.set_visible(False)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    cb = fig.colorbar(im, cax=cax)
    cb.update_ticks()
    cb.set_label("TopoFD channel contribution", fontsize=14)
    cb.ax.tick_params(labelsize=12)

    ax.set_xticks(range(N_CHANNELS))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=14)
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_labels, fontsize=14)
    ax.grid(False)

    for i in range(n_pairs):
        for j in range(N_CHANNELS):
            v = vals[i, j]
            if not np.isfinite(v):
                continue
            brightness = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            text_color = "white" if brightness > 0.45 else "black"
            ax.text(j, i, f"{v:.1f}",
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", color=text_color)

    fig.tight_layout()
    out = OUTPUT_DIR / "tfd_channel_contributions_absolute_v3.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
