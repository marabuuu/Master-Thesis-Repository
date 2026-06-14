"""Combine the four PoC tile panels into a single report-ready figure.

Rows (top → bottom): Zero / Noise / Orthogonal / Genomic
Left block:  Breast Cancer Cohort  (3 TCGA-BRCA tiles)
Right block: Liver Cancer Cohort   (3 TCGA-LIHC tiles)

Usage (from Master-Thesis-Repository/ with venv active):
    python -m src.visualization.poc_combined_panel
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

_BASE = Path("/mnt/bulk-saturn/maralampert/genhist/experiments")

PANELS = [
    ("Zero",       _BASE / "20260607_poc_128_zero_30M/panel.png"),
    ("Noise",      _BASE / "20260607_poc_128_noise_30M/panel.png"),
    ("Orthogonal", _BASE / "20260605_poc_128_orthogonal_nonorm_30M/panel.png"),
    ("Genomic",    _BASE / "20260607_poc_128_rna_norm_30M/panel.png"),
]

OUT = _BASE / "poc_panel_combined.png"


def _load_tile_region(path: Path) -> np.ndarray:
    """Load a panel PNG, crop the existing small label area at the top,
    return the tile strip as an RGB uint8 array."""
    img = np.array(Image.open(path).convert("RGB"))
    h = img.shape[0]
    # The label area occupies the top ~23 % of the figure (0.22 / 1.22 inches).
    # Add a small buffer; anything below this is pure tile content.
    crop_top = int(round(h * 0.245))
    return img[crop_top:]


def main() -> None:
    rows = [(label, _load_tile_region(p)) for label, p in PANELS]

    # All strips should have the same shape — verify.
    shapes = [r[1].shape for r in rows]
    assert len(set(s[1] for s in shapes)) == 1, f"Panel widths differ: {shapes}"

    strip_h, strip_w = rows[0][1].shape[:2]

    # Locate the breast / liver split inside the strip.
    # From make_panel.py: gap_center=8px in canvas coords.
    # At 300 dpi the canvas pixel ratio ≈ strip_w / 784 (≈2.42×).
    # The midpoint of the 8-pixel gap is at canvas x = half_w_px + 4 = 392.
    # Fraction: 392 / 784 = 0.50 exactly.
    # So the separator is dead-centre; left = BRCA, right = LIHC.
    sep_frac = 0.5   # fraction of strip width where gap centre falls
    brca_centre = sep_frac / 2          # ≈ 0.25
    lihc_centre = sep_frac + (1 - sep_frac) / 2  # ≈ 0.75

    n_rows = len(rows)

    # Figure layout: narrow label column on left, main image column, spacer row on top
    label_col_w = 0.9   # inches for row labels
    img_col_w   = strip_w / 300  # inches (300 dpi → 1:1 pixel mapping)
    header_h    = 0.35  # inches for cohort headers
    row_h       = strip_h / 300
    total_w = label_col_w + img_col_w
    total_h = header_h + n_rows * row_h + (n_rows - 1) * 0.06  # small gaps

    fig = plt.figure(figsize=(total_w, total_h), dpi=300)

    # GridSpec: 1 header row + n_rows image rows
    gs = gridspec.GridSpec(
        n_rows + 1, 2,
        figure=fig,
        width_ratios=[label_col_w, img_col_w],
        height_ratios=[header_h] + [row_h] * n_rows,
        hspace=0.04,
        wspace=0.0,
        left=0.0, right=1.0, top=1.0, bottom=0.0,
    )

    # ── Header row: cohort labels ─────────────────────────────────────────────
    ax_hdr = fig.add_subplot(gs[0, 1])
    ax_hdr.axis("off")
    ax_hdr.text(
        brca_centre, 0.35, "Breast Cancer Cohort",
        ha="center", va="center", fontsize=9, fontweight="bold",
        transform=ax_hdr.transAxes, color="0.2",
    )
    ax_hdr.text(
        lihc_centre, 0.35, "Liver Cancer Cohort",
        ha="center", va="center", fontsize=9, fontweight="bold",
        transform=ax_hdr.transAxes, color="0.2",
    )

    # ── Image rows ────────────────────────────────────────────────────────────
    for i, (label, strip) in enumerate(rows):
        # Row label (left column)
        ax_lbl = fig.add_subplot(gs[i + 1, 0])
        ax_lbl.axis("off")
        ax_lbl.text(
            0.95, 0.5, label,
            ha="right", va="center",
            fontsize=9, fontweight="bold",
            transform=ax_lbl.transAxes, color="0.15",
        )

        # Tile image (right column)
        ax_img = fig.add_subplot(gs[i + 1, 1])
        ax_img.imshow(strip, aspect="auto", interpolation="nearest")
        ax_img.axis("off")

    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
