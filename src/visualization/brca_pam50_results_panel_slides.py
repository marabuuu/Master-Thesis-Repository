#!/usr/bin/env python3
"""
BRCA PAM50 Results — slide-format variants of brca_pam50_results_panel.py

The report panel (brca_pam50_results_panel.py) packs everything into one
wide, roughly-square figure — great on a page, awkward on a slide. This
script keeps the same content but restacks it vertically (narrow + tall)
and splits it into two separate figures so each can be dropped into a
slide independently:

  brca_pam50_conditioning_slide.png — the 3 conditioning-metric line plots,
      stacked 3x1 instead of 1x3.
  brca_pam50_cfg_comparison_slide.png — the 3 CFG blocks, each with its FID
      heatmap stacked ABOVE its tile grid instead of beside it, plus a
      shared horizontal colourbar at the bottom instead of a tall vertical
      strip on the right.

Usage (from Master-Thesis-Repository/ with venv active):
    python -m src.visualization.brca_pam50_results_panel_slides
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from src.visualization.training_plots import _aggregate_by_step, load_gda_tfevents
from src.visualization.brca_pam50_results_panel import (
    _GENOMIC_BASE,
    _GENOMIC_RUN,
    _RC,
    FID_VMAX,
    FID_VMIN,
    C_GAP,
    C_SEP,
    C_SIG,
    _ema_series,
    _load_fid,
    _oslo_cmap,
    _plot_metric,
    _select_zips,
    _load_tile,
    _LUMA_PATIENT_INDICES,
    _BASAL_PATIENT_INDICES,
    _TILE_IDX,
)

OUT_COND = _GENOMIC_BASE / "brca_pam50_conditioning_slide.png"
OUT_CFG = _GENOMIC_BASE / "brca_pam50_cfg_comparison_slide.png"


# ── Figure 1: conditioning metrics, stacked 3x1 ─────────────────────────────

def build_conditioning_slide() -> None:
    plt.rcParams.update(_RC)

    print("Loading genomic run TFEvents …")
    gen_data = load_gda_tfevents(_GENOMIC_RUN)

    gen_sig_s, gen_sig_v = gen_data.get("cond/signal", ([], []))
    _gap_rs, _gap_rv = gen_data.get("cond/gap", ([], []))
    gen_gap_s, gen_gap_v = _aggregate_by_step(_gap_rs, _gap_rv)
    gen_sep_s, gen_sep_v = gen_data.get("cond/basal_luma_sep", ([], []))

    def _col_ylim(steps, vals, *, is_gap):
        fin = [v for v in vals if np.isfinite(v)]
        if not fin:
            return 0.0, 1e-4
        if not is_gap:
            s_M = [s / 1e6 for s in steps]
            _, v_ema = _ema_series(s_M, fin, alpha=0.05)
            col_max = max(v_ema) if v_ema else max(fin)
        else:
            col_max = max(fin)
        col_min = min(fin)
        top = col_max * 1.45
        bot = col_min * 1.20 if col_min < 0 else -top * 0.08
        return bot, top

    rows_data = [
        ("Cond. signal", gen_sig_s, gen_sig_v, C_SIG, False),
        ("Cond. gap", [float(s) for s in gen_gap_s], gen_gap_v, C_GAP, True),
        ("Cond. sep.", gen_sep_s, gen_sep_v, C_SEP, False),
    ]

    fig = plt.figure(figsize=(8, 18))
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.32,
                            left=0.20, right=0.96, top=0.98, bottom=0.05)

    for ri, (ylabel, steps, vals, color, is_gap) in enumerate(rows_data):
        y_bot, y_top = _col_ylim(steps, vals, is_gap=is_gap)
        ax = fig.add_subplot(gs[ri, 0])
        _plot_metric(ax, steps, vals, color, is_gap=is_gap, y_top=y_top, y_bot=y_bot)
        ax.set_ylabel(ylabel)
        if ri == 2:
            ax.set_xlabel("Samples (×10⁶)")

    OUT_COND.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_COND, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved → {OUT_COND}")
    plt.close(fig)


# ── Figure 2: CFG comparison, FID heatmap stacked above its tile grid ──────

def build_cfg_comparison_slide() -> None:
    plt.rcParams.update(_RC)

    print("Loading FID matrices …")
    cfg_fids: Dict[int, Tuple[np.ndarray, List[str], List[str]]] = {}
    for cfg in (1, 3, 5):
        p = _GENOMIC_BASE / f"generated_tiles_cfg{cfg}" / "fid_matrix_official.json"
        cfg_fids[cfg] = _load_fid(p)

    print("Loading sample tiles …")
    cfg_tiles: Dict[int, Dict[str, List[np.ndarray]]] = {}
    for cfg in (1, 3, 5):
        cfg_tiles[cfg] = {}
        for subtype in ("LumA", "Basal"):
            d = _GENOMIC_BASE / f"generated_tiles_cfg{cfg}" / "generated" / subtype
            idx_override = _LUMA_PATIENT_INDICES if subtype == "LumA" else _BASAL_PATIENT_INDICES
            cfg_tiles[cfg][subtype] = [
                _load_tile(p, tile_idx=_TILE_IDX) for p in _select_zips(d, indices=idx_override)
            ]

    fig = plt.figure(figsize=(11, 26))
    outer = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[3.0, 3.0, 3.0, 0.12],
        hspace=0.22,
        left=0.14, right=0.95, top=0.98, bottom=0.02,
    )

    for cfg_i, cfg in enumerate((1, 3, 5)):
        matrix, row_labels, col_labels = cfg_fids[cfg]
        tiles = cfg_tiles[cfg]

        block = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[cfg_i, 0],
            height_ratios=[1.3, 1.6], hspace=0.30,
        )

        # ── FID heatmap on top, centred over the tile grid's middle column ──
        fid_slot = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=block[0, 0], width_ratios=[0.3, 1.4, 0.3],
        )
        ax_fid = fig.add_subplot(fid_slot[0, 1])
        ax_fid.imshow(matrix, cmap=_oslo_cmap, vmin=FID_VMIN, vmax=FID_VMAX, aspect="equal")
        ax_fid.set_xticks(range(len(col_labels)))
        ax_fid.set_yticks(range(len(row_labels)))
        ax_fid.set_xticklabels(col_labels, fontsize=27)
        ax_fid.set_yticklabels(row_labels, fontsize=27)
        ax_fid.set_ylabel("Real", fontsize=28, labelpad=3)
        ax_fid.set_xlabel("Generated", fontsize=28, labelpad=3)
        ax_fid.set_title(f"CFG = {cfg}", fontsize=32, fontweight="bold", pad=10)

        for ii in range(matrix.shape[0]):
            for jj in range(matrix.shape[1]):
                v = matrix[ii, jj]
                nv = (v - FID_VMIN) / (FID_VMAX - FID_VMIN)
                ax_fid.text(jj, ii, f"{v:.1f}", ha="center", va="center",
                            fontsize=30, fontweight="bold",
                            color="white" if nv < 0.5 else "black")

        # ── tile grid below: 2 rows (subtype) x 3 cols (examples) ───────
        tile_grid = gridspec.GridSpecFromSubplotSpec(
            2, 3, subplot_spec=block[1, 0], wspace=0.05, hspace=0.06,
        )
        for sub_i, subtype in enumerate(("LumA", "Basal")):
            for j, tile_img in enumerate(tiles[subtype]):
                ax_t = fig.add_subplot(tile_grid[sub_i, j])
                ax_t.imshow(tile_img, interpolation="bilinear")
                ax_t.axis("off")
                if j == 0:
                    ax_t.text(
                        0.04, 0.96, subtype, transform=ax_t.transAxes,
                        ha="left", va="top", fontsize=22, fontweight="bold",
                        color="white",
                        bbox=dict(facecolor="black", alpha=0.45, pad=2,
                                  boxstyle="round,pad=0.2"),
                    )

    # ── shared horizontal FID colourbar at the bottom ───────────────────
    ax_cbar = fig.add_subplot(outer[3, 0])
    sm = plt.cm.ScalarMappable(cmap=_oslo_cmap, norm=plt.Normalize(vmin=FID_VMIN, vmax=FID_VMAX))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation="horizontal")
    cbar.set_label("FID", fontsize=28, labelpad=6)
    cbar.ax.tick_params(labelsize=24)

    OUT_CFG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_CFG, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved → {OUT_CFG}")
    plt.close(fig)


def main() -> None:
    build_conditioning_slide()
    build_cfg_comparison_slide()


if __name__ == "__main__":
    main()
