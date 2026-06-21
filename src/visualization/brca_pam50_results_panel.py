#!/usr/bin/env python3
"""
BRCA PAM50 Results Panel

Comprehensive figure for the BRCA PAM50 conditioning experiment:
  Section A — Conditioning metrics (2 rows):
    Row 0 (Orthogonal, 1-hot): placeholder — training in progress
    Row 1 (Genomic, RNA-seq):  signal / gap / Basal–LumA separation

  Section B — Controlled tile comparison (3 rows):
    Row 2: FID 2×2 heatmaps, one per CFG scale (cfg=1, cfg=3, cfg=5)
    Row 3: LumA tiles — SAME patient, SAME noise, three CFG scales
    Row 4: Basal tiles — SAME patient, SAME noise, three CFG scales

    Columns = CFG scales.  Rows = subtypes.
    Same patient across all columns → noise is held fixed; only guidance changes.

Usage (from Master-Thesis-Repository/ with venv active):
    python -m src.visualization.brca_pam50_results_panel
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from PIL import Image

# ── path setup ────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]  # Master-Thesis-Repository/
_EXP = _REPO.parent / "experiments"
sys.path.insert(0, str(_REPO))

from src.visualization.training_plots import (
    _PAPER_RC,
    _aggregate_by_step,
    _batlow_colors,
    _ema_series,
    _strip_spines,
    _subsample_series,
    load_gda_tfevents,
)

# ── fixed paths ───────────────────────────────────────────────────────────────
_GENOMIC_RUN = _EXP / "20260607_brca_pam50_cfg_v2_256" / "gda"
_GENOMIC_BASE = _EXP / "20260607_brca_pam50_cfg_v2_256"
_ORTHOGONAL_RUN = _EXP / "20260614_brca_pam50_cfg_v2_1hot_256" / "gda"
OUT = _GENOMIC_BASE / "brca_pam50_results_panel.png"

# ── colours — match make_paper_panels.py ─────────────────────────────────────
C_SIG, C_GAP, C_SEP = _batlow_colors([0.20, 0.55, 0.78])

try:
    from cmcrameri import cm as cmc  # type: ignore[import]
    _oslo_cmap = cmc.oslo
except Exception:
    _oslo_cmap = plt.cm.Blues_r  # type: ignore[attr-defined]

# Fixed FID colour range matching the PoC run (27–81)
FID_VMIN, FID_VMAX = 27.0, 81.0

_RC: Dict = {
    **_PAPER_RC,
    "font.size": 20,
    "axes.labelsize": 19,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
}

_TILE_IDX = 20
# LumA index 17 (TCGA-AO-A0JF) generates near-solid tiles at idx=20 (lap_var≈15).
# Index 18 (TCGA-AQ-A7U7) is sharp (lap_var≈4394) and visually representative.
_LUMA_PATIENT_INDICES: list[int] | None = [0, 18, 34]
_BASAL_PATIENT_INDICES: list[int] | None = None  # auto

# ── helpers ───────────────────────────────────────────────────────────────────

def _sci_fmt(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))


def _row_label(ax: plt.Axes, name: str, fontsize: int = 20) -> None:
    """Row label rotated 90° to the left of the axis."""
    ax.text(
        -0.28, 0.5, name,
        transform=ax.transAxes,
        rotation=90, va="center", ha="right",
        fontsize=fontsize, fontweight="bold", color="0.15",
    )


def _plot_metric(
    ax: plt.Axes,
    steps: List[float],
    vals: List[float],
    color: str,
    *,
    is_gap: bool = False,
    y_top: Optional[float] = None,
    y_bot: float = 0.0,
) -> None:
    """Raw + EMA-smoothed conditioning metric on a shared y-scale."""
    if vals:
        s_M = [s / 1e6 for s in steps]
        s_sub, v_sub = _subsample_series(s_M, list(vals), 1_500)
        ax.plot(s_sub, v_sub, lw=0.4, alpha=0.20, color=color, rasterized=True)
        s_ema, v_ema = _ema_series(s_M, list(vals), alpha=0.05)
        ax.plot(s_ema, v_ema, lw=2.2, color=color)

    ax.axhline(0.0, color="0.65", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylim(bottom=y_bot, top=y_top)
    _sci_fmt(ax)
    _strip_spines(ax)


def _placeholder_panel(ax: plt.Axes, y_top: Optional[float], y_bot: float) -> None:
    """Grey placeholder for a run that is still training."""
    ax.set_facecolor("#f5f5f5")
    ax.text(0.5, 0.54, "Training in progress",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=18, color="0.50", fontstyle="italic")
    ax.text(0.5, 0.36, "— placeholder —",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=16, color="0.65")
    ax.axhline(0.0, color="0.65", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylim(bottom=y_bot, top=y_top)
    _sci_fmt(ax)
    ax.set_xticks([])
    _strip_spines(ax)


def _load_fid(json_path: Path) -> Tuple[np.ndarray, List[str], List[str]]:
    with open(json_path) as fh:
        data = json.load(fh)
    fid_map = data["fid_matrix"]
    rows, cols = data["rows"], data["cols"]
    matrix = np.array([[fid_map[f"{r}_vs_{c}"] for c in cols] for r in rows])
    return (
        matrix,
        [r.replace("real_", "") for r in rows],
        [c.replace("gen_", "") for c in cols],
    )


def _load_tile(zip_path: Path, tile_idx: int = 20) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(z.namelist())
        with z.open(names[min(tile_idx, len(names) - 1)]) as f:
            return np.array(Image.open(io.BytesIO(f.read())).convert("RGB"))


def _select_zips(tiles_dir: Path, n: int = 3, indices: list[int] | None = None) -> List[Path]:
    zips = sorted(tiles_dir.glob("*.zip"))
    if indices is not None:
        return [zips[i] for i in indices]
    step = max(1, len(zips) // n)
    return [zips[i * step] for i in range(n)]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    plt.rcParams.update(_RC)

    # ── load conditioning metrics ─────────────────────────────────────────────
    print("Loading genomic run TFEvents …")
    gen_data = load_gda_tfevents(_GENOMIC_RUN)

    gen_sig_s, gen_sig_v = gen_data.get("cond/signal", ([], []))
    _gap_rs, _gap_rv      = gen_data.get("cond/gap", ([], []))
    gen_gap_s, gen_gap_v = _aggregate_by_step(_gap_rs, _gap_rv)
    gen_sep_s, gen_sep_v = gen_data.get("cond/basal_luma_sep", ([], []))

    print("Loading orthogonal (1-hot) run TFEvents …")
    orth_data = load_gda_tfevents(_ORTHOGONAL_RUN)

    orth_sig_s, orth_sig_v = orth_data.get("cond/signal", ([], []))
    _ogap_rs, _ogap_rv     = orth_data.get("cond/gap", ([], []))
    orth_gap_s, orth_gap_v = _aggregate_by_step(_ogap_rs, _ogap_rv)
    orth_sep_s, orth_sep_v = orth_data.get("cond/brca_lihc_sep", ([], []))

    def _col_ylim_pair(steps_a, vals_a, steps_b, vals_b, *, is_gap):
        all_vals = []
        all_maxes = []
        for steps, vals in ((steps_a, vals_a), (steps_b, vals_b)):
            if not vals:
                continue
            fin = [v for v in vals if np.isfinite(v)]
            if not fin:
                continue
            all_vals.extend(fin)
            if not is_gap:
                s_M = [s / 1e6 for s in steps]
                _, v_ema = _ema_series(s_M, fin, alpha=0.05)
                all_maxes.append(max(v_ema) if v_ema else max(fin))
            else:
                all_maxes.append(max(fin))
        if not all_vals:
            return 0.0, 1e-4
        col_max = max(all_maxes)
        col_min = min(all_vals)
        top = col_max * 1.25
        bot = col_min * 1.20 if col_min < 0 else -top * 0.08
        return bot, top

    ylims = [
        _col_ylim_pair(gen_sig_s, gen_sig_v, orth_sig_s, orth_sig_v, is_gap=False),
        _col_ylim_pair([float(s) for s in gen_gap_s], gen_gap_v,
                       [float(s) for s in orth_gap_s], orth_gap_v, is_gap=True),
        _col_ylim_pair(gen_sep_s, gen_sep_v, orth_sep_s, orth_sep_v, is_gap=False),
    ]
    y_bots = [b for b, _ in ylims]
    y_tops = [t for _, t in ylims]

    # ── load FID matrices ─────────────────────────────────────────────────────
    print("Loading FID matrices …")
    cfg_fids: Dict[int, Tuple[np.ndarray, List[str], List[str]]] = {}
    for cfg in (1, 3, 5):
        p = _GENOMIC_BASE / f"generated_tiles_cfg{cfg}" / "fid_matrix_official.json"
        cfg_fids[cfg] = _load_fid(p)

    # ── load sample tiles ─────────────────────────────────────────────────────
    # _select_zips sorts patients alphabetically; same count in every CFG dir
    # → identical patient at each column position across all CFG rows.
    # Same tile_idx within a patient zip → same batch seed → same starting noise.
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

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 28))

    outer = gridspec.GridSpec(
        5, 2,
        figure=fig,
        height_ratios=[1.0, 1.0, 2.0, 2.0, 2.0],
        width_ratios=[1, 0.025],
        hspace=0.38,
        wspace=0.03,
        left=0.09, right=0.97, top=0.97, bottom=0.03,
    )

    col_ylabels = ["Conditioning signal", "Conditioning gap", "Cond. separation"]

    # ── Row 0: Orthogonal (1-hot) conditioning ─────────────────────────────────
    inner0 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[0, :], wspace=0.42
    )
    orth_row_data = [
        (orth_sig_s,                     orth_sig_v, C_SIG, False),
        ([float(s) for s in orth_gap_s], orth_gap_v, C_GAP, True),
        (orth_sep_s,                     orth_sep_v, C_SEP, False),
    ]
    orth_axes = []
    for ci, (steps, vals, color, is_gap) in enumerate(orth_row_data):
        ax = fig.add_subplot(inner0[ci])
        orth_axes.append(ax)
        _plot_metric(ax, steps, vals, color,
                     is_gap=is_gap, y_top=y_tops[ci], y_bot=y_bots[ci])
        ax.set_ylabel(col_ylabels[ci])
        ax.tick_params(labelbottom=False)

    _row_label(orth_axes[0], "Orthogonal")

    # ── Row 1: Genomic conditioning ───────────────────────────────────────────
    inner1 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1, :], wspace=0.42
    )
    row1_data = [
        (gen_sig_s,                     gen_sig_v, C_SIG, False),
        ([float(s) for s in gen_gap_s], gen_gap_v, C_GAP, True),
        (gen_sep_s,                     gen_sep_v, C_SEP, False),
    ]
    gen_axes = []
    for ci, (steps, vals, color, is_gap) in enumerate(row1_data):
        ax = fig.add_subplot(inner1[ci])
        gen_axes.append(ax)
        _plot_metric(ax, steps, vals, color,
                     is_gap=is_gap, y_top=y_tops[ci], y_bot=y_bots[ci])
        ax.set_ylabel(col_ylabels[ci])
        ax.set_xlabel("Samples (×10⁶)")

    _row_label(gen_axes[0], "Genomic")

    # ── Rows 2–4: FID heatmap + sample tiles per CFG scale ───────────────────
    for cfg_i, cfg in enumerate((1, 3, 5)):
        matrix, row_labels, col_labels = cfg_fids[cfg]
        tiles = cfg_tiles[cfg]

        # 2 sub-rows (LumA / Basal) × 4 cols (FID | tile | tile | tile)
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 4,
            subplot_spec=outer[2 + cfg_i, 0],
            wspace=0.05, hspace=0.06,
            width_ratios=[2.0, 1, 1, 1],
        )

        # FID heatmap spanning both sub-rows
        ax_fid = fig.add_subplot(inner[:, 0])
        ax_fid.imshow(matrix, cmap=_oslo_cmap,
                      vmin=FID_VMIN, vmax=FID_VMAX, aspect="equal")

        ax_fid.set_xticks(range(len(col_labels)))
        ax_fid.set_yticks(range(len(row_labels)))
        ax_fid.set_xticklabels(col_labels, fontsize=17)
        ax_fid.set_yticklabels(row_labels, fontsize=17)
        ax_fid.set_ylabel("Real", fontsize=17, labelpad=3)
        ax_fid.set_xlabel("Generated", fontsize=17, labelpad=3)

        for ii in range(matrix.shape[0]):
            for jj in range(matrix.shape[1]):
                v = matrix[ii, jj]
                nv = (v - FID_VMIN) / (FID_VMAX - FID_VMIN)
                ax_fid.text(jj, ii, f"{v:.1f}",
                            ha="center", va="center", fontsize=16,
                            fontweight="bold",
                            color="white" if nv < 0.5 else "black")

        _row_label(ax_fid, f"cfg = {cfg}", fontsize=20)

        # Sample tiles — subtype label embedded in the first tile of each row
        for sub_i, subtype in enumerate(("LumA", "Basal")):
            for j, tile_img in enumerate(tiles[subtype]):
                ax_t = fig.add_subplot(inner[sub_i, 1 + j])
                ax_t.imshow(tile_img, interpolation="bilinear")
                ax_t.axis("off")
                if j == 0:
                    ax_t.text(
                        0.04, 0.96, subtype,
                        transform=ax_t.transAxes,
                        ha="left", va="top",
                        fontsize=16, fontweight="bold", color="white",
                        bbox=dict(facecolor="black", alpha=0.45,
                                  pad=2, boxstyle="round,pad=0.2"),
                    )

    # ── Shared FID colourbar (col 1, rows 2–4) ────────────────────────────────
    ax_cbar = fig.add_subplot(outer[2:, 1])
    sm = plt.cm.ScalarMappable(
        cmap=_oslo_cmap, norm=plt.Normalize(vmin=FID_VMIN, vmax=FID_VMAX)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label("FID", fontsize=16, labelpad=4)
    cbar.ax.tick_params(labelsize=14)

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved → {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
