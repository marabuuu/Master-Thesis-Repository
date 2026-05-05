#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TFD Separability Visualization
================================

Three complementary views of the TopoFD class-separability results:

1. **Channel contribution heatmap** (:func:`plot_channel_contributions`)
   Shows which cell-type channels drive the pairwise TFD for each
   subtype pair.  Both absolute TFD and row-normalised (relative
   contribution) variants are produced.

2. **Radar / spider chart** (:func:`plot_radar_channel_contributions`)
   One polygon per subtype pair, axes = cell types, values = relative
   channel contributions.  Each polygon is coloured as the 50/50 RGB
   blend of its two constituent subtype colours (same ``romaO`` palette
   used across the pipeline), so pairs can be cross-referenced visually.

3. **Cell-type composition boxplots** (:func:`plot_cell_type_boxplots`)
   Reads the segmentation mask ZIPs, computes the pixel-fraction of
   each cell type per tile, and draws side-by-side boxplots so
   histological subtypes can be compared directly.

Usage (pipeline)
-----------------
python run_pipeline.py --config src/config.yaml --stage tfd_separability_viz
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .core import (
    HEATMAP_CMAP,
    CATEGORICAL_CMAP,
    _check_matplotlib,
    _check_seaborn,
    build_label_palette,
    get_crameri_cmap,
    save_figure,
    setup_style,
)
from .spatial_metrics_viz import (
    compute_spatial_metrics_per_tile,
    plot_knn_metrics_comparison,
    plot_ripley_L_by_subtype,
    plot_voronoi_distribution,
)

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.figure import Figure
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:  # pragma: no cover
    pass

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)


def _extract_patient_id(name: str) -> str:
    """Extract the TCGA-XX-XXXX barcode from a filename or bare patient string."""
    stem = Path(name).stem.upper()
    m = re.match(r"(TCGA-[A-Z0-9]+-[A-Z0-9]+)", stem)
    if m:
        return m.group(1)
    for sep in ("_", "."):
        stem = stem.replace(sep, "-")
    while "--" in stem:
        stem = stem.replace("--", "-")
    parts = stem.split("-")
    if len(parts) >= 3 and parts[0].startswith("TCGA"):
        return "-".join(parts[:3])
    m = re.search(r"(?i)(TCGA-[A-Z0-9]+-[A-Z0-9]+)", name)
    return m.group(1).upper() if m else name


# -----------------------------------------------------------------------
# Cell-type metadata
# -----------------------------------------------------------------------

CHANNEL_NAMES: Dict[int, str] = {
    0: "Background",
    1: "Neutrophils",
    2: "Epithelium",
    3: "Lymphocytes",
    4: "Plasma Cells",
    5: "Eosinophils",
    6: "Connective",
}

#: Canonical display order for PAM50 subtypes
PAM50_ORDER = ["Basal", "Her2", "LumA", "LumB", "Normal"]


# ===================================================================
# § 1  Channel contribution heatmap
# ===================================================================

def plot_channel_contributions(
    results_json: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    cmap_name: str = HEATMAP_CMAP,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 200,
    verbose: bool = True,
) -> None:
    """Save two heatmaps showing per-channel TFD contributions.

    Parameters
    ----------
    results_json : path
        Path to ``tfd_separability_results.json``.
    output_dir : path
        Directory for output PNGs.
    cmap_name : str
        Crameri colourmap for the heatmap cells (default: ``lajolla``).
    figsize : (width, height) or None
        Figure dimensions; auto-sized from the data when ``None``.
    dpi : int
        Save resolution.
    verbose : bool
    """
    _check_matplotlib()
    setup_style()

    results_json = Path(results_json)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(results_json) as f:
        data = json.load(f)

    per_channel: Dict[str, Dict[str, float]] = data["pairwise_per_channel"]
    n_channels = len(CHANNEL_NAMES)

    # Build a DataFrame: rows=pairs, columns=channels
    pair_labels: List[str] = []
    rows: List[List[float]] = []

    for key, ch_vals in per_channel.items():
        a, b = key.split("__vs__")
        pair_labels.append(f"{a} vs {b}")
        row = [float(ch_vals.get(str(c), float("nan"))) for c in range(n_channels)]
        rows.append(row)

    col_labels = [CHANNEL_NAMES[c] for c in range(n_channels)]
    df_abs = pd.DataFrame(rows, index=pair_labels, columns=col_labels)

    # Normalised (row-sum = 1): relative contribution per pair
    row_sums = df_abs.sum(axis=1).replace(0, np.nan)
    df_rel = df_abs.div(row_sums, axis=0)

    if figsize is None:
        n_pairs = len(pair_labels)
        figsize = (max(9, n_channels * 1.4), max(5, n_pairs * 0.7 + 1.5))

    # Use a warm sequential colormap (magma) to improve visual contrast
    try:
        cmap = plt.get_cmap("magma")
    except Exception:
        cmap = get_crameri_cmap(cmap_name)

    for suffix, df, title_suffix, fmt in [
        ("absolute", df_abs, "Absolute TFD per Channel", "{:.0f}"),
        ("relative", df_rel, "Relative Channel Contribution", "{:.2f}"),
    ]:
        fig, ax = plt.subplots(figsize=figsize)
        vals = df.values.astype(float)
        # Use nearest interpolation to avoid white grid artefacts between cells
        im = ax.imshow(vals, cmap=cmap, aspect="auto", interpolation="nearest")
        # Remove axes spines so heatmap appears as a contiguous grid
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.08)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(
            "TFD" if suffix == "absolute" else "Fraction of pair TFD",
            fontsize=10,
        )

        ax.set_xticks(range(n_channels))
        ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(pair_labels)))
        ax.set_yticklabels(pair_labels, fontsize=9)
        ax.grid(False)
        ax.set_title(
            f"TFD Channel Contributions — {title_suffix}",
            fontsize=12, pad=8,
        )

        # Annotate cells
        vmax = np.nanmax(vals) if np.any(np.isfinite(vals)) else 1.0
        for i in range(len(pair_labels)):
            for j in range(n_channels):
                v = vals[i, j]
                if not np.isfinite(v):
                    continue
                brightness = v / vmax if vmax > 0 else 0
                text_color = "white" if brightness > 0.55 else "black"
                ax.text(
                    j, i, fmt.format(v),
                    ha="center", va="center",
                    fontsize=7, color=text_color,
                )

        fig.tight_layout()
        out = output_dir / f"tfd_channel_contributions_{suffix}.png"
        save_figure(fig, out, dpi=dpi)
        if verbose:
            print(f"[OK] Saved {suffix} channel heatmap → {out}")


# ===================================================================
# § 2  Radar / spider chart
# ===================================================================

def _blend_colors(hex_a: str, hex_b: str) -> Tuple[float, float, float]:
    """Return the perceptual midpoint of two hex colours as an RGB triple."""
    import matplotlib.colors as mcolors
    ra, ga, ba = mcolors.to_rgb(hex_a)
    rb, gb, bb = mcolors.to_rgb(hex_b)
    return ((ra + rb) / 2, (ga + gb) / 2, (ba + bb) / 2)


def plot_radar_channel_contributions(
    results_json: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    cmap_name: str = CATEGORICAL_CMAP,
    include_background: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 200,
    verbose: bool = True,
) -> None:
    """Save a radar (spider) chart of per-channel TFD contributions.

    Each polygon represents one PAM50 subtype pair.  The axes are the
    cell-type channels and the radial values are the **normalised**
    channel contributions (each polygon's values sum to 1), so shapes
    are directly comparable even when the absolute TFD magnitudes differ.

    Polygon colours are the 50/50 RGB blend of the two constituent
    subtype colours drawn from the same Crameri ``romaO`` palette used
    across the pipeline, so every pair can be traced back to its two
    subtypes visually.  A subtype colour key is placed below the chart
    for cross-reference.

    Parameters
    ----------
    results_json : path
        Path to ``tfd_separability_results.json``.
    output_dir : path
        Directory for the output PNG.
    cmap_name : str
        Crameri categorical colourmap for individual subtype colours
        (default: ``romaO``).
    include_background : bool
        Include the background channel (index 0) as a radar axis.
    figsize : (w, h) or None
        Auto-sized when ``None``.
    dpi : int
    verbose : bool
    """
    _check_matplotlib()
    setup_style()
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    results_json = Path(results_json)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(results_json) as f:
        data = json.load(f)

    class_names: List[str] = data["class_names"]
    per_channel: Dict[str, Dict[str, float]] = data["pairwise_per_channel"]
    pairwise_tfd: Dict[str, float] = data["pairwise_tfd"]

    # --- Channels to display ---
    ch_start = 0 if include_background else 1
    plot_channels = [(ch, name) for ch, name in CHANNEL_NAMES.items() if ch >= ch_start]
    ch_indices = [ch for ch, _ in plot_channels]
    ch_labels = [name for _, name in plot_channels]
    N = len(ch_indices)

    # --- Individual subtype colours (same palette as rest of pipeline) ---
    subtypes_present = sorted(class_names, key=lambda s: PAM50_ORDER.index(s) if s in PAM50_ORDER else 99)
    subtype_palette = build_label_palette(np.array(subtypes_present), cmap_name=cmap_name)

    # --- Radar geometry ---
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + angles[:1]   # close the polygon

    if figsize is None:
        figsize = (9.0, 9.0)

    fig = plt.figure(figsize=figsize)
    # Reserve bottom strip for the subtype colour key
    ax = fig.add_axes([0.08, 0.18, 0.84, 0.75], polar=True)

    pair_legend_handles: List = []

    for key, ch_vals in per_channel.items():
        a, b = key.split("__vs__")

        # Normalise contributions for this pair (relative channel importance)
        raw = np.array([float(ch_vals.get(str(c), 0.0)) for c in ch_indices])
        total = raw.sum()
        norm = raw / total if total > 0 else np.zeros(N)
        norm_closed = norm.tolist() + norm[:1].tolist()

        # Blended colour
        col_a = subtype_palette.get(a, "#888888")
        col_b = subtype_palette.get(b, "#888888")
        blended = _blend_colors(col_a, col_b)

        # Absolute TFD for the legend label
        abs_tfd = pairwise_tfd.get(key, float("nan"))
        tfd_str = f"{abs_tfd / 1e3:.0f}k" if np.isfinite(abs_tfd) else "—"
        label = f"{a} vs {b}  (TFD {tfd_str})"

        ax.plot(angles_closed, norm_closed, color=blended, linewidth=1.8, zorder=3)
        ax.fill(angles_closed, norm_closed, color=blended, alpha=0.10, zorder=2)

        pair_legend_handles.append(
            Line2D([0], [0], color=blended, linewidth=2.0, label=label)
        )

    # --- Axis styling ---
    ax.set_xticks(angles)
    ax.set_xticklabels(ch_labels, size=10)
    ax.set_rlabel_position(0)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.set_title(
        "Cell-Type Channel Contributions to TFD\n(relative, per subtype pair)",
        fontsize=12, pad=18,
    )

    # Subtle grid
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_facecolor("#fafafa")

    # --- Pair legend (right of chart, inside figure) ---
    pair_legend = fig.legend(
        handles=pair_legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.96),
        fontsize=8,
        title="Subtype pairs",
        title_fontsize=9,
        framealpha=0.85,
        edgecolor="grey",
    )

    # --- Subtype colour key (bottom strip) ---
    key_ax = fig.add_axes([0.08, 0.03, 0.84, 0.10])
    key_ax.set_axis_off()
    n_st = len(subtypes_present)
    x_positions = np.linspace(0.05, 0.95, n_st)
    for x_pos, subtype in zip(x_positions, subtypes_present):
        col = subtype_palette[subtype]
        key_ax.add_patch(
            plt.Rectangle(
                (x_pos - 0.04, 0.55), 0.08, 0.35,
                color=col, transform=key_ax.transAxes,
                clip_on=False,
            )
        )
        key_ax.text(
            x_pos, 0.25, subtype,
            ha="center", va="top",
            fontsize=9, transform=key_ax.transAxes,
        )
    key_ax.text(
        0.5, 1.05, "Individual subtype colours (blend of two = pair colour)",
        ha="center", va="bottom",
        fontsize=8, color="grey",
        transform=key_ax.transAxes,
    )

    out = output_dir / "tfd_radar_channel_contributions.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if verbose:
        print(f"[OK] Saved radar chart → {out}")


# ===================================================================
# § 3  Cell-type composition boxplots
# ===================================================================

#: Crameri colourmap used for the fixed cell-type colour assignment.
#: Kept separate from the subtype palette so the two never clash.
CELL_TYPE_CMAP = "batlowS"


def _collect_tissue_fractions(
    masks_dir: Path,
    patient_ids: List[str],
    max_tiles: Optional[int],
    seed: int = 42,
) -> np.ndarray:
    """Return (N_tiles, n_channels) tissue-area fraction array.

    For each tile mask (C, H, W):
    1. Per-pixel argmax assigns each pixel to its most probable class.
    2. Each channel's pixel count is divided by the total **non-background**
       pixel count (channel 0 excluded), yielding fractions that sum to 1
       across channels 1–6.
    3. Tiles where no non-background pixels are found are dropped.

    Normalising by non-background pixels removes the dominant background
    channel (~90 % of tile area) and gives stable, comparable fractions
    that directly reflect the tissue composition of each tile.
    """
    rng = np.random.default_rng(seed)
    n_ch_total = len(CHANNEL_NAMES)
    all_fracs: List[np.ndarray] = []

    per_patient: Optional[int] = (
        max(1, int(np.ceil(max_tiles / len(patient_ids))))
        if max_tiles is not None and len(patient_ids) > 0
        else None
    )

    for pid in patient_ids:
        zip_path = masks_dir / f"{pid}.zip"
        if not zip_path.exists():
            continue
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                tile_names = [n for n in zf.namelist() if n.endswith("_cls.npy")]
                if per_patient is not None and len(tile_names) > per_patient:
                    chosen = rng.choice(len(tile_names), per_patient, replace=False)
                    tile_names = [tile_names[i] for i in chosen]

                for name in tile_names:
                    try:
                        with zf.open(name) as f:
                            mask = np.load(BytesIO(f.read()))
                        if mask.ndim == 2:
                            continue
                        # Ensure (C, H, W)
                        if mask.ndim == 3 and mask.shape[0] >= mask.shape[1]:
                            mask = np.transpose(mask, (2, 0, 1))

                        n_ch = mask.shape[0]
                        pixel_classes = np.argmax(mask, axis=0).ravel()
                        counts = np.bincount(pixel_classes, minlength=n_ch).astype(float)

                        # Normalise by non-background pixels only
                        total_tissue = counts[1:].sum()
                        if total_tissue == 0:
                            continue
                        fracs = counts / total_tissue
                        all_fracs.append(fracs[:n_ch_total])
                    except Exception:
                        logger.debug("Failed to load %s from %s", name, zip_path)
        except Exception:
            logger.debug("Failed to open %s", zip_path)

    if not all_fracs:
        return np.empty((0, n_ch_total), dtype=float)

    arr = np.stack(all_fracs)
    if max_tiles is not None and arr.shape[0] > max_tiles:
        idx = rng.choice(arr.shape[0], max_tiles, replace=False)
        arr = arr[idx]
    return arr


def _build_cell_type_palette(
    ch_labels: List[str],
    cmap_name: str = CELL_TYPE_CMAP,
) -> Dict[str, str]:
    """Map cell-type names to maximally distinct Crameri colours.

    Samples the colourmap at positions spread evenly between 0.05 and 0.95
    (avoiding the identical-looking endpoints of sequential maps), then
    assigns colours in alphabetical label order so the mapping is always
    the same regardless of which channels are plotted.
    """
    import matplotlib as mpl
    from .core import get_crameri_cmap

    sorted_labels = sorted(ch_labels)
    n = len(sorted_labels)
    positions = np.linspace(0.05, 0.95, n)
    cmap = get_crameri_cmap(cmap_name)
    colours = [mpl.colors.to_hex(cmap(float(p))) for p in positions]
    return {label: col for label, col in zip(sorted_labels, colours)}


def plot_cell_type_boxplots(
    masks_dir: Union[str, Path],
    metadata_csv: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    patient_col: str = "Patient_ID",
    subtype_col: str = "Majority_Subtype_mRNA",
    classes: Optional[List[str]] = None,
    max_tiles_per_subtype: int = 2000,
    include_background: bool = False,
    cell_type_cmap: str = CELL_TYPE_CMAP,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> None:
    """Save side-by-side boxplots of cell-type tissue-area fractions per subtype.

    Layout
    ------
    One subplot per PAM50 subtype (five panels in a single row).  Within
    each panel the x-axis shows each cell type and the y-axis shows the
    **fraction of non-background tissue area** occupied by that cell type
    (expressed as a percentage).  The y-axis scale is **shared across all
    panels** so subtypes can be compared directly.

    Each cell type is assigned a fixed colour from the Crameri ``batlow``
    palette sampled at evenly-spread positions (0.05–0.95), avoiding the
    near-identical endpoint colours that plagued the previous approach.
    The same colour appears in every subtype panel.

    Parameters
    ----------
    masks_dir : path
        Directory containing per-patient ``<pid>.zip`` mask archives.
    metadata_csv : path
        CSV mapping patient IDs to PAM50 subtypes.
    output_dir : path
        Where to save the output PNG.
    n_bootstrap: int = 99,
        Column names in ``metadata_csv``.
    classes : list[str] or None
        Subtypes to include in PAM50 canonical order; ``None`` = all.
    max_tiles_per_subtype : int
        Maximum tiles sampled per subtype, distributed proportionally
        across patients.  ``None`` = use all available tiles.
    include_background : bool
        Include the background channel (index 0).  Default ``False``.
    cell_type_cmap : str
        Crameri colourmap for the fixed cell-type colours (default:
        ``batlow``).  Sampled at 0.05–0.95 to maximise distinctiveness.
    figsize : (w, h) or None
        Auto-sized when ``None``.
    dpi : int
    seed : int
    verbose : bool
    """
    _check_seaborn()
    setup_style()

    _pid = _extract_patient_id

    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Patient → subtype mapping ---
    df_meta = pd.read_csv(metadata_csv)
    df_meta["_pid"] = df_meta[patient_col].apply(_pid)
    patient_to_subtype: Dict[str, str] = dict(
        zip(df_meta["_pid"], df_meta[subtype_col])
    )

    available = sorted(df_meta[subtype_col].dropna().unique().tolist())
    if classes is not None:
        subtypes = [c for c in PAM50_ORDER if c in classes]
        subtypes += [c for c in classes if c not in PAM50_ORDER]
    else:
        subtypes = [c for c in PAM50_ORDER if c in available]
        subtypes += [c for c in available if c not in PAM50_ORDER]

    class_to_patients: Dict[str, List[str]] = {s: [] for s in subtypes}
    for pid, st in patient_to_subtype.items():
        if st in class_to_patients:
            class_to_patients[st].append(pid)

    # --- Channels to plot ---
    ch_start = 0 if include_background else 1
    channels_to_plot = [
        (ch, name) for ch, name in CHANNEL_NAMES.items() if ch >= ch_start
    ]
    ch_indices = [ch for ch, _ in channels_to_plot]
    ch_labels  = [name for _, name in channels_to_plot]

    # Fixed colour per cell type — spread evenly across colourmap (0.05–0.95)
    # to avoid near-identical endpoint colours
    cell_type_palette = _build_cell_type_palette(ch_labels, cmap_name=cell_type_cmap)

    # --- Collect tissue-area fractions per subtype ---
    records: List[Dict] = []
    for subtype in subtypes:
        pids = class_to_patients[subtype]
        if not pids:
            continue
        if verbose:
            print(f"  [{subtype}] Loading masks for {len(pids)} patients…")
        fracs_arr = _collect_tissue_fractions(
            masks_dir, pids, max_tiles_per_subtype, seed=seed,
        )
        if fracs_arr.shape[0] == 0:
            logger.warning("%s: no masks found in %s", subtype, masks_dir)
            continue
        if verbose:
            print(f"  [{subtype}] {fracs_arr.shape[0]} tiles processed")

        for tile_fracs in fracs_arr:
            for ch in ch_indices:
                if ch < len(tile_fracs):
                    records.append(
                        {
                            "Subtype": subtype,
                            "Cell Type": CHANNEL_NAMES[ch],
                            "Channel": ch,
                            "Tissue Fraction": float(tile_fracs[ch]),
                        }
                    )

    if not records:
        logger.error("No tissue fraction data collected — aborting boxplot.")
        return

    df_long = pd.DataFrame(records)

    # --- Figure: 1 row × n_subtypes cols, shared y-axis ---
    n_subtypes = len(subtypes)
    if figsize is None:
        figsize = (n_subtypes * 4.2, 5.2)

    fig, axes = plt.subplots(
        1, n_subtypes,
        figsize=figsize,
        sharey=True,
        squeeze=False,
    )

    for col_idx, subtype in enumerate(subtypes):
        ax = axes[0][col_idx]
        subset = df_long[df_long["Subtype"] == subtype]
        present = [name for name in ch_labels if name in subset["Cell Type"].values]

        sns.boxplot(
            data=subset,
            x="Cell Type",
            y="Tissue Fraction",
            order=present,
            palette=cell_type_palette,
            linewidth=0.8,
            fliersize=1.5,
            flierprops=dict(alpha=0.3, marker="o", markersize=1.5),
            ax=ax,
        )
        ax.set_title(subtype, fontsize=12, fontweight="bold", pad=6)
        ax.set_xlabel("")
        ax.set_ylabel("Fraction of tissue area" if col_idx == 0 else "")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(axis="x", labelrotation=40)

        if col_idx > 0:
            ax.tick_params(axis="y", labelleft=False)

    fig.suptitle(
        "Cell-Type Tissue Composition per PAM50 Subtype",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    out = output_dir / "tfd_cell_type_boxplots.png"
    save_figure(fig, out, dpi=dpi)
    if verbose:
        print(f"[OK] Saved cell-type boxplots → {out}")


# ===================================================================
# § 5  Ternary tissue composition plot
# ===================================================================

#: Channel indices for the three ternary components
_EPITHELIUM_CH   = 2
_IMMUNE_CHANNELS = (1, 3, 4, 5)   # Neutrophils, Lymphocytes, Plasma Cells, Eosinophils
_CONNECTIVE_CH   = 6


def _ternary_to_cart(a: float, b: float, c: float) -> Tuple[float, float]:
    """Barycentric (a, b, c) → Cartesian (x, y).

    Vertex positions
    ----------------
    * a = 1  Epithelium  → (0,   0)       bottom-left
    * b = 1  Immune      → (1,   0)       bottom-right
    * c = 1  Connective  → (0.5, √3/2)   top
    """
    x = b + 0.5 * c
    y = (np.sqrt(3) / 2) * c
    return float(x), float(y)


def _draw_ternary_axes(ax: "Figure", step: float = 0.25) -> None:
    """Render triangle border, interior gridlines, vertex labels, and edge ticks."""
    # Triangle border
    verts = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)]
    pts = [_ternary_to_cart(*v) for v in verts]
    xs, ys = zip(*pts)
    ax.plot(list(xs), list(ys), "k-", lw=1.5, zorder=5)

    grid_kw = dict(color="#aaaaaa", ls="--", lw=0.5, alpha=0.7, zorder=1)
    tick_vals = np.arange(step, 1.0, step)

    for k in tick_vals:
        # iso-a (constant Epithelium): left-edge point → bottom-edge point
        p0, p1 = _ternary_to_cart(k, 0, 1 - k), _ternary_to_cart(k, 1 - k, 0)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], **grid_kw)

        # iso-b (constant Immune): right-edge point → bottom-edge point
        p0, p1 = _ternary_to_cart(0, k, 1 - k), _ternary_to_cart(1 - k, k, 0)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], **grid_kw)

        # iso-c (constant Connective): left-edge point → right-edge point
        p0, p1 = _ternary_to_cart(1 - k, 0, k), _ternary_to_cart(0, 1 - k, k)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], **grid_kw)

        pct = f"{int(round(k * 100))}%"
        # Epithelium ticks on bottom edge at (a=k, b=1-k, c=0)
        x, y = _ternary_to_cart(k, 1 - k, 0)
        ax.text(x, y - 0.04, pct, ha="center", va="top", fontsize=7, color="#666666")
        # Connective ticks on left edge at (a=1-k, b=0, c=k)
        x, y = _ternary_to_cart(1 - k, 0, k)
        ax.text(x - 0.03, y, pct, ha="right", va="center", fontsize=7, color="#666666")
        # Immune ticks on right edge at (a=0, b=k, c=1-k)
        x, y = _ternary_to_cart(0, k, 1 - k)
        ax.text(x + 0.03, y, pct, ha="left", va="center", fontsize=7, color="#666666")

    # Vertex labels
    vx, vy = _ternary_to_cart(1, 0, 0)
    ax.text(vx - 0.05, vy - 0.06, "Epithelium\n(tumor)",
            ha="right", va="top", fontsize=10, fontweight="bold")
    vx, vy = _ternary_to_cart(0, 1, 0)
    ax.text(vx + 0.05, vy - 0.06, "Immune\ncells",
            ha="left", va="top", fontsize=10, fontweight="bold")
    vx, vy = _ternary_to_cart(0, 0, 1)
    ax.text(vx, vy + 0.06, "Connective\n(stroma)",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(-0.18, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_ternary_composition(
    masks_dir: Union[str, Path],
    metadata_csv: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    patient_col: str = "Patient_ID",
    subtype_col: str = "Majority_Subtype_mRNA",
    classes: Optional[List[str]] = None,
    max_tiles_per_subtype: int = 2000,
    cmap_name: str = CATEGORICAL_CMAP,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> None:
    """Save a ternary (triangle) plot of per-tile tissue composition.

    Each tile is a single point in a three-component triangle whose axes are:

    * **Epithelium** — channel 2 fraction of non-background tissue area
    * **Immune**     — channels 1 + 3 + 4 + 5 combined fraction
      (Neutrophils + Lymphocytes + Plasma Cells + Eosinophils)
    * **Connective** — channel 6 fraction

    These three components always sum to 1, so every tile maps to exactly
    one point in the triangle.  PAM50 subtypes are shown with the same
    Crameri ``romaO`` colours used throughout the pipeline.  A large diamond
    marker at the per-subtype median provides a group-level summary.

    Parameters
    ----------
    masks_dir : path
        Directory of per-patient ``<pid>.zip`` mask archives.
    metadata_csv : path
        CSV with patient → PAM50 subtype mapping.
    output_dir : path
        Where to save ``tfd_ternary_composition.png``.
    patient_col, subtype_col : str
        Column names in ``metadata_csv``.
    classes : list[str] or None
        Subtypes to include; ``None`` = all found in metadata.
    max_tiles_per_subtype : int
        Tile budget per subtype.
    cmap_name : str
        Crameri categorical colourmap for subtype colours (default: ``romaO``).
    figsize : (w, h) or None
        Auto-sized when ``None``.
    dpi : int
    seed : int
    verbose : bool
    """
    _check_matplotlib()
    setup_style()
    from matplotlib.lines import Line2D
    _pid = _extract_patient_id

    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Patient → subtype mapping
    df_meta = pd.read_csv(metadata_csv)
    df_meta["_pid"] = df_meta[patient_col].apply(_pid)
    patient_to_subtype = dict(zip(df_meta["_pid"], df_meta[subtype_col]))

    available = sorted(df_meta[subtype_col].dropna().unique().tolist())
    if classes is not None:
        subtypes = [c for c in PAM50_ORDER if c in classes]
        subtypes += [c for c in classes if c not in PAM50_ORDER]
    else:
        subtypes = [c for c in PAM50_ORDER if c in available]
        subtypes += [c for c in available if c not in PAM50_ORDER]

    class_to_patients: Dict[str, List[str]] = {s: [] for s in subtypes}
    for p, st in patient_to_subtype.items():
        if st in class_to_patients:
            class_to_patients[st].append(p)

    subtype_palette = build_label_palette(np.array(subtypes), cmap_name=cmap_name)

    # Collect ternary coordinates per subtype
    ternary_data: Dict[str, np.ndarray] = {}
    for subtype in subtypes:
        pids = class_to_patients[subtype]
        if not pids:
            continue
        if verbose:
            print(f"  [{subtype}] Loading masks for {len(pids)} patients…")
        fracs_arr = _collect_tissue_fractions(masks_dir, pids, max_tiles_per_subtype, seed=seed)
        if fracs_arr.shape[0] == 0:
            logger.warning("%s: no masks found in %s", subtype, masks_dir)
            continue

        epi  = fracs_arr[:, _EPITHELIUM_CH].copy()
        imm  = fracs_arr[:, sorted(_IMMUNE_CHANNELS)].sum(axis=1)
        conn = fracs_arr[:, _CONNECTIVE_CH].copy()

        # Renormalise to strict sum=1 (fracs already sum to 1 across 1-6 but
        # background is excluded, so epi+imm+conn covers all non-background)
        total = epi + imm + conn
        valid = total > 0
        epi[valid]  /= total[valid]
        imm[valid]  /= total[valid]
        conn[valid] /= total[valid]

        ternary_data[subtype] = np.stack(
            [epi[valid], imm[valid], conn[valid]], axis=1
        )
        if verbose:
            print(f"  [{subtype}] {ternary_data[subtype].shape[0]} tiles → ternary coordinates")

    if not ternary_data:
        logger.error("No ternary data collected — aborting ternary plot.")
        return

    if figsize is None:
        figsize = (9.0, 8.5)

    fig, ax = plt.subplots(figsize=figsize)
    _draw_ternary_axes(ax)

    legend_handles: List = []
    for subtype in subtypes:
        if subtype not in ternary_data:
            continue
        abc = ternary_data[subtype]   # (N, 3): [epi, imm, conn]
        color = subtype_palette[subtype]

        cart = np.array([_ternary_to_cart(row[0], row[1], row[2]) for row in abc])
        ax.scatter(cart[:, 0], cart[:, 1],
                   s=4, c=[color], alpha=0.25, linewidths=0, zorder=2)

        # Median re-normalised so it lies on the simplex
        med = np.median(abc, axis=0)
        if med.sum() > 0:
            med = med / med.sum()
        mx, my = _ternary_to_cart(*med)
        ax.scatter([mx], [my], s=140, c=[color], marker="D",
                   edgecolors="white", linewidths=1.2, zorder=6)

        legend_handles.append(
            Line2D([0], [0], marker="D", color="none",
                   markerfacecolor=color, markeredgecolor="white",
                   markersize=9, label=subtype)
        )

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=9,
        title="PAM50 subtype\n(◆ = median)",
        title_fontsize=9,
        framealpha=0.88,
        edgecolor="grey",
    )
    ax.set_title(
        "Tissue Composition per Tile — Ternary Plot\n"
        "(Epithelium · Immune · Connective)",
        fontsize=12, pad=10,
    )

    out = output_dir / "tfd_ternary_composition.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if verbose:
        print(f"[OK] Saved ternary composition plot → {out}")


# ===================================================================
# § 6  Spatial topology: nearest-neighbour distances
# ===================================================================

def _collect_spatial_stats(
    masks_dir: Path,
    patient_ids: List[str],
    non_bg_channels: List[int],
    max_tiles: Optional[int],
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Return per-tile nearest-neighbour distances between cell types.

    For each tile:
    1. Per-pixel argmax assigns each pixel to the most probable class.
    2. Connected-component centroids are extracted for each non-background
       channel via ``scipy.ndimage``.
    3. Two arrays are built:

       * ``nn_intra`` (N, n_ch): mean distance from each cell to its nearest
         same-type neighbour.  ``NaN`` when fewer than 2 cells of that type
         are present in the tile.
       * ``nn_cross`` (N, n_ch, n_ch): ``[i, j]`` = mean distance from every
         cell of type *i* to the nearest cell of type *j* (``NaN`` if either
         type has zero cells; diagonal = intra-type, requires ≥ 2 cells).

    All distances are in pixels.  ``nn_intra`` equals the diagonal of
    ``nn_cross``.

    Parameters
    ----------
    masks_dir : Path
    patient_ids : list[str]
    non_bg_channels : list[int]
        Channel indices to analyse (typically 1–6; background excluded).
    max_tiles : int or None
        Total tile budget across all patients; ``None`` = use all.
    seed : int
    """
    try:
        from scipy.ndimage import label as nd_label
        from scipy.ndimage import center_of_mass as nd_com
        from scipy.spatial import KDTree
    except ImportError as exc:
        raise ImportError("scipy is required for spatial NN analysis") from exc

    rng = np.random.default_rng(seed)
    n_ch = len(non_bg_channels)
    nn_intra_all: List[np.ndarray] = []
    nn_cross_all: List[np.ndarray] = []

    per_patient: Optional[int] = (
        max(1, int(np.ceil(max_tiles / len(patient_ids))))
        if max_tiles is not None and len(patient_ids) > 0
        else None
    )

    for pid in patient_ids:
        zip_path = masks_dir / f"{pid}.zip"
        if not zip_path.exists():
            continue
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                tile_names = [n for n in zf.namelist() if n.endswith("_cls.npy")]
                if per_patient is not None and len(tile_names) > per_patient:
                    chosen = rng.choice(len(tile_names), per_patient, replace=False)
                    tile_names = [tile_names[i] for i in chosen]

                for name in tile_names:
                    try:
                        with zf.open(name) as f:
                            mask = np.load(BytesIO(f.read()))
                        if mask.ndim == 2:
                            continue
                        if mask.ndim == 3 and mask.shape[0] >= mask.shape[1]:
                            mask = np.transpose(mask, (2, 0, 1))

                        pixel_classes = np.argmax(mask, axis=0)

                        # Centroids per channel via connected components
                        centroids: List[Optional[np.ndarray]] = []
                        for ch in non_bg_channels:
                            binary = (pixel_classes == ch)
                            labeled, n_comp = nd_label(binary)
                            if n_comp == 0:
                                centroids.append(None)
                                continue
                            coms = nd_com(binary, labeled, list(range(1, n_comp + 1)))
                            centroids.append(np.array(coms))  # (n_comp, 2) in (row, col)

                        # Cross-type NN matrix; diagonal = intra-type
                        nn_cross_tile = np.full((n_ch, n_ch), np.nan)
                        for i, cents_a in enumerate(centroids):
                            if cents_a is None or len(cents_a) == 0:
                                continue
                            for j, cents_b in enumerate(centroids):
                                if cents_b is None or len(cents_b) == 0:
                                    continue
                                if i == j:
                                    if len(cents_b) < 2:
                                        continue
                                    tree = KDTree(cents_b)
                                    dists, _ = tree.query(cents_a, k=2)
                                    nn_cross_tile[i, j] = dists[:, 1].mean()
                                else:
                                    tree = KDTree(cents_b)
                                    dists, _ = tree.query(cents_a, k=1)
                                    nn_cross_tile[i, j] = dists.mean()  # k=1 → 1D

                        nn_intra_all.append(np.diag(nn_cross_tile).copy())
                        nn_cross_all.append(nn_cross_tile)

                    except Exception:
                        logger.debug("Spatial stats failed: %s in %s", name, zip_path)
        except Exception:
            logger.debug("Cannot open %s", zip_path)

    if not nn_intra_all:
        return {
            "nn_intra": np.empty((0, n_ch), dtype=float),
            "nn_cross": np.empty((0, n_ch, n_ch), dtype=float),
        }

    nn_intra_arr = np.stack(nn_intra_all)
    nn_cross_arr = np.stack(nn_cross_all)
    if max_tiles is not None and nn_intra_arr.shape[0] > max_tiles:
        idx = rng.choice(nn_intra_arr.shape[0], max_tiles, replace=False)
        nn_intra_arr = nn_intra_arr[idx]
        nn_cross_arr = nn_cross_arr[idx]
    return {"nn_intra": nn_intra_arr, "nn_cross": nn_cross_arr}


def _collect_spatial_stats_for_subtypes(
    masks_dir: Path,
    subtypes: List[str],
    patient_to_subtype: Dict[str, str],
    non_bg_channels: List[int],
    max_tiles: Optional[int],
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Collect spatial NN stats for each subtype; returns ``{subtype: stats}``."""
    class_to_patients: Dict[str, List[str]] = {s: [] for s in subtypes}
    for p, st in patient_to_subtype.items():
        if st in class_to_patients:
            class_to_patients[st].append(p)

    stats_by_subtype: Dict[str, Dict[str, np.ndarray]] = {}
    for subtype in subtypes:
        pids = class_to_patients[subtype]
        if not pids:
            continue
        if verbose:
            print(f"  [{subtype}] Computing spatial NN stats ({len(pids)} patients)…")
        stats = _collect_spatial_stats(
            masks_dir, pids, non_bg_channels, max_tiles, seed=seed
        )
        if stats["nn_intra"].shape[0] == 0:
            logger.warning("%s: no spatial stats collected", subtype)
            continue
        stats_by_subtype[subtype] = stats
        if verbose:
            print(f"  [{subtype}] {stats['nn_intra'].shape[0]} tiles processed")
    return stats_by_subtype


def _collect_spatial_topology_for_subtypes(
    masks_dir: Path,
    subtypes: List[str],
    patient_to_subtype: Dict[str, str],
    non_bg_channels: List[int],
    max_tiles: Optional[int],
    n_bootstrap: int = 99,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, List[Dict[str, object]]]:
    """Collect tile-level topology metrics for each subtype.

    Returns a mapping ``{subtype: [tile_metrics, ...]}`` compatible with the
    spatial summary plot functions in ``spatial_metrics_viz.py``.
    """
    rng = np.random.default_rng(seed)
    class_to_patients: Dict[str, List[str]] = {s: [] for s in subtypes}
    for p, st in patient_to_subtype.items():
        if st in class_to_patients:
            class_to_patients[st].append(p)

    stats_by_subtype: Dict[str, List[Dict[str, object]]] = {}
    per_patient: Optional[int] = (
        max(1, int(np.ceil(max_tiles / len(subtypes))))
        if max_tiles is not None and len(subtypes) > 0
        else None
    )

    for subtype in subtypes:
        pids = class_to_patients[subtype]
        if not pids:
            continue
        if verbose:
            print(f"  [{subtype}] Computing spatial topology stats ({len(pids)} patients)…")

        tile_metrics: List[Dict[str, object]] = []
        for pid in pids:
            zip_path = masks_dir / f"{pid}.zip"
            if not zip_path.exists():
                continue
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    tile_names = [n for n in zf.namelist() if n.endswith("_cls.npy")]
                    if per_patient is not None and len(tile_names) > per_patient:
                        chosen = rng.choice(len(tile_names), per_patient, replace=False)
                        tile_names = [tile_names[i] for i in chosen]

                    for name in tile_names:
                        try:
                            with zf.open(name) as f:
                                mask = np.load(BytesIO(f.read()))
                            if mask.ndim == 2:
                                continue
                            if mask.ndim == 3 and mask.shape[0] >= mask.shape[1]:
                                mask = np.transpose(mask, (2, 0, 1))
                            if mask.ndim != 3:
                                continue

                            # Use the union of all non-background classes as the
                            # foreground cell mask for topology metrics.
                            foreground = np.sum(mask[non_bg_channels, :, :], axis=0)
                            metrics = compute_spatial_metrics_per_tile(
                                foreground,
                                channel_idx=-1,
                                bounding_box=(foreground.shape[1], foreground.shape[0]),
                                n_bootstrap=n_bootstrap,
                            )
                            tile_metrics.append(metrics)
                        except Exception:
                            logger.debug("Topology stats failed: %s in %s", name, zip_path)
            except Exception:
                logger.debug("Cannot open %s", zip_path)

        if tile_metrics:
            stats_by_subtype[subtype] = tile_metrics
            if verbose:
                print(f"  [{subtype}] {len(tile_metrics)} tiles processed")

    return stats_by_subtype


def plot_nn_distance_violins(
    stats_by_subtype: Dict[str, Dict[str, np.ndarray]],
    non_bg_channels: List[int],
    output_dir: Union[str, Path],
    *,
    cell_type_cmap: str = CELL_TYPE_CMAP,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 200,
    verbose: bool = True,
) -> None:
    """Save violin plots of intra-type nearest-neighbour distances per subtype.

    Layout mirrors the cell-type boxplots: one panel per PAM50 subtype,
    x-axis = cell type, y-axis = mean distance (px) from each cell to its
    nearest same-type neighbour.  Tiles where a type has fewer than two
    cells are dropped (``NaN``).  The y-axis is shared for direct comparison.

    Uses the same Crameri ``batlow`` cell-type palette as the composition
    boxplots so colours are consistent across all figures.

    Parameters
    ----------
    stats_by_subtype : dict
        Output of :func:`_collect_spatial_stats_for_subtypes`.
    non_bg_channels : list[int]
        Channel indices included in the stats (typically 1–6).
    output_dir : path
        Where to save ``tfd_nn_distance_violins.png``.
    cell_type_cmap : str
    figsize : (w, h) or None
    dpi : int
    verbose : bool
    """
    _check_seaborn()
    setup_style()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subtypes = [s for s in PAM50_ORDER if s in stats_by_subtype]
    subtypes += [s for s in stats_by_subtype if s not in PAM50_ORDER]
    ch_labels = [CHANNEL_NAMES[ch] for ch in non_bg_channels]
    cell_type_palette = _build_cell_type_palette(ch_labels, cmap_name=cell_type_cmap)

    # Long-form DataFrame; drop NaN (tiles with < 2 cells of that type)
    records: List[Dict] = []
    for subtype in subtypes:
        nn_intra = stats_by_subtype[subtype]["nn_intra"]  # (N, n_ch)
        for i, ch in enumerate(non_bg_channels):
            vals = nn_intra[:, i]
            for v in vals[np.isfinite(vals)]:
                records.append({
                    "Subtype": subtype,
                    "Cell Type": CHANNEL_NAMES[ch],
                    "NN Distance (px)": float(v),
                })

    if not records:
        logger.error("No intra-type NN data collected — skipping violin plot.")
        return

    df_long = pd.DataFrame(records)
    n_subtypes = len(subtypes)
    if figsize is None:
        figsize = (n_subtypes * 4.2, 5.2)

    fig, axes = plt.subplots(
        1, n_subtypes, figsize=figsize, sharey=True, squeeze=False
    )

    for col_idx, subtype in enumerate(subtypes):
        ax = axes[0][col_idx]
        subset = df_long[df_long["Subtype"] == subtype]
        present = [name for name in ch_labels if name in subset["Cell Type"].values]

        if subset.empty:
            ax.set_title(subtype, fontsize=12, fontweight="bold", pad=6)
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            continue

        sns.violinplot(
            data=subset,
            x="Cell Type",
            y="NN Distance (px)",
            order=present,
            palette=cell_type_palette,
            linewidth=0.8,
            inner="quartile",
            cut=0,
            ax=ax,
        )
        ax.set_title(subtype, fontsize=12, fontweight="bold", pad=6)
        ax.set_xlabel("")
        ax.set_ylabel("Mean NN distance (px)" if col_idx == 0 else "")
        ax.tick_params(axis="x", labelrotation=40)
        if col_idx > 0:
            ax.tick_params(axis="y", labelleft=False)

    fig.suptitle(
        "Intra-Type Nearest-Neighbour Distance per PAM50 Subtype\n"
        "(mean distance from each cell to its nearest same-type neighbour)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    out = output_dir / "tfd_nn_distance_violins.png"
    save_figure(fig, out, dpi=dpi)
    if verbose:
        print(f"[OK] Saved NN distance violin plots → {out}")


def plot_cross_type_proximity_heatmaps(
    stats_by_subtype: Dict[str, Dict[str, np.ndarray]],
    non_bg_channels: List[int],
    output_dir: Union[str, Path],
    *,
    cmap_name: str = HEATMAP_CMAP,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 200,
    verbose: bool = True,
) -> None:
    """Save side-by-side heatmaps of cross-cell-type nearest-neighbour distances.

    One heatmap per PAM50 subtype.  Cell *(i, j)* shows the **median**
    (over all tiles) of the mean distance from a cell of type *i* to the
    nearest cell of type *j*.  The diagonal is the intra-type clustering
    distance.

    Key biological readouts
    -----------------------
    * **Lymphocyte → Epithelium**: shorter in high-TIL subtypes (Basal/Her2)
      → tumour-immune interface proximity.
    * **Connective → Epithelium**: reflects desmoplastic stromal response
      (typically elevated in LumA/LumB).
    * **Lymphocyte → Lymphocyte** (diagonal): small = aggregated TIL clusters;
      large = dispersed infiltration.

    A shared colour scale across all panels makes subtype differences
    immediately visible.  Colourmap: Crameri ``lajolla`` (same as the
    channel contribution heatmaps).

    Parameters
    ----------
    stats_by_subtype : dict
        Output of :func:`_collect_spatial_stats_for_subtypes`.
    non_bg_channels : list[int]
        Channel indices included in the stats (typically 1–6).
    output_dir : path
        Where to save ``tfd_cross_type_proximity.png``.
    cmap_name : str
    figsize : (w, h) or None
    dpi : int
    verbose : bool
    """
    _check_matplotlib()
    setup_style()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subtypes = [s for s in PAM50_ORDER if s in stats_by_subtype]
    subtypes += [s for s in stats_by_subtype if s not in PAM50_ORDER]
    ch_labels = [CHANNEL_NAMES[ch] for ch in non_bg_channels]
    n_ch = len(non_bg_channels)
    # Prefer a sequential warm colormap for distances
    try:
        cmap = plt.get_cmap("magma")
    except Exception:
        cmap = get_crameri_cmap(cmap_name)

    # Per-subtype median cross-type distance matrix
    matrices: Dict[str, np.ndarray] = {}
    for subtype in subtypes:
        if subtype not in stats_by_subtype:
            continue
        nn_cross = stats_by_subtype[subtype]["nn_cross"]  # (N, n_ch, n_ch)
        matrices[subtype] = np.nanmedian(nn_cross, axis=0)

    if not matrices:
        logger.error("No cross-type proximity data — skipping heatmaps.")
        return

    # Shared colour scale — 5th to 95th percentile of all finite values
    all_vals = np.concatenate([m.ravel() for m in matrices.values()])
    all_vals = all_vals[np.isfinite(all_vals)]
    vmin = float(np.percentile(all_vals, 5))
    vmax = float(np.percentile(all_vals, 95))

    n_subtypes = len(subtypes)
    if figsize is None:
        figsize = (n_subtypes * 3.6 + 1.2, 5.2)

    fig, axes = plt.subplots(1, n_subtypes, figsize=figsize, squeeze=False)
    norm_range = vmax - vmin + 1e-9
    im_ref = None

    for col_idx, subtype in enumerate(subtypes):
        ax = axes[0][col_idx]
        if subtype not in matrices:
            ax.set_visible(False)
            continue
        mat = matrices[subtype]
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="equal", interpolation="nearest")
        im_ref = im
        # Hide spines so cell boundaries are not emphasised by white lines
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks(range(n_ch))
        ax.set_xticklabels(ch_labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(n_ch))
        ax.set_yticklabels(ch_labels if col_idx == 0 else [], fontsize=8)
        ax.grid(False)
        ax.set_title(subtype, fontsize=11, fontweight="bold", pad=5)

        for i in range(n_ch):
            for j in range(n_ch):
                v = mat[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=7, color="#888888")
                    continue
                txt_col = "white" if (v - vmin) / norm_range > 0.6 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color=txt_col)

    if im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=axes[0, -1], fraction=0.046, pad=0.10)
        cbar.set_label("Median NN distance (px)", fontsize=9)

    fig.suptitle(
        "Cross–Cell-Type Nearest-Neighbour Distances per PAM50 Subtype\n"
        "row = source type  ·  col = target type  ·  median over tiles (px)",
        fontsize=11, y=1.03,
    )
    fig.tight_layout()

    out = output_dir / "tfd_cross_type_proximity.png"
    save_figure(fig, out, dpi=dpi)
    if verbose:
        print(f"[OK] Saved cross-type proximity heatmaps → {out}")


# ===================================================================
# § 4  Pipeline entry point
# ===================================================================

def run_tfd_separability_viz(cfg: dict, verbose: bool = True) -> None:
    """Config-driven entry point for TFD separability visualisation.

    Expected config keys (``tfd_separability_viz`` section of ``config.yaml``)
    ---------------------------------------------------------------------------
    results_json : str
        Path to ``tfd_separability_results.json``.
    masks_dir : str
        Directory of per-patient segmentation mask ZIPs.
    metadata_csv : str
        CSV with patient → subtype mapping.
    patient_col : str
        Column name for patient IDs (default: ``Patient_ID``).
    subtype_col : str
        Column name for PAM50 subtypes (default: ``Majority_Subtype_mRNA``).
    classes : list[str] or null
        Subtypes to include; ``null`` = all in metadata CSV.
    output_dir : str
        Where to write output PNGs.
    max_tiles_per_subtype : int
        Maximum tiles sampled per subtype for boxplots (default: 2000).
    include_background : bool
        Include background channel in boxplots (default: false).
    cmap_categorical : str
        Crameri categorical colourmap for subtype colours (default: romaO).
    cmap_heatmap : str
        Crameri sequential colourmap for the heatmap (default: lajolla).
    figsize_heatmap : [w, h] or null
    figsize_boxplot : [w, h] or null
    dpi : int
        Save resolution (default: 200).
    max_tiles_spatial : int
        Tile budget per subtype for spatial NN computations (default: 500).
        Kept separate from ``max_tiles_per_subtype`` because connected-component
        extraction is significantly slower than pixel-fraction counting.
    plots : list[str]
        Which plots to generate.  Choices:
        ``channel_contributions``, ``radar_contributions``,
        ``cell_type_boxplots``, ``ternary_composition``,
        ``nn_distance_violins``, ``cross_type_proximity``,
        ``ripley_L_by_subtype``, ``voronoi_distribution``,
        ``knn_metrics``.
    """
    results_json = cfg.get("results_json")
    masks_dir = cfg.get("masks_dir")
    metadata_csv = cfg.get("metadata_csv")
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    plots = cfg.get("plots", ["channel_contributions", "radar_contributions", "cell_type_boxplots"])
    dpi = cfg.get("dpi", 200)

    if "radar_contributions" in plots:
        if not results_json:
            logger.warning("'results_json' not set — skipping radar_contributions plot.")
        else:
            if verbose:
                print("\n[TFD-Viz] Generating radar channel contribution chart…")
            figsize_raw = cfg.get("figsize_radar")
            figsize = tuple(figsize_raw) if figsize_raw else None
            plot_radar_channel_contributions(
                results_json=results_json,
                output_dir=output_dir,
                cmap_name=cfg.get("cmap_categorical", CATEGORICAL_CMAP),
                include_background=cfg.get("include_background", False),
                figsize=figsize,
                dpi=dpi,
                verbose=verbose,
            )

    if "channel_contributions" in plots:
        if not results_json:
            logger.warning("'results_json' not set — skipping channel_contributions plot.")
        else:
            if verbose:
                print("\n[TFD-Viz] Generating channel contribution heatmap…")
            figsize_raw = cfg.get("figsize_heatmap")
            figsize = tuple(figsize_raw) if figsize_raw else None
            plot_channel_contributions(
                results_json=results_json,
                output_dir=output_dir,
                cmap_name=cfg.get("cmap_heatmap", HEATMAP_CMAP),
                figsize=figsize,
                dpi=dpi,
                verbose=verbose,
            )

    if "cell_type_boxplots" in plots:
        if not masks_dir or not metadata_csv:
            logger.warning(
                "'masks_dir' or 'metadata_csv' not set — skipping cell_type_boxplots."
            )
        else:
            if verbose:
                print("\n[TFD-Viz] Generating cell-type composition boxplots…")
            figsize_raw = cfg.get("figsize_boxplot")
            figsize = tuple(figsize_raw) if figsize_raw else None
            plot_cell_type_boxplots(
                masks_dir=masks_dir,
                metadata_csv=metadata_csv,
                output_dir=output_dir,
                patient_col=cfg.get("patient_col", "Patient_ID"),
                subtype_col=cfg.get("subtype_col", "Majority_Subtype_mRNA"),
                classes=cfg.get("classes"),
                max_tiles_per_subtype=cfg.get("max_tiles_per_subtype", 2000),
                include_background=cfg.get("include_background", False),
                cell_type_cmap=cfg.get("cmap_cell_types", CELL_TYPE_CMAP),
                figsize=figsize,
                dpi=dpi,
                verbose=verbose,
            )

    if "ternary_composition" in plots:
        if not masks_dir or not metadata_csv:
            logger.warning(
                "'masks_dir' or 'metadata_csv' not set — skipping ternary_composition."
            )
        else:
            if verbose:
                print("\n[TFD-Viz] Generating ternary tissue composition plot…")
            figsize_raw = cfg.get("figsize_ternary")
            figsize = tuple(figsize_raw) if figsize_raw else None
            plot_ternary_composition(
                masks_dir=masks_dir,
                metadata_csv=metadata_csv,
                output_dir=output_dir,
                patient_col=cfg.get("patient_col", "Patient_ID"),
                subtype_col=cfg.get("subtype_col", "Majority_Subtype_mRNA"),
                classes=cfg.get("classes"),
                max_tiles_per_subtype=cfg.get("max_tiles_per_subtype", 2000),
                cmap_name=cfg.get("cmap_categorical", CATEGORICAL_CMAP),
                figsize=figsize,
                dpi=dpi,
                verbose=verbose,
            )

    # --- Spatial NN plots (B + C): shared data collection ---
    _need_spatial = (
        "nn_distance_violins" in plots
        or "cross_type_proximity" in plots
        or "ripley_L_by_subtype" in plots
        or "voronoi_distribution" in plots
        or "knn_metrics" in plots
    )
    if _need_spatial:
        if not masks_dir or not metadata_csv:
            logger.warning(
                "'masks_dir' or 'metadata_csv' not set — skipping spatial NN plots."
            )
        else:
            _patient_col = cfg.get("patient_col", "Patient_ID")
            _subtype_col = cfg.get("subtype_col", "Majority_Subtype_mRNA")
            _df_meta = pd.read_csv(metadata_csv)
            _df_meta["_pid"] = _df_meta[_patient_col].apply(_extract_patient_id)
            _p2s = dict(zip(_df_meta["_pid"], _df_meta[_subtype_col]))
            _available = sorted(_df_meta[_subtype_col].dropna().unique().tolist())
            _classes = cfg.get("classes")
            if _classes is not None:
                _subtypes = [c for c in PAM50_ORDER if c in _classes]
                _subtypes += [c for c in _classes if c not in PAM50_ORDER]
            else:
                _subtypes = [c for c in PAM50_ORDER if c in _available]
                _subtypes += [c for c in _available if c not in PAM50_ORDER]

            _non_bg = list(range(1, 7))
            _max_sp = cfg.get("max_tiles_spatial", 500)
            _ripley_bootstrap = cfg.get("ripley_n_bootstrap", 99)
            if verbose:
                print(
                    f"\n[TFD-Viz] Computing spatial NN statistics "
                    f"(max {_max_sp} tiles/subtype via scipy connected components; "
                    f"Ripley bootstrap={_ripley_bootstrap})…"
                )

            _stats = _collect_spatial_stats_for_subtypes(
                masks_dir=Path(masks_dir),
                subtypes=_subtypes,
                patient_to_subtype=_p2s,
                non_bg_channels=_non_bg,
                max_tiles=_max_sp,
                seed=cfg.get("seed", 42),
                verbose=verbose,
            )

            if "nn_distance_violins" in plots:
                if verbose:
                    print("\n[TFD-Viz] Generating intra-type NN distance violin plots…")
                figsize_raw = cfg.get("figsize_nn_violins")
                figsize = tuple(figsize_raw) if figsize_raw else None
                plot_nn_distance_violins(
                    stats_by_subtype=_stats,
                    non_bg_channels=_non_bg,
                    output_dir=output_dir,
                    cell_type_cmap=cfg.get("cmap_cell_types", CELL_TYPE_CMAP),
                    figsize=figsize,
                    dpi=dpi,
                    verbose=verbose,
                )

            if "cross_type_proximity" in plots:
                if verbose:
                    print("\n[TFD-Viz] Generating cross-type proximity heatmaps…")
                figsize_raw = cfg.get("figsize_proximity")
                figsize = tuple(figsize_raw) if figsize_raw else None
                plot_cross_type_proximity_heatmaps(
                    stats_by_subtype=_stats,
                    non_bg_channels=_non_bg,
                    output_dir=output_dir,
                    cmap_name=cfg.get("cmap_heatmap", HEATMAP_CMAP),
                    figsize=figsize,
                    dpi=dpi,
                    verbose=verbose,
                )

            # The newer spatial summary plots need tile-level topology metrics,
            # so we collect them separately from the NN summary stats.
            _topology_stats: Optional[Dict[str, List[Dict[str, object]]]] = None
            if "ripley_L_by_subtype" in plots:
                if verbose:
                    print("\n[TFD-Viz] Generating Ripley L(r) curves by subtype…")
                if _topology_stats is None:
                    _topology_stats = _collect_spatial_topology_for_subtypes(
                        masks_dir=Path(masks_dir),
                        subtypes=_subtypes,
                        patient_to_subtype=_p2s,
                        non_bg_channels=_non_bg,
                        max_tiles=_max_sp,
                        n_bootstrap=_ripley_bootstrap,
                        seed=cfg.get("seed", 42),
                        verbose=verbose,
                    )
                plot_ripley_L_by_subtype(
                    results=_topology_stats,
                    output_dir=output_dir,
                    dpi=dpi,
                )

            if "voronoi_distribution" in plots:
                if verbose:
                    print("\n[TFD-Viz] Generating Voronoi area distributions…")
                if _topology_stats is None:
                    _topology_stats = _collect_spatial_topology_for_subtypes(
                        masks_dir=Path(masks_dir),
                        subtypes=_subtypes,
                        patient_to_subtype=_p2s,
                        non_bg_channels=_non_bg,
                        max_tiles=_max_sp,
                        n_bootstrap=_ripley_bootstrap,
                        seed=cfg.get("seed", 42),
                        verbose=verbose,
                    )
                plot_voronoi_distribution(
                    results=_topology_stats,
                    output_dir=output_dir,
                    dpi=dpi,
                )

            if "knn_metrics" in plots:
                if verbose:
                    print("\n[TFD-Viz] Generating kNN connectivity metrics…")
                if _topology_stats is None:
                    _topology_stats = _collect_spatial_topology_for_subtypes(
                        masks_dir=Path(masks_dir),
                        subtypes=_subtypes,
                        patient_to_subtype=_p2s,
                        non_bg_channels=_non_bg,
                        max_tiles=_max_sp,
                        seed=cfg.get("seed", 42),
                        verbose=verbose,
                    )
                plot_knn_metrics_comparison(
                    results=_topology_stats,
                    output_dir=output_dir,
                    dpi=dpi,
                )
