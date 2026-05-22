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
from PIL import Image

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
    aggregate_metrics_per_patient,
    compare_metrics_across_subtypes,
    compute_spatial_metrics_per_tile,
    plot_knn_metrics_comparison,
    plot_ripley_L_auc_comparison,
    plot_ripley_L_by_subtype,
    plot_voronoi_distribution,
    run_subtype_tests,
    _globally_correct_stats,
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

ALL_TFD_VIZ_PLOTS = [
    "channel_contributions",
    "radar_contributions",
    "cell_type_boxplots",
    "tile_mask_examples",
    "ternary_composition",
    "nn_distance_violins",
    "cross_type_proximity",
    "ripley_L_by_subtype",
    "voronoi_distribution",
    "knn_metrics",
]


def _stat_star(q: float, eta_sq: float = 0.0, eta_sq_threshold: float = 0.01) -> str:
    """Significance stars requiring both a BH q-value gate and an effect-size gate.

    Parameters
    ----------
    q : float
        Globally BH-corrected Kruskal-Wallis q-value (``kruskal_q`` from
        :func:`_globally_correct_stats`).
    eta_sq : float
        η² = H / (n − 1) from :func:`run_subtype_tests`.
    eta_sq_threshold : float
        Minimum η² to annotate.  Default 0.01 ("small" by Cohen's convention).
        This prevents trivially small but statistically detectable differences
        (inflated by large n) from generating misleading stars.
    """
    if q >= 0.05 or eta_sq < eta_sq_threshold:
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    return "*"


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
#: Light, pastel-like Crameri colormap for better separation from black background.
#: Similar aesthetic to Set2 but using a perceptually-optimized categorical map.
CELL_TYPE_CMAP = "lipari"

# Canonical channel order used across all cell-type plots.
CELL_TYPE_ORDER = [
    CHANNEL_NAMES[0],
    CHANNEL_NAMES[1],
    CHANNEL_NAMES[2],
    CHANNEL_NAMES[3],
    CHANNEL_NAMES[4],
    CHANNEL_NAMES[5],
    CHANNEL_NAMES[6],
]


def _collect_tissue_fractions_by_patient(
    masks_dir: Path,
    patient_ids: List[str],
    max_per_patient: Optional[int],
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Return ``{pid: (n_tiles, n_channels)}`` tissue-area fraction arrays.

    For each tile mask (C, H, W):

    1. Per-pixel argmax assigns each pixel to its most probable class.
    2. Each channel's pixel count is divided by the total **non-background**
       pixel count (channel 0 excluded), yielding fractions that sum to 1
       across channels 1–6.
    3. Tiles where no non-background pixels are found are dropped.

    Patient ID is preserved as the outer key so callers can aggregate to
    patient level before statistical testing (call :func:`_patient_medians`)
    or flatten back to tile level for visualization (call
    :func:`_flatten_pid_arrays`).
    """
    rng = np.random.default_rng(seed)
    n_ch_total = len(CHANNEL_NAMES)
    result: Dict[str, np.ndarray] = {}

    for pid in patient_ids:
        zip_path = masks_dir / f"{pid}.zip"
        if not zip_path.exists():
            continue
        tile_fracs: List[np.ndarray] = []
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                tile_names = [n for n in zf.namelist() if n.endswith("_cls.npy")]
                if max_per_patient is not None and len(tile_names) > max_per_patient:
                    chosen = rng.choice(len(tile_names), max_per_patient, replace=False)
                    tile_names = [tile_names[i] for i in chosen]

                for name in tile_names:
                    try:
                        with zf.open(name) as f:
                            mask = np.load(BytesIO(f.read()))
                        if mask.ndim == 2:
                            continue
                        if mask.ndim == 3 and mask.shape[0] >= mask.shape[1]:
                            mask = np.transpose(mask, (2, 0, 1))

                        n_ch = mask.shape[0]
                        pixel_classes = np.argmax(mask, axis=0).ravel()
                        counts = np.bincount(pixel_classes, minlength=n_ch).astype(float)
                        total_tissue = counts[1:].sum()
                        if total_tissue == 0:
                            continue
                        fracs = counts / total_tissue
                        tile_fracs.append(fracs[:n_ch_total])
                    except Exception:
                        logger.debug("Failed to load %s from %s", name, zip_path)
        except (zipfile.BadZipFile, OSError) as e:
            logger.warning("Corrupted or unreadable mask zip file %s: %s. Skipping.", zip_path, e)
        except Exception as e:
            logger.debug("Failed to open %s: %s", zip_path, e)

        if tile_fracs:
            result[pid] = np.stack(tile_fracs)  # (n_tiles, n_ch)
    return result


def _flatten_pid_arrays(pid_to_tiles: Dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate ``{pid: (n_tiles, n_ch)}`` into a single ``(N, n_ch)`` array."""
    arrays = [arr for arr in pid_to_tiles.values() if len(arr) > 0]
    if not arrays:
        return np.empty((0, len(CHANNEL_NAMES)), dtype=float)
    return np.concatenate(arrays, axis=0)


def _patient_medians(
    pid_to_tiles: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Aggregate ``{pid: (n_tiles, n_ch)}`` → ``{pid: (n_ch,)}`` via nanmedian."""
    return {
        pid: np.nanmedian(tiles, axis=0)
        for pid, tiles in pid_to_tiles.items()
        if len(tiles) > 0
    }


def build_cell_type_palette(
    include_background: bool = False,
    cmap_name: str = CELL_TYPE_CMAP,
) -> Dict[str, str]:
    """Return the canonical fixed colour mapping for cell-type plots.

    The mapping is stable across all plots so the same cell type always
    receives the same colour, regardless of which subset of channels is
    displayed.
    
    Samples from the lighter part of the colormap (0.3–0.95) to ensure
    good contrast against black backgrounds (esp. for darkly-colored cell types).
    """
    import matplotlib as mpl
    from .core import get_crameri_cmap

    cmap = get_crameri_cmap(cmap_name)
    palette: Dict[str, str] = {}

    if include_background:
        palette[CHANNEL_NAMES[0]] = "#111111"

    # Sample at non-uniform positions to ensure colour diversity across the hue range
    # Positions chosen to give distinct colours: cool blues, purples, rose, coral, peachy, cream
    positions = [0.25, 0.38, 0.50, 0.62, 0.75, 0.90]
    for ch, pos in zip(range(1, 7), positions):
        palette[CHANNEL_NAMES[ch]] = mpl.colors.to_hex(cmap(float(pos)))

    return palette


def _build_cell_type_palette(
    ch_labels: List[str],
    cmap_name: str = CELL_TYPE_CMAP,
) -> Dict[str, str]:
    """Backward-compatible wrapper around :func:`build_cell_type_palette`."""
    palette = build_cell_type_palette(
        include_background=CHANNEL_NAMES[0] in ch_labels,
        cmap_name=cmap_name,
    )
    return {label: palette[label] for label in ch_labels if label in palette}


def _build_high_contrast_mask_palette() -> Dict[str, str]:
    """Return a high-contrast fixed palette for discrete mask rendering."""
    return {
        "Background": "#2b2b2b",
        "Neutrophils": "#e69f00",
        "Epithelium": "#56b4e9",
        "Lymphocytes": "#009e73",
        "Plasma Cells": "#f0e442",
        "Eosinophils": "#0072b2",
        "Connective": "#d55e00",
    }


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
    cell_type_palette = build_cell_type_palette(
        include_background=include_background,
        cmap_name=cell_type_cmap,
    )

    # --- Collect tissue-area fractions per subtype ---
    records: List[Dict] = []
    pid_fracs_by_subtype: Dict[str, Dict[str, np.ndarray]] = {}

    for subtype in subtypes:
        pids = class_to_patients[subtype]
        if not pids:
            continue
        if verbose:
            print(f"  [{subtype}] Loading masks for {len(pids)} patients…")

        per_patient = (
            max(1, max_tiles_per_subtype // len(pids))
            if max_tiles_per_subtype is not None else None
        )
        pid_to_tiles = _collect_tissue_fractions_by_patient(
            masks_dir, pids, per_patient, seed=seed,
        )
        if not pid_to_tiles:
            logger.warning("%s: no masks found in %s", subtype, masks_dir)
            continue

        pid_fracs_by_subtype[subtype] = pid_to_tiles
        fracs_arr = _flatten_pid_arrays(pid_to_tiles)
        if verbose:
            print(f"  [{subtype}] {fracs_arr.shape[0]} tiles from {len(pid_to_tiles)} patients")

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

    # --- Patient-level statistical tests (must run before plotting for annotations) ---
    # One value per patient = median of that patient's tile fractions.
    stats_results: Dict[str, Dict] = {}
    for ch, ch_name in channels_to_plot:
        patient_vals: Dict[str, np.ndarray] = {
            subtype: np.array([
                float(np.nanmedian(tiles[:, ch]))
                for tiles in pid_to_tiles.values()
                if ch < tiles.shape[1]
            ])
            for subtype, pid_to_tiles in pid_fracs_by_subtype.items()
        }
        stats_results[ch_name] = run_subtype_tests(patient_vals, metric_name=ch_name)
    # Apply one global BH pass across all cell-type KW tests and all pairwise tests.
    _globally_correct_stats(stats_results)

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
            hue="Cell Type",
            order=present,
            palette=cell_type_palette,
            legend=False,
            linewidth=0.8,
            fliersize=1.5,
            flierprops=dict(alpha=0.3, marker="o", markersize=1.5),
            ax=ax,
        )
        ax.set_title(subtype, fontsize=12, fontweight="bold", pad=6)
        ax.set_xlabel("")
        ax.set_ylabel("Fraction of nuclei area" if col_idx == 0 else "")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(axis="x", labelrotation=40)

        if col_idx > 0:
            ax.tick_params(axis="y", labelleft=False)

    # --- Significance annotations (globally corrected KW + effect-size gate) ---
    y_max = axes[0][0].get_ylim()[1]
    y_min = axes[0][0].get_ylim()[0]
    y_span = y_max - y_min
    star_y = y_max + 0.02 * y_span

    for col_idx, subtype in enumerate(subtypes):
        ax = axes[0][col_idx]
        subset = df_long[df_long["Subtype"] == subtype]
        present = [name for name in ch_labels if name in subset["Cell Type"].values]
        for x_idx, name in enumerate(present):
            res = stats_results.get(name, {})
            star = _stat_star(
                q=res.get("kruskal_q", res.get("kruskal_p", 1.0)),
                eta_sq=res.get("eta_sq", 0.0),
            )
            if star:
                ax.text(
                    x_idx, star_y, star,
                    ha="center", va="bottom", fontsize=7, color="#333333",
                )
        ax.set_ylim(top=y_max + 0.14 * y_span)

    fig.suptitle(
        "Cell-Type Nuclei Composition per PAM50 Subtype",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.text(
        0.5, -0.01,
        "Stars: global BH q<0.05 and η²≥0.01 (KW across PAM50 subtypes, "
        "patient-level medians).  * q<0.05  ** q<0.01  *** q<0.001",
        ha="center", va="top", fontsize=7, color="#555555", style="italic",
    )

    out = output_dir / "tfd_cell_type_boxplots.png"
    save_figure(fig, out, dpi=dpi)
    if verbose:
        print(f"[OK] Saved cell-type boxplots → {out}")

    stats_path = output_dir / "tfd_cell_type_composition_stats.json"
    with open(stats_path, "w") as _f:
        json.dump(stats_results, _f, indent=2)
    if verbose:
        print(f"[OK] Saved composition stats → {stats_path}")


def _normalise_multichannel_mask(mask: np.ndarray) -> np.ndarray:
    """Return a mask in ``(C, H, W)`` layout."""
    mask = np.asarray(mask)
    if mask.ndim == 2:
        return mask[np.newaxis, ...]
    if mask.ndim != 3:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")
    if mask.shape[0] <= 16 and mask.shape[0] < mask.shape[-1]:
        return mask
    return np.transpose(mask, (2, 0, 1))


def _mask_to_label_image(mask: np.ndarray) -> np.ndarray:
    """Collapse a multichannel mask to a single label map via argmax."""
    mask_cf = _normalise_multichannel_mask(mask)
    if mask_cf.ndim != 3:
        raise ValueError(f"Expected a 3D mask after normalisation, got {mask_cf.shape}")
    return np.argmax(mask_cf, axis=0).astype(np.int32)


def _find_zip_member(
    zip_names: List[str],
    sample_prefix: str,
    tile_stem: str,
    suffixes: Tuple[str, ...],
) -> Optional[str]:
    """Find a member path inside a tile ZIP archive."""
    for suffix in suffixes:
        candidate = f"{sample_prefix}/{tile_stem}{suffix}"
        if candidate in zip_names:
            return candidate
    basename_matches = [
        name for name in zip_names
        if name.startswith(f"{sample_prefix}/")
        and Path(name).stem == tile_stem
    ]
    return basename_matches[0] if basename_matches else None


def _count_mask_tiles(mask_zip_path: Path) -> int:
    try:
        with zipfile.ZipFile(mask_zip_path, "r") as zf:
            return sum(name.endswith("_cls.npy") for name in zf.namelist())
    except (zipfile.BadZipFile, OSError, Exception) as e:
        logger.warning("Corrupted or unreadable mask zip file %s: %s. Skipping.", mask_zip_path, e)
        return 0


def _select_example_from_mask_zip(
    mask_zip_path: Path,
    tiles_dir: Path,
) -> Optional[Dict[str, object]]:
    """Pick the most tissue-rich tile from one patient mask ZIP."""
    best: Optional[Dict[str, object]] = None
    try:
        with zipfile.ZipFile(mask_zip_path, "r") as mask_zip:
            mask_names = sorted(name for name in mask_zip.namelist() if name.endswith("_cls.npy"))
            for mask_member in mask_names:
                if "__tile_" not in mask_member:
                    continue
                sample_prefix, remainder = mask_member.split("__tile_", 1)
                tile_stem = f"tile_{remainder.removesuffix('_cls.npy')}"
                tile_zip_path = tiles_dir / f"{sample_prefix}.zip"
                if not tile_zip_path.exists():
                    continue

                with mask_zip.open(mask_member) as handle:
                    mask = np.load(handle)
                labels = _mask_to_label_image(mask)
                foreground_fraction = float(np.mean(labels > 0))

                if best is None or foreground_fraction > float(best["foreground_fraction"]):
                    with zipfile.ZipFile(tile_zip_path, "r") as tile_zip:
                        tile_member = _find_zip_member(
                            zip_names=tile_zip.namelist(),
                            sample_prefix=sample_prefix,
                            tile_stem=tile_stem,
                            suffixes=(".png", ".jpg", ".jpeg", ".tif", ".tiff"),
                        )
                        if tile_member is None:
                            continue
                        with tile_zip.open(tile_member) as tile_handle:
                            tile_img = Image.open(tile_handle).convert("RGB")
                            tile_img.load()

                    best = {
                        "sample_prefix": sample_prefix,
                        "tile_member": tile_member,
                        "mask_member": mask_member,
                        "tile_zip_path": tile_zip_path,
                        "mask_zip_path": mask_zip_path,
                        "tile_image": tile_img,
                        "mask_labels": labels,
                        "foreground_fraction": foreground_fraction,
                    }
    except (zipfile.BadZipFile, OSError, Exception) as e:
        logger.warning("Corrupted or unreadable mask zip file %s: %s. Skipping.", mask_zip_path, e)
    return best


def plot_tile_mask_examples(
    tiles_dir: Union[str, Path],
    masks_dir: Union[str, Path],
    metadata_csv: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    patient_col: str = "Patient_ID",
    subtype_col: str = "Majority_Subtype_mRNA",
    classes: Optional[List[str]] = None,
    cmap_name: str = CELL_TYPE_CMAP,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> Path:
    """Plot one representative H&E tile and mask per subtype.

    The figure uses two rows: the top row shows the RGB tile, and the bottom
    row shows the discrete classification mask with a fixed legend so the same
    cell type is always mapped to the same colour across the whole project.
    """
    _check_matplotlib()
    setup_style()
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap

    rng = np.random.default_rng(seed)

    tiles_dir = Path(tiles_dir)
    masks_dir = Path(masks_dir)
    metadata_csv = Path(metadata_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not tiles_dir.exists():
        raise FileNotFoundError(f"tiles_dir does not exist: {tiles_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks_dir does not exist: {masks_dir}")

    df_meta = pd.read_csv(metadata_csv)
    df_meta["_pid"] = df_meta[patient_col].apply(_extract_patient_id)
    patient_to_subtype: Dict[str, str] = dict(zip(df_meta["_pid"], df_meta[subtype_col]))
    available = sorted(df_meta[subtype_col].dropna().unique().tolist())

    if classes is not None:
        subtypes = [c for c in PAM50_ORDER if c in classes]
        subtypes += [c for c in classes if c not in PAM50_ORDER]
    else:
        subtypes = [c for c in PAM50_ORDER if c in available]
        subtypes += [c for c in available if c not in PAM50_ORDER]

    class_to_patients: Dict[str, List[str]] = {s: [] for s in subtypes}
    for pid, subtype in patient_to_subtype.items():
        if subtype in class_to_patients:
            class_to_patients[subtype].append(pid)

    examples: List[Dict[str, object]] = []
    for subtype in subtypes:
        patients = list(class_to_patients.get(subtype, []))
        if not patients:
            continue
        if len(patients) > 1:
            patients = list(rng.permutation(sorted(patients)))
        else:
            patients = sorted(patients)

        best_patient: Optional[Path] = None
        best_count = -1
        for pid in patients:
            mask_zip_path = masks_dir / f"{pid}.zip"
            if not mask_zip_path.exists():
                continue
            count = _count_mask_tiles(mask_zip_path)
            if count > best_count:
                best_patient = mask_zip_path
                best_count = count

        if best_patient is None:
            continue

        example = _select_example_from_mask_zip(best_patient, tiles_dir)
        if example is None:
            continue
        example["subtype"] = subtype
        example["patient_id"] = _extract_patient_id(best_patient.name)
        examples.append(example)

    if not examples:
        raise RuntimeError("No matched tile/mask examples were found.")

    n_cols = len(examples)
    if figsize is None:
        figsize = (max(12.0, 3.8 * n_cols), 7.5)

    palette = _build_high_contrast_mask_palette()
    cmap_values = [palette[CHANNEL_NAMES[i]] for i in range(7)]
    cmap = ListedColormap(cmap_values, name="cell_types_fixed")

    fig, axes = plt.subplots(2, n_cols, figsize=figsize, squeeze=False)
    fig.subplots_adjust(wspace=0.04, hspace=0.08, bottom=0.18)

    for col_idx, example in enumerate(examples):
        tile_ax = axes[0][col_idx]
        mask_ax = axes[1][col_idx]

        tile_img = np.asarray(example["tile_image"])
        mask_labels = np.asarray(example["mask_labels"])

        tile_ax.imshow(tile_img)
        tile_ax.set_axis_off()
        tile_ax.set_title(
            f"{example['subtype']}\n{example['patient_id']}",
            fontsize=11,
            fontweight="bold",
            pad=6,
        )

        mask_ax.imshow(mask_labels, cmap=cmap, vmin=0, vmax=6, interpolation="nearest")
        mask_ax.set_axis_off()
        mask_ax.set_xlabel("classification mask", fontsize=9, labelpad=10)

    axes[0][0].set_ylabel("H&E tile", fontsize=10)
    axes[1][0].set_ylabel("mask", fontsize=10)

    legend_order = CELL_TYPE_ORDER
    legend_handles = [
        mpatches.Patch(color=palette[label], label=f"{idx} {label}")
        for idx, label in enumerate(legend_order)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        framealpha=0.95,
        edgecolor="grey",
        bbox_to_anchor=(0.5, 0.02),
        title="Mask classes",
        title_fontsize=10,
        fontsize=9,
    )

    fig.suptitle(
        "Representative H&E tiles and matching classification masks",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    out = output_dir / "tile_mask_examples.png"
    save_figure(fig, out, dpi=dpi)
    if verbose:
        print(f"[OK] Saved tile/mask gallery → {out}")
    return out


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
        per_patient = (
            max(1, max_tiles_per_subtype // len(pids))
            if max_tiles_per_subtype is not None else None
        )
        pid_to_tiles = _collect_tissue_fractions_by_patient(
            masks_dir, pids, per_patient, seed=seed,
        )
        if not pid_to_tiles:
            logger.warning("%s: no masks found in %s", subtype, masks_dir)
            continue
        fracs_arr = _flatten_pid_arrays(pid_to_tiles)

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

def _collect_spatial_stats_by_patient(
    masks_dir: Path,
    patient_ids: List[str],
    non_bg_channels: List[int],
    max_per_patient: Optional[int],
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Return ``{pid: {nn_intra, nn_cross}}`` nearest-neighbour distance arrays.

    For each tile:

    1. Per-pixel argmax assigns each pixel to the most probable class.
    2. Connected-component centroids are extracted per non-background channel
       via ``scipy.ndimage`` (4-connected / von Neumann neighbourhood).
    3. Two arrays are built per tile:

       * ``nn_intra`` ``(n_ch,)``: mean distance from each cell to its nearest
         same-type neighbour (NaN when < 2 cells of that type are present).
       * ``nn_cross`` ``(n_ch, n_ch)``: entry ``[i, j]`` = mean distance from
         every cell of type *i* to the nearest cell of type *j* (NaN if either
         type has zero cells; diagonal equals ``nn_intra``).

    Patient ID is kept as the outer key so callers can compute patient-level
    medians for statistical testing separately from flattening for visualization.

    Parameters
    ----------
    masks_dir : Path
    patient_ids : list[str]
    non_bg_channels : list[int]
        Channel indices to analyse (typically 1–6; background excluded).
    max_per_patient : int or None
        Tile budget per patient; ``None`` = use all tiles.
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
    result: Dict[str, Dict[str, np.ndarray]] = {}

    for pid in patient_ids:
        zip_path = masks_dir / f"{pid}.zip"
        if not zip_path.exists():
            continue

        nn_intra_pid: List[np.ndarray] = []
        nn_cross_pid: List[np.ndarray] = []

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                tile_names = [n for n in zf.namelist() if n.endswith("_cls.npy")]
                if max_per_patient is not None and len(tile_names) > max_per_patient:
                    chosen = rng.choice(len(tile_names), max_per_patient, replace=False)
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

                        centroids: List[Optional[np.ndarray]] = []
                        for ch in non_bg_channels:
                            binary = (pixel_classes == ch)
                            labeled, n_comp = nd_label(binary)
                            if n_comp == 0:
                                centroids.append(None)
                                continue
                            coms = nd_com(binary, labeled, list(range(1, n_comp + 1)))
                            centroids.append(np.array(coms))

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
                                    nn_cross_tile[i, j] = dists.mean()

                        nn_intra_pid.append(np.diag(nn_cross_tile).copy())
                        nn_cross_pid.append(nn_cross_tile)

                    except Exception:
                        logger.debug("Spatial stats failed: %s in %s", name, zip_path)
        except Exception:
            logger.debug("Cannot open %s", zip_path)

        if nn_intra_pid:
            result[pid] = {
                "nn_intra": np.stack(nn_intra_pid),   # (n_tiles, n_ch)
                "nn_cross": np.stack(nn_cross_pid),   # (n_tiles, n_ch, n_ch)
            }
    return result


def _collect_spatial_stats_for_subtypes(
    masks_dir: Path,
    subtypes: List[str],
    patient_to_subtype: Dict[str, str],
    non_bg_channels: List[int],
    max_tiles: Optional[int],
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Collect spatial NN stats for each subtype.

    Returns
    -------
    dict[subtype, dict] with keys:

    * ``nn_intra``         ``(N_tiles, n_ch)``       — flat tile array for violin plots
    * ``nn_cross``         ``(N_tiles, n_ch, n_ch)``  — flat tile array for heatmaps
    * ``patient_nn_intra`` ``{pid: (n_ch,)}``         — per-patient medians for stats
    * ``patient_nn_cross`` ``{pid: (n_ch, n_ch)}``    — per-patient medians for stats
    """
    class_to_patients: Dict[str, List[str]] = {s: [] for s in subtypes}
    for p, st in patient_to_subtype.items():
        if st in class_to_patients:
            class_to_patients[st].append(p)

    n_ch = len(non_bg_channels)
    stats_by_subtype: Dict[str, Dict[str, np.ndarray]] = {}

    for subtype in subtypes:
        pids = class_to_patients[subtype]
        if not pids:
            continue
        if verbose:
            print(f"  [{subtype}] Computing spatial NN stats ({len(pids)} patients)…")

        per_patient = (
            max(1, max_tiles // len(pids)) if max_tiles is not None else None
        )
        pid_stats = _collect_spatial_stats_by_patient(
            masks_dir, pids, non_bg_channels, per_patient, seed=seed,
        )
        if not pid_stats:
            logger.warning("%s: no spatial stats collected", subtype)
            continue

        # Flatten for visualization
        nn_intra_flat = np.concatenate(
            [d["nn_intra"] for d in pid_stats.values()], axis=0
        )
        nn_cross_flat = np.concatenate(
            [d["nn_cross"] for d in pid_stats.values()], axis=0
        )

        # Patient-level medians for statistical testing
        patient_nn_intra = {
            pid: np.nanmedian(d["nn_intra"], axis=0)  # (n_ch,)
            for pid, d in pid_stats.items()
        }
        patient_nn_cross = {
            pid: np.nanmedian(d["nn_cross"], axis=0)  # (n_ch, n_ch)
            for pid, d in pid_stats.items()
        }

        stats_by_subtype[subtype] = {
            "nn_intra":          nn_intra_flat,
            "nn_cross":          nn_cross_flat,
            "patient_nn_intra":  patient_nn_intra,
            "patient_nn_cross":  patient_nn_cross,
        }
        if verbose:
            print(
                f"  [{subtype}] {nn_intra_flat.shape[0]} tiles "
                f"from {len(pid_stats)} patients"
            )
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
) -> Dict[str, Dict[str, object]]:
    """Collect tile-level topology metrics for each subtype.

    Returns
    -------
    dict[subtype, dict] with keys:

    * ``"tiles"``       ``list[dict]``            — flat tile metrics for visualization
    * ``"per_patient"`` ``{pid: list[dict]}``     — per-patient tile lists for
      patient-level aggregation and statistical testing via
      :func:`~spatial_metrics_viz.aggregate_metrics_per_patient` and
      :func:`~spatial_metrics_viz.compare_metrics_across_subtypes`.
    """
    rng = np.random.default_rng(seed)
    class_to_patients: Dict[str, List[str]] = {s: [] for s in subtypes}
    for p, st in patient_to_subtype.items():
        if st in class_to_patients:
            class_to_patients[st].append(p)

    stats_by_subtype: Dict[str, Dict[str, object]] = {}

    for subtype in subtypes:
        pids = class_to_patients[subtype]
        if not pids:
            continue
        # Cap per patient so each subtype gets ~max_tiles total (mirrors _collect_spatial_stats_for_subtypes)
        per_patient: Optional[int] = (
            max(1, max_tiles // len(pids)) if max_tiles is not None else None
        )
        if verbose:
            print(f"  [{subtype}] Computing spatial topology stats ({len(pids)} patients)…")

        per_patient_tiles: Dict[str, List[Dict[str, object]]] = {}
        for pid in pids:
            zip_path = masks_dir / f"{pid}.zip"
            if not zip_path.exists():
                continue
            pid_tile_metrics: List[Dict[str, object]] = []
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

                            foreground = np.sum(mask[non_bg_channels, :, :], axis=0)
                            metrics = compute_spatial_metrics_per_tile(
                                foreground,
                                channel_idx=-1,
                                bounding_box=(foreground.shape[1], foreground.shape[0]),
                                n_bootstrap=n_bootstrap,
                            )
                            pid_tile_metrics.append(metrics)
                        except Exception:
                            logger.debug("Topology stats failed: %s in %s", name, zip_path)
            except Exception:
                logger.debug("Cannot open %s", zip_path)

            if pid_tile_metrics:
                per_patient_tiles[pid] = pid_tile_metrics

        if per_patient_tiles:
            all_tiles = [m for tiles in per_patient_tiles.values() for m in tiles]
            stats_by_subtype[subtype] = {
                "tiles":       all_tiles,
                "per_patient": per_patient_tiles,
            }
            if verbose:
                print(
                    f"  [{subtype}] {len(all_tiles)} tiles "
                    f"from {len(per_patient_tiles)} patients"
                )

    return stats_by_subtype


def plot_nn_distance_violins(
    stats_by_subtype: Dict[str, Dict[str, np.ndarray]],
    non_bg_channels: List[int],
    output_dir: Union[str, Path],
    *,
    cell_type_cmap: str = CELL_TYPE_CMAP,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 200,
    log_scale: bool = True,
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
    cell_type_palette = build_cell_type_palette(
        include_background=False,
        cmap_name=cell_type_cmap,
    )

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

    # --- Patient-level statistical tests (run before plotting for annotations) ---
    nn_stats: Dict[str, Dict] = {}
    for i, ch in enumerate(non_bg_channels):
        ch_name = CHANNEL_NAMES[ch]
        patient_vals: Dict[str, np.ndarray] = {
            subtype: np.array([
                float(v[i])
                for v in stats_by_subtype[subtype].get("patient_nn_intra", {}).values()
                if np.isfinite(v[i])
            ])
            for subtype in subtypes
            if subtype in stats_by_subtype
        }
        nn_stats[ch_name] = run_subtype_tests(patient_vals, metric_name=ch_name)
    # Apply one global BH pass across all cell-type KW tests and all pairwise tests.
    _globally_correct_stats(nn_stats)

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
        if log_scale:
            ax.set_yscale("log")
            if col_idx == 0:
                ax.set_ylabel("Mean NN distance (px, log scale)")
        if col_idx > 0:
            ax.tick_params(axis="y", labelleft=False)

    # --- Significance annotations (globally corrected KW + effect-size gate) ---
    y_max = axes[0][0].get_ylim()[1]
    y_min = axes[0][0].get_ylim()[0]
    if log_scale:
        star_y = y_max * 1.05
        new_y_top = y_max * 1.20
    else:
        y_span = y_max - y_min
        star_y = y_max + 0.02 * y_span
        new_y_top = y_max + 0.14 * y_span

    for col_idx, subtype in enumerate(subtypes):
        ax = axes[0][col_idx]
        subset = df_long[df_long["Subtype"] == subtype]
        present = [name for name in ch_labels if name in subset["Cell Type"].values]
        for x_idx, name in enumerate(present):
            res = nn_stats.get(name, {})
            star = _stat_star(
                q=res.get("kruskal_q", res.get("kruskal_p", 1.0)),
                eta_sq=res.get("eta_sq", 0.0),
            )
            if star:
                ax.text(
                    x_idx, star_y, star,
                    ha="center", va="bottom", fontsize=7, color="#333333",
                )
        ax.set_ylim(top=new_y_top)

    fig.suptitle(
        "Intra-Type Nearest-Neighbour Distance per PAM50 Subtype\n"
        "(mean distance from each cell to its nearest same-type neighbour)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.text(
        0.5, -0.01,
        "Stars: global BH q<0.05 and η²≥0.01 (KW across PAM50 subtypes, "
        "patient-level medians).  * q<0.05  ** q<0.01  *** q<0.001",
        ha="center", va="top", fontsize=7, color="#555555", style="italic",
    )

    out = output_dir / "tfd_nn_distance_violins.png"
    save_figure(fig, out, dpi=dpi)
    if verbose:
        print(f"[OK] Saved NN distance violin plots → {out}")

    stats_path = output_dir / "tfd_nn_distance_stats.json"
    with open(stats_path, "w") as _f:
        json.dump(nn_stats, _f, indent=2)
    if verbose:
        print(f"[OK] Saved NN distance stats → {stats_path}")


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

    # Per-subtype median cross-type distance matrix.
    # Prefer patient-level medians (median of per-patient medians) to avoid
    # patients with many tiles dominating the result.
    matrices: Dict[str, np.ndarray] = {}
    for subtype in subtypes:
        if subtype not in stats_by_subtype:
            continue
        patient_cross = stats_by_subtype[subtype].get("patient_nn_cross", {})
        if patient_cross:
            mat_stack = np.stack(list(patient_cross.values()))  # (n_patients, n_ch, n_ch)
            matrices[subtype] = np.nanmedian(mat_stack, axis=0)
        else:
            matrices[subtype] = np.nanmedian(
                stats_by_subtype[subtype]["nn_cross"], axis=0
            )

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
        cbar = fig.colorbar(im_ref, ax=axes.ravel().tolist(), fraction=0.046, pad=0.02)
        cbar.set_label("Median NN distance (px)", fontsize=9)

    fig.suptitle(
        "Cross–Cell-Type Nearest-Neighbour Distances per PAM50 Subtype\n"
        "row = source type  ·  col = target type  ·  median of patient medians (px)",
        fontsize=11, y=1.03,
    )
    fig.tight_layout()

    out = output_dir / "tfd_cross_type_proximity.png"
    save_figure(fig, out, dpi=dpi)
    if verbose:
        print(f"[OK] Saved cross-type proximity heatmaps → {out}")

    # --- Patient-level statistical tests for each (source, target) cell-type pair ---
    cross_stats: Dict[str, Dict] = {}
    for i, src in enumerate(ch_labels):
        for j, tgt in enumerate(ch_labels):
            if i == j:
                continue  # intra-type distances are covered by nn_distance_stats
            key = f"{src}_to_{tgt}"
            patient_vals: Dict[str, np.ndarray] = {
                subtype: np.array([
                    float(mat[i, j])
                    for mat in stats_by_subtype[subtype].get("patient_nn_cross", {}).values()
                    if np.isfinite(mat[i, j])
                ])
                for subtype in subtypes
                if subtype in stats_by_subtype
            }
            cross_stats[key] = run_subtype_tests(patient_vals, metric_name=key)
    # Global BH across all 30 source→target pair families.
    _globally_correct_stats(cross_stats)

    stats_path = output_dir / "tfd_cross_type_proximity_stats.json"
    with open(stats_path, "w") as _f:
        json.dump(cross_stats, _f, indent=2)
    if verbose:
        print(f"[OK] Saved cross-type proximity stats → {stats_path}")


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
    tiles_dir : str
        Directory containing per-sample tile ZIP archives.
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
        ``cell_type_boxplots``, ``tile_mask_examples``, ``ternary_composition``,
        ``nn_distance_violins``, ``cross_type_proximity``,
        ``ripley_L_by_subtype``, ``voronoi_distribution``,
        ``knn_metrics``.
    """
    results_json = cfg.get("results_json")
    masks_dir = cfg.get("masks_dir")
    tiles_dir = cfg.get("tiles_dir")
    metadata_csv = cfg.get("metadata_csv")
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_cfg = cfg.get("plots")
    if plots_cfg is None:
        plots = ["channel_contributions", "radar_contributions", "cell_type_boxplots"]
    elif isinstance(plots_cfg, str):
        if plots_cfg.lower() in {"all", "*", "true"}:
            plots = list(ALL_TFD_VIZ_PLOTS)
        else:
            plots = [plots_cfg]
    else:
        plots = list(plots_cfg)
        if any(str(p).lower() in {"all", "*", "true"} for p in plots):
            plots = list(ALL_TFD_VIZ_PLOTS)

    # Keep order deterministic and drop invalid labels silently.
    _plot_set = {str(p) for p in plots}
    plots = [p for p in ALL_TFD_VIZ_PLOTS if p in _plot_set]
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

    if "tile_mask_examples" in plots:
        if not tiles_dir or not masks_dir or not metadata_csv:
            logger.warning(
                "'tiles_dir', 'masks_dir' or 'metadata_csv' not set — skipping tile_mask_examples."
            )
        else:
            if verbose:
                print("\n[TFD-Viz] Generating tile/mask example gallery…")
            plot_tile_mask_examples(
                tiles_dir=tiles_dir,
                masks_dir=masks_dir,
                metadata_csv=metadata_csv,
                output_dir=output_dir,
                patient_col=cfg.get("patient_col", "Patient_ID"),
                subtype_col=cfg.get("subtype_col", "Majority_Subtype_mRNA"),
                classes=cfg.get("classes"),
                cmap_name=cfg.get("cmap_cell_types", CELL_TYPE_CMAP),
                figsize=tuple(cfg["figsize_tile_mask_examples"]) if cfg.get("figsize_tile_mask_examples") else None,
                dpi=dpi,
                seed=cfg.get("seed", 42),
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

    # --- Spatial plots: shared patient/subtype setup, then two separate passes ---
    # _need_nn_stats:  NN distance + cross-type proximity  (fast per-tile summaries)
    # _need_topology:  Ripley L(r), Voronoi, kNN           (slow connected-component pass)
    # These are collected independently so selecting only ripley_L_by_subtype does
    # not trigger the unnecessary NN-stats collection (which caused time-limit errors).
    _need_nn_stats = "nn_distance_violins" in plots or "cross_type_proximity" in plots
    _need_topology = (
        "ripley_L_by_subtype" in plots
        or "ripley_L_auc_by_subtype" in plots
        or "voronoi_distribution" in plots
        or "knn_metrics" in plots
    )

    if _need_nn_stats or _need_topology:
        if not masks_dir or not metadata_csv:
            logger.warning(
                "'masks_dir' or 'metadata_csv' not set — skipping spatial plots."
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

            # ── NN stats (intra-type distances, cross-type proximity) ──────────
            _stats = None
            if _need_nn_stats:
                if verbose:
                    print(
                        f"\n[TFD-Viz] Computing spatial NN statistics "
                        f"(max {_max_sp} tiles/subtype)…"
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

            if "nn_distance_violins" in plots and _stats is not None:
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
                    log_scale=cfg.get("nn_violins_log_scale", True),
                    verbose=verbose,
                )

            if "cross_type_proximity" in plots and _stats is not None:
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

            # ── Topology stats (Ripley, Voronoi, kNN) — slow connected-component pass ──
            _topology_stats: Optional[Dict[str, Dict[str, object]]] = None
            if _need_topology:
                if verbose:
                    print(
                        f"\n[TFD-Viz] Computing spatial topology statistics "
                        f"(max {_max_sp} tiles/subtype; "
                        f"Ripley bootstrap={_ripley_bootstrap})…"
                    )
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

            if "ripley_L_by_subtype" in plots and _topology_stats is not None:
                if verbose:
                    print("\n[TFD-Viz] Generating Ripley L(r) curves by subtype…")
                plot_ripley_L_by_subtype(
                    results=_topology_stats,
                    output_dir=output_dir,
                    dpi=dpi,
                )

            if "ripley_L_auc_by_subtype" in plots and _topology_stats is not None:
                if verbose:
                    print("\n[TFD-Viz] Generating Ripley L(r) AUC comparison by subtype…")
                plot_ripley_L_auc_comparison(
                    results=_topology_stats,
                    output_dir=output_dir,
                    dpi=dpi,
                )

            if "voronoi_distribution" in plots and _topology_stats is not None:
                if verbose:
                    print("\n[TFD-Viz] Generating Voronoi area distributions…")
                plot_voronoi_distribution(
                    results=_topology_stats,
                    output_dir=output_dir,
                    cmap_name=cfg.get("cmap_categorical", CATEGORICAL_CMAP),
                    dpi=dpi,
                )

            if "knn_metrics" in plots and _topology_stats is not None:
                if verbose:
                    print("\n[TFD-Viz] Generating kNN connectivity metrics…")
                plot_knn_metrics_comparison(
                    results=_topology_stats,
                    output_dir=output_dir,
                    cmap_name=cfg.get("cmap_categorical", CATEGORICAL_CMAP),
                    dpi=dpi,
                )

            # After all topology plots are done, run patient-level stats
            if _topology_stats is not None:
                if verbose:
                    print("\n[TFD-Viz] Computing topology statistical tests…")
                subtype_patient_metrics = {
                    subtype: {
                        pid: aggregate_metrics_per_patient(tile_list)
                        for pid, tile_list in subtype_data["per_patient"].items()
                    }
                    for subtype, subtype_data in _topology_stats.items()
                }
                topo_test_results = compare_metrics_across_subtypes(subtype_patient_metrics)
                topo_stats_path = output_dir / "tfd_topology_stats.json"
                with open(topo_stats_path, "w") as _f:
                    json.dump(topo_test_results, _f, indent=2)
                if verbose:
                    print(f"[OK] Saved topology stats → {topo_stats_path}")
