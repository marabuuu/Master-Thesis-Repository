"""Verify that tumor-tile filtering correctly places tiles inside annotation polygons.

Picks three random ZIP files from BRCA-tumor-tiles-final, reconstructs the
annotation polygon and tile footprints in target-pixel space, then produces a
side-by-side figure per slide showing:

  - Red  : annotation polygon (tumor region)
  - Blue : all tiles from the corresponding input ZIP (full set before filtering)
  - Green: tiles kept in the output ZIP (tumor-only subset)

If all green tiles overlap the red annotation, filtering is working correctly.

Usage
-----
    python -m src.evaluation.verify_tumor_tile_selection

All paths are hard-coded to the project layout on the saturn cluster.
"""
from __future__ import annotations

import os
import re
import random
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shapely.geometry as sg
from shapely.affinity import scale
from shapely.ops import unary_union
from shapely.validation import make_valid

# ---------------------------------------------------------------------------
# Project-specific paths
# ---------------------------------------------------------------------------
OUTPUT_ZIP_DIR = Path("../data/BRCA-tumor-tiles-final")
INPUT_ZIP_DIR  = Path("../data/TCGA")
ANNOTATION_DIR = Path("../data/annotations-BRCA")
SAVE_DIR       = Path(__file__).parent  # src/evaluation/

# Coordinate-system constants (must match the filtering script invocation)
TARGET_MPP      = 0.5   # target microns-per-pixel used during filtering
NATIVE_SLIDE_MPP = 0.25  # native TCGA 40x slide MPP (annotation space)
TILE_SIZE_PX    = 512   # tile edge length in target pixels

N_SLIDES = 3  # number of random slides to verify

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_tile_coords(filename: str) -> Optional[tuple[float, float]]:
    """Extract (x, y) in microns from filenames like tile_(14602.488, 1537.104).png."""
    m = re.search(r'\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)', filename)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _read_annotation(csv_path: Path) -> sg.MultiPolygon:
    """Parse X_base/Y_base CSV into a MultiPolygon (native slide pixel coords)."""
    with open(csv_path) as f:
        lines = f.readlines()

    headers = [h.strip() for h in lines[0].split(',')]
    if 'X_base' not in headers or 'Y_base' not in headers:
        raise ValueError(f"Missing X_base/Y_base columns in {csv_path}")

    ix = headers.index('X_base')
    iy = headers.index('Y_base')

    polygons: list[sg.Polygon] = []
    coords: list[tuple[float, float]] = []

    for line in lines[1:]:
        parts = line.split(',')
        if parts[ix].strip() in ('X_base', '') or parts[iy].strip() in ('Y_base', ''):
            if len(set(coords)) >= 3:
                polygons.append(sg.Polygon(coords))
            coords = []
            continue
        try:
            coords.append((float(parts[ix]), float(parts[iy])))
        except ValueError:
            continue

    if len(set(coords)) >= 3:
        polygons.append(sg.Polygon(coords))

    valid_polys = []
    for p in polygons:
        if not p.is_valid:
            p = p.buffer(0)
        if not p.is_valid:
            p = make_valid(p)
        if isinstance(p, sg.Polygon) and not p.is_empty:
            valid_polys.append(p)
        elif isinstance(p, sg.MultiPolygon):
            valid_polys.extend(p.geoms)

    return sg.MultiPolygon(valid_polys)


def _tile_polygons_from_names(
    names: list[str],
) -> list[sg.Polygon]:
    """Convert a list of tile filenames to 512×512 rectangles in target-pixel space.

    Tile-filename coordinates are in MICRONS.  Converting to target pixels:
        x_target = x_um / TARGET_MPP
    """
    polys = []
    for name in names:
        c = _parse_tile_coords(os.path.basename(name))
        if c is None:
            continue
        x_um, y_um = c
        x0 = x_um / TARGET_MPP
        y0 = y_um / TARGET_MPP
        polys.append(sg.box(x0, y0, x0 + TILE_SIZE_PX, y0 + TILE_SIZE_PX))
    return polys


def _list_zip_pngs(zip_path: Path) -> list[str]:
    """Return basenames of all .png entries inside a ZIP archive."""
    with zipfile.ZipFile(zip_path) as zf:
        return [os.path.basename(n) for n in zf.namelist() if n.lower().endswith('.png')]


def _find_annotation_csv(slide_stem: str) -> Optional[Path]:
    """Match the output-zip stem to an annotation CSV by longest common prefix."""
    csv_files = list(ANNOTATION_DIR.glob("*.csv"))
    best: Optional[Path] = None
    best_len = 0
    for csv_path in csv_files:
        prefix = csv_path.stem  # filename without .csv
        if slide_stem.startswith(prefix) and len(prefix) > best_len:
            best = csv_path
            best_len = len(prefix)
    return best


def _find_input_zip(slide_stem: str) -> Optional[Path]:
    """Match the output-zip stem to an input ZIP in INPUT_ZIP_DIR."""
    candidates = list(INPUT_ZIP_DIR.glob(f"{slide_stem}.zip"))
    if candidates:
        return candidates[0]
    # Try matching by common prefix (output zip may have extra hash suffix)
    for p in INPUT_ZIP_DIR.iterdir():
        if p.suffix == '.zip' and slide_stem.startswith(p.stem):
            return p
    return None


# ---------------------------------------------------------------------------
# Plot one slide
# ---------------------------------------------------------------------------

def _plot_slide(
    ax: plt.Axes,
    slide_name: str,
    ann_poly: sg.MultiPolygon,
    all_polys: list[sg.Polygon],
    sel_polys: list[sg.Polygon],
) -> None:
    """Draw annotation + all tiles + selected tiles onto ax."""

    # Scale annotation from native slide pixels → target pixels
    sf = NATIVE_SLIDE_MPP / TARGET_MPP  # = 0.5
    ann_target = scale(ann_poly, xfact=sf, yfact=sf, origin=(0, 0))
    if not ann_target.is_valid:
        ann_target = ann_target.buffer(0)

    def _fill_poly(geom: sg.base.BaseGeometry, **kwargs):
        if isinstance(geom, sg.Polygon):
            x, y = geom.exterior.xy
            ax.fill(x, y, **kwargs)
        elif isinstance(geom, sg.MultiPolygon):
            for g in geom.geoms:
                x, y = g.exterior.xy
                ax.fill(x, y, **kwargs)

    # Annotation
    _fill_poly(ann_target, alpha=0.35, fc='red', ec='darkred', lw=0.8, label='Annotation (tumor region)')

    # All tiles — draw only bounding boxes as thin blue outlines for readability
    for poly in all_polys:
        x, y = poly.exterior.xy
        ax.plot(x, y, color='royalblue', lw=0.3, alpha=0.4)

    # Selected tumor tiles — filled green
    for poly in sel_polys:
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.55, fc='limegreen', ec='darkgreen', lw=0.5)

    ax.invert_yaxis()
    ax.set_aspect('equal')
    short = re.match(r'(TCGA-[^.]+)', slide_name)
    display_name = short.group(1) if short else slide_name[:40]
    ax.set_title(
        f"{display_name}\n"
        f"all tiles: {len(all_polys)}  |  selected: {len(sel_polys)}",
        fontsize=7,
        pad=4,
    )
    ax.tick_params(labelsize=6)
    ax.set_xlabel("x (target px)", fontsize=6)
    ax.set_ylabel("y (target px)", fontsize=6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    output_zips = sorted(p for p in OUTPUT_ZIP_DIR.iterdir() if p.suffix == '.zip')
    if len(output_zips) < N_SLIDES:
        raise RuntimeError(f"Fewer than {N_SLIDES} output zips found in {OUTPUT_ZIP_DIR}")

    random.seed(42)
    chosen = random.sample(output_zips, N_SLIDES)

    fig, axes = plt.subplots(1, N_SLIDES, figsize=(6 * N_SLIDES, 8))
    if N_SLIDES == 1:
        axes = [axes]

    for ax, out_zip in zip(axes, chosen):
        slide_stem = out_zip.stem
        short_name = slide_stem[:50]
        print(f"\n{'='*60}")
        print(f"Slide: {slide_stem}")

        # --- annotation ---
        ann_csv = _find_annotation_csv(slide_stem)
        if ann_csv is None:
            ax.set_title(f"{short_name}\n[no annotation CSV found]", fontsize=7)
            print("  WARNING: no annotation CSV found, skipping.")
            continue
        print(f"  Annotation CSV: {ann_csv.name}")
        ann_poly = _read_annotation(ann_csv)

        # --- selected (output) tiles ---
        sel_names = _list_zip_pngs(out_zip)
        sel_polys = _tile_polygons_from_names(sel_names)
        print(f"  Selected tiles : {len(sel_polys)}")

        # --- all (input) tiles ---
        in_zip = _find_input_zip(slide_stem)
        if in_zip is not None:
            print(f"  Input ZIP      : {in_zip.name}")
            all_names = _list_zip_pngs(in_zip)
            all_polys = _tile_polygons_from_names(all_names)
        else:
            print("  WARNING: no matching input ZIP found; showing only selected tiles.")
            all_polys = sel_polys  # fallback

        print(f"  All tiles      : {len(all_polys)}")

        # --- compute quick overlap stats ---
        sf = NATIVE_SLIDE_MPP / TARGET_MPP
        ann_target = scale(ann_poly, xfact=sf, yfact=sf, origin=(0, 0)).buffer(0)
        n_inside = sum(
            1 for p in sel_polys
            if ann_target.intersection(p).area / p.area >= 0.5
        )
        print(f"  Selected tiles ≥50% inside annotation: {n_inside}/{len(sel_polys)}")

        _plot_slide(ax, slide_stem, ann_poly, all_polys, sel_polys)

    # --- legend ---
    legend_handles = [
        mpatches.Patch(fc='red',       ec='darkred',  alpha=0.5, label='Annotation (tumor region)'),
        mpatches.Patch(fc='none',      ec='royalblue', alpha=0.6, label='All tiles (input ZIP)'),
        mpatches.Patch(fc='limegreen', ec='darkgreen', alpha=0.6, label='Selected tumor tiles (output ZIP)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "Tumor-Tile Selection",
        fontsize=11, y=1.01,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = SAVE_DIR / "tumor_tile_selection_verification.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved verification figure → {out_path}")


if __name__ == "__main__":
    main()
