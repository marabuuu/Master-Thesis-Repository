"""Create a 4-row thesis-quality tile grid from a FID evaluation directory.

Row order:
  1. Real TCGA-BRCA
  2. Generated TCGA-BRCA
  3. Real TCGA-LIHC
  4. Generated TCGA-LIHC

Tiles are sampled randomly (fixed seed) from the zip archives and displayed
at their native 128×128 resolution — no upscaling.

Usage
-----
    python -m src.visualization.tile_grid \\
        --eval-dir experiments/20260605_poc_128_orthogonal_nonorm_30M/fid_evaluation_scale1_last \\
        --cols 10 \\
        --out experiments/20260605_poc_128_orthogonal_nonorm_30M/tile_grid.png

Options
-------
    --eval-dir   Root dir with real/ and generated/ subdirs of per-patient zips
    --cols       Number of tiles per row (default: 10)
    --seed       Random seed for tile sampling (default: 42)
    --gap        Gap between tiles in pixels (default: 2)
    --out        Output PNG path (default: <eval-dir>/tile_grid.png)
"""

from __future__ import annotations

import argparse
import io
import random
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image


_ROW_DEFS = [
    ("real",      "TCGA-BRCA", "Real Breast"),
    ("generated", "TCGA-BRCA", "Generated Breast"),
    ("real",      "TCGA-LIHC", "Real Liver"),
    ("generated", "TCGA-LIHC", "Generated Liver"),
]


def _sample_tiles(zip_dir: Path, n: int, seed: int) -> list[np.ndarray]:
    """Sample up to *n* tile images (as H×W×3 uint8 arrays) from zip_dir."""
    rng = random.Random(seed)
    all_zips = sorted(zip_dir.glob("*.zip"))
    rng.shuffle(all_zips)

    tiles: list[np.ndarray] = []
    for zp in all_zips:
        if len(tiles) >= n:
            break
        try:
            with zipfile.ZipFile(zp) as zf:
                names = [nm for nm in zf.namelist()
                         if nm.lower().endswith((".png", ".jpg", ".jpeg"))]
                rng.shuffle(names)
                for nm in names:
                    if len(tiles) >= n:
                        break
                    data = zf.read(nm)
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    tiles.append(np.asarray(img))
        except zipfile.BadZipFile:
            continue

    return tiles[:n]


def make_grid(eval_dir: Path, cols: int, seed: int, gap: int) -> np.ndarray:
    rows_imgs: list[list[np.ndarray]] = []
    for split, cohort, _ in _ROW_DEFS:
        src = eval_dir / split / cohort
        tiles = _sample_tiles(src, cols, seed)
        if not tiles:
            raise FileNotFoundError(f"No tiles found in {src}")
        rows_imgs.append(tiles)

    tile_h, tile_w = rows_imgs[0][0].shape[:2]
    n_rows = len(_ROW_DEFS)

    grid_h = n_rows * tile_h + (n_rows - 1) * gap
    grid_w = cols * tile_w + (cols - 1) * gap
    canvas = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)

    for r, row_tiles in enumerate(rows_imgs):
        y = r * (tile_h + gap)
        for c, tile in enumerate(row_tiles):
            x = c * (tile_w + gap)
            h, w = tile.shape[:2]
            canvas[y:y + h, x:x + w] = tile

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Thesis tile grid from FID eval dir")
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gap", type=int, default=2,
                        help="Gap between tiles in pixels (filled white)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    eval_dir = args.eval_dir.resolve()
    out_path = args.out or eval_dir / "tile_grid.png"

    print(f"Sampling {args.cols} tiles per row from {eval_dir} ...")
    grid = make_grid(eval_dir, cols=args.cols, seed=args.seed, gap=args.gap)

    tile_h = grid.shape[0] // len(_ROW_DEFS)  # approximate
    label_width_in = 1.6
    grid_w_in = grid.shape[1] / 128  # 1 inch per tile at 128 dpi
    grid_h_in = grid.shape[0] / 128

    fig, ax = plt.subplots(
        figsize=(label_width_in + grid_w_in, grid_h_in),
        dpi=300,
    )
    ax.imshow(grid, interpolation="nearest", aspect="equal")
    ax.axis("off")

    # Row labels — centred vertically on each row
    n_rows = len(_ROW_DEFS)
    tile_h_px = (grid.shape[0] - (n_rows - 1) * args.gap) / n_rows
    for r, (_, _, label) in enumerate(_ROW_DEFS):
        y_centre = (r * (tile_h_px + args.gap) + tile_h_px / 2) / grid.shape[0]
        ax.text(
            -0.01, 1.0 - y_centre,
            label,
            transform=ax.transAxes,
            ha="right", va="center",
            fontsize=10, fontfamily="sans-serif",
        )

    # Thin horizontal separators between rows
    for r in range(1, n_rows):
        y_sep = (r * tile_h_px + (r - 1) * args.gap + args.gap / 2) / grid.shape[0]
        ax.axhline(
            y=y_sep * grid.shape[0],
            color="#aaaaaa", linewidth=0.4, xmin=0, xmax=1,
        )

    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
