"""Quick visual comparison of breast vs liver tiles — no upscaling.

Stitches tiles together with PIL so every pixel in the output is one
source pixel. Open the saved PNG in any image viewer at 100% zoom.

Usage:
    python scripts/compare_tiles.py <breast_zip> <liver_zip> [--n 3] [--seed 42] [--out tiles.png]
"""

import argparse
import io
import random
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RESAMPLE = Image.Resampling.LANCZOS


def sample_pngs(zip_path: str, n: int, seed: int) -> list:
    with zipfile.ZipFile(zip_path) as zf:
        names = [m for m in zf.namelist() if m.lower().endswith(".png")]
    random.seed(seed)
    chosen = random.sample(names, min(n, len(names)))
    imgs = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in chosen:
            imgs.append(Image.open(io.BytesIO(zf.read(name))).convert("RGB"))
    return imgs


def make_grid(rows: list, labels: list, display_size: int = 256, gap: int = 4) -> Image.Image:
    """Stitch rows × cols images into a grid, all tiles shown at display_size.

    Upscaling uses NEAREST so actual pixel content is visible without smoothing.
    Downscaling uses LANCZOS.
    """
    n_cols = len(rows[0])
    n_rows = len(rows)
    label_h = 16

    total_w = n_cols * display_size + (n_cols - 1) * gap
    total_h = n_rows * (display_size + label_h) + (n_rows - 1) * gap

    canvas = Image.new("RGB", (total_w, total_h), color=(30, 30, 30))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)

    for r, (imgs, label) in enumerate(zip(rows, labels)):
        y0 = r * (display_size + label_h + gap)
        draw.text((2, y0), label, fill=(220, 220, 220), font=font)
        for c, img in enumerate(imgs):
            w, h = img.size
            if w < display_size:
                resample = Image.NEAREST   # upscale: show real pixels, no smoothing
            else:
                resample = Image.LANCZOS   # downscale: anti-aliased
            tile = img.resize((display_size, display_size), resample)
            canvas.paste(tile, (c * (display_size + gap), y0 + label_h))

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("breast_zip")
    parser.add_argument("liver_zip")
    parser.add_argument("--breast-zip-orig", default=None, help="512px breast zip for comparison row")
    parser.add_argument("--liver-zip-orig",  default=None, help="512px liver zip for comparison row")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="tile_comparison.png")
    args = parser.parse_args()

    rows, labels = [], []

    if args.breast_zip_orig:
        orig = sample_pngs(args.breast_zip_orig, args.n, args.seed)
        # downscale originals in memory with same filter so tile indices match
        downscaled = [img.resize((128, 128), RESAMPLE) for img in orig]
        rows  += [orig, downscaled]
        labels += ["Breast 512px (orig)", "Breast 128px (downscaled)"]
    else:
        rows.append(sample_pngs(args.breast_zip, args.n, args.seed))
        labels.append("Breast 128px")

    if args.liver_zip_orig:
        orig = sample_pngs(args.liver_zip_orig, args.n, args.seed)
        downscaled = [img.resize((128, 128), RESAMPLE) for img in orig]
        rows  += [orig, downscaled]
        labels += ["Liver 512px (orig)", "Liver 128px (downscaled)"]
    else:
        rows.append(sample_pngs(args.liver_zip, args.n, args.seed))
        labels.append("Liver 128px")

    grid = make_grid(rows=rows, labels=labels)
    grid.save(args.out)
    print(f"Saved {grid.size[0]}×{grid.size[1]} px → {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
