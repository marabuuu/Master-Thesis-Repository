#!/usr/bin/env python
"""
Dry-run tool: select tiles inside ROIs from tile zip files produced by a tiler.

This reads each zip's `tiler_params.json` to infer `tile_size_px` and
`tile_size_um` (thus slide MPP), builds polygons for the tiles present in
the zip, loads the matching annotation CSV (uses `read_annotations` from
`data_prep.utils`) and reports which tiles would be selected according to
the same intersection logic used by `get_tiles_within_rois.py` (keep tiles
with >= 0.6 overlap). No files are extracted when run in `--dry-run` mode.
"""
import argparse
import json
import os
import re
import zipfile
from pathlib import Path

from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import scale
import shapely.geometry as sg
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as MatplotlibPolygon
from tqdm import tqdm

from preprocessing.utils import read_annotations, find_substring_in_list


def read_tiler_params_from_zip(zip_path, params_name="tiler_params.json"):
    # try a few common names for the params file
    candidates = [params_name, "tile_params.json", "params.json", "tiler_params.json"]
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in candidates:
            try:
                with zf.open(name) as fh:
                    return json.load(fh)
            except KeyError:
                continue
    return None


def list_tile_files_in_zip(zip_path, tile_ext=".png"):
    exts = [tile_ext.lower(), tile_ext.upper(), tile_ext.lower().replace('.', '')]
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        files = [n for n in names if any(n.lower().endswith(e) for e in exts)]
        # if none found, try common image extensions
        if not files:
            for e in ('.png', '.jpg', '.jpeg', '.tif'):
                files = [n for n in names if n.lower().endswith(e)]
                if files:
                    break
        return files


def parse_tile_origin_from_name(name, tile_size_px):
    """Try regex patterns to extract either pixel coords or row/col indices.
    Returns (x_px, y_px) in pixel coordinates or None if parsing fails.
    Heuristic: if parsed numbers look large (> tile_size_px*10) treat them as
    pixel coordinates; otherwise treat as row/col and multiply by tile_size_px.
    """
    basename = Path(name).stem
    # support integer or float coordinates, with optional surrounding 'tile_(x, y)'
    patterns = [r"^\(?\s*(?P<x>-?\d+\.?\d*)\s*[_.,-]\s*(?P<y>-?\d+\.?\d*)\s*\)?$",
                r"tile[_-]?\(?\s*(?P<x>-?\d+\.?\d*)\s*[, _.-]\s*(?P<y>-?\d+\.?\d*)\s*\)?$",
                r"r(?P<r>\d+)_c(?P<c>\d+)$",
                r"(?P<r>\d+)x(?P<c>\d+)$",
                r"^(?P<x>-?\d+)$"]

    for pat in patterns:
        m = re.search(pat, basename)
        if not m:
            continue
        gd = m.groupdict()
        if "x" in gd and "y" in gd and gd["x"] is not None and gd["y"] is not None:
            # allow float coordinates in filenames
            try:
                a = float(gd["x"]) if ('.' in gd["x"] or 'e' in gd["x"].lower()) else float(gd["x"])
                b = float(gd["y"]) if ('.' in gd["y"] or 'e' in gd["y"].lower()) else float(gd["y"])
            except Exception:
                a = float(int(gd["x"]))
                b = float(int(gd["y"]))
            # heuristic: if numbers are large, assume pixel coords
            if abs(a) > tile_size_px * 10 or abs(b) > tile_size_px * 10:
                return a, b
            else:
                return a * tile_size_px, b * tile_size_px

        if "r" in gd and "c" in gd and gd["r"] is not None and gd["c"] is not None:
            r = int(gd["r"])
            c = int(gd["c"])
            return c * tile_size_px, r * tile_size_px

    return None


def build_polygons_for_tiles(tile_files, tile_size_px, parse_fn=parse_tile_origin_from_name):
    polys = []
    origins = []
    for name in tile_files:
        coords = parse_fn(name, tile_size_px)
        if coords is None:
            polys.append(None)
            origins.append(None)
            continue
        x, y = coords
        poly = Polygon([(x, y), (x + tile_size_px, y), (x + tile_size_px, y + tile_size_px), (x, y + tile_size_px)])
        polys.append(poly)
        origins.append((x, y))
    return polys, origins


def dry_run_on_zip(zip_path, roi_dir, target_mpp, generate_plots, tile_ext):
    zpath = Path(zip_path)
    params = read_tiler_params_from_zip(zpath)
    if params is None:
        print(f"No tiler params found in {zpath}, using defaults (tile_size_px=512, mpp=0.5)")
        tile_size_px = 512
        tile_size_um = 0.5 * tile_size_px
    else:
        # infer tile size and mpp
        tile_size_px = int(params.get("tile_size_px", params.get("tile_size", 512)))
        tile_size_um = float(params.get("tile_size_um", params.get("tile_size_microns", 0.5 * tile_size_px)))
    slide_mpp = tile_size_um / tile_size_px

    # find matching annotation csv; use zip stem as slide id
    zip_stem = zpath.stem
    try:
        roi_files = os.listdir(roi_dir)
        # first try matching using slide_path from params (more reliable when available)
        roi_fname_list = []
        if params is not None:
            slide_path = params.get('slide_path') or params.get('slide') or None
            if slide_path:
                slide_base = Path(slide_path).stem
                roi_fname_list = find_substring_in_list(roi_files, slide_base)
        # fall back to substring match on zip stem
        if not roi_fname_list:
            roi_fname_list = find_substring_in_list(roi_files, zip_stem)
        # if not found, try prefix matching on the first N chars (fallback)
        if not roi_fname_list:
            N = 16
            prefix = zip_stem[:N]
            # prefer exact-start matches (case-sensitive), then case-insensitive
            roi_fname_list = [f for f in roi_files if f.startswith(prefix)]
            if not roi_fname_list:
                lpref = prefix.lower()
                roi_fname_list = [f for f in roi_files if f.lower().startswith(lpref)]
        if not roi_fname_list:
            # try more permissive matching: strip non-alphanumerics and match prefix
            def alnum(s):
                return ''.join(ch for ch in s if ch.isalnum())
            zs = alnum(zip_stem)[:16].lower()
            roi_fname_list = [f for f in roi_files if alnum(f)[:16].lower() == zs]
        if not roi_fname_list:
            print(f"No annotation CSV found for {zip_stem}, skipping (no match)")
            return False
        if len(roi_fname_list) > 1:
            raise RuntimeError(f"Multiple annotation CSV candidates for {zip_stem}: {roi_fname_list}")
        roi_path = os.path.join(roi_dir, roi_fname_list[0])
        # quick filesystem check: if CSV is empty (size 0) skip immediately
        try:
            if os.path.getsize(roi_path) == 0:
                print(f"Annotation CSV is empty (size 0): {roi_path}; skipping")
                return "empty_csv"
        except OSError:
            # if file can't be stat'ed, let later code handle it
            pass
    except Exception as e:
        print(f"Error finding annotation for {zip_stem}: {e}")
        return False

    # read annotations (reuses existing utility expected by original script)
    try:
        ann, _ = read_annotations(roi_path)
    except ValueError as e:
        # empty or invalid annotation file; caller should count and skip this zip
        print(f"Empty or invalid annotation CSV for {zip_stem}: {e}")
        return "empty_csv"

    scale_factor = slide_mpp / target_mpp
    scaled_annPolys = scale(ann, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))

    # list tiles and build polygons
    tile_files = list_tile_files_in_zip(zpath, tile_ext=tile_ext)
    if not tile_files:
        print(f"No tile images found inside zip {zpath.name}; trying to continue with empty tile list")
    polys, origins = build_polygons_for_tiles(tile_files, tile_size_px)

    # build results list
    selected = []
    for name, poly in zip(tile_files, polys):
        if poly is None:
            continue
        # quick bbox check
        if not poly.bounds or not scaled_annPolys.bounds:
            continue
        if not sg.box(*poly.bounds).intersects(sg.box(*scaled_annPolys.bounds)):
            continue
        # exact intersection
        if isinstance(scaled_annPolys, Polygon) and poly.intersects(scaled_annPolys):
            inter = scaled_annPolys.intersection(poly)
            if inter.area / poly.area >= 0.6:
                selected.append(name)
        elif isinstance(scaled_annPolys, MultiPolygon):
            for p in scaled_annPolys.geoms:
                if poly.intersects(p):
                    inter = p.intersection(poly)
                    if inter.area / poly.area >= 0.6:
                        selected.append(name)
                        break

    print(f"Zip {zpath.name}: {len(selected)}/{len(tile_files)} tiles selected (dry-run)")

    if generate_plots:
        out_plots_dir = zpath.parent / "plots_zip"
        out_plots_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        if isinstance(scaled_annPolys, sg.MultiPolygon):
            for polygon in scaled_annPolys.geoms:
                x, y = polygon.exterior.xy
                ax.fill(x, y, alpha=0.5, fc='r')
        else:
            x, y = scaled_annPolys.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='r')

        for name, poly in zip(tile_files, polys):
            if poly is None:
                continue
            x, y = poly.exterior.xy
            if name in selected:
                ax.fill(x, y, alpha=0.7, fc='g')
            else:
                ax.fill(x, y, alpha=0.2, fc='b')

            minx, miny, maxx, maxy = poly.bounds
            ax.add_patch(MatplotlibPolygon([[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny]], fill=None, edgecolor='g', linestyle='--'))

        legend_elements = [Patch(facecolor='red', edgecolor='r', alpha=0.5, label='Annotated Tissue'),
                           Patch(facecolor='green', edgecolor='g', alpha=0.7, label='Selected Tiles'),
                           Patch(facecolor='blue', edgecolor='b', alpha=0.2, label='All Tiles')]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.invert_yaxis()
        plt.savefig(out_plots_dir / f"{zip_stem}_dryrun.png")
        plt.close()

    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip-dir", required=True, help="Folder containing zip files (one zip per slide)")
    p.add_argument("--roi-dir", required=True, help="Folder with annotation CSVs")
    p.add_argument("--target-mpp", type=float, default=256 / 512, help="Target MPP used by downstream pipeline")
    p.add_argument("--tile-ext", default=".png")
    p.add_argument("--generate-plots", action="store_true")
    p.add_argument("--dry-run", action="store_true", default=True, help="If set, do not extract/copy tiles, only report")
    p.add_argument("--num-workers", type=int, default=1)
    args = p.parse_args()

    zips = sorted(Path(args.zip_dir).glob("*.zip"))
    if not zips:
        print("No zip files found in", args.zip_dir)
        return

    empty_csv_count = 0
    skipped_count = 0
    for z in tqdm(zips, desc="Processing zips"):
        try:
            res = dry_run_on_zip(z, args.roi_dir, args.target_mpp, args.generate_plots, args.tile_ext)
        except Exception as exc:
            print(f"Error processing {z.name}: {exc}")
            skipped_count += 1
            continue
        if res == "empty_csv":
            empty_csv_count += 1
            continue
        if not res:
            skipped_count += 1

    print(f"Empty CSV annotation files skipped: {empty_csv_count}")
    if skipped_count:
        print(f"Other slides skipped due to errors or missing data: {skipped_count}")


if __name__ == "__main__":
    main()
