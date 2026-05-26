"""Stream tar shards to tumor-only zip archives without staging to disk.

Each tar shard may contain multiple slides. For each slide directory found
in the tar: read tile coordinates from the .coords.json sidecar files, run
the ROI overlap test in memory, then write only the passing tiles straight
into a per-slide output zip. The full shard is never extracted.

The tar is opened twice per shard: once to read filenames + coords.json
(no image data), and once to stream only the selected PNGs to the output zips.

Usage (YAML config):
    python -m src.preprocessing.tar_to_tumor_zip --config src/config.yaml

Usage (CLI):
    python -m src.preprocessing.tar_to_tumor_zip \\
        --tar-dir /bulk/shards \\
        --roi-dir /data/annotations \\
        --save-dir $WORKSPACE_PATH/data/poc-tumor-tiles \\
        --target-mpp 0.5 --native-slide-mpp 0.25
"""

import argparse
import concurrent.futures
import json
import logging
import os
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from shapely.affinity import scale
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from tqdm import tqdm

from preprocessing.utils import match_zip_to_csv, read_annotations

load_dotenv()

_TAR_EXTS = ('.tar.gz', '.tgz', '.tar.bz2', '.tar')  # longest-first for suffix stripping

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _resolve_config_paths(cfg, repo_root):
    """Resolve relative paths in a config dict (same convention as get_tiles_within_rois)."""
    path_keys = {'tar_dir', 'roi_dir', 'save_dir', 'zip_dir', 'out_dir', 'output_dir', 'logdir'}

    def _resolve(v):
        if not isinstance(v, str) or v.startswith('/'):
            return v
        normalized = v[2:] if v.startswith('./') else v
        if normalized.startswith(('data/', 'dataframes/', 'experiments/')):
            candidate = (repo_root.parent / normalized).resolve()
            if candidate.exists() or not (repo_root / v).exists():
                return str(candidate)
        return str((repo_root / v).resolve())

    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if isinstance(v, str) and (k in path_keys or v.startswith('./') or v.startswith('../')):
                cfg[k] = _resolve(v)
            elif isinstance(v, dict):
                _resolve_config_paths(v, repo_root)
    return cfg


# ---------------------------------------------------------------------------
# Tar utilities
# ---------------------------------------------------------------------------

def _collect_tar_slides(tar_path):
    """Return {slide_dir_name: [(png_member_path, basename, x, y), ...]}

    Opens the tar once. For each .png tile, reads its .coords.json sidecar to
    get the actual (x, y) coordinates — more reliable than parsing filenames.
    Slides with no parseable coords are silently skipped.
    """
    slide_tiles = {}

    try:
        with tarfile.open(tar_path, 'r:*') as tf:
            members = {m.name: m for m in tf.getmembers() if m.isfile()}

            for member_path, m in members.items():
                if not member_path.endswith('.png'):
                    continue

                parts = member_path.split('/')
                if len(parts) < 3:
                    continue
                slide_dir = parts[1]
                base = parts[-1]

                coords_path = member_path[:-4] + '.coords.json'
                if coords_path not in members:
                    continue

                try:
                    f = tf.extractfile(members[coords_path])
                    if f is None:
                        continue
                    coords = json.load(f)
                    x, y = float(coords['x']), float(coords['y'])
                    slide_tiles.setdefault(slide_dir, []).append((member_path, base, x, y))
                except Exception as e:
                    logging.debug('Failed to read coords for %s: %s', member_path, e)

    except Exception as e:
        logging.warning('Failed to open tar %s: %s', tar_path, e)

    return slide_tiles


# ---------------------------------------------------------------------------
# Geometry / selection (no I/O)
# ---------------------------------------------------------------------------

def _eval_coord_hypotheses(tile_xy, ann_raw, slide_mpp, target_mpp, tile_size_px, max_samples=200):
    """Score (coords_in_microns, coord_is_center) hypotheses given known (x, y) pairs.

    Returns list of result dicts sorted by frac_covered descending.

    Annotations are always in native slide pixels and are always scaled by
    slide_mpp / target_mpp regardless of the tile-coordinate hypothesis.
    The hypothesis only controls how tile coords are converted to target pixels.
    """
    results = []
    sample = tile_xy[:max_samples]

    # Annotation is always in native slide pixels → scale to target pixels once.
    ann_sf = slide_mpp / target_mpp
    ann_poly_base = scale(ann_raw, xfact=ann_sf, yfact=ann_sf, origin=(0, 0))

    for coords_in_microns, coord_is_center in [(True, False), (True, True), (False, False), (False, True)]:
        try:
            ann_poly = ann_poly_base
            t_sz = tile_size_px  # tiles are extracted at target_mpp so size is always tile_size_px target px

            polys, centers = [], []
            for x, y in sample:
                xt = float(x) / target_mpp if coords_in_microns else float(x) * slide_mpp / target_mpp
                yt = float(y) / target_mpp if coords_in_microns else float(y) * slide_mpp / target_mpp
                x0 = xt - t_sz / 2 if coord_is_center else xt
                y0 = yt - t_sz / 2 if coord_is_center else yt
                polys.append(Polygon([(x0, y0), (x0 + t_sz, y0), (x0 + t_sz, y0 + t_sz), (x0, y0 + t_sz)]))
                centers.append((xt if coord_is_center else xt + t_sz / 2,
                                 yt if coord_is_center else yt + t_sz / 2))

            if not polys:
                continue

            intersects = [ann_poly.intersection(p) for p in polys]
            hits = [i for i in intersects if i.area > 0]
            ann_area = ann_poly.area
            frac = unary_union(hits).area / ann_area if hits and ann_area > 0 else 0.0
            ovlp_ratios = [i.area / p.area for i, p in zip(intersects, polys) if p.area > 0 and i.area > 0]
            results.append({
                'hypothesis': f'coords_in_microns={coords_in_microns}, center={coord_is_center}',
                'frac_covered': frac,
                'mean_overlap': float(np.mean(ovlp_ratios)) if ovlp_ratios else 0.0,
                'n_intersect': len(hits),
                'n_center_in': sum(1 for cx, cy in centers if ann_poly.contains(Point(cx, cy))),
            })
        except Exception as e:
            logging.debug('Hypothesis eval error: %s', e)

    return sorted(results, key=lambda r: r['frac_covered'], reverse=True)


def _select_tumor_tiles(tile_data, ann_raw, native_slide_mpp, target_mpp, tile_size_px,
                        area_threshold, selection_mode, annotation_buffer):
    """Return (selected_basenames, diagnostics_dict). Pure geometry — no I/O.

    tile_data: list of (member_path, basename, x, y)
    Coordinates come from .coords.json and are in an unknown unit (microns or
    native slide pixels) — the hypothesis evaluation auto-detects which.
    """
    tile_xy = [(x, y) for _, _, x, y in tile_data]
    basenames = [base for _, base, _, _ in tile_data]

    eval_res = _eval_coord_hypotheses(tile_xy, ann_raw, native_slide_mpp, target_mpp, tile_size_px)
    if eval_res:
        h = eval_res[0]['hypothesis']
        coords_in_microns = 'coords_in_microns=True' in h
        coord_is_center = 'center=True' in h
        logging.info('Auto-chosen hypothesis: %s (frac_covered=%.3f)', h, eval_res[0]['frac_covered'])
    else:
        coords_in_microns, coord_is_center = True, False

    # Annotation is always in native slide pixels → target pixels.
    ann_sf = native_slide_mpp / target_mpp
    scaled_ann = scale(ann_raw, xfact=ann_sf, yfact=ann_sf, origin=(0, 0))
    if not scaled_ann.is_valid:
        scaled_ann = scaled_ann.buffer(0)

    t_sz = tile_size_px  # tiles extracted at target_mpp, so size is always tile_size_px target px

    polygons = []
    for x, y in tile_xy:
        xt = float(x) / target_mpp if coords_in_microns else float(x) * native_slide_mpp / target_mpp
        yt = float(y) / target_mpp if coords_in_microns else float(y) * native_slide_mpp / target_mpp
        x0 = xt - t_sz / 2 if coord_is_center else xt
        y0 = yt - t_sz / 2 if coord_is_center else yt
        polygons.append(Polygon([(x0, y0), (x0 + t_sz, y0), (x0 + t_sz, y0 + t_sz), (x0, y0 + t_sz)]))

    # sanity check: annotation must overlap the tile region
    ann_check = scaled_ann if scaled_ann.is_valid else scaled_ann.buffer(0)
    try:
        tiles_box = Polygon([
            (min(p.bounds[0] for p in polygons), min(p.bounds[1] for p in polygons)),
            (max(p.bounds[2] for p in polygons), min(p.bounds[1] for p in polygons)),
            (max(p.bounds[2] for p in polygons), max(p.bounds[3] for p in polygons)),
            (min(p.bounds[0] for p in polygons), max(p.bounds[3] for p in polygons)),
        ])
        ovlp_ratio = tiles_box.intersection(ann_check).area / ann_check.area if ann_check.area > 0 else 0.0
    except Exception:
        ovlp_ratio = 0.0

    if ovlp_ratio < 0.1:
        logging.warning('Only %.1f%% of annotation overlaps tile region — likely MPP mismatch. Skipping.', ovlp_ratio * 100)
        return [], {'skipped': True, 'reason': 'coordinate_mismatch', 'overlap_ratio': float(ovlp_ratio)}

    ann_buf = scaled_ann.buffer(float(annotation_buffer)) if annotation_buffer and annotation_buffer > 0 else scaled_ann

    sel_center, sel_overlap = [], []
    for basename, poly in zip(basenames, polygons):
        try:
            minx, miny, maxx, maxy = poly.bounds
            cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
            if isinstance(ann_buf, (Polygon, MultiPolygon)) and ann_buf.contains(Point(cx, cy)):
                sel_center.append(basename)
            intsect = scaled_ann.intersection(poly)
            if intsect.area > 0 and intsect.area / poly.area >= area_threshold:
                sel_overlap.append(basename)
        except Exception:
            pass

    if selection_mode == 'center':
        selected = list(set(sel_center))
    elif selection_mode == 'overlap':
        selected = list(set(sel_overlap))
    else:
        selected = list(set(sel_center) | set(sel_overlap))

    return selected, {
        'total_tiles': len(tile_data),
        'selected': len(selected),
        'center_hits': len(sel_center),
        'overlap_hits': len(sel_overlap),
        'selection_mode': selection_mode,
        'hypothesis': eval_res[0]['hypothesis'] if eval_res else 'default',
    }


# ---------------------------------------------------------------------------
# Per-shard worker
# ---------------------------------------------------------------------------

def process_tar(tar_fname, tar_dir, save_dir, roi_dir, target_mpp, area_threshold,
                selection_mode, annotation_buffer, native_slide_mpp, tile_size_px):
    """Process one tar shard: filter each slide within it to tumor tiles.

    Writes one output zip per slide into save_dir. Opens the tar twice:
    once to read filenames + coords.json, once to stream selected PNGs.
    """
    tar_path = os.path.join(tar_dir, tar_fname)
    slide_tiles = _collect_tar_slides(tar_path)

    if not slide_tiles:
        print(f'No slides with parseable tiles in {tar_fname}')
        return 0

    roi_csv_list = os.listdir(roi_dir)
    written = 0

    for slide_dir, tile_data in slide_tiles.items():
        # use slide_dir (not tar_fname) for annotation matching and output naming
        out_zip = os.path.join(save_dir, f'{slide_dir}.zip')
        if os.path.exists(out_zip):
            # Verify the existing zip is not a corrupt partial from a prior crash.
            try:
                with zipfile.ZipFile(out_zip, 'r') as _zf:
                    names = _zf.namelist()
                    if names:
                        with _zf.open(names[0]) as _f:
                            _f.read(256)  # probe compressed data of first tile
                logging.info('Skipping %s: valid zip already exists (%d tiles)', slide_dir, len(names))
                written += 1
                continue
            except Exception as _e:
                logging.warning('Corrupt zip for %s (%s) — re-creating', slide_dir, _e)
                os.remove(out_zip)

        # match annotation CSV by slide directory name
        try:
            roi_csv = match_zip_to_csv(slide_dir, roi_csv_list)
            if roi_csv is None:
                print(f'  {slide_dir[:40]}: no annotation CSV matched, skipping')
                continue
        except Exception as e:
            print(f'  {slide_dir[:40]}: annotation match error: {e}')
            continue

        try:
            ann_raw, _ = read_annotations(os.path.join(roi_dir, roi_csv))
        except Exception as e:
            print(f'  {slide_dir[:40]}: failed to read annotations: {e}')
            continue

        selected, diag = _select_tumor_tiles(
            tile_data, ann_raw, native_slide_mpp, target_mpp, tile_size_px,
            area_threshold, selection_mode, annotation_buffer,
        )
        print(f'  {slide_dir[:50]}: {diag}')

        if not selected:
            print(f'  {slide_dir[:40]}: no tumor tiles selected')
            continue

        # stream selected PNGs from tar → zip, no staging to disk.
        # Write to a .tmp file first; rename to final path only on clean
        # completion — so a crash/OOM never leaves a corrupt partial zip.
        selected_set = set(selected)
        member_map = {base: full for full, base, _, _ in tile_data}
        out_zip_tmp = f'{out_zip}.{os.getpid()}.tmp'

        try:
            with (
                tarfile.open(tar_path, 'r:*') as tf,
                zipfile.ZipFile(out_zip_tmp, 'w', zipfile.ZIP_DEFLATED) as zf,
            ):
                for base, full_name in member_map.items():
                    if base not in selected_set:
                        continue
                    try:
                        f = tf.extractfile(tf.getmember(full_name))
                        if f is not None:
                            zf.writestr(os.path.join(slide_dir, base), f.read())
                    except Exception as e:
                        logging.warning('Failed to copy %s: %s', base, e)
            os.replace(out_zip_tmp, out_zip)  # atomic on POSIX
            written += 1
            print(f'  {slide_dir[:50]}: wrote {len(selected)} tiles → {os.path.basename(out_zip)}')
        except Exception as e:
            print(f'  {slide_dir[:40]}: error writing zip: {e}')
            for path in (out_zip_tmp, out_zip):
                if os.path.exists(path):
                    os.remove(path)

    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Filter tar-sharded tiles to tumor-only zip archives'
    )
    parser.add_argument('--tar-dir', type=str, default=None,
                        help='Directory containing tar shards (may be outside workspace)')
    parser.add_argument('--roi-dir', type=str, default=None,
                        help='Directory containing tumor annotation CSVs')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Output directory for tumor-only zip files')
    parser.add_argument('--target-mpp', type=float, default=0.5)
    parser.add_argument('--native-slide-mpp', type=float, default=0.25,
                        help='Native slide MPP; annotations are in this pixel space (TCGA 40x = 0.25)')
    parser.add_argument('--tile-size-px', type=float, default=256.0,
                        help='Tile size in pixels at extraction resolution (HEST default: 256)')
    parser.add_argument('--area-threshold', type=float, default=0.6)
    parser.add_argument('--selection-mode', choices=['center', 'overlap', 'both'], default='overlap')
    parser.add_argument('--annotation-buffer', type=float, default=0.0)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config; values read from the tar_to_tumor_zip section')

    temp_args, _ = parser.parse_known_args()
    if temp_args.config:
        try:
            config_path = Path(temp_args.config).expanduser().resolve()
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            if isinstance(cfg, dict) and 'tar_to_tumor_zip' in cfg:
                cfg = cfg['tar_to_tumor_zip'] or {}
            repo_root = config_path.parent
            if repo_root.name == 'src':
                repo_root = repo_root.parent
            _resolve_config_paths(cfg, repo_root)
            parser.set_defaults(**cfg)
        except Exception as e:
            print(f'Failed to load config {temp_args.config}: {e}')

    args = parser.parse_args()

    if not args.tar_dir or not args.roi_dir or not args.save_dir:
        parser.error('--tar-dir, --roi-dir, and --save-dir are required (via CLI or config)')

    os.makedirs(args.save_dir, exist_ok=True)

    tar_fnames = sorted(
        f for f in os.listdir(args.tar_dir)
        if any(f.lower().endswith(ext) for ext in _TAR_EXTS)
    )
    print(f'Found {len(tar_fnames)} tar shards in {args.tar_dir}')

    total_written = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_tar,
                fname,
                args.tar_dir,
                args.save_dir,
                args.roi_dir,
                args.target_mpp,
                args.area_threshold,
                args.selection_mode,
                args.annotation_buffer,
                args.native_slide_mpp,
                args.tile_size_px,
            ): fname
            for fname in tar_fnames
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc='Processing shards'
        ):
            try:
                total_written += future.result()
            except Exception as exc:
                print(f'{futures[future]} raised: {exc}')

    print(f'Done: wrote {total_written} slide zip(s) across {len(tar_fnames)} shards.')


if __name__ == '__main__':
    main()
