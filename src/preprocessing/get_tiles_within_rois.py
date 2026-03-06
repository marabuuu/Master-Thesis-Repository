"""Filter pre-extracted tiles or WSIs using region-of-interest (ROI) annotations.

This script was adapted from the mopadi repository to work with tiles stored in
zip archives. It accepts either a directory of zip files and copies/extracts tiles 
that overlap annotated regions by a configurable threshold.

Example usage from the command line::

    python -m src.preprocessing.get_tiles_within_rois \
        --zip-dir /path/to/tiles/zips \
        --roi-dir /path/to/annotations \
        --save-dir /path/to/output \
        --target-mpp 0.5 \
        --native-slide-mpp 0.25 \
        --plots --area-threshold 0.6 --selection-mode overlap

The script supports parallel processing via `--num-workers` and outputs a
zip archive and optional plot for each processed slide.
"""
# taken and adapted from mopadi repository to work with pre-extracted tiles in zip files
# https://github.com/KatherLab/mopadi/blob/main/src/mopadi/data_prep/get_tiles_within_rois.py
import os
import re
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import shapely.geometry as sg
import numpy as np
from shapely.affinity import scale
from shapely.affinity import translate
from shapely.ops import unary_union
from shapely.geometry import Point
import logging
import concurrent.futures
import shutil
from tqdm import tqdm
from preprocessing.utils import parse_tile_coords, read_zip_tile_list, read_tiler_params, get_mpp_from_tiler_params, match_zip_to_csv, read_annotations, extract_selected_from_zip
from shapely import speedups
from matplotlib.patches import Polygon as MatplotlibPolygon
import rtree
from dotenv import load_dotenv
import zipfile
import tarfile
import argparse
import json
import yaml  # pyyaml is required for configuration file support


load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


def process_tile(tile_data):
    tile_fname, tile_polygon, scaled_annPolys, index, img_dir, tiles_folder, filtered_tiles_dir, polygons = tile_data    

    try:
        #intersection = scaled_annPolys.intersection(tile_polygon) #for debugging
        #print(f"Tile ID: {tile_fname}, Intersection Area: {intersection.area}, Tile Area: {tile_polygon.area}, Ratio: {intersection.area / tile_polygon.area}")
        intersecting_ids = list(index.intersection(scaled_annPolys.bounds))
        #print(f"Tile bounds: {tile_polygon.bounds}, Intersecting IDs: {intersecting_ids}")
        if len(intersecting_ids) > 0 and index.count(tile_polygon.bounds) > 0:
            if isinstance(scaled_annPolys, Polygon) and tile_polygon.intersects(scaled_annPolys):
                intersection = scaled_annPolys.intersection(tile_polygon)
                #print(f"Intersection area: {intersection.area}, Tile area: {tile_polygon.area}")

                if (intersection.area / tile_polygon.area) >= 0.6 and not os.path.exists(os.path.join(filtered_tiles_dir, tile_fname)):
                    shutil.copy(os.path.join(img_dir, tiles_folder, tile_fname), filtered_tiles_dir)
                    #print(f"{os.path.join(img_dir, tiles_folder, tile_fname)} copied to {filtered_tiles_dir}")

            elif isinstance(scaled_annPolys, MultiPolygon):
                for polygon in scaled_annPolys.geoms:
                    if tile_polygon.intersects(polygon):
                        intersection = polygon.intersection(tile_polygon)
                        #print(f"Intersection area: {intersection.area}, Tile area: {tile_polygon.area}")

                        if (intersection.area / tile_polygon.area) >= 0.6 and not os.path.exists(os.path.join(filtered_tiles_dir, tile_fname)):
                            shutil.copy(os.path.join(img_dir, tiles_folder, tile_fname), filtered_tiles_dir)
                            #print(f"{os.path.join(img_dir, tiles_folder, tile_fname)} copied to {filtered_tiles_dir}")
        else:
            print(f"Length of intersecting_ids is 0: No bounding box intersection found for tile {tile_fname}")
    except Exception as err:
        print(f"There was an error with a polygon, which was not caught by existing checks, skipping this tile. Error: {err}")


def process_zip(zip_fname, zip_dir, save_dir, roi_dir, target_mpp, generate_plots, area_threshold=0.6, selection_mode='center', annotation_buffer=0.0, native_slide_mpp=0.25):
    """
    Process a ZIP archive of pre-extracted tiles.
    
    Key coordinate systems:
    - Tile filenames contain coordinates in MICRONS
    - Annotation CSV contains coordinates in NATIVE SLIDE PIXELS
    - We convert everything to TARGET PIXELS for comparison (matching original WSI workflow)
    
    Args:
        native_slide_mpp: The native slide's microns-per-pixel (e.g., 0.25 for 40x TCGA slides).
                         Annotations are in this pixel space.
    """
    tiles_folder = os.path.splitext(zip_fname)[0]
    filtered_tiles_dir = os.path.join(save_dir, tiles_folder)

    if os.path.exists(filtered_tiles_dir) and len(os.listdir(filtered_tiles_dir)) > 0:
        print(f'Skipping {tiles_folder}: output directory already exists and is non-empty')
        return True

    zip_path = os.path.join(zip_dir, zip_fname)
    if not os.path.exists(zip_path):
        print(f'ZIP file not found: {zip_path}, skipping...')
        return False

    # read tiler params - this gives us the EXTRACTION MPP (target), not native slide MPP
    params = read_tiler_params(zip_path)
    extraction_mpp = get_mpp_from_tiler_params(params)  # typically 0.5 for HEST
    if extraction_mpp is None:
        print(f'Could not determine extraction MPP for {zip_fname}, skipping...')
        return False
    
    print(f"Extraction MPP (from tiler_params): {extraction_mpp}")
    print(f"Native slide MPP (for annotations): {native_slide_mpp}")
    print(f"Target MPP: {target_mpp}")

    # tile size in px
    tile_size_px = float(params.get('tile_size_px', 512))

    # list tiles inside zip
    tile_fnames = read_zip_tile_list(zip_path)
    if not tile_fnames:
        print(f'No tile files found in {zip_fname}, skipping...')
        return False
    # Print raw tile coordinate bounds
    try:
        # Build a cleaned list of parsed coordinates (avoid calling parse_tile_coords twice)
        coords_raw = []
        for fn in tile_fnames:
            c = parse_tile_coords(fn)
            if c is None:
                continue
            # ensure coordinate is indexable with two elements
            if not (isinstance(c, (list, tuple)) and len(c) >= 2):
                continue
            try:
                x_val = float(c[0])
                y_val = float(c[1])
            except Exception:
                continue
            coords_raw.append((x_val, y_val))

        if coords_raw:
            xs = [c[0] for c in coords_raw]
            ys = [c[1] for c in coords_raw]
            try:
                print(f"Raw tile coordinate bounds: x=({min(xs)}, {max(xs)}), y=({min(ys)}, {max(ys)})")
                print(f"Raw tile coordinate centroid: ({np.mean(xs)}, {np.mean(ys)})")
            except Exception as e:
                print(f"Failed to compute simple statistics for raw coords: {e}")
        else:
            print("No valid tile coordinates parsed from filenames")
    except Exception as e:
        print(f"Failed to print raw tile coordinate bounds: {e}")

    os.makedirs(filtered_tiles_dir, exist_ok=True)

    # find corresponding annotations file
    try:
        # pass the full basename length so matching considers the full identifier (DX1/DX2)
        full_prefix_len = len(os.path.splitext(zip_fname)[0])
        roi_fname = match_zip_to_csv(zip_fname, os.listdir(roi_dir), max_prefix=full_prefix_len)
        if roi_fname is None:
            print('Could not find corresponding CSV file; the slide will be skipped...')
            return False
    except Exception as err:
        print(f"Exception during CSV file reading: {err}; the slide will be skipped")
        return False

    try:
        ann_raw, _ = read_annotations(os.path.join(roi_dir, roi_fname))
        # Print raw annotation bounds
        try:
            ann_bounds = ann_raw.bounds if hasattr(ann_raw, 'bounds') else None
            ann_centroid = (ann_raw.centroid.x, ann_raw.centroid.y) if hasattr(ann_raw, 'centroid') else None
            print(f"Raw annotation bounds: {ann_bounds}")
            print(f"Raw annotation centroid: {ann_centroid}")
        except Exception as e:
            print(f"Failed to print raw annotation bounds: {e}")
    except Exception as err:
        print(f"Error reading annotations for slide {tiles_folder}: {err}")
        return False

    # Evaluate several coordinate interpretation hypotheses to help debugging
    def evaluate_coord_hypotheses(tile_fnames, ann_raw, slide_mpp, target_mpp, tile_size_px, max_samples=None):
        try:
            from math import isfinite
            results = []
            # compute target tile size in pixels
            try:
                tile_size_target = float(tile_size_px) * (slide_mpp / target_mpp)
            except Exception:
                tile_size_target = tile_size_px

            hypotheses = [
                (True, False),
                (True, True),
                (False, False),
                (False, True),
            ]

            sample_list = tile_fnames if max_samples is None else tile_fnames[:max_samples]

            for coords_in_microns, coord_is_center in hypotheses:
                # scale annotation polygon into target-pixel space according to this hypothesis
                try:
                    if coords_in_microns:
                        ann_poly = scale(ann_raw, xfact=(1.0 / target_mpp), yfact=(1.0 / target_mpp), origin=(0, 0))
                    else:
                        ann_poly = scale(ann_raw, xfact=(slide_mpp / target_mpp), yfact=(slide_mpp / target_mpp), origin=(0, 0))
                except Exception:
                    ann_poly = ann_raw
                polys = []
                centers = []
                for fn in sample_list:
                    c = parse_tile_coords(fn)
                    if c is None:
                        continue
                    x, y = c
                    if coords_in_microns:
                        # coords given in microns -> convert to target pixels
                        x_t = float(x) / float(target_mpp)
                        y_t = float(y) / float(target_mpp)
                    else:
                        # coords given in slide pixels -> convert to target pixels
                        x_t = float(x) * (slide_mpp / target_mpp)
                        y_t = float(y) * (slide_mpp / target_mpp)

                    if coord_is_center:
                        x0 = x_t - tile_size_target / 2.0
                        y0 = y_t - tile_size_target / 2.0
                        center = (x_t, y_t)
                    else:
                        x0 = x_t
                        y0 = y_t
                        center = (x_t + tile_size_target / 2.0, y_t + tile_size_target / 2.0)

                    poly = Polygon([(x0, y0), (x0 + tile_size_target, y0), (x0 + tile_size_target, y0 + tile_size_target), (x0, y0 + tile_size_target)])
                    polys.append(poly)
                    centers.append(center)

                if not polys:
                    continue

                # compute metrics
                n_tiles = len(polys)
                intersects = [p.intersection(ann_poly) for p in polys]
                n_intersect = sum(1 for i in intersects if i.area > 0)
                # centers inside annotation
                n_center_in = sum(1 for c in centers if ann_poly.contains(Point(c)))
                try:
                    union_area = unary_union([i for i in intersects if i.area > 0]).area
                    ann_area = ann_poly.area
                    frac_ann_covered = union_area / ann_area if ann_area > 0 else 0.0
                except Exception:
                    frac_ann_covered = 0.0

                overlap_ratios = [ (i.area / p.area) for i, p in zip(intersects, polys) if p.area > 0 and i.area > 0 ]
                mean_overlap = float(np.mean(overlap_ratios)) if overlap_ratios else 0.0

                results.append({
                    'hypothesis': f"coords_in_microns={coords_in_microns}, center={coord_is_center}",
                    'n_tiles': n_tiles,
                    'n_intersect': n_intersect,
                    'n_center_in': n_center_in,
                    'frac_ann_covered': frac_ann_covered,
                    'mean_overlap': mean_overlap,
                })

            # sort by frac_ann_covered desc
            results = sorted(results, key=lambda r: r['frac_ann_covered'], reverse=True)
            return results
        except Exception as e:
            print(f"Failed to evaluate hypotheses: {e}")
            return []

    # run diagnostic evaluation (small sample to keep it fast)
    # pass raw annotations (unscaled) so hypotheses test both unit interpretations
    eval_results = evaluate_coord_hypotheses(tile_fnames, ann_raw, extraction_mpp, target_mpp, tile_size_px, max_samples=200)
    if eval_results:
        print("Coordinate hypothesis evaluation (sorted by fraction annotation covered):")
        for r in eval_results:
            print(r)

    # Use hypothesis evaluation to pick best interpretation (coords in microns vs pixels, center vs top-left)
    coords_in_microns_flag = None
    coord_is_center_flag = False
    try:
        if eval_results:
            best = eval_results[0]
            # parse hypothesis string like 'coords_in_microns=True, center=False'
            h = best['hypothesis']
            coords_in_microns_flag = 'coords_in_microns=True' in h
            coord_is_center_flag = 'center=True' in h
            print(f"Auto-chosen coordinate hypothesis from evaluation: {h}")
        else:
            coords_in_microns_flag = None
            coord_is_center_flag = False
    except Exception as e:
        print(f"Failed to pick hypothesis: {e}")

    # =========================================================================
    # COORDINATE CONVERSION STRATEGY (matching original WSI script):
    # - Tile coords in ZIP are in MICRONS -> convert to TARGET PIXELS
    # - Annotation coords are in NATIVE SLIDE PIXELS -> convert to TARGET PIXELS
    # - Everything compared in TARGET PIXEL space
    # - Tile polygons are 512x512 TARGET PIXELS (same as original script)
    # =========================================================================
    
    # Convert tile coordinates from microns to target pixels
    # tile_coord_microns / target_mpp = tile_coord_target_pixels
    polygons = []
    for fn in tile_fnames:
        c = parse_tile_coords(fn)
        if c is None:
            continue
        x_um, y_um = c
        
        # Convert from microns to target pixels
        x_px = float(x_um) / float(target_mpp)
        y_px = float(y_um) / float(target_mpp)
        
        # If coordinate represents center, shift to top-left
        if coord_is_center_flag:
            x_px -= tile_size_px / 2.0
            y_px -= tile_size_px / 2.0
        
        # Create 512x512 pixel polygon (same as original script)
        poly = Polygon([
            (x_px, y_px), 
            (x_px + tile_size_px, y_px), 
            (x_px + tile_size_px, y_px + tile_size_px), 
            (x_px, y_px + tile_size_px)
        ])
        polygons.append(poly)
    
    # Print tile polygon bounds for diagnostics
    if polygons:
        all_tile_minx = min(p.bounds[0] for p in polygons)
        all_tile_miny = min(p.bounds[1] for p in polygons)
        all_tile_maxx = max(p.bounds[2] for p in polygons)
        all_tile_maxy = max(p.bounds[3] for p in polygons)
        print(f"Tile polygon bounds (target px): minx={all_tile_minx:.1f}, miny={all_tile_miny:.1f}, maxx={all_tile_maxx:.1f}, maxy={all_tile_maxy:.1f}")
    
    # Scale annotations from NATIVE SLIDE PIXELS to TARGET PIXELS
    # This matches the original script: scale_factor = slide_mpp / target_mpp
    # But here slide_mpp is the NATIVE slide MPP, not the extraction MPP
    try:
        scale_factor = native_slide_mpp / target_mpp
        scaled_annPolys = scale(ann_raw, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
        print(f"Converted annotation from native slide pixels to target pixels (scale factor = {scale_factor})")
        print(f"Annotation bounds (target px): {scaled_annPolys.bounds}")
    except Exception as e:
        print(f"Failed to scale annotations: {e}")
        scaled_annPolys = ann_raw

    # Ensure scaled_annPolys is a valid geometry object (avoid None)
    if scaled_annPolys is None:
        scaled_annPolys = Polygon()
    
    tiles_with_polygons = list(zip(tile_fnames, polygons))

    # Auto-align annotation to tile region if there's poor/no overlap
    # This handles the case where tile coords are relative (starting at 0,0) 
    # while annotation coords are in full slide space
    try:
        tiles_overall_bounds = (
            min([p.bounds[0] for p in polygons]), 
            min([p.bounds[1] for p in polygons]), 
            max([p.bounds[2] for p in polygons]), 
            max([p.bounds[3] for p in polygons])
        )
        ann_bounds = scaled_annPolys.bounds  # (minx, miny, maxx, maxy)
        
        # Check if annotation overlaps with tile region
        tiles_box = Polygon([
            (tiles_overall_bounds[0], tiles_overall_bounds[1]),
            (tiles_overall_bounds[2], tiles_overall_bounds[1]),
            (tiles_overall_bounds[2], tiles_overall_bounds[3]),
            (tiles_overall_bounds[0], tiles_overall_bounds[3])
        ])
        
        overlap = tiles_box.intersection(scaled_annPolys)
        overlap_ratio = overlap.area / scaled_annPolys.area if scaled_annPolys.area > 0 else 0
        
        print(f"Tiles region: {tiles_overall_bounds}")
        print(f"Annotation region: {ann_bounds}")
        print(f"Overlap ratio (annotation covered by tiles): {overlap_ratio:.3f}")
        
        # If less than 10% of annotation overlaps with tiles, translate annotation to tile region
        if overlap_ratio < 0.1:
            # Translate annotation so its min corner aligns with tile region min corner
            # (This assumes annotation should be somewhere within the tile extraction area)
            dx = tiles_overall_bounds[0] - ann_bounds[0]
            dy = tiles_overall_bounds[1] - ann_bounds[1]
            print(f"Poor overlap detected. Translating annotation by dx={dx:.1f}, dy={dy:.1f} to align with tile region.")
            scaled_annPolys = translate(scaled_annPolys, xoff=dx, yoff=dy)
            print(f"Annotation bounds after translation: {scaled_annPolys.bounds}")
        else:
            print(f"Good overlap ({overlap_ratio:.1%}), no translation needed.")
    except Exception as e:
        print(f"Auto-alignment check failed: {e}")

    if generate_plots:
        os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)

        fig, ax = plt.subplots()

        # Plot annotations safely: handle empty, Polygon, MultiPolygon, or other geometry types
        try:
            if scaled_annPolys is None or getattr(scaled_annPolys, 'is_empty', False):
                print(f"No annotation geometry to plot for {tiles_folder}")
            else:
                # If MultiPolygon-like, iterate geometries
                if isinstance(scaled_annPolys, (MultiPolygon,)) or (hasattr(scaled_annPolys, 'geoms') and not isinstance(scaled_annPolys, Polygon)):
                    for polygon in getattr(scaled_annPolys, 'geoms', []):
                        if polygon is None or getattr(polygon, 'is_empty', False):
                            continue
                        try:
                            x, y = polygon.exterior.xy
                            ax.fill(x, y, alpha=0.5, fc='r', label='Annotated Tissue')
                        except Exception:
                            continue
                else:
                    ext = getattr(scaled_annPolys, 'exterior', None)
                    if ext is not None:
                        try:
                            x, y = ext.xy
                            ax.fill(x, y, alpha=0.5, fc='r', ec='Annotated Tissue')
                        except Exception:
                            pass
        except Exception as e:
            print(f"Failed plotting annotation geometry: {e}")

        # Plot tiles (skip any invalid polygons)
        for tile in polygons:
            try:
                if tile is None or getattr(tile, 'is_empty', False):
                    continue
                x, y = tile.exterior.xy
                ax.fill(x, y, alpha=0.5, fc='b', ec='none', label='Extracted Tiles')

                minx, miny, maxx, maxy = tile.bounds
                ax.add_patch(MatplotlibPolygon([[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny]], fill=None, edgecolor='b', linestyle='--'))
            except Exception:
                continue

        legend_elements = [Patch(facecolor='red', edgecolor='r', alpha=0.5, label='Annotated Tissue'),
                           Patch(facecolor='blue', edgecolor='b', alpha=0.5, label='All Tiles')]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.invert_yaxis()

        plt.savefig(os.path.join(save_dir, "plots", f'{tiles_folder}.png'))
        plt.close()

    # spatial index
    index = rtree.index.Index()
    for idx, polygon in enumerate(polygons):
        index.insert(idx, polygon.bounds)

    # Selection: compute both center-based and overlap-based selections
    selected_by_center = []
    selected_by_overlap = []
    centers = {}
    for tile_fname, tile_polygon in tiles_with_polygons:
        try:
            minx, miny, maxx, maxy = tile_polygon.bounds
            center = ((minx + maxx) / 2.0, (miny + maxy) / 2.0)
            centers[tile_fname] = center

            # center-based test (optionally buffer annotation)
            ann_test = scaled_annPolys
            if annotation_buffer and annotation_buffer > 0:
                ann_test = scaled_annPolys.buffer(float(annotation_buffer))
            if isinstance(ann_test, (Polygon, MultiPolygon)) and ann_test.contains(Point(center)):
                selected_by_center.append(tile_fname)

            # overlap-based test
            try:
                intersection = scaled_annPolys.intersection(tile_polygon)
                if intersection.area > 0 and (intersection.area / tile_polygon.area) >= area_threshold:
                    selected_by_overlap.append(tile_fname)
            except Exception:
                pass
        except Exception as err:
            print(f"Error testing tile intersection for {tile_fname}: {err}")

    # choose final selected list based on selection_mode
    if selection_mode == 'center':
        selected = list(set(selected_by_center))
    elif selection_mode == 'overlap':
        selected = list(set(selected_by_overlap))
    elif selection_mode == 'both':
        selected = list(set(selected_by_center) | set(selected_by_overlap))
    else:
        selected = list(set(selected_by_center))

    # diagnostics metrics
    try:
        # union area of selected tiles
        sel_polys = [p for fn, p in tiles_with_polygons if fn in selected]
        union_sel = unary_union(sel_polys) if sel_polys else None
        tiles_union_area = union_sel.area if union_sel is not None else 0.0
        ann_area = scaled_annPolys.area
        frac_ann_covered = (tiles_union_area / ann_area) if ann_area > 0 else 0.0
        n_center_in = len(selected_by_center)
        n_overlap = len(selected_by_overlap)
        mean_overlap = 0.0
        if sel_polys:
            overlaps = []
            for p in sel_polys:
                i = p.intersection(scaled_annPolys)
                if p.area > 0:
                    overlaps.append(i.area / p.area)
            mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
        print(f"Selection metrics: selected={len(selected)}, center_hits={n_center_in}, overlap_hits={n_overlap}, frac_ann_covered={frac_ann_covered:.3f}, mean_overlap={mean_overlap:.3f}")

        # write diagnostics JSON into output folder
        try:
            diag = {
                'selected_count': len(selected),
                'center_hits': n_center_in,
                'overlap_hits': n_overlap,
                'frac_ann_covered': float(frac_ann_covered),
                'mean_overlap': float(mean_overlap),
                'selection_mode': selection_mode,
                'annotation_buffer': float(annotation_buffer),
                'area_threshold': float(area_threshold)
            }
            diag_path = os.path.join(filtered_tiles_dir, 'diagnostics_selection.json')
            with open(diag_path, 'w') as f:
                json.dump(diag, f, indent=2)
        except Exception as e:
            print(f"Failed to write selection diagnostics: {e}")
    except Exception as e:
        print(f"Failed to compute selection diagnostics: {e}")

    # extract selected tiles from zip
    if selected:
        extract_selected_from_zip(zip_path, selected, filtered_tiles_dir)
    else:
        print(f'No tiles selected for {tiles_folder}')
        # Print diagnostics to help debug scaling mismatch
        try:
            print(f"Total tiles in zip: {len(tile_fnames)}; selected: {len(selected)}")
            print(f"Annotation bounds (target px): {scaled_annPolys.bounds}")
            # print a few tile bounds samples
            sample_bounds = [p.bounds for p in polygons[:5]]
            print(f"Sample tile bounds (first 5, target px): {sample_bounds}")
            # full tile bounds
            try:
                all_mins = [p.bounds[0] for p in polygons] + [p.bounds[1] for p in polygons]
                all_maxs = [p.bounds[2] for p in polygons] + [p.bounds[3] for p in polygons]
                tiles_overall_bounds = (min([p.bounds[0] for p in polygons]), min([p.bounds[1] for p in polygons]), max([p.bounds[2] for p in polygons]), max([p.bounds[3] for p in polygons]))
                print(f"Tiles overall bounds (target px): {tiles_overall_bounds}")
                # compute tile centroid
                tile_centroid_x = (tiles_overall_bounds[0] + tiles_overall_bounds[2]) / 2.0
                tile_centroid_y = (tiles_overall_bounds[1] + tiles_overall_bounds[3]) / 2.0
                print(f"Tile centroid (target px): ({tile_centroid_x:.1f}, {tile_centroid_y:.1f})")
                print(f"Annotation centroid (target px): ({scaled_annPolys.centroid.x:.1f}, {scaled_annPolys.centroid.y:.1f})")
                print(f"Centroid delta (tile - annotation): ({tile_centroid_x - scaled_annPolys.centroid.x:.1f}, {tile_centroid_y - scaled_annPolys.centroid.y:.1f})")
            except Exception as e:
                print(f"Failed to compute overall tile bounds: {e}")

            # print tiler params for inspection
            try:
                print(f"Tiler params: {params}")
            except Exception:
                pass
        except Exception as e:
            print(f"Failed to print diagnostics: {e}")

    # create zip archive
    archive_path = os.path.join(save_dir, f"{tiles_folder}.zip")
    try:
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(filtered_tiles_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join(tiles_folder, os.path.relpath(file_path, filtered_tiles_dir))
                    zf.write(file_path, arcname)
    except Exception as e:
        print(f'Error creating archive for {tiles_folder}: {e}')

    # Clean up: remove the temporary folder with extracted tiles (keep only zip and plot)
    try:
        shutil.rmtree(filtered_tiles_dir)
    except Exception as e:
        print(f'Failed to clean up temporary folder {filtered_tiles_dir}: {e}')

    return True


def main():

    parser = argparse.ArgumentParser(description='Filter tiles within ROIs from WSIs or ZIP tile archives')
    parser.add_argument('--zip-dir', type=str, default=None, help='Directory containing zip files of tiles')
    parser.add_argument('--slide-dir', type=str, default=None, help='Directory containing WSI files (original behavior)')
    parser.add_argument('--img-dir', type=str, default=None, help='Directory containing tile folders (original behavior)')
    parser.add_argument('--roi-dir', type=str, help='Directory containing annotation CSVs')
    parser.add_argument('--save-dir', type=str, help='Directory to write filtered tiles / archives')
    parser.add_argument('--target-mpp', type=float, default=(256/512), help='Target MPP to scale annotations to')
    parser.add_argument('--native-slide-mpp', type=float, default=0.25, help='Native slide MPP (microns/pixel). TCGA 40x slides are typically 0.25. Annotations are in native slide pixels.')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--plots', action='store_true')
    parser.add_argument('--area-threshold', type=float, default=0.6)
    parser.add_argument('--selection-mode', type=str, choices=['center', 'overlap', 'both'], default='center', help='Tile selection mode: center, overlap, or both')
    parser.add_argument('--annotation-buffer', type=float, default=0.0, help='Buffer (in target-pixels) to expand annotations for center test')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML configuration file')

    # allow loading defaults from config prior to final argument parsing
    temp_args, _ = parser.parse_known_args()
    if temp_args.config:
        try:
            with open(temp_args.config) as f:
                cfg = yaml.safe_load(f) or {}
            # support hierarchical config files where arguments live under a
            # "preprocessing" section (makes it easier to combine with other
            # workflows). if so, pull that sub-dictionary.
            if isinstance(cfg, dict) and 'preprocessing' in cfg:
                cfg = cfg['preprocessing'] or {}
            parser.set_defaults(**cfg)
        except Exception as e:
            print(f"Failed to load config file {temp_args.config}: {e}")

    args = parser.parse_args()

    # ensure essential parameters are supplied either via CLI or config
    if not args.roi_dir or not args.save_dir:
        parser.error('Both --roi-dir and --save-dir must be provided (via command line or config file)')

    target_mpp = args.target_mpp
    num_workers = args.num_workers
    generate_plots = args.plots
    roi_dir = args.roi_dir
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    correctly_extracted_slide_nr = 0

    if args.zip_dir:
        zip_dir = args.zip_dir
        zip_fnames = [f for f in os.listdir(zip_dir) if f.lower().endswith('.zip')]

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    process_zip,
                    fname,
                    zip_dir,
                    save_dir,
                    roi_dir,
                    target_mpp,
                    generate_plots,
                    args.area_threshold,
                    args.selection_mode,
                    args.annotation_buffer,
                    args.native_slide_mpp
                ): fname for fname in zip_fnames
            }

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing zips..."):
                slide_name = futures[future]
                try:
                    if future.result():
                        correctly_extracted_slide_nr += 1
                except Exception as exc:
                    print(f'ZIP {slide_name} generated an exception: {exc}')

        print(f'Correctly processed {correctly_extracted_slide_nr} zip archives.')

    else:
        # fallback to original WSI + tiles folder mode
        if args.slide_dir is None or args.img_dir is None:
            raise ValueError('Either --zip-dir or both --slide-dir and --img-dir must be provided')

        slide_dir = args.slide_dir
        img_dir = args.img_dir

        slides_fnames = os.listdir(slide_dir)

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    fname,
                    slide_dir,
                    img_dir,
                    save_dir,
                    roi_dir,
                    target_mpp,
                    generate_plots
                ): fname for fname in slides_fnames
            }

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing slides..."):
                slide_name = futures[future]
                try:
                    if future.result():
                        correctly_extracted_slide_nr += 1
                except Exception as exc:
                    print(f'Slide {slide_name} generated an exception: {exc}')

        print(f'Correctly processed {correctly_extracted_slide_nr} slides.')

if __name__ == "__main__":
    main()