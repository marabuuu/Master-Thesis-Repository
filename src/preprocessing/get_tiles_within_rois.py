import os
import re
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import shapely.geometry as sg
import numpy as np
from shapely.affinity import scale
import logging
import concurrent.futures
import shutil
from tqdm import tqdm
from preprocessing.utils import *
from shapely import speedups
from matplotlib.patches import Polygon as MatplotlibPolygon
import rtree
from dotenv import load_dotenv
import argparse
import zipfile
from pathlib import Path
import tempfile
import tarfile

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")

# helpers for zip-based input
try:
    from preprocessing.get_tiles_within_rois_zip import (
        read_tiler_params_from_zip,
        list_tile_files_in_zip,
        build_polygons_for_tiles,
    )
except Exception:
    # if import fails, zip-mode won't be available
    read_tiler_params_from_zip = None
    list_tile_files_in_zip = None
    build_polygons_for_tiles = None





def extract_tile_from_zip(zip_path, tile_name, dest_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open(tile_name) as fh:
                data = fh.read()
        with open(dest_path, 'wb') as out:
            out.write(data)
        return True
    except Exception as e:
        print(f"Failed to extract {tile_name} from {zip_path}: {e}")
        return False


def process_zip(zip_path, save_dir, roi_dir, target_mpp, generate_plots, tile_ext='.png'):
    # Similar logic to the dry-run but actually extract selected tiles
    zpath = Path(zip_path)
    params = None
    if read_tiler_params_from_zip is not None:
        params = read_tiler_params_from_zip(zpath)

    if params is None:
        tile_size_px = 512
        tile_size_um = 0.5 * tile_size_px
    else:
        tile_size_px = int(params.get('tile_size_px', params.get('tile_size', 512)))
        tile_size_um = float(params.get('tile_size_um', params.get('tile_size_microns', 0.5 * tile_size_px)))

    slide_mpp = tile_size_um / tile_size_px

    # find annotation CSV
    zip_stem = zpath.stem
    roi_files = os.listdir(roi_dir)
    roi_fname_list = []
    if params is not None:
        slide_path = params.get('slide_path') or params.get('slide') or None
        if slide_path:
            slide_base = Path(slide_path).stem
            roi_fname_list = find_substring_in_list(roi_files, slide_base)
    if not roi_fname_list:
        roi_fname_list = find_substring_in_list(roi_files, zip_stem)
    if not roi_fname_list:
        prefix = zip_stem[:16]
        roi_fname_list = [f for f in roi_files if f.startswith(prefix)]
    if not roi_fname_list:
        print(f"No annotation CSV found for {zip_stem}, skipping zip")
        return False
    if len(roi_fname_list) > 1:
        raise RuntimeError(f"Multiple annotation CSV candidates for {zip_stem}: {roi_fname_list}")

    roi_path = os.path.join(roi_dir, roi_fname_list[0])
    # skip empty CSVs
    try:
        if os.path.getsize(roi_path) == 0:
            print(f"Annotation CSV is empty (size 0): {roi_path}; skipping")
            return False
    except OSError:
        pass

    try:
        ann, _ = read_annotations(roi_path)
    except Exception as e:
        print(f"Failed to read annotations for {zip_stem}: {e}")
        return False

    scaled_annPolys = scale(ann, xfact=slide_mpp/target_mpp, yfact=slide_mpp/target_mpp, origin=(0,0))

    # list tiles
    tile_files = []
    if list_tile_files_in_zip is not None:
        tile_files = list_tile_files_in_zip(zpath, tile_ext=tile_ext)
    else:
        with zipfile.ZipFile(zpath, 'r') as zf:
            tile_files = [n for n in zf.namelist() if n.lower().endswith(tile_ext)]

    polys, origins = build_polygons_for_tiles(tile_files, tile_size_px)

    # build rtree index (kept for potential future optimizations)
    index = rtree.index.Index()
    num_polys = 0
    for idx, poly in enumerate(polys):
        if poly is None:
            continue
        num_polys += 1
        index.insert(idx, poly.bounds)

    # Debug/logging: report counts
    print(f"Zip {zpath.name}: total files={len(tile_files)}, polygons_built={num_polys}")
    try:
        print(f"Scaled annotation type: {type(scaled_annPolys)}, bounds: {getattr(scaled_annPolys, 'bounds', None)}")
    except Exception:
        pass

    # select tiles
    selected = []
    for name, poly in zip(tile_files, polys):
        if poly is None:
            continue
        if not poly.bounds or not scaled_annPolys.bounds:
            continue
        if not sg.box(*poly.bounds).intersects(sg.box(*scaled_annPolys.bounds)):
            continue
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

    print(f"Zip {zpath.name}: selected {len(selected)} tiles")

    # generate plot if requested (helps debug selection)
    if generate_plots:
        out_plots_dir = Path(save_dir) / "plots_zip"
        out_plots_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        try:
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
            ax.invert_yaxis()
            plt.savefig(out_plots_dir / f"{zip_stem}_selection.png")
        except Exception as e:
            print(f"Failed to generate plot for {zip_stem}: {e}")
        finally:
            plt.close()

    # if no selected tiles, skip creating tar
    if not selected:
        print(f"No tiles selected for {zip_stem}; skipping extraction")
        return True

    # extract selected tiles to a temp dir and create a tar.gz
    tmpdir = tempfile.mkdtemp(prefix=f"{zip_stem}_", dir=save_dir)
    try:
        with zipfile.ZipFile(zpath, 'r') as zf:
            for name in selected:
                try:
                    data = zf.read(name)
                except KeyError:
                    # sometimes entries have leading ./ or different path; try basename
                    try:
                        data = zf.read(os.path.basename(name))
                    except Exception as e:
                        print(f"Failed to read {name} from {zpath.name}: {e}")
                        continue
                outpath = os.path.join(tmpdir, os.path.basename(name))
                with open(outpath, 'wb') as fh:
                    fh.write(data)

        tar_path = os.path.join(save_dir, f"{zip_stem}.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            for fname in os.listdir(tmpdir):
                tar.add(os.path.join(tmpdir, fname), arcname=fname)

        print(f"Created tar file: {tar_path} with {len(os.listdir(tmpdir))} tiles")
    finally:
        # cleanup temp dir
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

    return True


def main():
    parser = argparse.ArgumentParser(description="Filter tiles inside ROIs from WSIs or tile zip files")
    parser.add_argument("--zip-input", action="store_true", help="Process zip files instead of WSIs")
    parser.add_argument("--zip-dir", help="Folder with zip files (one per slide)")
    parser.add_argument("--slide-dir", help="Folder with whole-slide images (for WSI mode)")
    parser.add_argument("--img-dir", help="Folder with extracted tile folders (for WSI mode)")
    parser.add_argument("--roi-dir", default=f"{ws_path}/data/TCGA-CRC/csv_annotations", help="Folder with annotation CSVs")
    parser.add_argument("--out-dir", default=f"{ws_path}/data/TCGA-CRC/tiles_512x512_05mpp-only-tum", help="Output folder for filtered tiles")
    parser.add_argument("--target-mpp", type=float, default=256 / 512)
    parser.add_argument("--generate-plots", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--tile-ext", default=".png")
    args = parser.parse_args()

    target_mpp = args.target_mpp
    num_workers = args.num_workers
    generate_plots = args.generate_plots
    roi_dir = args.roi_dir
    save_dir = args.out_dir
    os.makedirs(save_dir, exist_ok=True)

    if args.zip_input:
        if not args.zip_dir:
            raise SystemExit("--zip-dir required when --zip-input is set")
        zips = sorted(Path(args.zip_dir).glob('*.zip'))
        if not zips:
            print('No zip files found in', args.zip_dir)
            return

        success = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_zip, str(z), save_dir, roi_dir, target_mpp, generate_plots, args.tile_ext): z for z in zips}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing zips..."):
                z = futures[future]
                try:
                    if future.result():
                        success += 1
                except Exception as exc:
                    print(f'Zip {z.name} generated an exception: {exc}')

        print(f'Correctly processed {success} zip slides.')
    else:
        if not args.slide_dir or not args.img_dir:
            raise SystemExit("--slide-dir and --img-dir required for WSI mode")
        slide_dir = args.slide_dir
        img_dir = args.img_dir
        slides_fnames = os.listdir(slide_dir)

        correctly_extracted_slide_nr = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    process_slide,
                    fname,
                    slide_dir,
                    img_dir,
                    save_dir,
                    roi_dir,
                    target_mpp,
                    generate_plots,
                ): fname
                for fname in slides_fnames
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