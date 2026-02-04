import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import re
import zipfile
import json
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import shapely.geometry as sg
from shapely.validation import make_valid
import traceback
import logging
import h5py
from typing import List, Optional


def preprocess_log1p_minmax(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p then per-gene min-max scaling to [0,1].

    Useful when decoder uses sigmoid and reconstruction target is expected in [0,1].
    """
    arr = np.log1p(df.values.astype(float))
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(arr)
    return pd.DataFrame(scaled, index=df.index, columns=df.columns)


def preprocess_log1p_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p then z-score per gene (zero mean, unit var).

    Useful when decoder is identity and reconstruction uses MSE.
    """
    arr = np.log1p(df.values.astype(float))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(arr)
    return pd.DataFrame(scaled, index=df.index, columns=df.columns)


def inspect_variance(df: pd.DataFrame) -> dict:
    """Return simple variance diagnostics for the dataframe.
    """
    gene_var = df.var(axis=0)
    sample_var = df.var(axis=1)
    return {
        'n_genes': df.shape[1],
        'n_samples': df.shape[0],
        'gene_var_summary': gene_var.describe().to_dict(),
        'n_zero_var_genes': int((gene_var == 0).sum()),
        'sample_var_summary': sample_var.describe().to_dict()
    }

def create_polygons(df):
    polygons = []
    for _, row in df.iterrows():
        x, y = row['coord_x'], row['coord_y']
        polygon = Polygon([(x, y), (x + 512, y), (x + 512, y + 512), (x, y + 512)])
        polygons.append(polygon)
    return polygons
    

def create_dataframe(arrays):
    augmented, coords, features, zoom = arrays
    df_coords = pd.DataFrame(coords, columns=['coord_x', 'coord_y'])
    df_features = pd.DataFrame(features, columns=[f'feature_{i+1}' for i in range(features.shape[1])])

    # If 'augmented' data exists, create a DataFrame for it and concatenate it with the others
    if augmented is not None:
        df_augmented = pd.DataFrame(augmented, columns=['augmented'])
        df = pd.concat([df_coords, df_augmented, df_features], axis=1)
    else:
        df = pd.concat([df_coords, df_features], axis=1)

    if zoom is not None:
        df_zoom = pd.DataFrame(zoom, columns=['zoom'])
        df = pd.concat([df_coords, df_zoom, df_features], axis=1)
    else:
        df = pd.concat([df_coords, df_features], axis=1)

    return df


def read_h5_file(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        # print(f.keys())  # print all keys

        coords = f['coords'][:]
        feats = f['feats'][:]

        # check if "augmented" data exists, and if so, read it
        if 'augmented' in f.keys():
            augmented = f['augmented'][:]
        else:
            augmented = None

        # check if "zoom" data exists, and if so, read it
        if 'zoom' in f.keys():
            zoom = f['zoom'][:]
        else:
            zoom = None

        return augmented, coords, feats, zoom


def write_to_h5(df, h5_file_path):
    with h5py.File(h5_file_path, 'w') as f:
        # Create datasets for 'coords', 'feats' and possibly 'augmented'
        coords = f.create_dataset('coords', data=df[['coord_x', 'coord_y']].values)
        feats = f.create_dataset('feats', data=df[[col for col in df.columns if 'feature_' in col]].values)

        # If 'augmented' column exists in the DataFrame, create a dataset for it
        if 'augmented' in df.columns:
            augmented = f.create_dataset('augmented', data=df['augmented'].values)
        

def read_annotations(annon_path):
    polygons = []
    rectcoords_list = []

    with open(annon_path, 'r') as f:
        lines = f.readlines()

    headers = [h.strip() for h in lines[0].split(',')]  # Assuming CSV is comma separated
    if 'X_base' not in headers or 'Y_base' not in headers:
        raise IndexError('Unable to find "X_base" and "Y_base" columns in CSV file.')

    index_x = headers.index('X_base')
    index_y = headers.index('Y_base')

    roi_coords = []
    for line in lines[1:]:  # Skip the header
        elements = line.split(',')
        if elements[index_x] == 'X_base' or elements[index_y] == 'Y_base':
            # If we encounter a new 'X_base' or 'Y_base', save the previous polygon (if exists)
            if roi_coords and len(set(roi_coords)) >= 3:  # Ensure we have at least 3 unique points
                polygons.append(sg.Polygon(roi_coords))
                rectcoords_list.append([
                    [max(coord[0] for coord in roi_coords), min(coord[0] for coord in roi_coords)],
                    [min(coord[1] for coord in roi_coords), max(coord[1] for coord in roi_coords)]
                ])
            # Start a new polygon
            roi_coords = []
            continue
        else:
            roi_coords.append((float(elements[index_x]), float(elements[index_y])))  # Convert coordinates to numeric

    # Save the last polygon
    if roi_coords and len(set(roi_coords)) >= 3:
        polygons.append(sg.Polygon(roi_coords))
        rectcoords_list.append([
            [max(coord[0] for coord in roi_coords), min(coord[0] for coord in roi_coords)],
            [min(coord[1] for coord in roi_coords), max(coord[1] for coord in roi_coords)]
        ])

    for polygon in polygons:
        # remove invalid polygons and apply buffer
        if isinstance(polygon, sg.Polygon):
            if contains_nan_or_inf(polygon):
                polygon = fix_invalid_polygon(polygon)
                print("Invalid polygon was fixed.")
            # try to catch possible topology exceptions, e.g. due to polygon intersecting with itself
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
                print("Invalid polygon was fixed.")
            if not polygon.is_valid:
                polygon = make_valid(polygon)
                print("Invalid polygon was fixed using make_valid.")

    # Combine the individual polygons into a single MultiPolygon object
    annPolys = sg.MultiPolygon(polygons)

    return annPolys, np.int32(rectcoords_list)

def find_substring_in_list(strings, substring):
    return [s for s in strings if substring in s]


def extract_coordinates(filename):
    """Extract coordinates from filename. If coordinates were written as X,Y """
    match = re.search(r'\((\d+),(\d+)\)', filename)    # adjust here to the correct pattern!
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return None

def create_polygons_from_filenames(filenames):
    """Extract coordinates from filename. Assumes that coordinates were written as X,Y """
    polygons = []
    for filename in filenames:
        coords = extract_coordinates(filename)
        if coords:
            x, y = coords  # adjust here depending how tile fnames were written!
            polygon = Polygon([(x, y), (x + 512, y), (x + 512, y + 512), (x, y + 512)])  # (x, y) top left
            polygons.append(polygon)
    return polygons

def contains_nan_or_inf(polygon):
    """Add checks for NaN/Inf values in polygon"""
    for x, y in polygon.exterior.coords:
        if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
            return True
    return False


def fix_invalid_polygon(polygon):
    new_coords = [(x, y) for x, y in polygon.exterior.coords if
                  not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y))]
    return sg.Polygon(new_coords)


def read_tiler_params(zip_path: str) -> dict:
    """Read tiler_params.json from a ZIP archive and return parsed dict.

    Returns empty dict on failure.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # find tiler_params.json (case-insensitive)
            candidates = [n for n in z.namelist() if n.lower().endswith('tiler_params.json')]
            if not candidates:
                logging.warning(f'No tiler_params.json found in {zip_path}')
                return {}
            with z.open(candidates[0]) as f:
                data = json.load(f)
                return data
    except Exception as e:
        logging.warning(f'Failed to read tiler_params from {zip_path}: {e}')
        return {}


def get_mpp_from_tiler_params(params: dict) -> Optional[float]:
    """Compute slide MPP (microns per pixel) from tiler params.

    Uses keys 'tile_size_um' and 'tile_size_px' when available.
    Returns None if not computable.
    """
    try:
        # common keys mentioned: 'tile_size_um', 'tile_size_px'
        if 'tile_size_um' in params and 'tile_size_px' in params:
            tile_size_um = float(params['tile_size_um'])
            tile_size_px = float(params['tile_size_px'])
            if tile_size_px == 0:
                return None
            return tile_size_um / tile_size_px

        # sometimes keys may be nested or named differently; try some fallbacks
        for k in ['tile_size', 'tile_px']:
            if k in params:
                # best-effort parse
                vals = params.get(k)
                try:
                    return float(vals[0]) / float(vals[1])
                except Exception:
                    continue
    except Exception:
        logging.exception('Error computing MPP from tiler params')
    return None


def parse_tile_coords(filename: str) -> Optional[tuple]:
    """Parse coordinates from filenames like tile_(14602.488, 1537.104).png.

    Returns (x, y) as floats or None if no match.
    """
    m = re.search(r'\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)', filename)
    if not m:
        return None
    try:
        x = float(m.group(1))
        y = float(m.group(2))
        return x, y
    except Exception:
        return None


def create_polygons_from_filenames(
    filenames: List[str],
    tile_size_px: float = 512.0,
    slide_mpp: Optional[float] = None,
    target_mpp: Optional[float] = None,
    coords_in_microns: Optional[bool] = None,
    coord_is_center: bool = False,
) -> List[Polygon]:
    """Create shapely Polygons for each tile filename using parsed coordinates.

    Coordinates and tile size are converted into the same "target pixel" space
    used by scaled annotations. If `slide_mpp` and `target_mpp` are provided,
    the tile size and coordinates will be transformed appropriately.

    - If `coords_in_microns` is True, parsed coords are interpreted as microns
      and converted to target pixels via `x_target = x_um / target_mpp`.
    - If `coords_in_microns` is False, parsed coords are interpreted as original
      slide pixels and converted to target pixels via
      `x_target = x_px * slide_mpp / target_mpp`.
    - If `coords_in_microns` is None, the function will attempt a simple
      heuristic to detect microns (large values or presence of decimals).

    When `slide_mpp` or `target_mpp` are not provided the function falls back
    to creating polygons directly from parsed coords and `tile_size_px`.
    """
    polygons = []

    # compute tile size in target pixels if possible
    tile_size_target = tile_size_px
    if slide_mpp is not None and target_mpp is not None:
        try:
            tile_size_target = float(tile_size_px) * (slide_mpp / target_mpp)
        except Exception:
            tile_size_target = tile_size_px

    for filename in filenames:
        coords = parse_tile_coords(filename)
        if not coords:
            continue
        x, y = coords

        # decide whether coords are microns or pixels
        use_microns = coords_in_microns
        if use_microns is None:
            # fallback heuristic: treat as microns when values are large or fractional
            if (abs(x) > 10000 or abs(y) > 10000) or (not float(x).is_integer() or not float(y).is_integer()):
                use_microns = True
            else:
                use_microns = False

        if use_microns:
            if target_mpp is None:
                x_t = float(x)
                y_t = float(y)
            else:
                # coordinates provided in microns -> convert to target pixels
                x_t = float(x) / float(target_mpp)
                y_t = float(y) / float(target_mpp)
        else:
            if slide_mpp is not None and target_mpp is not None:
                # coordinates provided in slide pixels -> convert to target pixels
                x_t = float(x) * (slide_mpp / target_mpp)
                y_t = float(y) * (slide_mpp / target_mpp)
            else:
                x_t = float(x)
                y_t = float(y)

        # if filename coords represent tile center, shift to top-left for polygon
        if coord_is_center:
            x0 = x_t - (tile_size_target / 2.0)
            y0 = y_t - (tile_size_target / 2.0)
        else:
            x0 = x_t
            y0 = y_t

        polygon = Polygon([(x0, y0), (x0 + tile_size_target, y0), (x0 + tile_size_target, y0 + tile_size_target), (x0, y0 + tile_size_target)])
        polygons.append(polygon)

    return polygons


def read_zip_tile_list(zip_path: str) -> List[str]:
    """Return list of tile filenames inside the ZIP (no extraction).

    Filters names that contain coordinate pattern.
    """
    names = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for n in z.namelist():
                if parse_tile_coords(os.path.basename(n)):
                    names.append(os.path.basename(n))
    except Exception:
        logging.exception(f'Failed to list ZIP contents for {zip_path}')
    return names


def extract_selected_from_zip(zip_path: str, filenames: List[str], out_dir: str) -> None:
    """Extract only the specified filenames from ZIP into out_dir.

    Filenames should be basenames; this will match any member ending with that basename.
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            members = z.namelist()
            # for each requested basename, find a member that endswith it
            for basename in filenames:
                matches = [m for m in members if m.endswith(basename)]
                if not matches:
                    logging.warning(f'{basename} not found in {zip_path}')
                    continue
                # extract first matching member
                z.extract(matches[0], path=out_dir)
    except Exception:
        logging.exception(f'Failed to extract selected tiles from {zip_path}')


def match_zip_to_csv(zip_name: str, csv_list: List[str], max_prefix: int = 20) -> Optional[str]:
    """Match zip filename to csv filename using prefix heuristics.

    Returns matched csv filename or None.
    """
    base = os.path.basename(zip_name)
    base_noext = os.path.splitext(base)[0]
    # 1) Try to find CSVs that start with progressively longer prefixes of the zip basename
    max_L = min(len(base_noext), max_prefix if max_prefix is not None else len(base_noext))
    for L in range(max_L, 7, -1):
        pref = base_noext[:L]
        candidates = [c for c in csv_list if os.path.basename(c).startswith(pref)]
        if candidates:
            return candidates[0]

    # 2) If that fails, try substring match using decreasing prefix lengths
    for L in range(max_L, 7, -1):
        pref = base_noext[:L]
        candidates = [c for c in csv_list if pref in os.path.basename(c)]
        if candidates:
            return candidates[0]

    # 3) As a last resort, try matching using the 'core' before the first dot (e.g. 'TCGA-...-DX1')
    parts = base_noext.split('.')
    if len(parts) >= 2:
        core2 = parts[0] + '.' + parts[1]
        candidates = [c for c in csv_list if os.path.basename(c).startswith(core2)]
        if candidates:
            return candidates[0]

    # fallback to single-part core match
    core = parts[0]
    candidates = [c for c in csv_list if os.path.basename(c).startswith(core) or core in os.path.basename(c)]
    if candidates:
        return candidates[0]

    # try substring containment
    for L in range(max_prefix, 7, -1):
        pref = base_noext[:L]
        candidates = [c for c in csv_list if pref in os.path.basename(c)]
        if candidates:
            return candidates[0]

    # try any csv that is contained in zip basename
    candidates = [c for c in csv_list if os.path.splitext(os.path.basename(c))[0] in base_noext]
    if candidates:
        return candidates[0]

    return None