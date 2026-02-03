import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import shapely.geometry as sg


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

# this function is taken from mopadi repository: https://github.com/KatherLab/mopadi/blob/main/src/mopadi/data_prep/utils.py
def read_annotations(annon_path):
    import csv
    from io import StringIO

    polygons = []
    rectcoords_list = []

    with open(annon_path, 'r', newline='') as f:
        text = f.read()

    if not text or not text.strip():
        raise ValueError(f"Annotation file is empty: {annon_path}")

    # detect delimiter
    try:
        dialect = csv.Sniffer().sniff(text.splitlines(True)[0])
        delim = dialect.delimiter
    except Exception:
        delim = ','

    reader = csv.reader(StringIO(text), delimiter=delim)
    rows = [r for r in reader if any(cell.strip() for cell in r)]
    if not rows:
        raise ValueError(f"No valid rows found in annotation file: {annon_path}")

    headers = [h.strip() for h in rows[0]]

    # Accept several common header name variants for X/Y
    x_candidates = ['X_base', 'X', 'x', 'X_coord', 'X_COORD', 'X0', 'x0', 'X_base_um']
    y_candidates = ['Y_base', 'Y', 'y', 'Y_coord', 'Y_COORD', 'Y0', 'y0', 'Y_base_um']

    index_x = None
    index_y = None
    for name in x_candidates:
        if name in headers:
            index_x = headers.index(name)
            break
    for name in y_candidates:
        if name in headers:
            index_y = headers.index(name)
            break

    # If we couldn't find named columns, try to auto-detect two numeric columns
    if index_x is None or index_y is None:
        # inspect first few data rows to find numeric columns
        numeric_counts = [0] * len(headers)
        sample_rows = rows[1: min(len(rows), 20)]
        for r in sample_rows:
            for i, cell in enumerate(r):
                try:
                    float(cell)
                    numeric_counts[i] += 1
                except Exception:
                    pass

        # pick the two columns with highest numeric counts (and at least some numeric entries)
        cand = [i for i, c in sorted(enumerate(numeric_counts), key=lambda x: -x[1]) if c > 0]
        if len(cand) >= 2:
            index_x, index_y = cand[0], cand[1]
        else:
            raise IndexError('Unable to find X/Y coordinate columns in CSV file.')

    roi_coords = []
    for elements in rows[1:]:  # Skip the header
        # guard against short rows
        if len(elements) <= max(index_x, index_y):
            continue
        vx = elements[index_x].strip()
        vy = elements[index_y].strip()
        # treat repeated header markers as polygon separators
        if vx == 'X_base' or vy == 'Y_base':
            if roi_coords and len(set(roi_coords)) >= 3:  # Ensure we have at least 3 unique points
                polygons.append(sg.Polygon(roi_coords))
                rectcoords_list.append([
                    [max(coord[0] for coord in roi_coords), min(coord[0] for coord in roi_coords)],
                    [min(coord[1] for coord in roi_coords), max(coord[1] for coord in roi_coords)]
                ])
            roi_coords = []
            continue

        try:
            x = float(vx)
            y = float(vy)
        except Exception:
            # skip malformed coordinate rows
            continue

        roi_coords.append((x, y))

    # Save the last polygon
    if roi_coords and len(set(roi_coords)) >= 3:
        polygons.append(sg.Polygon(roi_coords))
        rectcoords_list.append([
            [max(coord[0] for coord in roi_coords), min(coord[0] for coord in roi_coords)],
            [min(coord[1] for coord in roi_coords), max(coord[1] for coord in roi_coords)]
        ])

    # Fallback: if no polygons were built from explicit separators, attempt to
    # group consecutive numeric rows into contour segments and build polygons.
    if not polygons:
        try:
            pts = []
            for elements in rows[1:]:
                if len(elements) <= max(index_x, index_y):
                    continue
                try:
                    x = float(elements[index_x].strip())
                    y = float(elements[index_y].strip())
                    pts.append((x, y))
                except Exception:
                    continue

            if len(pts) >= 3:
                arr = np.array(pts)
                # distances between consecutive points
                diffs = arr[1:] - arr[:-1]
                d = np.sqrt((diffs ** 2).sum(axis=1))
                med = float(np.median(d)) if len(d) > 0 else 0.0
                p95 = float(np.percentile(d, 95)) if len(d) > 0 else 0.0
                # heuristic threshold: tuneable; guard minimum absolute
                threshold = max(med * 10.0, p95 * 3.0, 1000.0)
                # split indices where jump > threshold
                split_idx = list(np.where(d > threshold)[0])
                start = 0
                for sidx in split_idx:
                    seg = pts[start:sidx + 1]
                    start = sidx + 1
                    if len(seg) >= 3 and len(set(seg)) >= 3:
                        polygons.append(sg.Polygon(seg))
                        rectcoords_list.append([
                            [max(c[0] for c in seg), min(c[0] for c in seg)],
                            [min(c[1] for c in seg), max(c[1] for c in seg)]
                        ])
                # final segment
                seg = pts[start:]
                if len(seg) >= 3 and len(set(seg)) >= 3:
                    polygons.append(sg.Polygon(seg))
                    rectcoords_list.append([
                        [max(c[0] for c in seg), min(c[0] for c in seg)],
                        [min(c[1] for c in seg), max(c[1] for c in seg)]
                    ])
        except Exception:
            # non-fatal: leave polygons empty so caller can handle
            pass

    fixed_polygons = []

    def normalize_to_polygons(item):
        """Return a list of Polygon objects given various possible item types."""
        out = []
        if item is None:
            return out
        # already a Polygon
        if isinstance(item, sg.Polygon):
            out.append(item)
            return out
        # a MultiPolygon -> extend
        if isinstance(item, sg.MultiPolygon):
            out.extend([g for g in item.geoms if isinstance(g, sg.Polygon)])
            return out
        # geometry collections or other shapely geometries
        try:
            geom_type = getattr(item, 'geom_type', None)
            if geom_type == 'Polygon':
                out.append(sg.Polygon(item.exterior.coords))
                return out
            if geom_type == 'MultiPolygon':
                out.extend([sg.Polygon(g.exterior.coords) for g in item.geoms if getattr(g, 'geom_type', None) == 'Polygon'])
                return out
        except Exception:
            pass

        # if it's a raw coordinate sequence (list/tuple/ndarray)
        if isinstance(item, (list, tuple, np.ndarray)):
            # flatten if nested
            if len(item) == 0:
                return out
            # try construct polygon directly
            try:
                poly = sg.Polygon(item)
                if isinstance(poly, sg.Polygon):
                    out.append(poly)
                    return out
            except Exception:
                # try iterating subitems
                for sub in item:
                    out.extend(normalize_to_polygons(sub))
                return out

        return out

    for polygon in polygons:
        normalized = normalize_to_polygons(polygon)
        for p in normalized:
            # attempt to repair if needed
            if contains_nan_or_inf(p):
                try:
                    p = fix_invalid_polygon(p)
                    print("Invalid polygon was fixed.")
                except NameError:
                    p = p.buffer(0)

            if not getattr(p, 'is_valid', True):
                p = p.buffer(0)

            if not getattr(p, 'is_valid', True):
                try:
                    p = make_valid(p)
                    print("Invalid polygon was fixed using make_valid.")
                except NameError:
                    pass

            # now accept Polygon, MultiPolygon, and GeometryCollection by extracting polygon parts
            if isinstance(p, sg.Polygon):
                if p.is_valid and not contains_nan_or_inf(p):
                    fixed_polygons.append(p)
                continue

            if isinstance(p, sg.MultiPolygon) or getattr(p, 'geom_type', None) == 'MultiPolygon':
                for g in getattr(p, 'geoms', []):
                    if isinstance(g, sg.Polygon) and g.is_valid and not contains_nan_or_inf(g):
                        fixed_polygons.append(g)
                continue

            # GeometryCollection or other collection types: extract polygon parts
            if getattr(p, 'geom_type', None) == 'GeometryCollection' or hasattr(p, 'geoms'):
                for g in getattr(p, 'geoms', []):
                    if isinstance(g, sg.Polygon):
                        if not g.is_valid:
                            g = g.buffer(0)
                        if g.is_valid and not contains_nan_or_inf(g):
                            fixed_polygons.append(g)
                continue

    if not fixed_polygons:
        # Final fallback: attempt to construct a polygon from the convex hull
        # of all numeric points in the file. This helps when annotations are
        # unordered vertex lists that cannot be split into valid contours.
        try:
            pts_all = []
            for elements in rows[1:]:
                if len(elements) <= max(index_x, index_y):
                    continue
                try:
                    x = float(elements[index_x].strip())
                    y = float(elements[index_y].strip())
                    pts_all.append((x, y))
                except Exception:
                    continue

            if len(pts_all) >= 3:
                mp = sg.MultiPoint(pts_all)
                hull = mp.convex_hull
                if isinstance(hull, sg.Polygon) and hull.is_valid and hull.area > 0:
                    # buffer(0) to attempt to fix minor geometry issues
                    ph = hull.buffer(0)
                    if ph.is_valid and ph.area > 0:
                        fixed_polygons.append(ph)
                        print(f"Convex-hull fallback produced polygon for: {annon_path}")
        except Exception:
            pass

    if not fixed_polygons:
        raise ValueError(f"No valid polygons parsed from annotation file: {annon_path}")

    # Combine the individual polygons into a single MultiPolygon object
    try:
        annPolys = sg.MultiPolygon(fixed_polygons)
    except Exception as e:
        raise ValueError(f"Failed to construct MultiPolygon from parsed polygons: {e}")

    return annPolys, np.int32(rectcoords_list)

# this function is taken from mopadi repository: https://github.com/KatherLab/mopadi/blob/main/src/mopadi/data_prep/utils.py
def find_substring_in_list(strings, substring):
    return [s for s in strings if substring in s]

def contains_nan_or_inf(polygon):
    """Add checks for NaN/Inf values in polygon"""
    for x, y in polygon.exterior.coords:
        if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
            return True
    return False