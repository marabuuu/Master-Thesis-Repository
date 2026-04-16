from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import yaml


VALID_SUBTYPES = {"Basal", "LumA"}


def canonical_patient_id(name: str) -> str:
    stem = Path(str(name)).stem.upper()
    for sep in ("_", "."):
        stem = stem.replace(sep, "-")
    while "--" in stem:
        stem = stem.replace("--", "-")
    parts = stem.split("-")
    if len(parts) >= 3 and parts[0].startswith("TCGA"):
        return "-".join(parts[:3])
    return stem


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def infer_label_csv_and_columns(cfg: dict, label_csv_path: Optional[str] = None) -> Tuple[str, str, str]:
    encoding_cfg = cfg.get("encoding", {}) if isinstance(cfg.get("encoding"), dict) else {}
    joint_cfg = cfg.get("joint_training", {}) if isinstance(cfg.get("joint_training"), dict) else {}
    reconstruction_cfg = cfg.get("reconstruction", {}) if isinstance(cfg.get("reconstruction"), dict) else {}

    csv_path = (
        label_csv_path
        or encoding_cfg.get("csv_path")
        or joint_cfg.get("csv_path")
        or reconstruction_cfg.get("csv_path")
    )
    if not csv_path:
        raise ValueError("Could not infer label CSV path. Pass --label-csv-path explicitly.")

    patient_col = (
        encoding_cfg.get("patient_col")
        or joint_cfg.get("patient_col")
        or reconstruction_cfg.get("patient_col")
        or "Patient_ID"
    )
    subtype_col = encoding_cfg.get("subtype_col") or "Majority_Subtype_mRNA"
    return str(csv_path), str(patient_col), str(subtype_col)


def load_patient_splits(path: str | Path) -> Dict[str, List[str]]:
    with open(path) as f:
        payload = json.load(f)

    result: Dict[str, List[str]] = {}
    for split in ("train", "val", "test"):
        if split not in payload:
            continue
        split_obj = payload[split]
        if isinstance(split_obj, dict):
            pts = split_obj.get("patients", [])
        else:
            pts = split_obj
        result[split] = [canonical_patient_id(p) for p in pts]
    return result


def load_subtype_table(csv_path: str | Path, patient_col: str, subtype_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if patient_col not in df.columns:
        raise KeyError(f"Patient column '{patient_col}' not found in {csv_path}")
    if subtype_col not in df.columns:
        raise KeyError(f"Subtype column '{subtype_col}' not found in {csv_path}")

    out = df[[patient_col, subtype_col]].copy()
    out[patient_col] = out[patient_col].astype(str).map(canonical_patient_id)
    out[subtype_col] = out[subtype_col].astype(str).str.strip()
    out = out[out[subtype_col].isin(VALID_SUBTYPES)].dropna().drop_duplicates(subset=[patient_col], keep="first")
    return out.rename(columns={patient_col: "patient_id", subtype_col: "subtype"})


def _iter_datasets(group: h5py.Group, prefix: str = "") -> List[Tuple[str, h5py.Dataset]]:
    out: List[Tuple[str, h5py.Dataset]] = []
    for key, value in group.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, h5py.Dataset):
            out.append((path, value))
        elif isinstance(value, h5py.Group):
            out.extend(_iter_datasets(value, path))
    return out


def _score_feature_candidate(name: str, arr: np.ndarray) -> float:
    lname = name.lower()
    score = 0.0
    if any(k in lname for k in ("feature", "feat", "embed", "embedding", "virchow")):
        score += 5.0
    if arr.ndim == 2:
        score += 3.0
    if arr.ndim == 2 and arr.shape[1] >= 32:
        score += 2.0
    if np.issubdtype(arr.dtype, np.number):
        score += 1.0
    return score


def _score_coord_candidate(name: str, arr: np.ndarray) -> float:
    lname = name.lower()
    score = 0.0
    if any(k in lname for k in ("coord", "coords", "xy", "position")):
        score += 5.0
    if arr.ndim == 2 and arr.shape[1] >= 2:
        score += 2.0
    if np.issubdtype(arr.dtype, np.number):
        score += 1.0
    return score


def _pick_feature_array(h5f: h5py.File) -> np.ndarray:
    datasets = _iter_datasets(h5f)
    candidates: List[Tuple[float, str, np.ndarray]] = []
    for name, ds in datasets:
        try:
            arr = ds[()]
        except Exception:
            continue
        if not isinstance(arr, np.ndarray):
            continue
        score = _score_feature_candidate(name, arr)
        if score <= 0:
            continue
        candidates.append((score, name, arr))

    if not candidates:
        raise ValueError("No candidate feature dataset found in H5 file")

    candidates.sort(key=lambda x: (x[0], x[2].ndim, x[2].shape[1] if x[2].ndim == 2 else -1), reverse=True)
    best = candidates[0][2]
    if best.ndim != 2:
        best = np.atleast_2d(best)
    return best.astype(np.float32)


def _pick_coord_array(h5f: h5py.File, n_tiles: int) -> Optional[np.ndarray]:
    datasets = _iter_datasets(h5f)
    candidates: List[Tuple[float, np.ndarray]] = []
    for name, ds in datasets:
        try:
            arr = ds[()]
        except Exception:
            continue
        if not isinstance(arr, np.ndarray):
            continue
        score = _score_coord_candidate(name, arr)
        if score <= 0:
            continue
        if arr.ndim == 2 and arr.shape[0] == n_tiles and arr.shape[1] >= 2:
            candidates.append((score + 5.0, arr))
        elif arr.ndim == 2 and arr.shape[1] >= 2:
            candidates.append((score, arr))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    arr = candidates[0][1]
    if arr.shape[0] != n_tiles:
        return None
    return arr[:, :2]


def load_patient_h5_features(h5_path: str | Path, patient_id: Optional[str] = None) -> Tuple[str, np.ndarray, Optional[np.ndarray]]:
    h5_path = Path(h5_path)
    pid = canonical_patient_id(patient_id or h5_path.stem)
    with h5py.File(h5_path, "r") as h5f:
        feats = _pick_feature_array(h5f)
        coords = _pick_coord_array(h5f, feats.shape[0])
    return pid, feats, coords


def build_tile_feature_table(
    features_dir: str | Path,
    subtype_df: pd.DataFrame,
    splits: Dict[str, List[str]],
    max_tiles_per_patient: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    features_dir = Path(features_dir)
    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")

    all_h5 = sorted(list(features_dir.glob("*.h5")) + list(features_dir.glob("*.hdf5")))
    if not all_h5:
        raise FileNotFoundError(f"No .h5/.hdf5 files found in: {features_dir}")

    split_of_patient: Dict[str, str] = {}
    for split, pts in splits.items():
        for p in pts:
            split_of_patient[canonical_patient_id(p)] = split

    subtype_map = dict(zip(subtype_df["patient_id"], subtype_df["subtype"]))

    rows: List[dict] = []
    rng = np.random.RandomState(seed)

    for h5_path in all_h5:
        pid, feats, coords = load_patient_h5_features(h5_path)
        if pid not in split_of_patient:
            continue
        if pid not in subtype_map:
            continue

        n = feats.shape[0]
        indices = np.arange(n)
        if max_tiles_per_patient is not None and n > max_tiles_per_patient:
            indices = rng.choice(indices, size=max_tiles_per_patient, replace=False)

        for i in indices:
            row = {
                "patient_id": pid,
                "split": split_of_patient[pid],
                "subtype": subtype_map[pid],
                "tile_index": int(i),
                "feature": feats[i],
            }
            if coords is not None:
                row["x"] = float(coords[i, 0])
                row["y"] = float(coords[i, 1])
            rows.append(row)

    if not rows:
        raise ValueError("No overlapping samples found after applying splits + subtype filters")

    df = pd.DataFrame(rows)
    return df


def encode_labels(labels: Iterable[str]) -> np.ndarray:
    y = np.array([1 if str(lbl) == "Basal" else 0 for lbl in labels], dtype=np.int64)
    return y


def split_to_arrays(df: pd.DataFrame, split_name: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    sub = df[df["split"] == split_name].reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError(f"No rows found for split '{split_name}'")
    x = np.stack(sub["feature"].to_list()).astype(np.float32)
    y = encode_labels(sub["subtype"].tolist())
    return x, y, sub
