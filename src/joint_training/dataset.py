"""
Dataset for joint genomic VAE + diffusion training.

Pairs raw gene expression vectors (from CSV) with tile images (from ZIP archives),
matching on patient ID (TCGA-XX-XXXX prefix).

Supports patient-level train/val/test splitting: pass ``patient_ids`` to restrict
the dataset to a specific set of patients.
"""

from __future__ import annotations

import json
import random
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def canonical_patient_id(name: str) -> str:
    """Extract TCGA-XX-XXXX from various filename formats."""
    stem = Path(name).stem.upper()
    for sep in ("_", "."):
        stem = stem.replace(sep, "-")
    while "--" in stem:
        stem = stem.replace("--", "-")
    parts = stem.split("-")
    if len(parts) >= 3 and parts[0].startswith("TCGA"):
        return "-".join(parts[:3])
    return stem


# ──────────────────────────────────────────────────────────────────────
#  Patient-level splitting
# ──────────────────────────────────────────────────────────────────────

def patient_split(
    patient_ids: list[str],
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split patient IDs into train / val / test sets.

    Returns a dict with keys "train", "val", "test", each mapping to a
    sorted list of patient IDs.
    """
    rng = np.random.RandomState(seed)
    ids = sorted(patient_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_test = max(1, int(n * test_fraction)) if test_fraction > 0 else 0
    n_val = max(1, int(n * val_fraction)) if val_fraction > 0 else 0
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"Not enough patients ({n}) for the requested split fractions "
            f"(val={val_fraction}, test={test_fraction})"
        )

    splits: dict[str, list[str]] = {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train : n_train + n_val]),
        "test": sorted(ids[n_train + n_val :]),
    }
    return splits


def save_split(splits: dict[str, list[str]], out_dir: str) -> str:
    """Persist the patient split to ``<out_dir>/patient_splits.json``."""
    out_path = Path(out_dir) / "patient_splits.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        k: {"patients": v, "n_patients": len(v)}
        for k, v in splits.items()
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return str(out_path)


def load_split(path: str) -> dict[str, list[str]]:
    """Load a previously saved split JSON."""
    with open(path) as f:
        payload = json.load(f)
    return {k: v["patients"] for k, v in payload.items()}


class GenomicTileDataset(Dataset):
    """
    Pairs raw gene expression vectors with tile images from ZIP archives.

    Each sample returns:
        img:        (3, H, W) tensor in [-1, 1]
        genomic:    (N_genes,) raw gene expression vector (float32)
        patient_id: str

    Parameters
    ----------
    csv_path : str
        Path to gene expression CSV. Columns = genes, last col or index = Patient_ID.
    tiles_zip_dir : str
        Directory containing per-sample ZIP archives of tile PNGs.
    img_size : int
        Tile resize target (square).
    patient_col : str
        Column name for patient IDs in the CSV.
    label_col : str or None
        If present, drop this column (e.g. subtype labels mixed into gene data).
    gene_list : list[str] or None
        If provided, restrict to these genes only.
    max_tiles_per_patient : int or None
        Cap tiles indexed per patient ZIP (None = use all).
    patient_ids : list[str] or None
        If provided, restrict to *only* these patients. Used to enforce
        patient-level splits (train / val / test).
    """

    def __init__(
        self,
        csv_path: str,
        tiles_zip_dir: str,
        img_size: int = 512,
        patient_col: str = "Patient_ID",
        label_col: Optional[str] = None,
        gene_list: Optional[list[str]] = None,
        max_tiles_per_patient: Optional[int] = None,
        patient_ids: Optional[list[str]] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.max_tiles_per_patient = max_tiles_per_patient

        # ── Load & preprocess gene expression ──────────────────────────
        df = pd.read_csv(csv_path)
        if patient_col in df.columns:
            df = df.set_index(patient_col)

        # Drop non-numeric label column if present
        if label_col and label_col in df.columns:
            df = df.drop(columns=[label_col])

        # Keep only numeric columns
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=1, how="all")  # drop all-NaN columns
        df = df.fillna(0.0)

        # Optional gene subsetting
        if gene_list is not None:
            available = [g for g in gene_list if g in df.columns]
            if available:
                df = df[available]

        # Preprocessing: log1p + z-score
        values = df.values.astype(np.float64)
        # Only apply log1p if data looks like raw counts (not already normalized)
        if np.median(values[values > 0]) > 2.0:
            values = np.log1p(values)
        means = values.mean(axis=0)
        stds = values.std(axis=0)
        stds[stds < 1e-8] = 1.0  # avoid division by zero
        values = (values - means) / stds

        self.gene_names = list(df.columns)
        self.n_genes = len(self.gene_names)
        self._genomic = {
            str(pid): torch.from_numpy(values[i].astype(np.float32))
            for i, pid in enumerate(df.index)
        }
        # Store normalization stats for reproducibility
        self._norm_means = means
        self._norm_stds = stds

        print(f"[GenomicTileDataset] Loaded {len(self._genomic)} patients, "
              f"{self.n_genes} genes")

        # ── Index tile ZIPs ────────────────────────────────────────────
        tiles_dir = Path(tiles_zip_dir).expanduser()
        zip_map: dict[str, list[Path]] = {}  # patient_id -> list of zip paths
        for zp in sorted(tiles_dir.glob("*.zip")):
            pid = canonical_patient_id(zp.name)
            zip_map.setdefault(pid, []).append(zp)

        # Match patients present in both genomic and tile data
        common = set(self._genomic) & set(zip_map)

        # Further restrict to allowed patient set (for train/val/test splits)
        if patient_ids is not None:
            allowed = set(patient_ids)
            common = common & allowed

        common_sorted = sorted(common)
        if not common_sorted:
            raise RuntimeError(
                f"No matching patients between CSV ({len(self._genomic)} patients) "
                f"and tile ZIPs ({len(zip_map)} zips) in {tiles_zip_dir}"
                + (f" (filter: {len(patient_ids)} patient_ids)"
                   if patient_ids is not None else "")
            )
        self.patient_ids = common_sorted
        print(f"[GenomicTileDataset] {len(common_sorted)} patients matched "
              f"(of {len(self._genomic)} genomic, {len(zip_map)} zip)")

        # Build flat list: (patient_id, zip_path, tile_name)
        self.samples: list[tuple[str, Path, str]] = []
        for pid in common_sorted:
            for zpath in zip_map[pid]:
                try:
                    with zipfile.ZipFile(zpath, "r") as zf:
                        names = [
                            n for n in zf.namelist()
                            if n.lower().endswith((".png", ".jpg", ".jpeg"))
                            and not n.startswith("__MACOSX")
                        ]
                except (zipfile.BadZipFile, OSError) as e:
                    print(f"  [WARN] Skipping bad zip {zpath.name}: {e}")
                    continue

                if max_tiles_per_patient is not None:
                    names = names[:max_tiles_per_patient]

                for name in names:
                    self.samples.append((pid, zpath, name))

        print(f"[GenomicTileDataset] {len(self.samples)} tile-genomic pairs total")

        # ── Image transform ────────────────────────────────────────────
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),                 # [0, 1]
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # [-1, 1]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, zpath, tile_name = self.samples[idx]

        # Load tile from ZIP
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                data = zf.read(tile_name)
            img = Image.open(BytesIO(data)).convert("RGB")
        except Exception:
            # On corruption, return a random other sample
            return self[random.randint(0, len(self) - 1)]

        img = self.transform(img)
        genomic = self._genomic[pid]

        return {"img": img, "genomic": genomic, "patient_id": pid}
