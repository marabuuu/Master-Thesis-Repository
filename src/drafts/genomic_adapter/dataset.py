"""
ZipTilesWithGenomicFeatures — dataset pairing ZIP-archived tiles with genomic vectors.

Moved here from src/drafts/mopadi_genomic/dataset.py.  The drafts copy is kept
for backwards compatibility with older checkpoints but this is the authoritative
version used by GDALitModel.
"""

from __future__ import annotations

import json
import logging
import os
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import h5py
import numpy as np
import torch

from mopadi.dataset import DefaultTilesDataset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patient-ID helpers
# ---------------------------------------------------------------------------

def patient_id_from_tile_path(path: str) -> str:
    """Extract the 3-token TCGA patient barcode from a tile path.

    Works for both ZIP-internal paths (``/path/zip.zip:tile.jpg``) and plain
    filesystem paths (``/path/TCGA-XX-XXXX.../tile.jpg``).

    Examples
    --------
    >>> patient_id_from_tile_path("/data/TCGA-3C-AALI-01Z-00-DX1.UUID.zip:tile.png")
    'TCGA-3C-AALI'
    >>> patient_id_from_tile_path("/data/TCGA-3C-AALI-01Z-00-DX1/tile.png")
    'TCGA-3C-AALI'
    """
    zip_or_dir = path.split(":")[0]
    basename = os.path.basename(zip_or_dir)
    stem = basename[:-4] if basename.lower().endswith(".zip") else basename
    barcode = stem.split(".")[0]
    parts = barcode.split("-")
    return "-".join(parts[:3])


def find_genomic_h5(patient_id: str, genomic_h5_dir: str) -> Optional[str]:
    """Return the H5 file path for *patient_id*, or None if not found."""
    base = Path(genomic_h5_dir)
    p = base / f"{patient_id}.h5"
    if p.exists():
        return str(p)
    for dx in range(1, 6):
        p = base / f"{patient_id}-DX{dx}.h5"
        if p.exists():
            return str(p)
    return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ZipTilesWithGenomicFeatures(DefaultTilesDataset):
    """Pairs ZIP-archived tile images with patient-level genomic conditioning vectors.

    Each ``__getitem__`` returns a dict with keys:
      ``img``      — (3, H, W) float32 in [-1, 1]
      ``feat``     — (n_genes,) float32 normalised gene expression
      ``coords``   — (2,) tile coordinates (from parent)
      ``filename`` — str tile path (from parent)
      ``subtype``  — str subtype/cohort label (for balanced sampling only)
    """

    def __init__(
        self,
        zip_dir: str,
        genomic_h5_dir: str,
        patient_splits_path: str,
        split: str = "train",
        max_tiles_by_subtype: Optional[Dict[str, Optional[int]]] = None,
        tile_sampling_seed: int = 42,
        img_size: int = 256,
        do_resize: bool = False,
        do_normalize: bool = True,
        cache_pickle_tiles_path: Optional[str] = None,
    ):
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"split must be one of train/val/test/all, got '{split}'")

        self._splits, self._subtype_map = _load_splits_and_subtypes(patient_splits_path)
        if split == "all":
            self._split_patients: Set[str] = set(self._splits.keys())
        else:
            self._split_patients = {
                pid for pid, fold in self._splits.items() if fold == split
            }

        log.info("ZipTilesWithGenomicFeatures: split='%s', %d patients", split, len(self._split_patients))

        self._img_size = img_size
        super().__init__(
            root_dirs=[zip_dir],
            split="none",
            max_tiles_per_patient=None,  # type: ignore[arg-type]
            as_tensor=True,
            do_normalize=do_normalize,
            do_resize=do_resize,
            img_size=img_size,
            process_only_zips=True,
            cache_pickle_tiles_path=cache_pickle_tiles_path,  # type: ignore[arg-type]
        )
        log.info("Parent scan found %d tiles total", len(self.tile_paths))

        self.tile_paths = [
            p for p in self.tile_paths
            if patient_id_from_tile_path(p) in self._split_patients
        ]
        log.info("After split filter: %d tiles", len(self.tile_paths))

        if max_tiles_by_subtype:
            self.tile_paths = _apply_tile_caps(
                tile_paths=self.tile_paths,
                subtype_map=self._subtype_map,
                caps=max_tiles_by_subtype,
                seed=tile_sampling_seed,
            )
            log.info("After tile capping: %d tiles", len(self.tile_paths))

        unique_patients = {patient_id_from_tile_path(p) for p in self.tile_paths}
        self._genomic_cache, missing = _build_genomic_cache(unique_patients, genomic_h5_dir)

        if missing:
            warnings.warn(
                f"{len(missing)} patients have no matching H5 file and will be excluded: "
                f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}",
                UserWarning,
                stacklevel=2,
            )
            self.tile_paths = [
                p for p in self.tile_paths
                if patient_id_from_tile_path(p) not in missing
            ]
            log.info("After removing patients with no H5: %d tiles remain", len(self.tile_paths))

        log.info(
            "Dataset ready: %d tiles, %d patients, %d genes",
            len(self.tile_paths),
            len(self._genomic_cache),
            next(iter(self._genomic_cache.values())).shape[0] if self._genomic_cache else 0,
        )

    def __getitem__(self, index: int) -> Dict:
        import zipfile as _zipfile
        import zlib as _zlib
        item: Optional[Dict] = None
        for offset in range(len(self.tile_paths)):
            try:
                candidate = super().__getitem__((index + offset) % len(self.tile_paths))
            except (_zipfile.BadZipFile, OSError, _zlib.error) as exc:
                log.warning("Skipping corrupt tile at index %d (offset %d): %s", index, offset, exc)
                continue
            img = candidate["img"]
            if img.shape[-2] == self._img_size and img.shape[-1] == self._img_size:
                item = candidate
                break
        if item is None:
            raise RuntimeError(f"All {len(self.tile_paths)} tiles are corrupt or wrong size starting at index {index}")
        pid = patient_id_from_tile_path(item["filename"])
        item["feat"] = self._genomic_cache[pid]
        item["subtype"] = self._subtype_map.get(pid, "unknown")
        return item


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_splits_and_subtypes(path: str) -> tuple[Dict[str, str], Dict[str, str]]:
    with open(path) as f:
        payload = json.load(f)
    splits: Dict[str, str] = {}
    subtype_map: Dict[str, str] = {}
    for fold, entries in payload.items():
        if fold.startswith("_"):
            continue
        for pid, meta in entries.items():
            if pid.startswith("_"):
                continue
            splits[pid] = fold
            if isinstance(meta, dict):
                subtype_map[pid] = meta.get("subtype", "unknown")
    return splits, subtype_map


def _apply_tile_caps(
    tile_paths: List[str],
    subtype_map: Dict[str, str],
    caps: Dict[str, Optional[int]],
    seed: int,
) -> List[str]:
    rng = random.Random(seed)
    patient_tiles: Dict[str, List[str]] = defaultdict(list)
    for path in tile_paths:
        patient_tiles[patient_id_from_tile_path(path)].append(path)
    kept: List[str] = []
    for pid, tiles in patient_tiles.items():
        subtype = subtype_map.get(pid)
        cap = caps.get(subtype) if subtype is not None else None  # type: ignore[arg-type]
        if cap is not None and len(tiles) > cap:
            tiles = rng.sample(tiles, cap)
        kept.extend(tiles)
    return sorted(kept)


def _build_genomic_cache(
    patient_ids: Set[str],
    genomic_h5_dir: str,
) -> tuple[Dict[str, torch.Tensor], Set[str]]:
    cache: Dict[str, torch.Tensor] = {}
    missing: Set[str] = set()
    for pid in patient_ids:
        h5_path = find_genomic_h5(pid, genomic_h5_dir)
        if h5_path is None:
            missing.add(pid)
            continue
        with h5py.File(h5_path, "r") as f:
            vec = np.asarray(f["feats"][:], dtype=np.float32).squeeze()  # type: ignore[index]
        cache[pid] = torch.from_numpy(vec)
    if missing:
        log.warning("%d / %d patients have no H5 file. Their tiles will be dropped.",
                    len(missing), len(patient_ids))
    return cache, missing
