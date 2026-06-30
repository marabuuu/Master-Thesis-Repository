"""
ZipTilesWithGenomicFeatures — MoPaDi-compatible dataset that pairs tile images
from ZIP archives with patient-level gene-expression conditioning vectors.

Key design decisions
--------------------
* Extends MoPaDi's ``DefaultTilesDataset`` so all ZIP-reading, image transforms
  and coordinate extraction are inherited without duplication.
* Patient ID is extracted from the ZIP filename (``TCGA-XX-XXXX`` = first 3
  hyphen-separated tokens of the barcode before the first dot):
      TCGA-3C-AALI-01Z-00-DX1.UUID.HASH.zip  →  TCGA-3C-AALI
* All H5 files are read once at ``__init__`` time into a RAM cache
  (≈1.6 MB for 800 BRCA patients × 512 genes).  No file handles are kept open
  during ``__getitem__``, which avoids descriptor leaks with multi-worker
  DataLoaders.
* Split filtering and per-subtype tile capping are applied after the
  parent ``__init__`` scan, so tile-path enumeration happens only once.
* Tiles for patients with no matching H5 file are silently removed and a
  warning is printed (expected for the ~5 % BRCA patients without RNA-seq).
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
# Patient-ID helpers (pure functions, independently testable)
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
    # Strip the internal-file portion of a zip path.
    zip_or_dir = path.split(":")[0]
    basename = os.path.basename(zip_or_dir)
    # Remove .zip suffix if present to get the stem.
    stem = basename[:-4] if basename.lower().endswith(".zip") else basename
    # The barcode is the part before the first dot (UUID separator).
    barcode = stem.split(".")[0]          # e.g. TCGA-3C-AALI-01Z-00-DX1
    parts = barcode.split("-")
    # Return first three hyphen-parts: TCGA, cohort, patient → TCGA-XX-XXXX
    return "-".join(parts[:3])


def find_genomic_h5(patient_id: str, genomic_h5_dir: str) -> Optional[str]:
    """Return the H5 file path for *patient_id*, or None if not found.

    Search order
    ------------
    1. ``{patient_id}.h5``          — single RNA-seq sample (most common)
    2. ``{patient_id}-DX1.h5``      — first of multiple RNA-seq samples
    3. ``{patient_id}-DX2.h5`` …    — further samples (up to DX5)

    For patients with multiple RNA-seq samples (``-DX1``, ``-DX2``) we always
    take the first available sample.  Mixing RNA-seq samples for the same
    patient is avoided by keeping the samples together in the same train/val/
    test fold (handled by the preprocessing pipeline).
    """
    base = Path(genomic_h5_dir)
    # Exact match first.
    p = base / f"{patient_id}.h5"
    if p.exists():
        return str(p)
    # Multi-sample fallback.
    for dx in range(1, 6):
        p = base / f"{patient_id}-DX{dx}.h5"
        if p.exists():
            return str(p)
    return None


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class ZipTilesWithGenomicFeatures(DefaultTilesDataset):
    """Dataset that pairs ZIP-archived tile images with genomic guidance vectors.

    Each ``__getitem__`` returns a dict with:

    ``img``      — ``(3, H, W)`` float32 tensor in ``[-1, 1]``
    ``feat``     — ``(n_genes,)`` float32 tensor (normalised gene expression)
    ``coords``   — ``(2,)`` float32 tile coordinates (passthrough from parent)
    ``filename`` — str tile path (passthrough from parent)

    Parameters
    ----------
    zip_dir:
        Directory containing per-patient ``*.zip`` tile archives.
    genomic_h5_dir:
        Directory containing per-patient ``*.h5`` gene-expression files.
    patient_splits_path:
        Path to ``patient_splits.json`` (produced by ``build_genomic_features``).
    split:
        One of ``'train'``, ``'val'``, ``'test'``, or ``'all'``.
        Only tiles whose patient ID belongs to this fold are included.
    max_tiles_by_subtype:
        ``{subtype: max_tiles_per_patient}`` cap.  ``None`` values (or an
        absent subtype key) mean no cap for that subtype.  Pass ``None``
        to disable capping entirely (useful for val/test).
    tile_sampling_seed:
        Seed for reproducible tile subsampling when caps are applied.
    img_size:
        Resize target (pixels).  Only applied if ``do_resize=True``.
    do_resize:
        Whether to resize tiles.  Default: ``False`` (no image FM needed).
    do_normalize:
        Normalise image to ``[-1, 1]`` (required for diffusion training).
    cache_pickle_tiles_path:
        Optional path to a pickle cache for tile-path enumeration.
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

        # ── Load patient splits ──────────────────────────────────────────
        self._splits, self._subtype_map = _load_splits_and_subtypes(
            patient_splits_path
        )
        if split == "all":
            self._split_patients: Set[str] = set(self._splits.keys())
        else:
            self._split_patients = {
                pid for pid, fold in self._splits.items() if fold == split
            }

        log.info(
            "ZipTilesWithGenomicFeatures: split='%s', %d patients",
            split, len(self._split_patients),
        )

        # ── Parent init: scan ZIP directory, build full tile-path list ───
        # Pass process_only_zips=True so MoPaDi's scanner reads tile names
        # from inside ZIP files without extraction.
        # max_tiles_per_patient=None → collect all tiles; we apply our own caps.
        self._img_size = img_size

        super().__init__(
            root_dirs=[zip_dir],
            split="none",               # we do our own split filtering below
            max_tiles_per_patient=None,
            as_tensor=True,
            do_normalize=do_normalize,
            do_resize=do_resize,
            img_size=img_size,
            process_only_zips=True,
            cache_pickle_tiles_path=cache_pickle_tiles_path,
        )
        log.info("Parent scan found %d tiles total", len(self.tile_paths))

        # ── Filter to split patients ─────────────────────────────────────
        self.tile_paths = [
            p for p in self.tile_paths
            if patient_id_from_tile_path(p) in self._split_patients
        ]
        log.info("After split filter: %d tiles", len(self.tile_paths))

        # ── Per-subtype tile capping ─────────────────────────────────────
        if max_tiles_by_subtype:
            self.tile_paths = _apply_tile_caps(
                tile_paths=self.tile_paths,
                subtype_map=self._subtype_map,
                caps=max_tiles_by_subtype,
                seed=tile_sampling_seed,
            )
            log.info("After tile capping: %d tiles", len(self.tile_paths))

        # ── Build genomic cache (H5 → RAM) ───────────────────────────────
        unique_patients = {
            patient_id_from_tile_path(p) for p in self.tile_paths
        }
        self._genomic_cache, missing = _build_genomic_cache(
            unique_patients, genomic_h5_dir
        )

        if missing:
            warnings.warn(
                f"{len(missing)} patients have no matching H5 file and will "
                f"be excluded from the dataset: {sorted(missing)[:10]}"
                f"{'...' if len(missing) > 10 else ''}",
                UserWarning,
                stacklevel=2,
            )
            self.tile_paths = [
                p for p in self.tile_paths
                if patient_id_from_tile_path(p) not in missing
            ]
            log.info(
                "After removing patients with no H5: %d tiles remain",
                len(self.tile_paths),
            )

        log.info(
            "Dataset ready: %d tiles, %d patients, %d genes",
            len(self.tile_paths),
            len(self._genomic_cache),
            next(iter(self._genomic_cache.values())).shape[0]
            if self._genomic_cache else 0,
        )

    # ------------------------------------------------------------------
    # Core data access
    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> Dict:
        # Walk forward until we find a readable, square tile.
        # Catches BadZipFile (bad central directory), OSError (I/O), and
        # zlib.error (corrupt compressed stream inside a tile) so a single bad
        # tile never kills a DataLoader worker.
        import zipfile as _zipfile
        import zlib as _zlib
        for offset in range(len(self.tile_paths)):
            try:
                item = super().__getitem__((index + offset) % len(self.tile_paths))
            except (_zipfile.BadZipFile, OSError, _zlib.error) as exc:
                log.warning(
                    "Skipping corrupt tile at index %d (offset %d): %s",
                    index, offset, exc,
                )
                continue
            img = item["img"]
            if img.shape[-2] == self._img_size and img.shape[-1] == self._img_size:
                break
        pid = patient_id_from_tile_path(item["filename"])
        item["feat"] = self._genomic_cache[pid]    # (n_genes,) float32 tensor
        item["subtype"] = self._subtype_map.get(pid, "unknown")
        return item


# ---------------------------------------------------------------------------
# Private helpers (pure functions — no coupling to dataset state)
# ---------------------------------------------------------------------------

def _load_splits_and_subtypes(
    path: str,
) -> tuple[Dict[str, str], Dict[str, str]]:
    """Load patient_splits.json into (splits, subtype_map) dicts."""
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
    """Apply per-subtype max-tiles-per-patient caps.

    Returns a new (sorted) list of tile paths with excess tiles removed.
    Sampling is reproducible via ``seed``.
    """
    rng = random.Random(seed)

    # Group tiles by patient.
    patient_tiles: Dict[str, List[str]] = defaultdict(list)
    for path in tile_paths:
        pid = patient_id_from_tile_path(path)
        patient_tiles[pid].append(path)

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
    """Read all relevant H5 files into a RAM dictionary.

    Returns
    -------
    cache:
        ``{patient_id: Tensor(n_genes,)}``
    missing:
        Patient IDs for which no H5 file was found.
    """
    cache: Dict[str, torch.Tensor] = {}
    missing: Set[str] = set()

    for pid in patient_ids:
        h5_path = find_genomic_h5(pid, genomic_h5_dir)
        if h5_path is None:
            missing.add(pid)
            continue
        with h5py.File(h5_path, "r") as f:
            vec = np.asarray(f["feats"][:], dtype=np.float32).squeeze()
        cache[pid] = torch.from_numpy(vec)

    if missing:
        log.warning(
            "%d / %d patients have no H5 file.  Their tiles will be dropped.",
            len(missing), len(patient_ids),
        )
    return cache, missing
