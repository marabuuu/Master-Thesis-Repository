"""Dataset pairing ZIP-archived tiles with genomic conditioning vectors."""

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
import torch.nn.functional as F

from mopadi.dataset import DefaultTilesDataset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patient-ID helpers
# ---------------------------------------------------------------------------

def patient_id_from_tile_path(path: str) -> str:
    """Extract TCGA-XX-XXXX patient barcode from a tile or zip path."""
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

# Fixed cohort → integer index, used for class_embed conditioning
# (nn.Embedding) and one_hot error messages.
COHORT_INDEX: Dict[str, int] = {"TCGA-BRCA": 0, "TCGA-LIHC": 1}


def _make_orthogonal_binary_codes(feat_dim: int, normalize: bool = True) -> Dict[str, torch.Tensor]:
    """Return alternating binary codes for BRCA (odd indices) and LIHC (even indices)."""
    if feat_dim < 2:
        raise ValueError(f"feat_dim must be at least 2, got {feat_dim}")
    if feat_dim % 2 != 0:
        raise ValueError(
            f"feat_dim must be even for alternating orthogonal codes, got {feat_dim}"
        )

    brca = torch.zeros(feat_dim, dtype=torch.float32)
    lihc = torch.zeros(feat_dim, dtype=torch.float32)
    brca[1::2] = 1.0
    lihc[0::2] = 1.0

    if normalize:
        brca = F.normalize(brca, p=2, dim=-1)
        lihc = F.normalize(lihc, p=2, dim=-1)

    dot = torch.dot(brca, lihc).item()
    log.info(
        "Synthetic one_hot conditioning: BRCA=0101..., LIHC=1010..., "
        "normalize=%s, dot=%.1f, norms=(%.3f, %.3f), feat_dim=%d",
        normalize, dot, float(brca.norm().item()), float(lihc.norm().item()), feat_dim,
    )
    return {
        "TCGA-BRCA": brca,
        "TCGA-LIHC": lihc,
    }


def _make_orthogonal_codes(
    class_names: list,
    feat_dim: int,
    normalize: bool = True,
) -> Dict[str, torch.Tensor]:
    """Return one orthogonal binary code per class via round-robin position assignment."""
    n = len(class_names)
    if n < 2:
        raise ValueError(f"Need at least 2 class names, got {n}")
    if feat_dim < n:
        raise ValueError(f"feat_dim ({feat_dim}) must be >= number of classes ({n})")

    codes = {}
    for i, name in enumerate(class_names):
        v = torch.zeros(feat_dim, dtype=torch.float32)
        v[i::n] = 1.0
        if normalize:
            v = F.normalize(v, p=2, dim=-1)
        codes[name] = v

    c0, c1 = codes[class_names[0]], codes[class_names[1]]
    log.info(
        "Orthogonal codes: %d classes %s, feat_dim=%d, normalize=%s, "
        "dot(0,1)=%.4f, norm0=%.3f",
        n, class_names, feat_dim, normalize,
        torch.dot(c0, c1).item(), float(c0.norm().item()),
    )
    return codes


class ZipTilesWithGenomicFeatures(DefaultTilesDataset):
    """Pairs ZIP-archived tile images with patient-level genomic conditioning vectors.

    Returns dicts with img, feat, coords, filename, subtype.
    conditioning_type selects what feat contains: real (RNA-seq), zeros,
    noise (random unit sphere), one_hot (orthogonal per cohort), or class_embed
    (integer index for nn.Embedding).
    """

    def __init__(
        self,
        zip_dir: str,
        genomic_h5_dir: Optional[str],
        patient_splits_path: str,
        split: str = "train",
        max_tiles_by_subtype: Optional[Dict[str, Optional[int]]] = None,
        tile_sampling_seed: int = 42,
        img_size: int = 256,
        do_resize: bool = False,
        do_normalize: bool = True,
        cache_pickle_tiles_path: Optional[str] = None,
        conditioning_type: str = "real",
        feat_dim: int = 512,
        normalize_feats: bool = True,
    ):
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"split must be one of train/val/test/all, got '{split}'")
        if conditioning_type not in {"real", "zeros", "noise", "one_hot", "class_embed"}:
            raise ValueError(f"conditioning_type must be real/zeros/noise/one_hot/class_embed, got '{conditioning_type}'")
        if conditioning_type == "real" and genomic_h5_dir is None:
            raise ValueError("genomic_h5_dir is required when conditioning_type='real'")

        self._conditioning_type = conditioning_type
        self._feat_dim = feat_dim

        self._splits, self._subtype_map = _load_splits_and_subtypes(patient_splits_path)
        if split == "all":
            self._split_patients: Set[str] = set(self._splits.keys())
        else:
            self._split_patients = {
                pid for pid, fold in self._splits.items() if fold == split
            }

        log.info("ZipTilesWithGenomicFeatures: split='%s', %d patients", split, len(self._split_patients))

        self._img_size = img_size
        self._orthogonal_codes: Optional[Dict[str, torch.Tensor]] = None
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
        if conditioning_type == "real":
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
        else:
            self._genomic_cache = {}
            if conditioning_type == "one_hot":
                unique_subtypes = sorted(set(self._subtype_map.values()))
                self._orthogonal_codes = _make_orthogonal_codes(
                    unique_subtypes, feat_dim, normalize=normalize_feats
                )
            else:
                self._orthogonal_codes = None
            log.info("Synthetic conditioning ('%s'): skipping H5 loading", conditioning_type)

        log.info(
            "Dataset ready: %d tiles, %d patients, conditioning='%s' feat_dim=%d",
            len(self.tile_paths),
            len(unique_patients),
            conditioning_type,
            feat_dim,
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
        item["subtype"] = self._subtype_map.get(pid, "unknown")

        if self._conditioning_type == "real":
            item["feat"] = self._genomic_cache[pid]
        elif self._conditioning_type == "zeros":
            item["feat"] = torch.zeros(self._feat_dim)
        elif self._conditioning_type == "noise":
            # Fresh unit-sphere random vector every call — no consistent signal by design.
            v = torch.randn(self._feat_dim)
            item["feat"] = v / v.norm().clamp(min=1e-8)
        elif self._conditioning_type == "class_embed":
            subtype = item["subtype"]
            if subtype not in COHORT_INDEX:
                raise KeyError(f"class_embed conditioning: unknown subtype '{subtype}'. Known: {list(COHORT_INDEX)}")
            item["feat"] = torch.tensor([COHORT_INDEX[subtype]], dtype=torch.long)
        elif self._conditioning_type == "one_hot":
            subtype = item["subtype"]
            if self._orthogonal_codes is None or subtype not in self._orthogonal_codes:
                known = list(self._orthogonal_codes.keys()) if self._orthogonal_codes is not None else list(COHORT_INDEX.keys())
                raise KeyError(
                    f"one_hot conditioning: unknown subtype '{subtype}'. "
                    f"Known: {known}"
                )
            item["feat"] = self._orthogonal_codes[subtype].clone()
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
