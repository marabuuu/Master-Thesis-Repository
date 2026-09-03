#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract cheap global color/texture features from H&E tiles.

This is a *shortcut probe*, not a real feature extractor: every feature here is
a trivial per-tile summary statistic (channel means/stds, a coarse color
histogram, a Laplacian-variance texture proxy). None of it captures learned
morphology the way Virchow2 does.

Purpose: if a linear classifier trained on these low-level stats alone can
already separate Basal vs LumA nearly as well as the Virchow2-based classifier,
that is evidence the Virchow2 classifier's separability (and its exaggeration
on generated tiles) may be partly driven by low-level style/color shortcuts
rather than genuine morphology — see the 20260618_subtype_classifier_cv
discussion. If the color-only classifier is much weaker than Virchow2 across
the board, that argues against a trivial-shortcut explanation.

The HDF5 layout mirrors ``extract_virchow2_features.py`` exactly (one file per
patient, dataset "features" of shape (N, D)) so it is a drop-in for the
existing subtype classifier pipeline: ``cv_subtype_classifier.py`` (flat
directory, real tiles) and ``evaluate_generated_tiles.py``
({Basal,LumA} subdirectories, generated tiles) both work unmodified.

Usage
-----
::

    python -m src.classifier.extract_color_texture_features \\
        --tiles-dir experiments/.../generated/Basal \\
        --output-dir experiments/.../color_features/Basal
"""
from __future__ import annotations

import argparse
import io
import logging
import zipfile
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import h5py
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

_TILE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_N_HIST_BINS = 8
FEATURE_DIM = 3 + 3 + 3 + 3 + 3 + 3 * _N_HIST_BINS  # = 39


def _laplacian_variance(gray: np.ndarray) -> float:
    """Cheap edge/texture-density proxy: variance of the discrete Laplacian."""
    kernel_sum = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(kernel_sum.var()) if kernel_sum.size > 0 else 0.0


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB [0,1] -> HSV [0,1] (avoids a colorsys per-pixel loop)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc > 0, delta / np.where(maxc > 0, maxc, 1.0), 0.0)

    rc = np.where(delta > 0, (maxc - r) / np.where(delta > 0, delta, 1.0), 0.0)
    gc = np.where(delta > 0, (maxc - g) / np.where(delta > 0, delta, 1.0), 0.0)
    bc = np.where(delta > 0, (maxc - b) / np.where(delta > 0, delta, 1.0), 0.0)

    h = np.zeros_like(maxc)
    h = np.where(maxc == r, bc - gc, h)
    h = np.where(maxc == g, 2.0 + rc - bc, h)
    h = np.where(maxc == b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h = np.where(delta == 0, 0.0, h)
    return np.stack([h, s, v], axis=-1)


def image_to_color_texture_features(img: Image.Image) -> np.ndarray:
    """Compute a ``(FEATURE_DIM,)`` float32 vector of global color/texture stats.

    Features: RGB mean/std (6), HSV mean/std for H/S/V (6), grayscale mean/std +
    Laplacian variance (3), and a coarse 8-bin-per-channel RGB color histogram
    (24), normalized to sum to 1 per channel.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    rgb_mean = arr.mean(axis=(0, 1))
    rgb_std = arr.std(axis=(0, 1))

    hsv = _rgb_to_hsv(arr)
    hsv_mean = hsv.mean(axis=(0, 1))
    hsv_std = hsv.std(axis=(0, 1))

    gray = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray_mean = float(gray.mean())
    gray_std = float(gray.std())
    lap_var = _laplacian_variance(gray)

    hist_feats = []
    for c in range(3):
        hist, _ = np.histogram(arr[..., c], bins=_N_HIST_BINS, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        hist /= max(hist.sum(), 1.0)
        hist_feats.append(hist)

    feat = np.concatenate([
        rgb_mean, rgb_std,
        hsv_mean, hsv_std,
        [gray_mean, gray_std, lap_var],
        *hist_feats,
    ]).astype(np.float32)
    assert feat.shape[0] == FEATURE_DIM, f"expected {FEATURE_DIM}, got {feat.shape[0]}"
    return feat


# ---------------------------------------------------------------------------
# Tile discovery (identical layout support to extract_virchow2_features.py)
# ---------------------------------------------------------------------------

def _iter_zip(zip_path: Path) -> Iterator[Image.Image]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if Path(name).suffix.lower() in _TILE_SUFFIXES:
                with zf.open(name) as fh:
                    yield Image.open(io.BytesIO(fh.read())).convert("RGB")


def _iter_dir(patient_dir: Path) -> Iterator[Image.Image]:
    for p in sorted(patient_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in _TILE_SUFFIXES:
            yield Image.open(p).convert("RGB")


def _discover_patients(tiles_dir: Path) -> List[Tuple[str, str, Path]]:
    from src.classifier.utils_subtype_data import canonical_patient_id

    patients: List[Tuple[str, str, Path]] = []
    for item in sorted(tiles_dir.iterdir()):
        if item.is_file() and item.suffix.lower() == ".zip":
            patients.append((canonical_patient_id(item.stem), "zip", item))
        elif item.is_dir():
            has_images = any(
                f.suffix.lower() in _TILE_SUFFIXES for f in item.iterdir() if f.is_file()
            )
            if has_images:
                patients.append((canonical_patient_id(item.name), "dir", item))
    return patients


def _extract_patient_features(source_type: str, source_path: Path) -> np.ndarray:
    iter_fn = _iter_zip if source_type == "zip" else _iter_dir
    feats = [image_to_color_texture_features(img) for img in iter_fn(source_path)]
    if not feats:
        raise ValueError(f"No tiles found in {source_path}")
    return np.stack(feats, axis=0)


def _save_h5(features: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("features", data=features.astype(np.float32), compression="gzip")


def extract_features_for_dir(
    tiles_dir: str | Path,
    output_dir: str | Path,
    skip_existing: bool = True,
) -> None:
    tiles_dir = Path(tiles_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patients = _discover_patients(tiles_dir)
    if not patients:
        raise FileNotFoundError(f"No patient ZIPs or tile directories found under {tiles_dir}")
    logger.info(f"Found {len(patients)} patient(s) in {tiles_dir}")

    processed = skipped = failed = 0
    for pid, source_type, source_path in patients:
        out_path = output_dir / f"{pid}.h5"
        if skip_existing and out_path.exists():
            logger.info(f"  {pid}: skip (exists)")
            skipped += 1
            continue
        try:
            features = _extract_patient_features(source_type, source_path)
            _save_h5(features, out_path)
            logger.info(f"  {pid}: {features.shape[0]} tiles -> {out_path.name}")
            processed += 1
        except Exception as exc:
            logger.error(f"  {pid}: FAILED — {exc}")
            failed += 1

    logger.info(
        f"Done. processed={processed} skipped={skipped} failed={failed} | output: {output_dir}"
    )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract global color/texture shortcut-probe features from H&E tiles")
    p.add_argument("--tiles-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--no-skip-existing", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    extract_features_for_dir(
        tiles_dir=args.tiles_dir,
        output_dir=args.output_dir,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
