#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract Virchow2 patch features from generated/reconstructed H&E tiles.

Reads per-patient ZIP archives produced by ``reconstruct_tiles.py`` (each ZIP
contains PNG tiles) or per-patient subdirectories, encodes every tile with the
frozen Virchow2 ViT, and writes one HDF5 file per patient::

    output_dir/
        TCGA-3C-AALI.h5   # dataset "features", shape (N, 1280), float32
        TCGA-A2-A04T.h5
        ...

The HDF5 layout is intentionally identical to the real-slide Virchow2 feature
files so that the existing subtype classifier pipeline
(``evaluate_subtype_classifier``) consumes them without modification.
Coordinate information is omitted because generated tiles do not have a spatial
position on a slide.

Input formats supported
-----------------------
* Per-patient ZIPs (default reconstruction output)::

      <tiles_dir>/<patient_id>.zip   -- PNG/JPG entries inside the archive

* Per-patient subdirectories (flat layout)::

      <tiles_dir>/<patient_id>/*.png

Usage (CLI)
-----------
::

    python -m src.classifier.extract_virchow2_features \\
        --tiles-dir experiments/.../reconstructed \\
        --output-dir experiments/.../synthetic_virchow2_features \\
        --batch-size 16 --device cuda

Usage (programmatic)
--------------------
::

    from src.classifier.extract_virchow2_features import extract_features_for_dir
    extract_features_for_dir(tiles_dir, output_dir, batch_size=16, device="cuda")
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
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

_TILE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Virchow2 feature extractor
# ---------------------------------------------------------------------------

class _Virchow2Extractor:
    """Frozen Virchow2 ViT encoder: 224×224 RGB tile → 1280-d class token.

    Replicates the extraction logic from
    ``mopadi/src/mopadi/model/extractor.py`` (FeatureExtractorVirchow2) so
    that synthetic-tile features are compatible with the real-slide H5 files
    used to train the subtype classifier.
    """

    def __init__(self, device: str = "cpu") -> None:
        try:
            import timm
            from timm.data import create_transform, resolve_data_config
            from timm.layers import SwiGLUPacked
        except ImportError as exc:
            raise ImportError(
                "timm is required: pip install timm"
            ) from exc

        logger.info("Loading Virchow2 from HuggingFace Hub (paige-ai/Virchow2)...")
        self.model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        ).eval().to(device)
        self.device = device

        for p in self.model.parameters():
            p.requires_grad = False

        self.transform = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )
        logger.info(f"Virchow2 ready on '{device}'")

    @torch.inference_mode()
    def encode_batch(self, pil_images: List[Image.Image]) -> np.ndarray:
        """Return class-token embeddings for a list of PIL images.

        Args:
            pil_images: RGB PIL images (any size; resized by the Virchow2
                        transform to 224×224 internally).

        Returns:
            ``np.ndarray`` of shape ``(N, 1280)``, dtype ``float32``.
        """
        batch = torch.stack(
            [self.transform(img) for img in pil_images]
        ).to(self.device)
        output = self.model(batch)   # (N, 261, 1280)  [cls + 256 patch + 4 reg]
        class_token = output[:, 0]   # (N, 1280)
        return class_token.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Tile discovery
# ---------------------------------------------------------------------------

def _iter_zip(zip_path: Path) -> Iterator[Tuple[str, Image.Image]]:
    """Yield ``(entry_name, PIL.Image)`` for every image inside a ZIP."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if Path(name).suffix.lower() in _TILE_SUFFIXES:
                with zf.open(name) as fh:
                    img = Image.open(io.BytesIO(fh.read())).convert("RGB")
                yield name, img


def _iter_dir(patient_dir: Path) -> Iterator[Tuple[str, Image.Image]]:
    """Yield ``(filename, PIL.Image)`` for every image in a flat directory."""
    for p in sorted(patient_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in _TILE_SUFFIXES:
            yield p.name, Image.open(p).convert("RGB")


def _discover_patients(tiles_dir: Path) -> List[Tuple[str, str, Path]]:
    """Return ``[(patient_id, source_type, source_path), ...]``.

    ``source_type`` is ``"zip"`` or ``"dir"``.
    ``patient_id`` is normalised via ``canonical_patient_id``.
    """
    from src.classifier.utils_subtype_data import canonical_patient_id

    patients: List[Tuple[str, str, Path]] = []
    for item in sorted(tiles_dir.iterdir()):
        if item.is_file() and item.suffix.lower() == ".zip":
            patients.append((canonical_patient_id(item.stem), "zip", item))
        elif item.is_dir():
            has_images = any(
                f.suffix.lower() in _TILE_SUFFIXES
                for f in item.iterdir()
                if f.is_file()
            )
            if has_images:
                patients.append((canonical_patient_id(item.name), "dir", item))
    return patients


# ---------------------------------------------------------------------------
# Per-patient extraction
# ---------------------------------------------------------------------------

def _extract_patient_features(
    extractor: _Virchow2Extractor,
    source_type: str,
    source_path: Path,
    batch_size: int,
) -> np.ndarray:
    """Extract Virchow2 features for all tiles of one patient.

    Returns array of shape ``(n_tiles, 1280)``, ``float32``.
    """
    iter_fn = _iter_zip if source_type == "zip" else _iter_dir
    chunks: List[np.ndarray] = []
    batch: List[Image.Image] = []

    for _name, img in iter_fn(source_path):
        batch.append(img)
        if len(batch) == batch_size:
            chunks.append(extractor.encode_batch(batch))
            batch = []

    if batch:
        chunks.append(extractor.encode_batch(batch))

    if not chunks:
        raise ValueError(f"No tiles found in {source_path}")

    return np.concatenate(chunks, axis=0)


def _save_h5(features: np.ndarray, out_path: Path) -> None:
    """Write ``(N, 1280)`` float32 feature array to HDF5 as dataset 'features'."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset(
            "features",
            data=features.astype(np.float32),
            compression="gzip",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features_for_dir(
    tiles_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = 32,
    device: Optional[str] = None,
    skip_existing: bool = True,
) -> None:
    """Extract Virchow2 features for all patients found in *tiles_dir*.

    Each patient produces one ``<patient_id>.h5`` file in *output_dir*
    containing a single HDF5 dataset named ``"features"`` with shape
    ``(n_tiles, 1280)`` and dtype ``float32``.  This file is directly
    consumable by ``evaluate_subtype_classifier`` without any code changes —
    just point ``features_dir`` at *output_dir*.

    Args:
        tiles_dir:     Directory with per-patient ZIP archives or
                       subdirectories of tile images.
        output_dir:    Destination directory for ``.h5`` files.
        batch_size:    Number of tiles per GPU forward pass.
        device:        ``"cuda"``, ``"cpu"``, or ``None`` for auto-detect.
        skip_existing: Skip patients whose ``.h5`` file already exists.
    """
    tiles_dir = Path(tiles_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    patients = _discover_patients(tiles_dir)
    if not patients:
        raise FileNotFoundError(
            f"No patient ZIPs or tile directories found under {tiles_dir}"
        )
    logger.info(f"Found {len(patients)} patient(s) in {tiles_dir}")

    extractor = _Virchow2Extractor(device=device)

    processed = skipped = failed = 0
    for pid, source_type, source_path in patients:
        out_path = output_dir / f"{pid}.h5"
        if skip_existing and out_path.exists():
            logger.info(f"  {pid}: skip (exists)")
            skipped += 1
            continue

        try:
            features = _extract_patient_features(
                extractor, source_type, source_path, batch_size
            )
            _save_h5(features, out_path)
            logger.info(f"  {pid}: {features.shape[0]} tiles → {out_path.name}")
            processed += 1
        except Exception as exc:
            logger.error(f"  {pid}: FAILED — {exc}")
            failed += 1

    logger.info(
        f"Done. processed={processed} skipped={skipped} failed={failed} "
        f"| output: {output_dir}"
    )


def run_virchow2_extraction(cfg: dict, verbose: bool = True) -> None:
    """Config-driven entry point called from ``run_pipeline.py``.

    Expected config section (``virchow2_extraction``)::

        tiles_dir:      path to per-patient ZIP archives / tile directories
        output_dir:     destination for per-patient .h5 feature files
        batch_size:     (optional, default 32)
        device:         (optional, auto-detected)
        skip_existing:  (optional, default true)
    """
    tiles_dir = cfg.get("tiles_dir")
    output_dir = cfg.get("output_dir")
    if not tiles_dir:
        raise ValueError("Missing 'virchow2_extraction.tiles_dir' in config.yaml")
    if not output_dir:
        raise ValueError("Missing 'virchow2_extraction.output_dir' in config.yaml")

    if verbose:
        logger.info(f"[Virchow2Extraction] tiles_dir: {tiles_dir}")
        logger.info(f"[Virchow2Extraction] output_dir: {output_dir}")

    extract_features_for_dir(
        tiles_dir=tiles_dir,
        output_dir=output_dir,
        batch_size=int(cfg.get("batch_size", 32)),
        device=cfg.get("device") or None,
        skip_existing=bool(cfg.get("skip_existing", True)),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract Virchow2 features from generated H&E tiles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tiles-dir", required=True,
        help="Directory with per-patient ZIP archives or tile subdirectories",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Destination directory for per-patient .h5 feature files",
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help="Number of tiles per forward pass",
    )
    p.add_argument(
        "--device", default=None,
        help="'cuda', 'cpu', or leave unset for auto-detect",
    )
    p.add_argument(
        "--no-skip-existing", action="store_true",
        help="Re-extract even if the .h5 file already exists",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    extract_features_for_dir(
        tiles_dir=args.tiles_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
