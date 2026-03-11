#!/usr/bin/env python3
"""
Cell Segmentation and Classification using DeepCMorph.

Run DeepCMorph on H&E tile images to produce per-tile nuclei segmentation
masks and per-cell-type classification maps.  The output is saved as NumPy
``.npy`` files compatible with the TopoFD evaluation pipeline.

**Input** – one of:
  * a directory of ZIP archives (one per patient, each containing 512×512 tiles)
  * a flat directory of tile images (.png / .jpg)

**Output** – for every tile a ``.npy`` file is saved with shape ``(H, W, C)``
where each channel is a binary mask for one cell type, plus (optionally) a
global nuclei segmentation mask ``*_seg.npy``.

Usage examples
--------------
# From a directory of ZIP tile archives
python -m src.classifier.segment_and_classify_cells \\
    --input-dir /data/tiles \\
    --output-dir /data/cell_masks \\
    --input-format zip \\
    --num-classes 32 \\
    --batch-size 8

# From a flat directory of PNG/JPG images
python -m src.classifier.segment_and_classify_cells \\
    --input-dir /data/flat_tiles \\
    --output-dir /data/cell_masks \\
    --input-format images \\
    --batch-size 16

# With a custom cell-type grouping (reducing 32 classes to N groups)
python -m src.classifier.segment_and_classify_cells \\
    --input-dir /data/tiles \\
    --output-dir /data/cell_masks \\
    --input-format zip \\
    --grouping-json /data/cell_type_groups.json

Dependencies
------------
    deepcmorph, torch, numpy, Pillow, tqdm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── image extensions accepted when scanning directories / ZIPs ──────────
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ===================================================================
# § 1  Tile discovery
# ===================================================================

def discover_tiles_from_zips(
    input_dir: Path,
    max_tiles_per_zip: Optional[int] = None,
) -> List[Tuple[Path, str]]:
    """Return a list of ``(zip_path, internal_name)`` for every tile image.

    Parameters
    ----------
    input_dir : Path
        Directory containing ``*.zip`` archives.
    max_tiles_per_zip : int or None
        If set, randomly sub-sample at most *N* tiles per ZIP (reproducible).

    Returns
    -------
    list of (Path, str)
    """
    zip_files = sorted(input_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No .zip files found in {input_dir}")

    tiles: List[Tuple[Path, str]] = []
    for zp in zip_files:
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                candidates = [
                    n for n in zf.namelist()
                    if Path(n).suffix.lower() in _IMAGE_EXTS
                ]
        except zipfile.BadZipFile:
            logger.warning("Skipping bad zip: %s", zp)
            continue

        if max_tiles_per_zip and len(candidates) > max_tiles_per_zip:
            rng = np.random.RandomState(hash(zp.name) % 2**32)
            candidates = list(rng.choice(candidates, max_tiles_per_zip, replace=False))

        for name in candidates:
            tiles.append((zp, name))

    logger.info("Discovered %d tiles from %d ZIP archives.", len(tiles), len(zip_files))
    return tiles


def discover_tiles_from_dir(input_dir: Path) -> List[Path]:
    """Return a sorted list of tile image paths from a flat directory."""
    tiles = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in _IMAGE_EXTS
    )
    if not tiles:
        raise FileNotFoundError(f"No image files found in {input_dir}")
    logger.info("Discovered %d tiles in %s.", len(tiles), input_dir)
    return tiles


# ===================================================================
# § 2  Tile loading helpers
# ===================================================================

def load_tile_from_zip(zip_path: Path, internal_name: str) -> Image.Image:
    """Open a single tile from inside a ZIP archive."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(internal_name) as f:
            return Image.open(BytesIO(f.read())).convert("RGB")


def load_tile_from_path(path: Path) -> Image.Image:
    """Open a tile image file from disk."""
    return Image.open(path).convert("RGB")


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to a float32 numpy array in [0, 1]."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return arr


# ===================================================================
# § 3  DeepCMorph wrapper
# ===================================================================

class DeepCMorphSegmenter:
    """Thin wrapper around DeepCMorph for batched cell segmentation.

    Parameters
    ----------
    num_classes : int
        Number of cell-type classes the model was trained with (default 32
        for the TCGA pre-trained weights).
    weights_dataset : str
        Which pre-trained weights to load (passed to
        ``model.load_weights(dataset=...)``).
    device : str
        ``"cuda"`` or ``"cpu"``.
    grouping : dict or None
        Optional mapping from *group name* → list of original class indices
        (0-based).  When provided the output classification maps are
        aggregated into fewer channels according to this grouping.
    """

    def __init__(
        self,
        num_classes: int = 32,
        weights_dataset: str = "TCGA",
        device: str = "cuda",
        grouping: Optional[Dict[str, List[int]]] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        # DeepCMorph is not a proper Python package – add its repo to
        # sys.path so ``from model import DeepCMorph`` resolves.
        # The repo is expected at  <workspace>/DeepCMorph  (next to
        # Master-Thesis-Repository).
        _deepcmorph_dir = str(
            Path(__file__).resolve().parents[3] / "DeepCMorph"
        )
        if _deepcmorph_dir not in sys.path:
            sys.path.insert(0, _deepcmorph_dir)

        try:
            from model import DeepCMorph  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "DeepCMorph is required.  Clone the repo next to "
                "Master-Thesis-Repository so that ../DeepCMorph/model.py "
                "exists, or add it to PYTHONPATH manually."
            ) from exc

        self.device = device
        self.num_classes = num_classes
        self.grouping = grouping

        logger.info(
            "Initialising DeepCMorph (num_classes=%d, weights=%s, device=%s)",
            num_classes, weights_dataset, device,
        )
        self.model = DeepCMorph(num_classes=num_classes)

        # The DeepCMorph code expects checkpoint paths like
        # "pretrained_models/… .pth" to be loadable from the current
        # working directory. When the DeepCMorph repo lives next to this
        # project we prefer to resolve the checkpoint path explicitly so
        # load_weights can find the file regardless of cwd.
        checkpoints_dir = Path(_deepcmorph_dir) / "pretrained_models"
        dataset_to_fname = {
            "COMBINED": "DeepCMorph_Datasets_Combined_41_classes_acc_8159.pth",
            "TCGA": "DeepCMorph_Pan_Cancer_32_classes_acc_8273.pth",
            "TCGA_REGULARIZED": "DeepCMorph_Pan_Cancer_Regularized_32_classes_acc_8200.pth",
            "CRC": "DeepCMorph_NCT_CRC_HE_Dataset_9_classes_acc_9699.pth",
        }

        path_to_checkpoint = None

        # If user provided an explicit checkpoint path, prefer that.
        if checkpoint_path:
            cand = Path(checkpoint_path)
            if not cand.exists():
                # Try resolving relative to the DeepCMorph repo
                alt = Path(_deepcmorph_dir) / checkpoint_path
                if alt.exists():
                    cand = alt
            if cand.exists():
                path_to_checkpoint = str(cand)
            else:
                logger.warning("Provided checkpoint path does not exist: %s", checkpoint_path)

        # Otherwise try to find pretrained checkpoint in the DeepCMorph repo
        if path_to_checkpoint is None and weights_dataset in dataset_to_fname:
            fname = dataset_to_fname[weights_dataset]
            # Search several possible locations for the checkpoint
            search_dirs = [
                checkpoints_dir,                                       # DeepCMorph/pretrained_models/
                Path(__file__).resolve().parents[3] / "models" / "DeepCMorph",  # models/DeepCMorph/
            ]
            for search_dir in search_dirs:
                candidate = search_dir / fname
                if candidate.exists():
                    path_to_checkpoint = str(candidate)
                    logger.info("Found checkpoint at %s", path_to_checkpoint)
                    break

        if path_to_checkpoint is not None:
            self.model.load_weights(path_to_checkpoints=path_to_checkpoint)
        else:
            # Fall back to the original API: let the model resolve the
            # relative path (this may fail if cwd is incompatible).
            self.model.load_weights(dataset=weights_dataset)
        # Move to device if the model exposes .to(); otherwise it stays as-is.
        if hasattr(self.model, "to"):
            self.model = self.model.to(device)

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #

    def predict(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run segmentation + classification on a single image.

        Parameters
        ----------
        image : np.ndarray
            RGB image with shape (H, W, 3), values in [0, 1] float32
            **or** uint8 in [0, 255] (the model handles both).

        Returns
        -------
        seg_map : np.ndarray, shape (H, W)
            Binary nuclei segmentation mask.
        cls_maps : np.ndarray, shape (H, W, C)
            Per-class binary segmentation maps.  *C* equals ``num_classes``
            unless ``grouping`` is set, in which case *C* equals the number
            of groups.
        """
        # Ensure input is a torch.Tensor on the correct device
        inp: torch.Tensor
        if isinstance(image, np.ndarray):
            t = torch.from_numpy(image)
            # Expect H,W,3 -> convert to C,H,W
            if t.ndim == 3 and t.shape[2] == 3:
                t = t.permute(2, 0, 1)
            # add batch dim
            inp = t.unsqueeze(0).float().to(self.device)
        elif torch.is_tensor(image):
            inp = image
            if inp.ndim == 3 and inp.shape[2] == 3:
                inp = inp.permute(2, 0, 1)
            if inp.ndim == 3:
                inp = inp.unsqueeze(0)
            inp = inp.float().to(self.device)
        else:
            raise TypeError("Unsupported image type for prediction")

        # Run model (may accept batched tensors)
        out = self.model(inp, return_segmentation_maps=True)

        # Model may return (seg_map, cls_maps) or a list/tuple; unpack safely
        if isinstance(out, (list, tuple)) and len(out) >= 2:
            seg_map_t, cls_maps_t = out[0], out[1]
        else:
            raise RuntimeError("Unexpected model output format from DeepCMorph")

        # Convert torch tensors to numpy arrays and normalise shapes
        def _to_numpy(x: object) -> np.ndarray:
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            arr = np.asarray(x)
            # Squeeze batch dim if present
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            return arr

        seg_map = _to_numpy(seg_map_t)
        cls_maps = _to_numpy(cls_maps_t)

        # If cls_maps is channels-first (C,H,W) convert to (H,W,C)
        expected_c = len(self.grouping) if self.grouping else self.num_classes
        if cls_maps.ndim == 3 and cls_maps.shape[0] == expected_c:
            cls_maps = np.transpose(cls_maps, (1, 2, 0))

        # Apply optional channel grouping (operates on numpy arrays)
        if self.grouping is not None:
            cls_maps = self._apply_grouping(cls_maps)

        return seg_map, cls_maps

    def predict_batch(
        self,
        images: Sequence[np.ndarray],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Run prediction on a list of images.

        DeepCMorph may or may not support native batching; this helper
        iterates over individual calls so the public API doesn't change.
        """
        results: List[Tuple[np.ndarray, np.ndarray]] = []
        for img in images:
            results.append(self.predict(img))
        return results

    # ------------------------------------------------------------------ #
    #  Grouping helper
    # ------------------------------------------------------------------ #

    def _apply_grouping(self, cls_maps: np.ndarray) -> np.ndarray:
        """Collapse per-class maps into grouped channels.

        Parameters
        ----------
        cls_maps : np.ndarray, shape (H, W, C_orig)

        Returns
        -------
        grouped : np.ndarray, shape (H, W, C_groups)
        """
        h, w = cls_maps.shape[:2]
        # Guard against static type-checker warnings: return original maps
        # when no grouping is provided.
        if not self.grouping:
            return cls_maps

        group_names = sorted(self.grouping.keys())
        grouped = np.zeros((h, w, len(group_names)), dtype=cls_maps.dtype)
        for gi, gname in enumerate(group_names):
            for original_idx in self.grouping[gname]:
                if original_idx < cls_maps.shape[2]:
                    grouped[..., gi] = np.maximum(
                        grouped[..., gi], cls_maps[..., original_idx]
                    )
        return grouped


# ===================================================================
# § 4  Output helpers
# ===================================================================

def _make_output_name(source_name: str, suffix: str = "") -> str:
    """Derive the .npy output filename from the tile source name.

    Examples
    --------
    >>> _make_output_name("tile_(1024, 512).png", "_cls")
    'tile_(1024, 512)_cls.npy'
    >>> _make_output_name("TCGA-3C-AALI/tile.jpg")
    'TCGA-3C-AALI__tile.npy'
    """
    # Flatten any directory structure inside a ZIP into underscores
    name = source_name.replace("/", "__").replace("\\", "__")
    stem = Path(name).stem
    return f"{stem}{suffix}.npy"


def _patient_id_from_zip(zip_path: Path) -> str:
    """Extract a TCGA-style patient ID from a ZIP filename.

    E.g.  ``TCGA-3C-AALI-01Z-00-DX1.<uuid>.<hash>.zip`` → ``TCGA-3C-AALI``
    """
    parts = zip_path.stem.split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return zip_path.stem


# ===================================================================
# § 5  Main processing loop
# ===================================================================

def process_tiles_from_zips(
    input_dir: Path,
    output_dir: Path,
    segmenter: DeepCMorphSegmenter,
    *,
    max_tiles_per_zip: Optional[int] = None,
    save_seg: bool = True,
    per_patient_dirs: bool = True,
) -> int:
    """Segment / classify all tiles found in ZIP archives.

    Parameters
    ----------
    input_dir : Path
        Directory with ``*.zip`` tile archives.
    output_dir : Path
        Root output directory.
    segmenter : DeepCMorphSegmenter
        Initialised segmenter.
    max_tiles_per_zip : int or None
        Optional cap on tiles per ZIP.
    save_seg : bool
        If True, also save the global nuclei segmentation ``*_seg.npy``.
    per_patient_dirs : bool
        If True, create a sub-directory per patient under *output_dir*.

    Returns
    -------
    int
        Total number of tiles processed.
    """
    tile_entries = discover_tiles_from_zips(input_dir, max_tiles_per_zip)

    # Group entries by zip for efficient sequential reads
    from collections import defaultdict
    zip_groups: Dict[Path, List[str]] = defaultdict(list)
    for zp, name in tile_entries:
        zip_groups[zp].append(name)

    total = 0
    for zp in tqdm(sorted(zip_groups), desc="ZIP archives", unit="zip"):
        names = zip_groups[zp]
        pid = _patient_id_from_zip(zp)

        # Determine output ZIP path: per-patient ZIPs or a single aggregate ZIP
        if per_patient_dirs:
            zip_name = f"{pid}.zip"
        else:
            zip_name = f"{output_dir.name}.zip"
        zip_path = output_dir / zip_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Open input ZIP and output ZIP for this patient
        with zipfile.ZipFile(zp, "r") as in_zf, \
             zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as out_zf:
            for tile_name in tqdm(names, desc=pid, leave=False, unit="tile"):
                try:
                    with in_zf.open(tile_name) as f:
                        img = Image.open(BytesIO(f.read())).convert("RGB")
                    arr = pil_to_numpy(img)

                    seg_map, cls_maps = segmenter.predict(arr)

                    # Save classification maps (H, W, C) – TopoFD compatible
                    cls_fname = _make_output_name(tile_name, "_cls")
                    bio = BytesIO()
                    np.save(bio, cls_maps)
                    out_zf.writestr(cls_fname, bio.getvalue())

                    # Optionally save nuclei segmentation
                    if save_seg:
                        seg_fname = _make_output_name(tile_name, "_seg")
                        bio2 = BytesIO()
                        np.save(bio2, seg_map)
                        out_zf.writestr(seg_fname, bio2.getvalue())

                    total += 1

                except Exception:
                    logger.exception("Failed on tile %s in %s", tile_name, zp)

    return total


def process_tiles_from_images(
    input_dir: Path,
    output_dir: Path,
    segmenter: DeepCMorphSegmenter,
    *,
    save_seg: bool = True,
) -> int:
    """Segment / classify tiles stored as loose images in a directory.

    Returns the number of tiles processed.
    """
    tile_paths = discover_tiles_from_dir(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # For flat images we create a single ZIP containing all outputs
    zip_name = f"{Path(input_dir).stem}.zip"
    zip_path = output_dir / zip_name
    total = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as out_zf:
        for tp in tqdm(tile_paths, desc="Tiles", unit="tile"):
            try:
                img = load_tile_from_path(tp)
                arr = pil_to_numpy(img)

                seg_map, cls_maps = segmenter.predict(arr)

                cls_fname = _make_output_name(tp.name, "_cls")
                bio = BytesIO()
                np.save(bio, cls_maps)
                out_zf.writestr(cls_fname, bio.getvalue())

                if save_seg:
                    seg_fname = _make_output_name(tp.name, "_seg")
                    bio2 = BytesIO()
                    np.save(bio2, seg_map)
                    out_zf.writestr(seg_fname, bio2.getvalue())

                total += 1

            except Exception:
                logger.exception("Failed on tile %s", tp)

    return total


# ===================================================================
# § 6  CLI
# ===================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cell segmentation & classification with DeepCMorph.  "
                    "Produces per-tile .npy masks suitable for TopoFD evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing tile images (ZIPs or loose images).",
    )
    p.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory where .npy output masks are saved.",
    )
    p.add_argument(
        "--input-format", type=str, choices=["zip", "images"], default="zip",
        help="How tiles are stored: 'zip' (STAMP archives) or 'images' (flat dir).",
    )
    p.add_argument(
        "--num-classes", type=int, default=32,
        help="Number of cell-type classes for DeepCMorph (default: 32 for TCGA).",
    )
    p.add_argument(
        "--weights-dataset", type=str, default="TCGA",
        help="Pre-trained weight set to load (default: 'TCGA').",
    )
    p.add_argument(
        "--checkpoint-path", type=str, default=None,
        help="Optional explicit path to a DeepCMorph .pth checkpoint file.",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Device for inference ('cuda' or 'cpu'). Auto-detected if omitted.",
    )
    p.add_argument(
        "--max-tiles-per-zip", type=int, default=None,
        help="Maximum number of tiles to process per ZIP archive.",
    )
    p.add_argument(
        "--grouping-json", type=str, default=None,
        help="Path to a JSON file mapping group names to lists of original "
             "class indices (0-based).  Used to reduce the 32 TCGA classes "
             "into fewer categories.",
    )
    p.add_argument(
        "--no-seg", action="store_true",
        help="Do not save the global nuclei segmentation mask.",
    )
    p.add_argument(
        "--flat-output", action="store_true",
        help="Save all output files in a single directory (no per-patient sub-dirs).",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose / debug logging.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Device auto-detection
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Grouping
    grouping: Optional[Dict[str, List[int]]] = None
    if args.grouping_json:
        with open(args.grouping_json, "r") as f:
            grouping = json.load(f)
        n_groups = len(grouping) if grouping else 0
        logger.info(
            "Loaded cell-type grouping with %d groups from %s",
            n_groups, args.grouping_json,
        )

    # Build segmenter
    segmenter = DeepCMorphSegmenter(
        num_classes=args.num_classes,
        weights_dataset=args.weights_dataset,
        device=device,
        grouping=grouping,
        checkpoint_path=args.checkpoint_path,
    )

    # Run
    if args.input_format == "zip":
        n = process_tiles_from_zips(
            input_dir=input_dir,
            output_dir=output_dir,
            segmenter=segmenter,
            max_tiles_per_zip=args.max_tiles_per_zip,
            save_seg=not args.no_seg,
            per_patient_dirs=not args.flat_output,
        )
    else:
        n = process_tiles_from_images(
            input_dir=input_dir,
            output_dir=output_dir,
            segmenter=segmenter,
            save_seg=not args.no_seg,
        )

    logger.info("Done – processed %d tiles.  Output in %s", n, output_dir)


if __name__ == "__main__":
    main()
