#!/usr/bin/env python3
"""
Visualization of segmentation results for reference and synthetic tiles.

Compares original tiles with their segmentation masks (nuclei + class maps)
against reconstructed tiles with their segmentation masks.
"""

from __future__ import annotations

import logging
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap, hsv_to_rgb
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from .core import setup_style, save_figure
    _HAS_CORE = True
except ImportError:
    try:
        from visualization.core import setup_style, save_figure
        _HAS_CORE = True
    except ImportError:
        _HAS_CORE = False

        def setup_style() -> None:  # type: ignore[misc]
            import matplotlib as _mpl
            _mpl.rcParams.update({
                "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
                "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
                "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
            })

        def save_figure(fig, save_path, close: bool = True, **kw) -> None:  # type: ignore[misc]
            import matplotlib.pyplot as _plt
            from pathlib import Path as _Path
            p = _Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, bbox_inches="tight", facecolor="white", edgecolor="none", dpi=300)
            print(f"[OK] Saved figure → {p}")
            if close:
                _plt.close(fig)


def _extract_tile_and_masks(
    tile_zip_path: Path,
    tile_name: str,
    mask_zip_path: Path,
    num_classes: int = 32,
) -> Tuple[Optional[Image.Image], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract a tile image and its corresponding segmentation masks from ZIP archives.
    
    Parameters
    ----------
    tile_zip_path : Path
        Path to ZIP archive containing tile images.
    tile_name : str
        Filename of the tile (e.g., "tile_(x, y).png").
    mask_zip_path : Path
        Path to ZIP archive containing segmentation masks.
    num_classes : int
        Number of cell-type classes.
    
    Returns
    -------
    tile : PIL.Image or None
        The tile image.
    seg_mask : np.ndarray or None
        Nuclei segmentation mask (H, W).
    cls_maps : np.ndarray or None
        Class maps (H, W, C) where C = num_classes.
    """
    tile_img = None
    seg_mask = None
    cls_maps = None
    
    # Extract tile image
    try:
        with zipfile.ZipFile(tile_zip_path, 'r') as zf:
            if tile_name in zf.namelist():
                tile_data = zf.read(tile_name)
                tile_img = Image.open(BytesIO(tile_data))
    except Exception as e:
        logger.warning(f"Failed to extract tile {tile_name} from {tile_zip_path.name}: {e}")
        return None, None, None
    
    # Extract masks from corresponding mask ZIP
    try:
        with zipfile.ZipFile(mask_zip_path, 'r') as zf:
            # Find corresponding masked files
            # e.g., "tile_(x, y).png" -> look for "tile_(x, y)_seg.npy" and "tile_(x, y)_cls.npy"
            tile_stem = Path(tile_name).stem  # Remove .png extension
            
            seg_name = f"{tile_stem}_seg.npy"
            cls_name = f"{tile_stem}_cls.npy"
            
            # Try to find these files in the ZIP
            npy_files = [f for f in zf.namelist() if f.endswith('.npy')]
            
            # Match by tile_stem (handling various naming conventions)
            for npy_file in npy_files:
                if seg_name in npy_file or (tile_stem in npy_file and "_seg" in npy_file):
                    try:
                        seg_data = zf.read(npy_file)
                        seg_mask = np.load(BytesIO(seg_data))
                        
                        # Handle axis ordering: if shape is (C, H, W) and C=1, reshape to (H, W)
                        if seg_mask.ndim == 3 and seg_mask.shape[0] == 1:
                            seg_mask = np.squeeze(seg_mask, axis=0)
                    except Exception as e:
                        logger.warning(f"Failed to load segmentation mask {npy_file}: {e}")
                
                if cls_name in npy_file or (tile_stem in npy_file and "_cls" in npy_file):
                    try:
                        cls_data = zf.read(npy_file)
                        cls_maps = np.load(BytesIO(cls_data))
                        
                        # Handle axis ordering: if shape is (C, H, W), transpose to (H, W, C)
                        if cls_maps.ndim == 3:
                            # Check if first axis is small (likely channels), and other axes are 512
                            if cls_maps.shape[0] < cls_maps.shape[1]:
                                cls_maps = np.transpose(cls_maps, (1, 2, 0))
                        
                        # Now truncate or pad to match num_classes
                        if cls_maps.shape[-1] != num_classes:
                            if cls_maps.ndim == 3:
                                if cls_maps.shape[-1] < num_classes:
                                    pad = num_classes - cls_maps.shape[-1]
                                    cls_maps = np.pad(cls_maps, ((0, 0), (0, 0), (0, pad)))
                                else:
                                    cls_maps = cls_maps[..., :num_classes]
                    except Exception as e:
                        logger.warning(f"Failed to load class maps {npy_file}: {e}")
    except Exception as e:
        logger.warning(f"Failed to extract masks from {mask_zip_path.name}: {e}")
    
    return tile_img, seg_mask, cls_maps


def _colorize_class_map(cls_maps: np.ndarray, num_classes: int = 32, seg_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Convert class maps (H, W, C) into a colored RGB image (H, W, 3).

    Each channel is treated as a binary mask for a cell type. Any pixel
    where *all* channels are zero is considered background and rendered black.
    
    Uses HSV color space to generate vibrant, maximally distinct colors for
    each cell type: black background, then colors with varying hues.

    Notes
    -----
    If the input is a soft map (not strictly binary), this uses argmax to
    pick the strongest class per pixel, but only where the max value is > 0.
    """
    # Validate input shape: should be (H, W, C)
    if cls_maps.ndim != 3:
        # Try to reshape or pad
        if cls_maps.ndim == 2:
            cls_maps = np.expand_dims(cls_maps, axis=-1)
        else:
            raise ValueError(f"Unexpected class maps shape: {cls_maps.shape}")
    
    # Ensure we have the expected number of classes
    if cls_maps.shape[2] != num_classes:
        if cls_maps.shape[2] < num_classes:
            pad = num_classes - cls_maps.shape[2]
            cls_maps = np.pad(cls_maps, ((0, 0), (0, 0), (0, pad)))
        else:
            cls_maps = cls_maps[..., :num_classes]
    
    H, W = cls_maps.shape[:2]

    # Determine per-pixel label (0..C-1), with -1 indicating background
    max_vals = np.max(cls_maps, axis=2)
    class_idx = np.argmax(cls_maps, axis=2)
    
    if seg_mask is not None:
        # Most reliable method: use the explicit nuclei mask to set background to black
        seg_2d = np.squeeze(seg_mask)
        # Handle both [0, 1] probability masks and [0, 255] masks by using a dynamic threshold
        threshold = 0.5 if seg_2d.max() <= 1.0 else 127.5
        class_idx[seg_2d < threshold] = -1
    else:
        # Fallback if no mask is provided
        class_idx[max_vals < 0.1] = -1

    # Generate vibrant, maximally distinct neon colors
    colors = [(0.0, 0.0, 0.0)]  # Index 0: black (background)
    
    # Use golden ratio to randomly distribute hues so sequentially numbered classes 
    # (like class 1, 2, 3) get completely different colors instead of similar ones.
    golden_ratio = 0.618033988749895
    
    for i in range(num_classes):
        # Multiply by golden ratio to disperse colors widely over the color wheel
        hue = (i * golden_ratio) % 1.0
        saturation = 1.0  # 100% saturation for neon
        value = 1.0       # 100% brightness for neon
        
        # Convert HSV to RGB
        rgb = hsv_to_rgb([hue, saturation, value])
        colors.append(tuple(rgb))
    
    cmap = ListedColormap(colors)

    # Map label -> RGB (background is index 0)
    colored = cmap(class_idx + 1)[:, :, :3]
    return colored


def _format_nuclei_segmentation(
    seg_mask: np.ndarray,
) -> np.ndarray:
    """
    Convert binary nuclei segmentation mask to displayable image.
    
    Per DeepCMorph paper: nuclei segmentation shows background as WHITE,
    nuclei as BLACK. This creates a clear visual representation.
    
    Parameters
    ----------
    seg_mask : np.ndarray, shape (H, W) or (H, W, 1)
        Binary segmentation mask where 0=background, 1=nuclei.
    
    Returns
    -------
    segmentation_display : np.ndarray, shape (H, W, 3)
        RGB image where 0 (background) -> WHITE (255),
        1 (nuclei) -> BLACK (0).
    """
    # Ensure segmentation mask is 2D
    seg_mask = np.asarray(seg_mask)
    if seg_mask.ndim > 2:
        seg_mask = np.squeeze(seg_mask)
    
    if seg_mask.ndim != 2:
        raise ValueError(f"After squeezing, seg_mask should be 2D but got shape {seg_mask.shape}")
    
    # Invert: background (0) -> 255 (white), nuclei (1) -> 0 (black)
    seg_inverted = (1 - seg_mask.astype(np.float32)) * 255.0
    
    # Convert to RGB (replicate across channels)
    segmentation_rgb = np.stack([seg_inverted] * 3, axis=-1).astype(np.uint8)
    
    return segmentation_rgb.astype(np.float32) / 255.0


def visualize_segmentation_results(
    reference_tiles_dir: str | Path,
    reference_masks_dir: str | Path,
    synthetic_tiles_dir: str | Path,
    synthetic_masks_dir: str | Path,
    output_dir: str | Path,
    num_classes: int = 32,
    num_samples: int = 1,
    figsize: Tuple[int, int] = (16, 12),
    verbose: bool = True,
) -> None:
    """
    Create comparison visualizations of reference vs synthetic segmentation.
    
    Parameters
    ----------
    reference_tiles_dir, synthetic_tiles_dir : path-like
        Directories containing tile ZIP archives.
    reference_masks_dir, synthetic_masks_dir : path-like
        Directories containing segmentation mask ZIP archives.
    output_dir : path-like
        Where to save visualization images.
    num_classes : int
        Number of cell-type classes (default: 32 for TCGA).
    num_samples : int
        Number of sample tiles to visualize per patient (default: 1).
    figsize : tuple
        Figure size for the visualization grid.
    verbose : bool
        Whether to print progress information.
    """
    ref_tiles_dir = Path(reference_tiles_dir)
    ref_masks_dir = Path(reference_masks_dir)
    syn_tiles_dir = Path(synthetic_tiles_dir)
    syn_masks_dir = Path(synthetic_masks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    # Get patient IDs from tile ZIPs
    ref_tile_zips = sorted(ref_tiles_dir.glob("*.zip"))
    syn_tile_zips = sorted(syn_tiles_dir.glob("*.zip"))
    
    if not ref_tile_zips or not syn_tile_zips:
        logger.warning("No ZIP files found in tile directories")
        return
    
    # Map patient IDs to ZIP paths
    from joint_training.dataset import canonical_patient_id
    
    ref_tile_map = {canonical_patient_id(z.name): z for z in ref_tile_zips}
    syn_tile_map = {canonical_patient_id(z.name): z for z in syn_tile_zips}
    
    ref_mask_map = {canonical_patient_id(z.name): z for z in sorted(ref_masks_dir.glob("*.zip"))}
    syn_mask_map = {canonical_patient_id(z.name): z for z in sorted(syn_masks_dir.glob("*.zip"))}
    
    # Find common patients
    common_patients = set(ref_tile_map.keys()) & set(syn_tile_map.keys())
    common_patients = common_patients & set(ref_mask_map.keys()) & set(syn_mask_map.keys())
    
    if not common_patients:
        logger.warning("No common patients found across tile and mask directories")
        return
    
    if verbose:
        print(f"\n[INFO] Creating segmentation visualizations for {len(common_patients)} patients...")
    
    # Collect all samples to visualize
    all_samples = []
    
    for patient_id in sorted(common_patients):
        ref_tile_zip = ref_tile_map[patient_id]
        ref_mask_zip = ref_mask_map[patient_id]
        syn_tile_zip = syn_tile_map[patient_id]
        syn_mask_zip = syn_mask_map[patient_id]
        
        # Extract tile names from reference ZIP
        try:
            with zipfile.ZipFile(ref_tile_zip, 'r') as zf:
                tile_names = [f for f in zf.namelist() if f.endswith(('.png', '.jpg', '.jpeg'))]
                # Sample up to num_samples tiles
                sampled_tiles = tile_names[:num_samples]
        except Exception as e:
            logger.warning(f"Failed to enumerate tiles in {ref_tile_zip.name}: {e}")
            continue
        
        for tile_name in sampled_tiles:
            all_samples.append({
                'patient_id': patient_id,
                'tile_name': tile_name,
                'ref_tile_zip': ref_tile_zip,
                'ref_mask_zip': ref_mask_zip,
                'syn_tile_zip': syn_tile_zip,
                'syn_mask_zip': syn_mask_zip,
            })
    
    if not all_samples:
        logger.warning("No samples to visualize")
        return
    
    # Create grid visualization
    n_samples = len(all_samples)
    n_cols = 6  # [ref_tile | ref_seg | ref_cls | syn_tile | syn_seg | syn_cls]
    n_rows = n_samples
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize[0], max(figsize[1], 2.5 * n_rows)),
        squeeze=False,
    )
    fig.suptitle(
        "Segmentation Comparison: Reference vs Synthetic",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for row, sample in enumerate(all_samples):
        patient_id = sample['patient_id']
        tile_name = sample['tile_name']
        
        # Extract reference data
        ref_tile, ref_seg, ref_cls = _extract_tile_and_masks(
            sample['ref_tile_zip'],
            tile_name,
            sample['ref_mask_zip'],
            num_classes=num_classes,
        )
        
        # Extract synthetic data
        syn_tile, syn_seg, syn_cls = _extract_tile_and_masks(
            sample['syn_tile_zip'],
            tile_name,
            sample['syn_mask_zip'],
            num_classes=num_classes,
        )
        
        if ref_tile is None or syn_tile is None:
            logger.warning(f"Failed to load sample {patient_id}/{tile_name}, skipping")
            continue
        
        # --- Reference tiles ---
        # Col 0: Reference tile
        axes[row, 0].imshow(ref_tile)
        axes[row, 0].set_title(f"{patient_id}\n{tile_name}", fontsize=8)
        axes[row, 0].axis('off')
        
        # Col 1: Reference segmentation (binary mask)
        if ref_seg is not None:
            seg_display = _format_nuclei_segmentation(ref_seg)
            axes[row, 1].imshow(seg_display)
            axes[row, 1].set_title("Ref Seg", fontsize=8)
        axes[row, 1].axis('off')
        
        # Col 2: Reference class map
        if ref_cls is not None:
            colored_cls = _colorize_class_map(ref_cls, num_classes=num_classes, seg_mask=ref_seg)
            if colored_cls.ndim == 3 and colored_cls.shape[0] > 1 and colored_cls.shape[1] > 1:
                axes[row, 2].imshow(colored_cls)
                axes[row, 2].set_title("Ref Classes", fontsize=8)
            else:
                axes[row, 2].text(0.5, 0.5, f"Invalid shape", 
                                 ha='center', va='center', transform=axes[row, 2].transAxes)
                logger.warning(f"ref_cls colorization produced invalid shape: {colored_cls.shape}")
        axes[row, 2].axis('off')
        
        # --- Synthetic tiles ---
        # Col 3: Synthetic tile
        axes[row, 3].imshow(syn_tile)
        axes[row, 3].set_title("Synthetic", fontsize=8)
        axes[row, 3].axis('off')
        
        # Col 4: Synthetic segmentation (binary mask)
        if syn_seg is not None:
            seg_display = _format_nuclei_segmentation(syn_seg)
            axes[row, 4].imshow(seg_display)
            axes[row, 4].set_title("Syn Seg", fontsize=8)
        axes[row, 4].axis('off')
        
        # Col 5: Synthetic class map
        if syn_cls is not None:
            colored_cls = _colorize_class_map(syn_cls, num_classes=num_classes, seg_mask=syn_seg)
            if colored_cls.ndim == 3 and colored_cls.shape[0] > 1 and colored_cls.shape[1] > 1:
                axes[row, 5].imshow(colored_cls)
                axes[row, 5].set_title("Syn Classes", fontsize=8)
            else:
                axes[row, 5].text(0.5, 0.5, f"Invalid shape", 
                                 ha='center', va='center', transform=axes[row, 5].transAxes)
                logger.warning(f"syn_cls colorization produced invalid shape: {colored_cls.shape}")
        axes[row, 5].axis('off')
    
    # Add column headers (use rcParam axes.titlesize via fontsize=None → default)
    col_titles = ["Reference\nTile", "Reference\nSegmentation", "Reference\nClasses",
                  "Synthetic\nTile", "Synthetic\nSegmentation", "Synthetic\nClasses"]
    for col, title in enumerate(col_titles):
        axes[0, col].text(0.5, 1.12, title, ha='center', va='bottom', fontweight='bold',
                          transform=axes[0, col].transAxes)
    
    plt.tight_layout()

    output_path = output_dir / "segmentation_comparison_grid.png"
    save_figure(fig, output_path, close=True)
    if verbose:
        print(f"[OK] Saved visualization to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize segmentation results for reference vs synthetic tiles."
    )
    parser.add_argument("--reference-tiles-dir", type=str, required=True)
    parser.add_argument("--reference-masks-dir", type=str, required=True)
    parser.add_argument("--synthetic-tiles-dir", type=str, required=True)
    parser.add_argument("--synthetic-masks-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-classes", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    visualize_segmentation_results(
        reference_tiles_dir=args.reference_tiles_dir,
        reference_masks_dir=args.reference_masks_dir,
        synthetic_tiles_dir=args.synthetic_tiles_dir,
        synthetic_masks_dir=args.synthetic_masks_dir,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        num_samples=args.num_samples,
        verbose=args.verbose,
    )
