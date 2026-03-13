#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Segmentation and Topological Fréchet Distance Pipeline

Orchestrates the complete pipeline:
  1. Discover tiles present in synthetic (reconstructed) set
  2. Segment ONLY the matching reference tiles
  3. Segment synthetic tiles
  4. Compute Topological Fréchet Distance between the two segmentation sets

This ensures we compare apples-to-apples: only tiles that exist in both
reference and synthetic sets are evaluated.

Supports config-file driven execution via YAML for flexible pipeline control.

Input: ZIP archives or flat directories
Output: ZIP archives of segmentation masks + TopoFD summary
"""

import os
import sys
import json
import logging
import zipfile
import yaml
from pathlib import Path
from typing import List, Set, Optional, Dict, Tuple, Any
import argparse
import subprocess
from collections import defaultdict

from .utils import extract_patient_id
from .topological_frechet_distance import compute_topofd_from_folders

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def discover_tiles_in_zip(zip_path: Path) -> Set[Tuple[str, str]]:
    """
    Discover all image tiles in a ZIP archive.
    
    Returns set of (patient_id, tile_basename) tuples.
    Patient ID is extracted from the ZIP filename (TCGA-XX-XXXX).
    """
    tiles = set()
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    # Extract patient ID from zip name
                    patient_id = extract_patient_id(zip_path.stem)
                    tile_basename = Path(name).name
                    tiles.add((patient_id, tile_basename))
    except Exception as e:
        logger.warning(f"Could not read ZIP {zip_path}: {e}")
    return tiles


def discover_tiles_in_dir(dir_path: Path) -> Set[Tuple[str, str]]:
    """
    Discover all image tiles in a flat directory.
    
    Returns set of (patient_id, tile_basename) tuples.
    For flat dirs, patient_id is derived from each file's patient marker or set to 'unknown'.
    """
    tiles = set()
    for file_path in dir_path.iterdir():
        if file_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}:
            patient_id = extract_patient_id(file_path.name)
            tiles.add((patient_id, file_path.name))
    return tiles


try:
    from .utils import extract_patient_id
except Exception:
    from quality_assurance.utils import extract_patient_id


def discover_synthetic_tiles(synthetic_dir: Path) -> Set[Tuple[str, str]]:
    """
    Discover all tiles in synthetic (reconstructed) directory.
    
    Handles both ZIP archives and flat directories.
    """
    synthetic_tiles = set()
    
    if not synthetic_dir.exists():
        raise FileNotFoundError(f"Synthetic directory not found: {synthetic_dir}")
    
    # Check if it contains ZIPs
    zip_files = list(synthetic_dir.glob('*.zip'))
    if zip_files:
        logger.info(f"Discovering synthetic tiles from {len(zip_files)} ZIP archives")
        for zip_path in zip_files:
            synthetic_tiles.update(discover_tiles_in_zip(zip_path))
    else:
        # Assume flat directory
        logger.info(f"Discovering synthetic tiles from flat directory")
        synthetic_tiles.update(discover_tiles_in_dir(synthetic_dir))
    
    logger.info(f"Found {len(synthetic_tiles)} tiles in synthetic set")
    return synthetic_tiles


def filter_reference_zips(
    reference_dir: Path,
    synthetic_tiles: Set[Tuple[str, str]],
) -> Dict[Path, List[str]]:
    """
    Filter reference ZIP archives to only include those patients present in the
    synthetic set.  We do not prune individual tiles because the reconstructed
    filenames differ; instead we match on patient ID alone.

    Returns mapping: zip_path -> list of all tile names (unfiltered).  Only
    ZIPs whose patient appears in ``synthetic_tiles`` are returned.
    """
    filtered = {}
    
    # derive set of patient IDs present in synthetic tiles (lowercased)
    patient_ids = {pid.lower() for (pid, _) in synthetic_tiles}
    logger.info(f"Synthetic patients: {sorted(patient_ids)}")
    
    zip_files = sorted(reference_dir.glob('*.zip'))
    logger.info(f"Filtering {len(zip_files)} reference ZIPs against synthetic patients")
    
    for zip_path in zip_files:
        patient_id = extract_patient_id(zip_path.stem)
        if patient_id.lower() in patient_ids:
            # keep entire archive unmodified
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    names = [n for n in zf.namelist()
                             if n.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
                if names:
                    filtered[zip_path] = names
                    logger.info(f"  keeping {zip_path.name} ({len(names)} tiles)")
            except Exception as e:
                logger.warning(f"Could not read ZIP {zip_path}: {e}")
    
    logger.info(f"Keeping {len(filtered)} reference ZIPs matching synthetic patients")
    return filtered


def create_filtered_reference_zip(
    input_zip: Path,
    output_dir: Path,
    tile_names: List[str],
) -> Path:
    """
    Create a new ZIP containing only the specified tiles from input_zip.
    
    Output ZIP is placed in output_dir with the same basename as input.
    """
    output_zip = output_dir / input_zip.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating filtered ZIP: {output_zip.name}")
    
    with zipfile.ZipFile(input_zip, 'r') as zf_in:
        with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf_out:
            for tile_name in tile_names:
                try:
                    data = zf_in.read(tile_name)
                    zf_out.writestr(tile_name, data)
                except Exception as e:
                    logger.warning(f"Could not copy {tile_name}: {e}")
    
    return output_zip


def run_segmentation(
    input_dir: Path,
    output_dir: Path,
    num_classes: int = 32,
    device: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> None:
    """
    Run DeepCMorph segmentation on tiles in input_dir.
    
    Assumes input_dir contains ZIP archives.

    The ``batch_size`` and ``num_workers`` parameters are accepted for
    compatibility with the configuration schema, but the underlying
    ``segment_and_classify_cells`` script currently does not expose
    corresponding command‑line options.  They are therefore ignored here.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # skip if there are no ZIP files to process
    zip_files = list(input_dir.glob('*.zip'))
    if not zip_files:
        logger.warning(f"No ZIP files found in {input_dir}; skipping segmentation")
        return
    
    logger.info(f"Running segmentation on {input_dir} ({len(zip_files)} archives)")
    # warn about ignored parameters
    if batch_size is not None or num_workers is not None:
        logger.debug(
            "batch_size=%s num_workers=%s provided but will be ignored",
            batch_size, num_workers,
        )
    
    # If no explicit checkpoint path given, search known locations
    if checkpoint_path is None:
        workspace_root = Path(__file__).resolve().parents[3]
        fname = "DeepCMorph_Pan_Cancer_32_classes_acc_8273.pth"
        search_dirs = [
            workspace_root / "DeepCMorph" / "pretrained_models",
            workspace_root / "models" / "DeepCMorph",
        ]
        for search_dir in search_dirs:
            candidate = search_dir / fname
            if candidate.exists():
                checkpoint_path = str(candidate)
                logger.info(f"Resolved TCGA checkpoint to {checkpoint_path}")
                break
        else:
            logger.warning(
                "Could not find TCGA checkpoint in any of %s; "
                "DeepCMorph will attempt to load with relative path",
                [str(d) for d in search_dirs],
            )
    
    cmd = [
        'python', '-m', 'src.classifier.segment_and_classify_cells',
        '--input-dir', str(input_dir),
        '--output-dir', str(output_dir),
        '--input-format', 'zip',
        '--num-classes', str(num_classes),
        '--weights-dataset', 'TCGA',
    ]
    
    if device is not None:
        cmd.extend(['--device', device])
    
    if checkpoint_path is not None:
        cmd.extend(['--checkpoint-path', checkpoint_path])
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
    
    if result.returncode != 0:
        raise RuntimeError(f"Segmentation failed with exit code {result.returncode}")
    
    logger.info(f"Segmentation complete. Output in {output_dir}")


def extract_masks_from_zips(masks_dir: Path) -> int:
    """
    Extract all .npy mask files from ZIP archives in the directory.
    
    If masks are already extracted, this is a no-op. If ZIPs exist,
    extract all .npy files FLATTENED to the root of masks_dir (no nested subdirs).
    This avoids memory overhead from nested directory traversal.
    
    Parameters
    ----------
    masks_dir : Path
        Directory potentially containing both ZIP archives and/or extracted .npy files
        
    Returns
    -------
    int
        Number of .npy files found after extraction (flattened in root)
    """
    masks_dir = Path(masks_dir)
    
    # Check if we already have .npy files at root level (flat structure)
    npy_files = list(masks_dir.glob('*.npy'))
    if npy_files:
        logger.info(f"Found {len(npy_files)} pre-extracted .npy files (flattened) in {masks_dir}")
        return len(npy_files)
    
    # Check for ZIP files
    zip_files = list(masks_dir.glob('*.zip'))
    if not zip_files:
        logger.warning(f"No .npy files or .zip archives found in {masks_dir}")
        return 0
    
    logger.info(f"Extracting masks from {len(zip_files)} ZIP archives (flattening to root)...")
    extracted_count = 0
    for zip_path in zip_files:
        try:
            logger.debug(f"  Extracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Extract only .npy files, flatten to root
                for member in zf.namelist():
                    if member.lower().endswith('.npy'):
                        # Extract to root with only the basename (no nested dirs)
                        data = zf.read(member)
                        output_path = masks_dir / Path(member).name
                        output_path.write_bytes(data)
                        extracted_count += 1
        except Exception as e:
            logger.warning(f"Failed to extract {zip_path.name}: {e}")
    
    logger.info(f"Extracted {extracted_count} .npy files from ZIPs (flattened to root)")
    return extracted_count


def run_topofd(
    reference_masks_dir: Path,
    synthetic_masks_dir: Path,
    output_dir: Path,
    n_bins: int = 100,
    n_layers: int = 1,
) -> None:
    """
    Run Topological Fréchet Distance computation.
    
    Handles both extracted .npy files and ZIP archives. If ZIPs are found,
    they are extracted first.
    
    Parameters
    ----------
    reference_masks_dir : Path
        Directory containing reference segmentation masks (.npy files or ZIPs)
    synthetic_masks_dir : Path
        Directory containing synthetic segmentation masks (.npy files or ZIPs)
    output_dir : Path
        Directory where TopoFD results JSON will be saved
    n_bins : int
        Persistence landscape bins (default: 100)
    n_layers : int
        Persistence landscape layers (default: 1)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Computing TopoFD between mask sets")
    logger.info(f"  Reference masks: {reference_masks_dir}")
    logger.info(f"  Synthetic masks: {synthetic_masks_dir}")
    logger.info(f"  Output directory: {output_dir}")
    
    # Extract masks from ZIPs if needed
    logger.info("\nPreparing reference masks...")
    ref_count = extract_masks_from_zips(reference_masks_dir)
    
    logger.info("\nPreparing synthetic masks...")
    syn_count = extract_masks_from_zips(synthetic_masks_dir)
    
    if ref_count == 0 or syn_count == 0:
        raise FileNotFoundError(
            f"No mask files found after extraction:\n"
            f"  Reference: {ref_count} files\n"
            f"  Synthetic: {syn_count} files"
        )
    
    # Compute TopoFD directly (no subprocess)
    logger.info(f"\nComputing Topological Fréchet Distance...")
    result = compute_topofd_from_folders(
        reference_dir=reference_masks_dir,
        generated_dir=synthetic_masks_dir,
        n_landscape_bins=n_bins,
        n_landscape_layers=n_layers,
        verbose=True,
    )
    
    # Save results to a JSON file in output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "topofd_results.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "topofd": float(result.topofd),
            "per_channel": {str(k): float(v) for k, v in result.per_channel.items()},
            "n_reference": result.n_reference,
            "n_generated": result.n_generated,
            "n_channels": result.n_channels,
        }, f, indent=2)
    
    logger.info(f"✅ TopoFD computation complete. Results saved to {output_file}")


def main(
    reference_zips_dir: Path,
    synthetic_tiles_dir: Path,
    reference_masks_output: Path,
    synthetic_masks_output: Path,
    topofd_output: Path,
    num_classes: int = 32,
    batch_size: int = 4,
    num_workers: int = 4,
    device: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    n_landscape_bins: int = 100,
    n_landscape_layers: int = 1,
    skip_segmentation: bool = False,
) -> None:
    """
    Main orchestration function.
    
    Parameters
    ----------
    reference_zips_dir : Path
        Directory containing reference tile ZIP archives
    synthetic_tiles_dir : Path
        Directory containing synthetic tiles (ZIPs or flat directory)
    reference_masks_output : Path
        Output directory for reference segmentation masks
    synthetic_masks_output : Path
        Output directory for synthetic segmentation masks
    topofd_output : Path
        Output directory for TopoFD results
    """
    
    logger.info("=" * 80)
    logger.info("Segmentation + TopoFD Pipeline")
    logger.info("=" * 80)
    logger.info(f"Reference ZIPs:      {reference_zips_dir}")
    logger.info(f"Synthetic tiles:     {synthetic_tiles_dir}")
    logger.info(f"Reference masks:     {reference_masks_output}")
    logger.info(f"Synthetic masks:     {synthetic_masks_output}")
    logger.info(f"TopoFD output:       {topofd_output}")
    
    # Step 1: Discover synthetic tiles
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Discovering synthetic tiles")
    logger.info("=" * 80)
    synthetic_tiles = discover_synthetic_tiles(synthetic_tiles_dir)
    logger.info(f"✅ Found {len(synthetic_tiles)} synthetic tiles")
    
    # Step 2: Filter reference ZIPs to only matching tiles
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Filtering reference ZIPs to matching tiles")
    logger.info("=" * 80)
    filtered_reference = filter_reference_zips(reference_zips_dir, synthetic_tiles)
    logger.info(f"✅ Filtered to {len(filtered_reference)} reference ZIPs")
    
    # Create filtered reference directory
    filtered_ref_dir = reference_zips_dir.parent / f"{reference_zips_dir.name}_filtered_matching"
    filtered_ref_dir.mkdir(parents=True, exist_ok=True)
    
    for zip_path, tile_names in filtered_reference.items():
        create_filtered_reference_zip(zip_path, filtered_ref_dir, tile_names)
    
    logger.info(f"✅ Created filtered reference ZIPs in {filtered_ref_dir}")
    
    # Step 3: Segment reference tiles
    if not skip_segmentation:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Segmenting reference tiles")
        logger.info("=" * 80)
        # ``batch_size`` and ``num_workers`` are kept in the call signature
        # for backwards compatibility with our config file, even though the
        # downstream script does not currently accept these options.
        run_segmentation(
            filtered_ref_dir,
            reference_masks_output,
            num_classes=num_classes,
            device=device,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        logger.info(f"✅ Reference segmentation complete")
        
        # Step 4: Segment synthetic tiles
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Segmenting synthetic tiles")
        logger.info("=" * 80)
        run_segmentation(
            synthetic_tiles_dir,
            synthetic_masks_output,
            num_classes=num_classes,
            device=device,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        logger.info(f"✅ Synthetic segmentation complete")
    
    # Step 5: Compute TopoFD
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Computing Topological Fréchet Distance")
    logger.info("=" * 80)
    run_topofd(
        reference_masks_output,
        synthetic_masks_output,
        topofd_output,
        n_bins=n_landscape_bins,
        n_layers=n_landscape_layers,
    )
    logger.info(f"✅ TopoFD computation complete")
    
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to {topofd_output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Segment tiles and compute Topological Fréchet Distance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Full pipeline from config
  python src/quality_assurance/segment_and_compute_topofd.py --config src/config.yaml
  
  # Skip segmentation, only compute TopoFD (masks already exist)
  python src/quality_assurance/segment_and_compute_topofd.py --config src/config.yaml --skip-segmentation
  
  # Override specific settings from command-line
  python src/quality_assurance/segment_and_compute_topofd.py --config src/config.yaml --device cuda:0
        """,
    )
    p.add_argument(
        '--config', type=str, required=True,
        help='Path to YAML config file (required)',
    )
    p.add_argument(
        '--skip-segmentation', action='store_true',
        help='Skip segmentation and only compute TopoFD (masks must already exist)',
    )
    # Optional overrides for specific parameters
    p.add_argument(
        '--device', type=str, default=None,
        help='Device for inference (overrides config)',
    )
    p.add_argument(
        '--checkpoint-path', type=str, default=None,
        help='Optional explicit path to DeepCMorph checkpoint (overrides config)',
    )
    return p


def load_config_from_yaml(config_path: Optional[str]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str or None
        Path to YAML config file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    if config_path is None:
        return {}
    
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_file}")
        return {}
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded config from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_file}: {e}")
        return {}


def merge_config_with_args(yaml_config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Extract segmentation section from YAML config and merge with CLI arguments.
    Command-line args take precedence over YAML config.
    
    Parameters
    ----------
    yaml_config : dict
        Full configuration loaded from YAML
    args : argparse.Namespace
        Command-line arguments
        
    Returns
    -------
    dict
        Merged configuration (segmentation section)
    """
    # Extract segmentation section from config
    segmentation_config = yaml_config.get('segmentation', {})
    merged = segmentation_config.copy()
    
    # Ensure we have the required paths
    if 'reference_tiles_dir' not in merged:
        merged['reference_tiles_dir'] = None
    if 'synthetic_tiles_dir' not in merged:
        merged['synthetic_tiles_dir'] = None
    if 'reference_masks_dir' not in merged:
        merged['reference_masks_dir'] = None
    if 'synthetic_masks_dir' not in merged:
        merged['synthetic_masks_dir'] = None
    
    # Extract topofd section (nested under segmentation)
    topofd_config = merged.get('topofd', {})
    if 'topofd' not in merged:
        merged['topofd'] = {}
    
    # Override with CLI args if provided
    if args.device is not None:
        merged['device'] = args.device
    if args.checkpoint_path is not None:
        merged['checkpoint_path'] = args.checkpoint_path
    
    merged['skip_segmentation'] = args.skip_segmentation
    
    return merged


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    
    # Config is required
    if not args.config:
        parser.error("--config is required")
    
    # Load config from YAML
    yaml_config = load_config_from_yaml(args.config)
    if not yaml_config:
        logger.error("Failed to load config file")
        sys.exit(1)
    
    # Merge YAML config with CLI args
    config = merge_config_with_args(yaml_config, args)
    
    # Extract required fields from segmentation section
    reference_zips_dir = config.get('reference_tiles_dir')
    synthetic_tiles_dir = config.get('synthetic_tiles_dir')
    reference_masks_output = config.get('reference_masks_dir')
    synthetic_masks_output = config.get('synthetic_masks_dir')
    
    # Extract topofd config
    topofd_config = config.get('topofd', {})
    topofd_output = topofd_config.get('output_dir')
    
    if not reference_zips_dir or not synthetic_tiles_dir:
        logger.error(
            "Missing required configuration in segmentation section:\n"
            "  - reference_tiles_dir\n"
            "  - synthetic_tiles_dir"
        )
        sys.exit(1)
    
    if not reference_masks_output or not synthetic_masks_output:
        logger.error(
            "Missing required configuration in segmentation section:\n"
            "  - reference_masks_dir\n"
            "  - synthetic_masks_dir"
        )
        sys.exit(1)
    
    if not topofd_output:
        logger.error(
            "Missing required configuration:\n"
            "  - segmentation.topofd.output_dir"
        )
        sys.exit(1)
    
    # Get optional parameters with defaults
    num_classes = config.get('num_classes', 32)
    batch_size = config.get('batch_size', 4)
    num_workers = config.get('num_workers', 4)
    device = config.get('device')
    checkpoint_path = config.get('checkpoint_path')
    n_landscape_bins = topofd_config.get('n_landscape_bins', 100)
    n_landscape_layers = topofd_config.get('n_landscape_layers', 1)
    skip_segmentation = config.get('skip_segmentation', False)
    
    if skip_segmentation:
        logger.info("\n⏭️  SKIP-SEGMENTATION MODE: Masks must already exist at:")
        logger.info(f"   - {reference_masks_output}")
        logger.info(f"   - {synthetic_masks_output}\n")
    
    try:
        main(
            reference_zips_dir=Path(reference_zips_dir),
            synthetic_tiles_dir=Path(synthetic_tiles_dir),
            reference_masks_output=Path(reference_masks_output),
            synthetic_masks_output=Path(synthetic_masks_output),
            topofd_output=Path(topofd_output),
            num_classes=num_classes,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            checkpoint_path=checkpoint_path,
            n_landscape_bins=n_landscape_bins,
            n_landscape_layers=n_landscape_layers,
            skip_segmentation=skip_segmentation,
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
