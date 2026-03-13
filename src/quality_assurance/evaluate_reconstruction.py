#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reconstruction Quality Evaluation Script

This script evaluates the quality of diffusion model reconstructions by comparing
original tiles with their genomic-conditioned reconstructions.

Supports:
- Loading tiles from zip files (original and reconstructed)
- Matching tiles by filename
- Computing MSE, PSNR, SSIM metrics
- Generating summary statistics and visualizations
- Saving results to CSV and plots to specified directories

Usage:
    python evaluate_reconstruction.py \\
        --original-zip-dir /path/to/original_tiles \\
        --reconstructed-zip-dir /path/to/reconstructed_tiles \\
        --output-dir ./evaluation_results \\
        --plot-dir ./plots

    # For specific patients
    python evaluate_reconstruction.py \\
        --original-zip-dir /path/to/original_tiles \\
        --reconstructed-zip-dir /path/to/reconstructed_tiles \\
        --patient-ids TCGA-BH-A0AU TCGA-LL-A5YL \\
        --output-dir ./evaluation_results
"""

import argparse
import csv
import json
import re
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Any

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from .metrics import compute_all_metrics
    from .utils import extract_patient_id
except Exception:
    # Allow running the script directly (not as a package) by falling
    # back to absolute import when the relative import fails.
    from quality_assurance.metrics import compute_all_metrics
    from quality_assurance.utils import extract_patient_id
try:
    from visualization.reconstruction_eval import plot_metrics_summary, plot_per_patient_metrics, plot_comparison_grid, plot_metric_correlation, plot_single_comparison
except ImportError:
    # visualization is optional
    pass

@dataclass
class TilePair:
    """Represents a matched pair of original and reconstructed tiles."""
    patient_id: str
    tile_name: str
    original: Image.Image
    reconstructed: Image.Image
    
    
@dataclass
class PatientResult:
    """Evaluation results for a single patient."""
    patient_id: str
    tile_results: List[Dict] = field(default_factory=list)
    
    @property
    def num_tiles(self) -> int:
        return len(self.tile_results)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics across all tiles for this patient."""
        if not self.tile_results:
            return {}
        
        mse_values = [t["mse"] for t in self.tile_results]
        psnr_values = [t["psnr"] for t in self.tile_results if np.isfinite(t["psnr"])]
        ssim_values = [t["ssim"] for t in self.tile_results]
        
        return {
            "mse_mean": float(np.mean(mse_values)),
            "mse_std": float(np.std(mse_values)),
            "psnr_mean": float(np.mean(psnr_values)) if psnr_values else float("nan"),
            "psnr_std": float(np.std(psnr_values)) if psnr_values else float("nan"),
            "ssim_mean": float(np.mean(ssim_values)),
            "ssim_std": float(np.std(ssim_values)),
        }


# Patient ID extraction now consolidated in utils.py
# Use: from .utils import extract_patient_id


# ------------------------------------------------------------------
#  Coordinate-based tile matching helpers
# ------------------------------------------------------------------

_COORD_RE = re.compile(r"\(([\d.]+)[,_\s]+([\d.]+)\)")


def _extract_coords(name: str) -> Optional[Tuple[float, float]]:
    """Extract ``(x, y)`` coordinates embedded in a tile filename.

    Handles originals like ``tile_(26386.952, 9222.624).png``
    and reconstructed like ``reconstructed_00005_tile_(26386.952,_9222.624).png``.
    """
    m = _COORD_RE.search(name)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return None


def find_matching_zips(
    original_dir: Path,
    reconstructed_dir: Path,
    patient_ids: Optional[List[str]] = None,
) -> List[Tuple[str, Path, Path]]:
    """
    Find matching zip files between original and reconstructed directories.
    
    Matching is done by patient ID extracted from filename.
    
    Args:
        original_dir: Directory containing original tile zips
        reconstructed_dir: Directory containing reconstructed tile zips
        patient_ids: Optional list of specific patient IDs to include
        
    Returns:
        List of (patient_id, original_zip_path, reconstructed_zip_path) tuples
    """
    # Build lookup of original zips by patient ID
    original_by_pid = {}
    for zip_path in original_dir.glob("*.zip"):
        pid = canonical_patient_id(zip_path.name)
        if pid not in original_by_pid:
            original_by_pid[pid] = zip_path
    
    # Build lookup of reconstructed zips by patient ID  
    recon_by_pid = {}
    for zip_path in reconstructed_dir.glob("*.zip"):
        pid = canonical_patient_id(zip_path.name)
        if pid not in recon_by_pid:
            recon_by_pid[pid] = zip_path
    
    # Find matches
    matched = []
    common_pids = set(original_by_pid.keys()) & set(recon_by_pid.keys())
    
    if patient_ids:
        target_pids = set(p.upper() for p in patient_ids)
        common_pids = common_pids & target_pids
    
    for pid in sorted(common_pids):
        matched.append((pid, original_by_pid[pid], recon_by_pid[pid]))
    
    return matched


def load_tiles_from_zip(zip_path: Path) -> Dict[str, Image.Image]:
    """
    Load all tiles from a zip file.
    
    Args:
        zip_path: Path to zip file
        
    Returns:
        Dictionary mapping tile name (basename) to PIL Image
    """
    tiles = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                with zf.open(name) as f:
                    img = Image.open(BytesIO(f.read())).convert("RGB")
                    # Use basename as key for matching
                    tile_name = Path(name).name
                    tiles[tile_name] = img
    return tiles


def match_tiles(
    original_tiles: Dict[str, Image.Image],
    reconstructed_tiles: Dict[str, Image.Image],
) -> List[Tuple[str, Image.Image, Image.Image]]:
    """Match tiles between original and reconstructed sets.

    First tries exact filename matching.  When that yields no hits the
    function falls back to **coordinate-based matching**: it extracts
    ``(x, y)`` coordinates embedded in the filenames and pairs tiles
    whose coordinates agree.  This is necessary for genomic
    reconstructions where filenames are prefixed with
    ``reconstructed_NNNNN_``.

    Args:
        original_tiles: Dict of original tile name -> image
        reconstructed_tiles: Dict of reconstructed tile name -> image

    Returns:
        List of (tile_name, original_image, reconstructed_image) tuples
    """
    matched = []

    # --- Strategy 1: exact basename match ---
    common_names = set(original_tiles.keys()) & set(reconstructed_tiles.keys())
    if common_names:
        for name in sorted(common_names):
            matched.append((name, original_tiles[name], reconstructed_tiles[name]))
        return matched

    # --- Strategy 2: coordinate-based matching ---
    # Build a lookup of original tiles by their embedded coordinates
    orig_by_coord: Dict[Tuple[float, float], Tuple[str, Image.Image]] = {}
    for name, img in original_tiles.items():
        coords = _extract_coords(name)
        if coords is not None:
            orig_by_coord[coords] = (name, img)

    for rname, rimg in sorted(reconstructed_tiles.items()):
        coords = _extract_coords(rname)
        if coords is not None and coords in orig_by_coord:
            oname, oimg = orig_by_coord[coords]
            matched.append((oname, oimg, rimg))

    if not matched:
        # --- Strategy 3: ordered pairing (last resort) ---
        orig_sorted = sorted(original_tiles.items())
        recon_sorted = sorted(reconstructed_tiles.items())
        for (oname, oimg), (_, rimg) in zip(orig_sorted, recon_sorted):
            matched.append((oname, oimg, rimg))

    return matched


class ReconstructionEvaluator:
    """
    Main class for evaluating reconstruction quality.
    
    Handles loading, matching, and computing metrics for tile pairs.
    """
    
    def __init__(
        self,
        original_zip_dir: str,
        reconstructed_zip_dir: str,
        patient_ids: Optional[List[str]] = None,
    ):
        """
        Initialize evaluator with directories containing tile zips.
        
        Args:
            original_zip_dir: Directory with original tile zips
            reconstructed_zip_dir: Directory with reconstructed tile zips
            patient_ids: Optional list of specific patient IDs to evaluate
        """
        self.original_dir = Path(original_zip_dir)
        self.reconstructed_dir = Path(reconstructed_zip_dir)
        self.patient_ids = patient_ids
        
        # Find matching zip files
        self.matched_zips = find_matching_zips(
            self.original_dir,
            self.reconstructed_dir,
            self.patient_ids,
        )
        
        # Results storage
        self.patient_results: Dict[str, PatientResult] = {}
        self.all_tile_results: List[Dict] = []
    
    @property
    def num_patients(self) -> int:
        return len(self.matched_zips)
    
    def iter_tile_pairs(self) -> Iterator[TilePair]:
        """
        Iterate over all matched tile pairs across all patients.
        
        Yields:
            TilePair objects for each matched original/reconstructed pair
        """
        for pid, orig_zip, recon_zip in self.matched_zips:
            original_tiles = load_tiles_from_zip(orig_zip)
            recon_tiles = load_tiles_from_zip(recon_zip)
            matched = match_tiles(original_tiles, recon_tiles)
            
            for tile_name, orig_img, recon_img in matched:
                yield TilePair(
                    patient_id=pid,
                    tile_name=tile_name,
                    original=orig_img,
                    reconstructed=recon_img,
                )
    
    def evaluate_all(
        self,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate all matched tile pairs and compute metrics.
        
        Args:
            show_progress: If True, show tqdm progress bar
            
        Returns:
            Dictionary with overall summary and per-patient results
        """
        # Reset results
        self.patient_results = {}
        self.all_tile_results = []
        
        # Count total pairs for progress bar
        total_pairs = 0
        for pid, orig_zip, recon_zip in self.matched_zips:
            orig_tiles = load_tiles_from_zip(orig_zip)
            recon_tiles = load_tiles_from_zip(recon_zip)
            matched = match_tiles(orig_tiles, recon_tiles)
            total_pairs += len(matched)
        
        # Evaluate each pair
        pbar = tqdm(total=total_pairs, desc="Evaluating tiles", disable=not show_progress)
        
        for pid, orig_zip, recon_zip in self.matched_zips:
            # Initialize patient result
            if pid not in self.patient_results:
                self.patient_results[pid] = PatientResult(patient_id=pid)
            
            # Load tiles
            original_tiles = load_tiles_from_zip(orig_zip)
            recon_tiles = load_tiles_from_zip(recon_zip)
            matched = match_tiles(original_tiles, recon_tiles)
            
            for tile_name, orig_img, recon_img in matched:
                # Compute metrics
                metrics = compute_all_metrics(orig_img, recon_img)
                
                # Store result
                result = {
                    "patient_id": pid,
                    "tile_name": tile_name,
                    **metrics,
                }
                
                self.patient_results[pid].tile_results.append(result)
                self.all_tile_results.append(result)
                pbar.update(1)
        
        pbar.close()
        
        # Compute overall summary
        overall_summary = self._compute_overall_summary()
        
        return {
            "summary": overall_summary,
            "per_patient": {
                pid: result.get_summary()
                for pid, result in self.patient_results.items()
            },
            "num_patients": len(self.patient_results),
            "num_tiles": len(self.all_tile_results),
        }
    
    def _compute_overall_summary(self) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics across all evaluated tiles."""
        if not self.all_tile_results:
            return {}
        
        mse_values = np.array([t["mse"] for t in self.all_tile_results])
        psnr_values = np.array([t["psnr"] for t in self.all_tile_results])
        ssim_values = np.array([t["ssim"] for t in self.all_tile_results])
        
        # Filter infinite PSNR values
        finite_psnr = psnr_values[np.isfinite(psnr_values)]
        
        return {
            "mse": {
                "mean": float(np.mean(mse_values)),
                "std": float(np.std(mse_values)),
                "min": float(np.min(mse_values)),
                "max": float(np.max(mse_values)),
                "median": float(np.median(mse_values)),
            },
            "psnr": {
                "mean": float(np.mean(finite_psnr)) if len(finite_psnr) > 0 else float("nan"),
                "std": float(np.std(finite_psnr)) if len(finite_psnr) > 0 else float("nan"),
                "min": float(np.min(finite_psnr)) if len(finite_psnr) > 0 else float("nan"),
                "max": float(np.max(finite_psnr)) if len(finite_psnr) > 0 else float("nan"),
                "median": float(np.median(finite_psnr)) if len(finite_psnr) > 0 else float("nan"),
            },
            "ssim": {
                "mean": float(np.mean(ssim_values)),
                "std": float(np.std(ssim_values)),
                "min": float(np.min(ssim_values)),
                "max": float(np.max(ssim_values)),
                "median": float(np.median(ssim_values)),
            },
        }
    
    def save_results(
        self,
        output_dir: str,
        save_csv: bool = True,
        save_json: bool = True,
    ) -> None:
        """
        Save evaluation results to files.
        
        Args:
            output_dir: Directory to save results
            save_csv: If True, save per-tile results to CSV
            save_json: If True, save summary to JSON
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save per-tile CSV
        if save_csv and self.all_tile_results:
            csv_path = output_path / "tile_metrics.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["patient_id", "tile_name", "mse", "psnr", "ssim"],
                )
                writer.writeheader()
                for result in self.all_tile_results:
                    writer.writerow({
                        "patient_id": result["patient_id"],
                        "tile_name": result["tile_name"],
                        "mse": result["mse"],
                        "psnr": result["psnr"],
                        "ssim": result["ssim"],
                    })
            print(f"[OK] Saved tile metrics to {csv_path}")
        
        # Save per-patient summary CSV
        if save_csv and self.patient_results:
            patient_csv_path = output_path / "patient_summary.csv"
            with open(patient_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "patient_id", "num_tiles",
                        "mse_mean", "mse_std",
                        "psnr_mean", "psnr_std",
                        "ssim_mean", "ssim_std",
                    ],
                )
                writer.writeheader()
                for pid, result in self.patient_results.items():
                    summary = result.get_summary()
                    writer.writerow({
                        "patient_id": pid,
                        "num_tiles": result.num_tiles,
                        **summary,
                    })
            print(f"[OK] Saved patient summary to {patient_csv_path}")
        
        # Save JSON summary
        if save_json:
            json_path = output_path / "evaluation_summary.json"
            summary = self._compute_overall_summary()
            output_data = {
                "overall_summary": summary,
                "per_patient_summary": {
                    pid: result.get_summary()
                    for pid, result in self.patient_results.items()
                },
                "statistics": {
                    "num_patients": len(self.patient_results),
                    "num_tiles": len(self.all_tile_results),
                },
            }
            with open(json_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"[OK] Saved evaluation summary to {json_path}")


def evaluate_patient_tiles(
    original_zip: str,
    reconstructed_zip: str,
) -> Dict[str, Any]:
    """
    Convenience function to evaluate tiles for a single patient.
    
    Args:
        original_zip: Path to original tiles zip
        reconstructed_zip: Path to reconstructed tiles zip
        
    Returns:
        Dictionary with per-tile metrics and summary
    """
    orig_tiles = load_tiles_from_zip(Path(original_zip))
    recon_tiles = load_tiles_from_zip(Path(reconstructed_zip))
    matched = match_tiles(orig_tiles, recon_tiles)
    
    results = []
    for tile_name, orig_img, recon_img in matched:
        metrics = compute_all_metrics(orig_img, recon_img)
        results.append({
            "tile_name": tile_name,
            **metrics,
        })
    
    # Compute summary
    if results:
        mse_values = [r["mse"] for r in results]
        psnr_values = [r["psnr"] for r in results if np.isfinite(r["psnr"])]
        ssim_values = [r["ssim"] for r in results]
        
        summary = {
            "mse_mean": float(np.mean(mse_values)),
            "psnr_mean": float(np.mean(psnr_values)) if psnr_values else float("nan"),
            "ssim_mean": float(np.mean(ssim_values)),
            "num_tiles": len(results),
        }
    else:
        summary = {}
    
    return {
        "tiles": results,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate reconstruction quality of diffusion model tiles"
    )
    
    parser.add_argument(
        "--original-zip-dir",
        type=str,
        required=True,
        help="Directory containing original tile zip files",
    )
    parser.add_argument(
        "--reconstructed-zip-dir",
        type=str,
        required=True,
        help="Directory containing reconstructed tile zip files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save evaluation results (CSV, JSON)",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory to save visualization plots (optional)",
    )
    parser.add_argument(
        "--patient-ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific patient IDs to evaluate (default: all matching)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip saving CSV files",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip saving JSON summary",
    )
    parser.add_argument(
        "--num-comparison-samples",
        type=int,
        default=16,
        help="Number of tile pairs to show in comparison grid (default: 16)",
    )
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "=" * 60)
    print("RECONSTRUCTION QUALITY EVALUATION")
    print("=" * 60)
    print(f"Original tiles:      {args.original_zip_dir}")
    print(f"Reconstructed tiles: {args.reconstructed_zip_dir}")
    print(f"Output directory:    {args.output_dir}")
    if args.plot_dir:
        print(f"Plot directory:      {args.plot_dir}")
    print("=" * 60 + "\n")
    
    # Initialize evaluator
    evaluator = ReconstructionEvaluator(
        original_zip_dir=args.original_zip_dir,
        reconstructed_zip_dir=args.reconstructed_zip_dir,
        patient_ids=args.patient_ids,
    )
    
    print(f"[OK] Found {evaluator.num_patients} matching patient(s)\n")
    
    if evaluator.num_patients == 0:
        print("[ERROR] No matching patients found between directories")
        return
    
    # Run evaluation
    results = evaluator.evaluate_all()
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Patients evaluated: {results['num_patients']}")
    print(f"Tiles evaluated:    {results['num_tiles']}")
    print()
    
    summary = results["summary"]
    print("Overall Metrics:")
    if summary and isinstance(summary, dict) and "mse" in summary:
        print(f"  MSE:  {summary['mse']['mean']:.2f} +/- {summary['mse']['std']:.2f}")
        print(f"  PSNR: {summary['psnr']['mean']:.2f} +/- {summary['psnr']['std']:.2f} dB")
        print(f"  SSIM: {summary['ssim']['mean']:.4f} +/- {summary['ssim']['std']:.4f}")
    else:
        print("  [WARN] No metric data available (no matched tiles were evaluated).")
    print("=" * 60 + "\n")
    
    # Save results
    evaluator.save_results(
        output_dir=args.output_dir,
        save_csv=not args.no_csv,
        save_json=not args.no_json,
    )
    
    # Generate plots if plot directory specified
    if args.plot_dir:
        try:
            # Try relative import first (package execution), then absolute fallback
            try:
                from visualization.reconstruction_eval import (
                    plot_metrics_summary,
                    plot_comparison_grid,
                    plot_per_patient_metrics,
                )
            except Exception:
                from visualization.reconstruction_eval import (
                    plot_metrics_summary,
                    plot_comparison_grid,
                    plot_per_patient_metrics,
                )

            plot_path = Path(args.plot_dir)
            plot_path.mkdir(parents=True, exist_ok=True)

            # Summary distribution plots
            plot_metrics_summary(
                evaluator.all_tile_results,
                save_path=plot_path / "metrics_distribution.png",
            )

            # Per-patient comparison
            plot_per_patient_metrics(
                evaluator.patient_results,
                save_path=plot_path / "per_patient_metrics.png",
            )

            # Comparison grid with sample tiles
            tile_pairs = list(evaluator.iter_tile_pairs())
            if tile_pairs:
                import random
                sample_pairs = random.sample(
                    tile_pairs,
                    min(args.num_comparison_samples, len(tile_pairs)),
                )
                plot_comparison_grid(
                    sample_pairs,
                    save_path=plot_path / "tile_comparison.png",
                )

            print(f"\n[OK] Saved plots to {args.plot_dir}")

        except Exception as e:
            print(f"[WARN] Could not generate plots: {e}")
    
    print("\n[DONE] Evaluation complete!\n")


if __name__ == "__main__":
    main()
