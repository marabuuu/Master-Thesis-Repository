#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run reconstruction quality evaluation using parameters from config.yaml

Usage:
    python -m quality_assurance.run_evaluation /path/to/config.yaml
    
    Or from the repo root:
    python src/quality_assurance/run_evaluation.py config.yaml
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if "evaluation" not in config:
        raise ValueError(f"No 'evaluation' section found in {config_path}")
    return config["evaluation"]


def run_evaluation(config: Dict[str, Any], verbose: bool = True) -> None:
    """Run reconstruction quality evaluation with config parameters."""
    from .evaluate_reconstruction import ReconstructionEvaluator
    
    # Required parameters
    original_zip_dir = config.get("original_zip_dir")
    reconstructed_zip_dir = config.get("reconstructed_zip_dir")
    output_dir = config.get("output_dir")
    
    if not all([original_zip_dir, reconstructed_zip_dir, output_dir]):
        raise ValueError(
            "config.evaluation must specify: "
            "original_zip_dir, reconstructed_zip_dir, output_dir"
        )
    
    # Type-safe after validation
    original_zip_dir = str(original_zip_dir)
    reconstructed_zip_dir = str(reconstructed_zip_dir)
    output_dir = str(output_dir)
    
    # Optional parameters
    patient_ids = config.get("patient_ids")
    save_csv = config.get("save_csv", True)
    save_json = config.get("save_json", True)
    plot_dir = config.get("plot_dir")
    num_comparison_samples = config.get("num_comparison_samples", 16)
    include_diff = config.get("include_diff_heatmap", True)
    
    if verbose:
        print("\n" + "="*70)
        print("RECONSTRUCTION QUALITY EVALUATION")
        print("="*70)
        print(f"Original tiles:      {original_zip_dir}")
        print(f"Reconstructed tiles: {reconstructed_zip_dir}")
        print(f"Output directory:    {output_dir}")
        if plot_dir:
            print(f"Plot directory:      {plot_dir}")
        if patient_ids:
            print(f"Patient IDs:         {', '.join(patient_ids)}")
        print("="*70 + "\n")
    
    # Initialize evaluator
    evaluator = ReconstructionEvaluator(
        original_zip_dir=original_zip_dir,
        reconstructed_zip_dir=reconstructed_zip_dir,
        patient_ids=patient_ids,
    )
    
    if verbose:
        print(f"[OK] Found {evaluator.num_patients} matching patient(s)\n")
    
    if evaluator.num_patients == 0:
        print("[ERROR] No matching patients found between directories")
        return
    
    # Run evaluation
    results = evaluator.evaluate_all(show_progress=verbose)
    
    # Print summary
    if verbose:
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"Patients evaluated: {results['num_patients']}")
        print(f"Tiles evaluated:    {results['num_tiles']}")
        print()
        
        summary = results["summary"]
        if summary and isinstance(summary, dict) and "mse" in summary:
            print("Overall Metrics:")
            print(f"  MSE:  {summary['mse']['mean']:.2f} +/- {summary['mse']['std']:.2f}")
            print(f"  PSNR: {summary['psnr']['mean']:.2f} +/- {summary['psnr']['std']:.2f} dB")
            print(f"  SSIM: {summary['ssim']['mean']:.4f} +/- {summary['ssim']['std']:.4f}")
        print("="*70 + "\n")
    
    # Save results
    evaluator.save_results(
        output_dir=output_dir,
        save_csv=save_csv,
        save_json=save_json,
    )
    
    # Generate plots if plot directory specified
    if plot_dir:
        try:
            from visualization.reconstruction_eval import (
                plot_metrics_summary,
                plot_comparison_grid,
                plot_per_patient_metrics,
                plot_metric_correlation,
            )
            
            plot_path = Path(plot_dir)
            plot_path.mkdir(parents=True, exist_ok=True)
            
            plot_types = config.get("plot_types", ["metrics_summary"])
            
            if "metrics_summary" in plot_types:
                plot_metrics_summary(
                    evaluator.all_tile_results,
                    save_path=plot_path / "metrics_distribution.png",
                )
            
            if "per_patient_metrics" in plot_types:
                plot_per_patient_metrics(
                    evaluator.patient_results,
                    save_path=plot_path / "per_patient_metrics.png",
                )
            
            if "comparison_grid" in plot_types:
                tile_pairs = list(evaluator.iter_tile_pairs())
                if tile_pairs:
                    import random
                    sample_pairs = random.sample(
                        tile_pairs,
                        min(num_comparison_samples, len(tile_pairs)),
                    )
                    plot_comparison_grid(
                        sample_pairs,
                        save_path=plot_path / "tile_comparison.png",
                        include_diff=include_diff,
                    )
            
            if "metric_correlation" in plot_types:
                plot_metric_correlation(
                    evaluator.all_tile_results,
                    save_path=plot_path / "metric_correlation.png",
                )
            
            if verbose:
                print(f"[OK] Saved plots to {plot_dir}\n")
        
        except Exception as e:
            print(f"[WARN] Could not generate plots: {e}\n")
    
    if verbose:
        print("[DONE] Evaluation complete!\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run reconstruction quality evaluation from config.yaml",
        prog="python -m quality_assurance.run_evaluation",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.yaml file (required)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        run_evaluation(config, verbose=not args.quiet)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
