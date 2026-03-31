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
    reconstructed_zip_dir = config.get("reconstructed_zip_dir")
    output_dir = config.get("output_dir")
    if not all([reconstructed_zip_dir, output_dir]):
        raise ValueError(
            "config.evaluation must specify: reconstructed_zip_dir, output_dir"
        )
    reconstructed_zip_dir = str(reconstructed_zip_dir)
    output_dir = str(output_dir)

    # Mode: image_guided (default) or random_noise
    mode = config.get("mode", "image_guided")

    # original_zip_dir is not used in random_noise mode
    original_zip_dir = config.get("original_zip_dir")
    if mode != "random_noise" and not original_zip_dir:
        raise ValueError(
            "config.evaluation.original_zip_dir is required for image_guided mode"
        )
    if original_zip_dir:
        original_zip_dir = str(original_zip_dir)

    # Optional parameters
    patient_ids = config.get("patient_ids")
    save_csv = config.get("save_csv", True)
    save_json = config.get("save_json", True)
    plot_dir = config.get("plot_dir")
    num_comparison_samples = config.get("num_comparison_samples", 16)
    include_diff = config.get("include_diff_heatmap", True)

    # Subtype metadata — loaded from an optional CSV
    metadata_csv = config.get("metadata_csv")
    subtype_col = config.get("subtype_col", "Majority_Subtype_mRNA")
    patient_col = config.get("patient_col", "Patient_ID")

    if verbose:
        print("\n" + "=" * 70)
        print("RECONSTRUCTION QUALITY EVALUATION")
        print("=" * 70)
        print(f"Mode:                {mode}")
        if original_zip_dir:
            print(f"Original tiles:      {original_zip_dir}")
        print(f"Reconstructed tiles: {reconstructed_zip_dir}")
        print(f"Output directory:    {output_dir}")
        if plot_dir:
            print(f"Plot directory:      {plot_dir}")
        if patient_ids:
            print(f"Patient IDs:         {', '.join(str(p) for p in patient_ids)}")
        if metadata_csv:
            print(f"Metadata CSV:        {metadata_csv}  (subtype_col={subtype_col})")
        print("=" * 70 + "\n")

    # ------------------------------------------------------------------ #
    #  Load subtype map from metadata CSV when provided                   #
    # ------------------------------------------------------------------ #
    subtype_map: Optional[Dict[str, str]] = None
    if metadata_csv:
        try:
            from visualization.reconstruction_eval import _load_subtype_map
            subtype_map = _load_subtype_map(metadata_csv, subtype_col, patient_col) or None
        except Exception as exc:
            print(f"[WARN] Could not load subtype map: {exc}")

    # ================================================================== #
    #  random_noise mode: no originals, just visualise generated tiles    #
    # ================================================================== #
    if mode == "random_noise":
        if plot_dir:
            try:
                import zipfile
                from io import BytesIO
                from PIL import Image as _Image
                from visualization.reconstruction_eval import plot_random_noise_grid

                plot_path = Path(plot_dir)
                plot_path.mkdir(parents=True, exist_ok=True)

                recon_dir = Path(reconstructed_zip_dir)
                generated_tiles = []
                for zpath in sorted(recon_dir.glob("*.zip")):
                    # Derive canonical patient ID from zip filename
                    import re as _re
                    stem = zpath.stem.upper()
                    m = _re.match(r"(TCGA-[A-Z0-9]+-[A-Z0-9]+)", stem)
                    pid = m.group(1) if m else stem
                    with zipfile.ZipFile(zpath, "r") as zf:
                        names = sorted(
                            n for n in zf.namelist()
                            if n.lower().endswith((".png", ".jpg", ".jpeg"))
                        )
                        for name in names[:num_comparison_samples]:
                            img = _Image.open(BytesIO(zf.read(name))).convert("RGB")
                            generated_tiles.append(
                                type("GT", (), {"patient_id": pid, "reconstructed": img})()
                            )
                    if len(generated_tiles) >= num_comparison_samples:
                        break

                plot_types = config.get("plot_types", ["random_noise_grid"])
                if "random_noise_grid" in plot_types or not plot_types:
                    plot_random_noise_grid(
                        generated_tiles[:num_comparison_samples],
                        save_path=plot_path / "random_noise_grid.png",
                        subtype_map=subtype_map,
                    )

                if verbose:
                    print(f"[OK] Saved plots to {plot_dir}\n")
            except Exception as exc:
                print(f"[WARN] Could not generate random_noise plots: {exc}\n")

        if verbose:
            print("[DONE] Evaluation complete (random_noise — no metrics computed)!\n")
        return

    # ================================================================== #
    #  image_guided mode: pair originals with reconstructions             #
    # ================================================================== #
    evaluator = ReconstructionEvaluator(
        original_zip_dir=original_zip_dir,  # type: ignore[arg-type]
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
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Patients evaluated: {results['num_patients']}")
        print(f"Tiles evaluated:    {results['num_tiles']}")
        print()
        summary = results["summary"]
        if summary and isinstance(summary, dict) and "mse" in summary:
            print("Overall Metrics:")
            print(f"  MSE:  {summary['mse']['mean']:.2f} +/- {summary['mse']['std']:.2f}")
            print(f"  PSNR: {summary['psnr']['mean']:.2f} +/- {summary['psnr']['std']:.2f} dB")
            print(f"  SSIM: {summary['ssim']['mean']:.4f} +/- {summary['ssim']['std']:.4f}")
        print("=" * 70 + "\n")

    # Save results
    evaluator.save_results(
        output_dir=output_dir,
        save_csv=save_csv,
        save_json=save_json,
    )

    # ------------------------------------------------------------------ #
    #  Detect cross-patient zip files in reconstructed_zip_dir           #
    #  (named {pid}__cond_{cond_pid}.zip by the reconstruction module)   #
    # ------------------------------------------------------------------ #
    import re as _re
    _CROSS_RE = _re.compile(r"^(.+?)__cond_(.+?)\.zip$", _re.IGNORECASE)
    cross_zips = []
    for zpath in Path(reconstructed_zip_dir).glob("*.zip"):
        m = _CROSS_RE.match(zpath.name)
        if m:
            cross_zips.append(zpath)

    # Generate plots
    if plot_dir:
        try:
            from visualization.reconstruction_eval import (
                plot_metrics_summary,
                plot_comparison_grid,
                plot_per_patient_metrics,
                plot_metric_correlation,
                plot_cross_patient_comparison,
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
                    subtype_map=subtype_map,
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
                        subtype_map=subtype_map,
                        mode=mode,
                    )

            if "metric_correlation" in plot_types:
                plot_metric_correlation(
                    evaluator.all_tile_results,
                    save_path=plot_path / "metric_correlation.png",
                )

            # cross_patient: load pairs from cross-patient zips when present
            if "cross_patient" in plot_types and cross_zips and original_zip_dir:
                import zipfile
                from io import BytesIO
                from PIL import Image as _Image
                from quality_assurance.evaluate_reconstruction import match_tiles, load_tiles_from_zip

                cross_pairs = []
                for zpath in cross_zips:
                    m_c = _CROSS_RE.match(zpath.name)
                    if not m_c:
                        continue
                    # Normalise to canonical TCGA-XX-XXXX
                    def _canon(s: str) -> str:
                        s = s.upper()
                        mc = _re.match(r"(TCGA-[A-Z0-9]+-[A-Z0-9]+)", s)
                        return mc.group(1) if mc else s
                    tile_pid = _canon(m_c.group(1))
                    cond_pid = _canon(m_c.group(2))
                    orig_zip = Path(original_zip_dir) / f"{tile_pid}.zip"
                    if not orig_zip.exists():
                        # Try glob fallback
                        candidates = list(Path(original_zip_dir).glob(f"{tile_pid}*.zip"))
                        orig_zip = candidates[0] if candidates else None
                    if not orig_zip or not orig_zip.exists():
                        continue
                    orig_tiles = load_tiles_from_zip(orig_zip)
                    recon_tiles = load_tiles_from_zip(zpath)
                    for tile_name, orig_img, recon_img in match_tiles(orig_tiles, recon_tiles)[:num_comparison_samples]:
                        cross_pairs.append(type("CP", (), {
                            "original": orig_img,
                            "reconstructed": recon_img,
                            "patient_id": tile_pid,
                            "conditioning_patient_id": cond_pid,
                        })())

                if cross_pairs:
                    plot_cross_patient_comparison(
                        cross_pairs,
                        save_path=plot_path / "cross_patient_comparison.png",
                        subtype_map=subtype_map,
                    )
                else:
                    print("[WARN] cross_patient plot requested but no cross-patient zips were found/matched")

            if verbose:
                print(f"[OK] Saved plots to {plot_dir}\n")

        except Exception as exc:
            print(f"[WARN] Could not generate plots: {exc}\n")

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
