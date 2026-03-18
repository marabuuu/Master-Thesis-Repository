#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified CLI for running all pipeline stages from config.yaml

Usage:
    python run_pipeline.py --config src/config.yaml --stage evaluation
    python run_pipeline.py --config src/config.yaml --stage preprocessing
    python run_pipeline.py --config src/config.yaml --stage all
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_stage(config: Dict[str, Any], stage: str, config_path: str = "", verbose: bool = True) -> None:
    """Run a specific pipeline stage based on config."""
    
    if stage == "evaluation":
        from src.quality_assurance import run_evaluation
        if "evaluation" not in config:
            raise ValueError("No 'evaluation' section in config.yaml")
        run_evaluation(config["evaluation"], verbose=verbose)
    
    elif stage == "preprocessing":
        print(f"[INFO] Preprocessing stage not yet configured in this CLI")
        print(f"       Run: python -m src.preprocessing.get_tiles_within_rois --config {config_path}")
    
    elif stage == "encoding":
        print(f"[INFO] Encoding stage not yet configured in this CLI")
        print(f"       Run: python -m src.encoding.encode_genomics --config {config_path}")
    
    elif stage == "extract_joint_latents":
        from src.joint_training.train import extract_latents
        if "extract_joint_latents" not in config:
            raise ValueError("No 'extract_joint_latents' section in config.yaml")
        if "joint_training" not in config:
            raise ValueError("No 'joint_training' section in config.yaml to provide model definitions")
            
        joint_cfg = config["joint_training"].copy()
        extract_cfg = config["extract_joint_latents"]
        
        # Enforce output directory override
        joint_cfg["latent_dir"] = extract_cfg.get("out_dir")
        
        if verbose:
            print(f"[INFO] Extracting joint latents from {extract_cfg.get('ckpt')}...")
            
        extract_latents(
            joint_cfg=joint_cfg,
            ckpt_path=extract_cfg.get("ckpt"),
            split=extract_cfg.get("split", "all"),
            verbose=verbose
        )
    
    elif stage == "visualize_latents":
        from src.visualization.visualize_latents import run_visualizations
        if "visualize_latents" not in config:
            raise ValueError("No 'visualize_latents' section in config.yaml")
        run_visualizations(config["visualize_latents"], verbose=verbose)
    
    elif stage == "joint_training":
        from src.joint_training.train import run_joint_training
        if "joint_training" not in config:
            raise ValueError("No 'joint_training' section in config.yaml")
        run_joint_training(config["joint_training"], verbose=verbose)

    elif stage == "training":
        print(f"[INFO] Training stage not yet configured in this CLI")
        print(f"       Run: python -m src.training.train_genomic_autoenc --config {config_path}")
    
    elif stage == "training_stats":
        # This stage wraps src.statistics.training_curves CLI using the provided config
        from src.statistics import training_curves
        if "training_stats" not in config:
            raise ValueError("No 'training_stats' section in config.yaml")
        old_argv = sys.argv
        try:
            sys.argv = [old_argv[0], "--config", config_path]
            training_curves.main()
        finally:
            sys.argv = old_argv

    elif stage == "sampling":
        print(f"[INFO] Sampling stage not yet configured in this CLI")
        print(f"       Run: python -m src.sampling.sample_from_model --config {config_path}")
    
    elif stage == "reconstruction":
        from src.reconstruction import run_reconstruction
        if "reconstruction" not in config:
            raise ValueError("No 'reconstruction' section in config.yaml")
        run_reconstruction(config["reconstruction"], verbose=verbose)
    
    elif stage == "segmentation":
        from src.classifier.segment_and_classify_cells import run_segmentation
        if "segmentation" not in config:
            raise ValueError("No 'segmentation' section in config.yaml")
        run_segmentation(config["segmentation"], verbose=verbose)
    
    elif stage == "all":
        print("[INFO] Running all stages in sequence...")
        for s in ["preprocessing", "encoding", "training", "sampling", "evaluation"]:
            print(f"\n{'='*70}")
            print(f"STAGE: {s.upper()}")
            print(f"{'='*70}\n")
            try:
                run_stage(config, s, config_path, verbose=verbose)
            except Exception as e:
                if "not yet configured" not in str(e):
                    print(f"[ERROR] Stage '{s}' failed: {e}")
                    return
    
    else:
        raise ValueError(
            f"Unknown stage: {stage}. Choose from: preprocessing, encoding, extract_joint_latents, visualize_latents, joint_training, training, training_stats, sampling, reconstruction, segmentation, evaluation, all"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Unified CLI for running pipeline stages from config.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation only
  python run_pipeline.py --config src/config.yaml --stage evaluation
  
  # Run all stages
  python run_pipeline.py --config src/config.yaml --stage all
  
  # With quiet output
  python run_pipeline.py --config src/config.yaml --stage evaluation --quiet
        """,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.yaml file (e.g., src/config.yaml)",
    )
    
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["preprocessing", "encoding", "extract_joint_latents", "visualize_latents", "joint_training", "training", "training_stats", "sampling", "reconstruction", "segmentation", "evaluation", "all"],
        help="Pipeline stage to run",
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        run_stage(config, args.stage, args.config, verbose=not args.quiet)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
