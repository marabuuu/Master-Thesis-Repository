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
    
    elif stage == "training":
        print(f"[INFO] Training stage not yet configured in this CLI")
        print(f"       Run: python -m src.training.train_genomic_autoenc --config {config_path}")
    
    elif stage == "sampling":
        print(f"[INFO] Sampling stage not yet configured in this CLI")
        print(f"       Run: python -m src.sampling.sample_from_model --config {config_path}")
    
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
        raise ValueError(f"Unknown stage: {stage}. Choose from: preprocessing, encoding, training, sampling, evaluation, all")


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
        choices=["preprocessing", "encoding", "training", "sampling", "evaluation", "all"],
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
