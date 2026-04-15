#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract Genomic Latent Features from Jointly Trained VAE.

This script loads a jointly-trained (VAE + Diffusion) checkpoint,
passes the genomic data of patients through the VAE encoder, and
saves the resulting `(1, 512)` latent vectors into one `.h5` file
per patient, using the dataset name 'feats'.

Usage:
    python -m src.encoding.extract_joint_latents --config src/config.yaml --ckpt /path/to/checkpoint.ckpt --out-dir /path/to/save/dir
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch

# Add parent of src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from joint_training.train import extract_latents
from utils.config_utils import resolve_config_paths as _resolve_config_paths, deep_update as _deep_update


def _expected_variant_from_section(section: str) -> str | None:
    if section.startswith("gene_token_cross_attention_"):
        return "gene_token_cross_attention_joint_training"
    if section.startswith("gene_token_transformer_"):
        return "gene_token_transformer_joint_training"
    if section.startswith("cross_attention_"):
        return "cross_attention_joint_training"
    if section == "joint_training":
        return "joint_training"
    return None

def main():
    parser = argparse.ArgumentParser(description="Extract Latents from Joint Training Checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml (e.g., src/config.yaml)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the joint training .ckpt file")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to save the resulting .h5 files")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"], help="Which data split to extract for")
    parser.add_argument(
        "--section",
        type=str,
        default="joint_training",
        help=(
            "Config section to extract from. Use 'gene_token_cross_attention_joint_training' for GTCA."
        ),
    )
    
    args = parser.parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)
    full_cfg = _resolve_config_paths(full_cfg, Path(__file__).resolve().parents[2])

    if args.section == "joint_training":
        joint_cfg = full_cfg.get("joint_training", full_cfg)
    else:
        if args.section not in full_cfg:
            raise KeyError(f"Section '{args.section}' not found in config")

        # GTCA/GTT variants are override sections on top of GTT base config.
        if args.section.startswith("gene_token_"):
            base_cfg = full_cfg.get("gene_token_transformer_joint_training", full_cfg.get("joint_training", {}))
        else:
            base_cfg = full_cfg.get("joint_training", {})

        joint_cfg = _deep_update(base_cfg, full_cfg[args.section])
    
    # Override the output directory for latents
    joint_cfg["latent_dir"] = args.out_dir

    print(f"Loading checkpoint: {args.ckpt}")
    print(f"Using config section: {args.section}")
    print(f"Extracting latents to: {args.out_dir} ...")
    expected_variant = _expected_variant_from_section(args.section)
    if expected_variant is not None:
        print(f"Expected checkpoint variant: {expected_variant}")
    
    # Uses the configured method from joint_training which we updated 
    # to output 'feats' of shape (1, 512).
    extract_latents(
        joint_cfg,
        ckpt_path=args.ckpt,
        split=args.split,
        verbose=True,
        expected_variant=expected_variant,
    )

if __name__ == "__main__":
    main()
