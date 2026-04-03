#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tile reconstruction module — reconstruct histopathology tiles from genomic features.

Subpackage of the joint training pipeline.
"""

from __future__ import annotations

__all__ = [
    "reconstruct_tiles",
    "investigate_noising",
    "run_reconstruction",
]


def run_reconstruction(rec_cfg: dict, verbose: bool = True) -> None:
    """
    Run tile reconstruction from genomic features (called from pipeline).
    
    Extracts configuration from rec_cfg dict and calls the main reconstruction pipeline.
    
    Parameters
    ----------
    rec_cfg : dict
        Reconstruction configuration from config.yaml (reconstruction section)
    verbose : bool
        Whether to print verbose output
    """
    from .reconstruct_tiles import main
    
    # Extract required parameters from config
    checkpoint_path = rec_cfg.get("checkpoint_path")
    csv_path = rec_cfg.get("csv_path")
    tiles_zip_dir = rec_cfg.get("tiles_zip_dir")
    output_dir = rec_cfg.get("output_dir")
    
    if not checkpoint_path:
        raise ValueError("Missing 'reconstruction.checkpoint_path' in config.yaml")
    if not csv_path:
        raise ValueError("Missing 'reconstruction.csv_path' in config.yaml")
    if not tiles_zip_dir:
        raise ValueError("Missing 'reconstruction.tiles_zip_dir' in config.yaml")
    if not output_dir:
        raise ValueError("Missing 'reconstruction.output_dir' in config.yaml")
    
    # Extract optional parameters
    patient_ids = rec_cfg.get("patient_ids", None)
    conditioning_patients = rec_cfg.get("conditioning_patients", None)
    patient_splits_path = rec_cfg.get("patient_splits_path", None)
    n_tiles_per_patient = rec_cfg.get("n_tiles_per_patient", 20)
    mode = rec_cfg.get("mode", "image_guided")
    investigate = rec_cfg.get("investigate", False)
    device = rec_cfg.get("device", None)
    guidance_scale = float(rec_cfg.get("guidance_scale", 1.0))
    
    if verbose:
        print(f"[Reconstruction] Loading config from reconstruction section")
        print(f"[Reconstruction] checkpoint: {checkpoint_path}")
        print(f"[Reconstruction] csv: {csv_path}")
        print(f"[Reconstruction] tiles_dir: {tiles_zip_dir}")
        print(f"[Reconstruction] output: {output_dir}")
        print(f"[Reconstruction] patients: {patient_ids if patient_ids else 'all'}")
        print(
            "[Reconstruction] conditioning_patients: "
            f"{conditioning_patients if conditioning_patients else 'same as tile patients'}"
        )
        print(f"[Reconstruction] mode: {mode}")
    
    # Call main reconstruction pipeline
    main(
        checkpoint_path=checkpoint_path,
        config_path="",  # Not used in main() when called from here
        gene_csv_path=csv_path,
        tiles_dir=tiles_zip_dir,
        save_dir=output_dir,
        patients=patient_ids,
        conditioning_patients=conditioning_patients,
        patient_splits_path=patient_splits_path,
        n_tiles_per_patient=n_tiles_per_patient,
        mode=mode,
        investigate=investigate,
        device=device,
        guidance_scale=guidance_scale,
    )
