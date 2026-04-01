"""
Classifier Module - Genomic Classification & Cell Segmentation Helpers

This module provides tools for training and evaluating classifiers
on genomic feature vectors (e.g., PAM50 subtype classification), as well as
cell segmentation and classification for histopathology tiles.

Components:
    - train_subtype_classifier: Train Basal-vs-LumA classifier on Virchow2 H5 features
    - evaluate_subtype_classifier: Evaluate subtype classifier on val/test splits
    - extract_virchow2_features: Extract Virchow2 features from generated tiles and
      save as per-patient H5 files compatible with the subtype classifier pipeline
    - segment_and_classify_cells: Run DeepCMorph cell segmentation on H&E tiles
      to produce per-cell-type binary masks (.npy) compatible with TopoFD

Usage:
    python -m src.classifier.train_subtype_classifier --help
    python -m src.classifier.evaluate_subtype_classifier --help
    python -m src.classifier.extract_virchow2_features --help
    python -m src.classifier.segment_and_classify_cells --help
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def run_virchow2_extraction(cfg: Dict[str, Any], verbose: bool = True) -> None:
    """Extract Virchow2 features from generated tiles and save as H5 files.

    Expected config section (``virchow2_extraction``):
      tiles_dir:      path to per-patient ZIP archives / tile directories
      output_dir:     destination for per-patient .h5 feature files
      batch_size:     optional int (default 32)
      device:         optional str (default auto-detect)
      skip_existing:  optional bool (default true)
    """
    from .extract_virchow2_features import run_virchow2_extraction as _run

    _run(cfg, verbose=verbose)


def run_subtype_classifier(cfg: Dict[str, Any], verbose: bool = True) -> None:
    """Run subtype classifier train/eval from config dict.

    Expected config section (``subtype_classifier``):
      mode: train | evaluate | both
      config_path: path/to/src/config.yaml
      features_dir: path/to/virchow2_h5_dir
      patient_splits_path: path/to/patient_splits.json
      output_dir: path/to/output_root
      label_csv_path: optional override
      max_tiles_per_patient: optional int
      seed: optional int
      train: {solver, C, max_iter, class_weight}
      evaluate: {model_path}
    """
    from .evaluate_subtype_classifier import main as eval_main
    from .train_subtype_classifier import main as train_main

    mode = str(cfg.get("mode", "both")).lower()
    if mode not in {"train", "evaluate", "both"}:
        raise ValueError("subtype_classifier.mode must be one of: train, evaluate, both")

    config_path = cfg.get("config_path")
    features_dir = cfg.get("features_dir")
    patient_splits_path = cfg.get("patient_splits_path")
    output_dir = cfg.get("output_dir")

    if not config_path:
        raise ValueError("Missing 'subtype_classifier.config_path' in config.yaml")
    if not features_dir:
        raise ValueError("Missing 'subtype_classifier.features_dir' in config.yaml")
    if not patient_splits_path:
        raise ValueError("Missing 'subtype_classifier.patient_splits_path' in config.yaml")
    if not output_dir:
        raise ValueError("Missing 'subtype_classifier.output_dir' in config.yaml")

    output_dir = str(Path(output_dir))
    train_dir = str(Path(output_dir) / "train")
    eval_dir = str(Path(output_dir) / "eval")

    common = {
        "config": str(config_path),
        "features_dir": str(features_dir),
        "patient_splits_path": str(patient_splits_path),
    }
    optional = {}
    if cfg.get("label_csv_path"):
        optional["label_csv_path"] = str(cfg.get("label_csv_path"))
    max_tiles = cfg.get("max_tiles_per_patient")
    if max_tiles is not None:
        optional["max_tiles_per_patient"] = int(max_tiles)
    seed = cfg.get("seed")
    if seed is not None:
        optional["seed"] = int(seed)

    if mode in {"train", "both"}:
        train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else {}
        train_args = {
            **common,
            **optional,
            "output_dir": train_dir,
            "solver": str(train_cfg.get("solver", "liblinear")),
            "C": float(train_cfg.get("C", 1.0)),
            "max_iter": int(train_cfg.get("max_iter", 2000)),
            "class_weight": str(train_cfg.get("class_weight", "balanced")),
        }
        if verbose:
            print("[SubtypeClassifier] Running TRAIN")
            print(f"[SubtypeClassifier] output_dir: {train_dir}")
        train_main(train_args)

    if mode in {"evaluate", "both"}:
        eval_cfg = cfg.get("evaluate", {}) if isinstance(cfg.get("evaluate"), dict) else {}
        default_model = str(Path(train_dir) / "subtype_linear_model.joblib")
        raw_model = eval_cfg.get("model_path", default_model)
        model_path = str(raw_model) if raw_model else default_model
        eval_args = {
            **common,
            **optional,
            "model_path": model_path,
            "output_dir": eval_dir,
        }
        if verbose:
            print("[SubtypeClassifier] Running EVALUATE")
            print(f"[SubtypeClassifier] model_path: {model_path}")
            print(f"[SubtypeClassifier] output_dir: {eval_dir}")
        eval_main(eval_args)
