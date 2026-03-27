#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

try:
    from classifier.utils_subtype_data import (
        build_tile_feature_table,
        infer_label_csv_and_columns,
        load_patient_splits,
        load_subtype_table,
        load_yaml,
        split_to_arrays,
    )
except ImportError:
    from classifier.utils_subtype_data import (
        build_tile_feature_table,
        infer_label_csv_and_columns,
        load_patient_splits,
        load_subtype_table,
        load_yaml,
        split_to_arrays,
    )


def _resolve_config_relative_path(path_value: str, config_path: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)

    config_dir = Path(config_path).resolve().parent
    repo_root = config_dir.parent if config_dir.name == "src" else config_dir
    repo_candidate = (repo_root / path).resolve()
    normalized = path_value[2:] if str(path_value).startswith("./") else str(path_value)

    # In this workspace, data/dataframes/experiments often live next to the repo root.
    if normalized.startswith(("data/", "dataframes/", "experiments/")):
        parent_candidate = (repo_root.parent / normalized).resolve()
        if parent_candidate.exists() or not repo_candidate.exists():
            return str(parent_candidate)

    return str(repo_candidate)


def tune_threshold(y_true: np.ndarray, p_pos: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t = 0.5
    best_score = -1.0
    for t in thresholds:
        preds = (p_pos >= t).astype(np.int64)
        score = balanced_accuracy_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Basal vs LumA linear classifier on Virchow2 H5 tile features")
    parser.add_argument("--config", type=str, required=True, help="Path to src/config.yaml")
    parser.add_argument("--features-dir", type=str, required=True, help="Directory with Virchow2 H5 files (one per patient)")
    parser.add_argument("--patient-splits-path", type=str, required=True, help="Path to patient_splits.json")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save trained classifier artifacts")
    parser.add_argument("--label-csv-path", type=str, default=None, help="Optional label CSV override")
    parser.add_argument("--max-tiles-per-patient", type=int, default=None, help="Optional tile cap per patient")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solver", type=str, default="liblinear", choices=["liblinear", "lbfgs", "saga"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--class-weight", type=str, default="balanced", choices=["balanced", "none"])
    return parser.parse_args()


def main(args: dict | argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    elif isinstance(args, dict):
        args = argparse.Namespace(**args)

    # When called programmatically from run_pipeline, optional CLI args may be
    # omitted from the dict. Normalize them to parser-equivalent defaults.
    if not hasattr(args, "label_csv_path"):
        args.label_csv_path = None
    if not hasattr(args, "max_tiles_per_patient"):
        args.max_tiles_per_patient = None
    if not hasattr(args, "seed"):
        args.seed = 42

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(args.config)
    label_csv_path, patient_col, subtype_col = infer_label_csv_and_columns(cfg, args.label_csv_path)
    label_csv_path = _resolve_config_relative_path(label_csv_path, args.config)
    splits = load_patient_splits(args.patient_splits_path)
    subtype_df = load_subtype_table(label_csv_path, patient_col, subtype_col)

    df = build_tile_feature_table(
        features_dir=args.features_dir,
        subtype_df=subtype_df,
        splits=splits,
        max_tiles_per_patient=args.max_tiles_per_patient,
        seed=args.seed,
    )

    x_train, y_train, train_df = split_to_arrays(df, "train")
    x_val, y_val, val_df = split_to_arrays(df, "val")
    x_test, y_test, test_df = split_to_arrays(df, "test")

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)

    clf = LogisticRegression(
        C=args.C,
        solver=args.solver,
        class_weight=None if args.class_weight == "none" else "balanced",
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    clf.fit(x_train_s, y_train)

    val_prob = clf.predict_proba(x_val_s)[:, 1]
    threshold = tune_threshold(y_val, val_prob)

    artifact = {
        "scaler": scaler,
        "classifier": clf,
        "threshold": threshold,
        "label_mapping": {"LumA": 0, "Basal": 1},
        "feature_dim": int(x_train_s.shape[1]),
        "config": {
            "solver": args.solver,
            "C": args.C,
            "max_iter": args.max_iter,
            "class_weight": args.class_weight,
            "seed": args.seed,
            "patient_col": patient_col,
            "subtype_col": subtype_col,
            "label_csv_path": label_csv_path,
            "features_dir": str(args.features_dir),
            "patient_splits_path": str(args.patient_splits_path),
        },
    }

    model_path = out_dir / "subtype_linear_model.joblib"
    joblib.dump(artifact, model_path)

    summary = {
        "n_tiles": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "n_patients": {
            "train": int(train_df["patient_id"].nunique()),
            "val": int(val_df["patient_id"].nunique()),
            "test": int(test_df["patient_id"].nunique()),
        },
        "class_balance_tiles": {
            "train_basal": int(y_train.sum()),
            "train_luma": int((1 - y_train).sum()),
            "val_basal": int(y_val.sum()),
            "val_luma": int((1 - y_val).sum()),
            "test_basal": int(y_test.sum()),
            "test_luma": int((1 - y_test).sum()),
        },
        "threshold": float(threshold),
        "model_path": str(model_path),
    }

    with open(out_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[SubtypeClassifier] Training complete")
    print(f"[SubtypeClassifier] Saved model: {model_path}")
    print(f"[SubtypeClassifier] Threshold (val tuned): {threshold:.4f}")


if __name__ == "__main__":
    main()
