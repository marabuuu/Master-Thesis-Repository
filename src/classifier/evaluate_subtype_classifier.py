#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Basal vs LumA linear classifier on Virchow2 H5 tile features")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--features-dir", type=str, required=True)
    parser.add_argument("--patient-splits-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--label-csv-path", type=str, default=None)
    parser.add_argument("--max-tiles-per-patient", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def tile_metrics(y_true: np.ndarray, p_pos: np.ndarray, threshold: float) -> Dict[str, float | list]:
    y_pred = (p_pos >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize="true")

    metrics: Dict[str, float | list] = {
        "roc_auc": float(roc_auc_score(y_true, p_pos)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, p_pos)) if len(np.unique(y_true)) > 1 else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_luma": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_luma": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_basal": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_basal": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
    }
    return metrics


def patient_level_metrics(split_df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    grouped = split_df.groupby("patient_id").agg(
        p_pos_mean=("p_pos", "mean"),
        p_pos_median=("p_pos", "median"),
        y=("y_true", "first"),
    )

    if len(grouped) == 0:
        return {}

    y_true = grouped["y"].to_numpy(dtype=np.int64)
    p_mean = grouped["p_pos_mean"].to_numpy(dtype=np.float64)
    pred_mean = (p_mean >= threshold).astype(np.int64)

    out = {
        "n_patients": int(len(grouped)),
        "roc_auc_mean": float(roc_auc_score(y_true, p_mean)) if len(np.unique(y_true)) > 1 else float("nan"),
        "balanced_accuracy_mean": float(balanced_accuracy_score(y_true, pred_mean)),
    }
    return out


def hard_tile_metrics(split_df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    p = split_df["p_pos"].to_numpy(dtype=np.float64)
    y = split_df["y_true"].to_numpy(dtype=np.int64)

    pred = (p >= threshold).astype(np.int64)
    confident = (p >= 0.8) | (p <= 0.2)

    tile_conf_frac = float(confident.mean()) if len(confident) > 0 else float("nan")

    confidence = np.abs(p - 0.5)
    q25 = np.quantile(confidence, 0.25) if len(confidence) > 0 else 0.0
    hard_mask = confidence <= q25
    hard_acc = float((pred[hard_mask] == y[hard_mask]).mean()) if hard_mask.any() else float("nan")

    per_patient_var = split_df.groupby("patient_id")["p_pos"].var(ddof=0).fillna(0.0)
    return {
        "confident_tile_fraction": tile_conf_frac,
        "bottom_quartile_confidence_accuracy": hard_acc,
        "mean_patient_probability_variance": float(per_patient_var.mean()) if len(per_patient_var) > 0 else float("nan"),
    }


def evaluate_split(
    artifact: dict,
    x: np.ndarray,
    y: np.ndarray,
    meta_df: pd.DataFrame,
) -> Dict[str, object]:
    scaler = artifact["scaler"]
    clf = artifact["classifier"]
    threshold = float(artifact.get("threshold", 0.5))

    x_scaled = scaler.transform(x)
    p_pos = clf.predict_proba(x_scaled)[:, 1]

    eval_df = meta_df[["patient_id", "subtype", "tile_index"]].copy()
    eval_df["y_true"] = y
    eval_df["p_pos"] = p_pos

    return {
        "tile": tile_metrics(y, p_pos, threshold),
        "patient": patient_level_metrics(eval_df, threshold),
        "hard_tiles": hard_tile_metrics(eval_df, threshold),
        "n_tiles": int(len(eval_df)),
        "n_patients": int(eval_df["patient_id"].nunique()),
    }


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

    artifact = joblib.load(args.model_path)

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

    results: Dict[str, object] = {}
    for split_name in ("val", "test"):
        x, y, meta = split_to_arrays(df, split_name)
        results[split_name] = evaluate_split(artifact, x, y, meta)

    with open(out_dir / "evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[SubtypeClassifier] Evaluation complete")
    print(f"[SubtypeClassifier] Saved: {out_dir / 'evaluation_metrics.json'}")


if __name__ == "__main__":
    main()
