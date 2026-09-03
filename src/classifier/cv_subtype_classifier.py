#!/usr/bin/env python3
"""5-fold cross-validated Basal vs LumA classifier with bootstrap CIs.

Trains a logistic regression on Virchow2 tile embeddings using
patient-level stratified k-fold cross-validation.  Produces:
  - per-fold and aggregated metrics with 95% bootstrap CIs
  - a final model trained on all data (for application to generated tiles)
  - ROC/PR curves with confidence bands, fold-comparison bar chart
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .utils_subtype_data import (
    build_tile_feature_table_all,
    canonical_patient_id,
    encode_labels,
    infer_label_csv_and_columns,
    load_subtype_table,
    load_yaml,
    patients_to_arrays,
)


def _resolve_path(path_value: str, config_path: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    config_dir = Path(config_path).resolve().parent
    repo_root = config_dir.parent if config_dir.name == "src" else config_dir
    normalized = path_value[2:] if str(path_value).startswith("./") else str(path_value)
    if normalized.startswith(("data/", "dataframes/", "experiments/")):
        parent_candidate = (repo_root.parent / normalized).resolve()
        if parent_candidate.exists() or not (repo_root / path).resolve().exists():
            return str(parent_candidate)
    return str((repo_root / path).resolve())


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _tile_metrics(y_true: np.ndarray, p_pos: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (p_pos >= threshold).astype(np.int64)
    has_both = len(np.unique(y_true)) > 1
    return {
        "roc_auc": float(roc_auc_score(y_true, p_pos)) if has_both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, p_pos)) if has_both else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_basal": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_basal": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_luma": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_luma": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
    }


def _patient_metrics(df: pd.DataFrame, threshold: float) -> Dict[str, Any]:
    grouped = df.groupby("patient_id").agg(
        p_mean=("p_pos", "mean"),
        y=("y_true", "first"),
    )
    y = grouped["y"].to_numpy(dtype=np.int64)
    p = grouped["p_mean"].to_numpy(dtype=np.float64)
    pred = (p >= threshold).astype(np.int64)
    has_both = len(np.unique(y)) > 1
    return {
        "n_patients": int(len(grouped)),
        "roc_auc": float(roc_auc_score(y, p)) if has_both else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def _tune_threshold(y_true: np.ndarray, p_pos: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_s = 0.5, -1.0
    for t in thresholds:
        s = balanced_accuracy_score(y_true, (p_pos >= t).astype(np.int64))
        if s > best_s:
            best_s = s
            best_t = float(t)
    return best_t


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def _run_cross_validation(
    tile_df: pd.DataFrame,
    n_splits: int,
    Cs: List[float],
    solver: str,
    max_iter: int,
    class_weight: str,
    seed: int,
    out_dir: Path,
) -> pd.DataFrame:
    """Run stratified k-fold CV at patient level; return out-of-fold predictions."""

    patient_subtypes = tile_df.groupby("patient_id")["subtype"].first()
    patient_ids = patient_subtypes.index.to_numpy()
    y_patients = encode_labels(patient_subtypes.values)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_parts: List[pd.DataFrame] = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(patient_ids, y_patients)):
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_pids = patient_ids[train_idx]
        test_pids = patient_ids[test_idx]

        x_train, y_train, _ = patients_to_arrays(tile_df, train_pids)
        x_test, y_test, test_meta = patients_to_arrays(tile_df, test_pids)

        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s = scaler.transform(x_test)

        cw = None if class_weight == "none" else "balanced"
        clf = LogisticRegressionCV(
            Cs=Cs, cv=5, solver=solver, class_weight=cw,
            scoring="balanced_accuracy", max_iter=max_iter,
            random_state=seed, n_jobs=-1,
        )
        clf.fit(x_train_s, y_train)

        p_pos = clf.predict_proba(x_test_s)[:, 1]
        fold_df = test_meta[["patient_id", "subtype", "tile_index"]].copy()
        fold_df["y_true"] = y_test
        fold_df["p_pos"] = p_pos
        fold_df["fold"] = fold

        fold_df.to_parquet(fold_dir / "predictions.parquet", index=False)

        threshold = _tune_threshold(y_test, p_pos)
        fold_metrics = {
            "fold": fold,
            "n_train_patients": int(len(train_pids)),
            "n_test_patients": int(len(test_pids)),
            "n_train_tiles": int(len(y_train)),
            "n_test_tiles": int(len(y_test)),
            "best_C": float(clf.C_[0]),
            "threshold": threshold,
            "tile": _tile_metrics(y_test, p_pos, threshold),
            "patient": _patient_metrics(fold_df, threshold),
        }
        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(fold_metrics, f, indent=2)

        print(f"  Fold {fold}: tile AUROC={fold_metrics['tile']['roc_auc']:.4f}  "
              f"patient AUROC={fold_metrics['patient']['roc_auc']:.4f}  "
              f"best_C={fold_metrics['best_C']}")

        oof_parts.append(fold_df)

    oof_df = pd.concat(oof_parts, ignore_index=True)
    oof_df.to_parquet(out_dir / "all_oof_predictions.parquet", index=False)
    return oof_df


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _bootstrap_confidence_intervals(
    oof_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Resample patients with replacement and compute CIs for key metrics."""

    patient_ids = oof_df["patient_id"].unique()
    rng = np.random.RandomState(seed)

    metric_names = [
        "tile_roc_auc", "tile_pr_auc", "tile_balanced_accuracy", "tile_f1_macro",
        "tile_precision_basal", "tile_recall_basal",
        "patient_roc_auc", "patient_balanced_accuracy",
    ]
    samples = {m: [] for m in metric_names}

    for _ in range(n_bootstrap):
        boot_pids = rng.choice(patient_ids, size=len(patient_ids), replace=True)

        parts = []
        for pid in boot_pids:
            parts.append(oof_df[oof_df["patient_id"] == pid])
        boot_df = pd.concat(parts, ignore_index=True)

        y = boot_df["y_true"].to_numpy(dtype=np.int64)
        p = boot_df["p_pos"].to_numpy(dtype=np.float64)

        if len(np.unique(y)) < 2:
            continue

        tm = _tile_metrics(y, p, threshold)
        pm = _patient_metrics(boot_df, threshold)

        samples["tile_roc_auc"].append(tm["roc_auc"])
        samples["tile_pr_auc"].append(tm["pr_auc"])
        samples["tile_balanced_accuracy"].append(tm["balanced_accuracy"])
        samples["tile_f1_macro"].append(tm["f1_macro"])
        samples["tile_precision_basal"].append(tm["precision_basal"])
        samples["tile_recall_basal"].append(tm["recall_basal"])
        samples["patient_roc_auc"].append(pm["roc_auc"])
        samples["patient_balanced_accuracy"].append(pm["balanced_accuracy"])

    result = {}
    for name, vals in samples.items():
        arr = np.array(vals)
        result[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
        }
    return result


# ---------------------------------------------------------------------------
# Final model
# ---------------------------------------------------------------------------

def _train_final_model(
    tile_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    Cs: List[float],
    solver: str,
    max_iter: int,
    class_weight: str,
    seed: int,
    out_dir: Path,
) -> None:
    """Train a single model on ALL data for application to generated tiles."""
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pids = tile_df["patient_id"].unique()
    x, y, _ = patients_to_arrays(tile_df, all_pids)

    scaler = StandardScaler()
    x_s = scaler.fit_transform(x)

    cw = None if class_weight == "none" else "balanced"
    clf = LogisticRegressionCV(
        Cs=Cs, cv=5, solver=solver, class_weight=cw,
        scoring="balanced_accuracy", max_iter=max_iter,
        random_state=seed, n_jobs=-1,
    )
    clf.fit(x_s, y)

    oof_y = oof_df["y_true"].to_numpy(dtype=np.int64)
    oof_p = oof_df["p_pos"].to_numpy(dtype=np.float64)
    threshold = _tune_threshold(oof_y, oof_p)

    artifact = {
        "scaler": scaler,
        "classifier": clf,
        "threshold": threshold,
        "label_mapping": {"LumA": 0, "Basal": 1},
        "feature_dim": int(x.shape[1]),
        "config": {
            "solver": solver,
            "Cs_searched": Cs,
            "best_C": float(clf.C_[0]),
            "max_iter": max_iter,
            "class_weight": class_weight,
            "seed": seed,
            "n_patients": int(len(all_pids)),
            "n_tiles": int(len(y)),
        },
    }
    model_path = out_dir / "subtype_linear_model.joblib"
    joblib.dump(artifact, model_path)

    summary = {
        "n_patients": int(len(all_pids)),
        "n_tiles": int(len(y)),
        "n_basal_tiles": int(y.sum()),
        "n_luma_tiles": int((1 - y).sum()),
        "best_C": float(clf.C_[0]),
        "threshold": threshold,
        "feature_dim": int(x.shape[1]),
        "model_path": str(model_path),
    }
    with open(out_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[CV] Final model: {model_path}  (C={clf.C_[0]}, threshold={threshold:.4f})")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_roc_with_ci(
    oof_df: pd.DataFrame,
    bootstrap_ci: Dict[str, Dict[str, float]],
    threshold: float,
    n_bootstrap: int,
    seed: int,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    y = oof_df["y_true"].to_numpy(dtype=np.int64)
    p = oof_df["p_pos"].to_numpy(dtype=np.float64)
    fpr, tpr, _ = roc_curve(y, p)
    auc_val = roc_auc_score(y, p)
    ci = bootstrap_ci["tile_roc_auc"]

    rng = np.random.RandomState(seed)
    patient_ids = oof_df["patient_id"].unique()
    base_fpr = np.linspace(0, 1, 200)
    tpr_samples = []
    for _ in range(n_bootstrap):
        boot_pids = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        parts = [oof_df[oof_df["patient_id"] == pid] for pid in boot_pids]
        bdf = pd.concat(parts, ignore_index=True)
        by = bdf["y_true"].to_numpy(dtype=np.int64)
        bp = bdf["p_pos"].to_numpy(dtype=np.float64)
        if len(np.unique(by)) < 2:
            continue
        b_fpr, b_tpr, _ = roc_curve(by, bp)
        tpr_samples.append(np.interp(base_fpr, b_fpr, b_tpr))

    tpr_stack = np.array(tpr_samples)
    tpr_lo = np.percentile(tpr_stack, 2.5, axis=0)
    tpr_hi = np.percentile(tpr_stack, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.fill_between(base_fpr, tpr_lo, tpr_hi, alpha=0.2, color="#1f77b4", label="95% CI")
    ax.plot(fpr, tpr, color="#1f77b4", lw=2,
            label=f"AUROC = {auc_val:.3f} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — 5-Fold CV (Basal vs LumA)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_pr_with_ci(
    oof_df: pd.DataFrame,
    bootstrap_ci: Dict[str, Dict[str, float]],
    threshold: float,
    n_bootstrap: int,
    seed: int,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    y = oof_df["y_true"].to_numpy(dtype=np.int64)
    p = oof_df["p_pos"].to_numpy(dtype=np.float64)
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)

    rng = np.random.RandomState(seed)
    patient_ids = oof_df["patient_id"].unique()
    base_rec = np.linspace(0, 1, 200)
    prec_samples = []
    for _ in range(n_bootstrap):
        boot_pids = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        parts = [oof_df[oof_df["patient_id"] == pid] for pid in boot_pids]
        bdf = pd.concat(parts, ignore_index=True)
        by = bdf["y_true"].to_numpy(dtype=np.int64)
        bp = bdf["p_pos"].to_numpy(dtype=np.float64)
        if len(np.unique(by)) < 2:
            continue
        b_prec, b_rec, _ = precision_recall_curve(by, bp)
        prec_samples.append(np.interp(base_rec, b_rec[::-1], b_prec[::-1]))

    prec_stack = np.array(prec_samples)
    prec_lo = np.percentile(prec_stack, 2.5, axis=0)
    prec_hi = np.percentile(prec_stack, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.fill_between(base_rec, prec_lo, prec_hi, alpha=0.2, color="#d62728", label="95% CI")
    ax.plot(rec, prec, color="#d62728", lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — 5-Fold CV (Basal vs LumA)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_fold_comparison(cv_dir: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    folds = []
    for fold_dir in sorted(cv_dir.glob("fold_*")):
        mp = fold_dir / "metrics.json"
        if mp.exists():
            with open(mp) as f:
                folds.append(json.load(f))
    if not folds:
        return

    metrics = ["roc_auc", "balanced_accuracy", "f1_macro"]
    labels = ["AUROC", "Bal. Accuracy", "F1 (macro)"]
    n_folds = len(folds)
    x = np.arange(len(metrics))
    width = 0.8 / n_folds

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, fm in enumerate(folds):
        vals = [fm["tile"].get(m, 0.0) for m in metrics]
        ax.bar(x + i * width, vals, width, label=f"Fold {fm['fold']}", alpha=0.85)

    ax.set_xticks(x + width * (n_folds - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Tile-Level Metrics per Fold")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_probability_histogram(oof_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for subtype, color in [("LumA", "#1f77b4"), ("Basal", "#d62728")]:
        vals = oof_df.loc[oof_df["subtype"] == subtype, "p_pos"].to_numpy()
        ax.hist(vals, bins=50, alpha=0.6, color=color, label=subtype, density=True)
    ax.set_xlabel("P(Basal)")
    ax.set_ylabel("Density")
    ax.set_title("Predicted P(Basal) by True Subtype — OOF Predictions")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(oof_df: pd.DataFrame, threshold: float, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    y = oof_df["y_true"].to_numpy(dtype=np.int64)
    p = oof_df["p_pos"].to_numpy(dtype=np.float64)
    y_pred = (p >= threshold).astype(np.int64)
    cm = confusion_matrix(y, y_pred, labels=[0, 1], normalize="true")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["LumA", "Basal"])
    ax.set_yticklabels(["LumA", "Basal"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_cv_pipeline(cfg: Dict[str, Any]) -> None:
    """Config-driven entry point for the CV classifier pipeline."""
    config_path = cfg.get("config_path", cfg.get("_config_path", ""))
    features_dir = cfg["features_dir"]
    output_dir = Path(cfg["output_dir"])
    max_tiles = cfg.get("max_tiles_per_patient")
    if max_tiles is not None:
        max_tiles = int(max_tiles)
    seed = int(cfg.get("seed", 42))
    n_folds = int(cfg.get("n_folds", 5))
    n_bootstrap = int(cfg.get("n_bootstrap", 1000))
    solver = str(cfg.get("solver", "liblinear"))
    Cs = list(cfg.get("Cs", [0.01, 0.1, 1.0, 10.0, 100.0]))
    max_iter = int(cfg.get("max_iter", 2000))
    class_weight = str(cfg.get("class_weight", "balanced"))

    # Load labels
    if config_path:
        full_cfg = load_yaml(config_path)
    else:
        full_cfg = {}
    label_csv_path = cfg.get("label_csv_path")
    patient_col = cfg.get("patient_col", None)
    subtype_col = cfg.get("subtype_col", None)
    if label_csv_path is None:
        label_csv_path, patient_col_inf, subtype_col_inf = infer_label_csv_and_columns(full_cfg, None)
        patient_col = patient_col or patient_col_inf
        subtype_col = subtype_col or subtype_col_inf
        if config_path:
            label_csv_path = _resolve_path(label_csv_path, config_path)
    else:
        patient_col = patient_col or "Patient_ID"
        subtype_col = subtype_col or "Majority_Subtype_mRNA"
        if config_path and not Path(label_csv_path).is_absolute():
            label_csv_path = _resolve_path(label_csv_path, config_path)

    print(f"[CV] Loading labels from: {label_csv_path}")
    subtype_df = load_subtype_table(label_csv_path, patient_col, subtype_col)
    print(f"[CV] {len(subtype_df)} patients with valid Basal/LumA labels")

    print(f"[CV] Loading tile features from: {features_dir}")
    tile_df = build_tile_feature_table_all(
        features_dir=features_dir,
        subtype_df=subtype_df,
        max_tiles_per_patient=max_tiles,
        seed=seed,
    )
    n_patients = tile_df["patient_id"].nunique()
    n_basal = tile_df[tile_df["subtype"] == "Basal"]["patient_id"].nunique()
    n_luma = tile_df[tile_df["subtype"] == "LumA"]["patient_id"].nunique()
    print(f"[CV] Loaded {len(tile_df)} tiles from {n_patients} patients "
          f"(Basal={n_basal}, LumA={n_luma})")

    # 1) Cross-validation
    cv_dir = output_dir / "cv_results"
    cv_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[CV] Running {n_folds}-fold cross-validation...")
    oof_df = _run_cross_validation(
        tile_df, n_splits=n_folds, Cs=Cs, solver=solver,
        max_iter=max_iter, class_weight=class_weight, seed=seed,
        out_dir=cv_dir,
    )

    # Aggregate OOF metrics
    oof_y = oof_df["y_true"].to_numpy(dtype=np.int64)
    oof_p = oof_df["p_pos"].to_numpy(dtype=np.float64)
    oof_threshold = _tune_threshold(oof_y, oof_p)

    agg_tile = _tile_metrics(oof_y, oof_p, oof_threshold)
    agg_patient = _patient_metrics(oof_df, oof_threshold)

    # 2) Bootstrap CIs
    print(f"\n[CV] Computing bootstrap CIs (n={n_bootstrap})...")
    boot_ci = _bootstrap_confidence_intervals(
        oof_df, n_bootstrap=n_bootstrap, seed=seed, threshold=oof_threshold,
    )

    cv_summary = {
        "n_folds": n_folds,
        "n_patients": n_patients,
        "n_tiles": int(len(oof_df)),
        "n_basal_patients": n_basal,
        "n_luma_patients": n_luma,
        "threshold": oof_threshold,
        "aggregate_tile_metrics": agg_tile,
        "aggregate_patient_metrics": agg_patient,
        "bootstrap_ci": boot_ci,
        "n_bootstrap": n_bootstrap,
    }
    with open(cv_dir / "cv_summary.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    print(f"\n[CV] Aggregate tile AUROC: {agg_tile['roc_auc']:.4f} "
          f"[{boot_ci['tile_roc_auc']['ci_lower']:.4f}, "
          f"{boot_ci['tile_roc_auc']['ci_upper']:.4f}]")
    print(f"[CV] Aggregate patient AUROC: {agg_patient['roc_auc']:.4f} "
          f"[{boot_ci['patient_roc_auc']['ci_lower']:.4f}, "
          f"{boot_ci['patient_roc_auc']['ci_upper']:.4f}]")

    # 3) Final model
    print("\n[CV] Training final model on all data...")
    _train_final_model(
        tile_df, oof_df, Cs=Cs, solver=solver, max_iter=max_iter,
        class_weight=class_weight, seed=seed,
        out_dir=output_dir / "final_model",
    )

    # 4) Plots
    print("\n[CV] Generating plots...")
    plots_dir = cv_dir / "plots"
    _plot_roc_with_ci(oof_df, boot_ci, oof_threshold, n_bootstrap, seed,
                      plots_dir / "roc_with_ci.png")
    _plot_pr_with_ci(oof_df, boot_ci, oof_threshold, n_bootstrap, seed,
                     plots_dir / "pr_with_ci.png")
    _plot_fold_comparison(cv_dir, plots_dir / "fold_comparison.png")
    _plot_probability_histogram(oof_df, plots_dir / "probability_histogram.png")
    _plot_confusion_matrix(oof_df, oof_threshold, plots_dir / "confusion_matrix.png")

    print(f"\n[CV] Done. Results saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-validated Basal vs LumA classifier with bootstrap CIs",
    )
    p.add_argument("--config", type=str, default="",
                    help="Path to src/config.yaml (for label CSV inference)")
    p.add_argument("--features-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--label-csv-path", type=str, default=None)
    p.add_argument("--patient-col", type=str, default=None)
    p.add_argument("--subtype-col", type=str, default=None)
    p.add_argument("--max-tiles-per-patient", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--solver", type=str, default="liblinear")
    p.add_argument("--Cs", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--class-weight", type=str, default="balanced")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = {
        "config_path": args.config,
        "features_dir": args.features_dir,
        "output_dir": args.output_dir,
        "label_csv_path": args.label_csv_path,
        "patient_col": args.patient_col,
        "subtype_col": args.subtype_col,
        "max_tiles_per_patient": args.max_tiles_per_patient,
        "seed": args.seed,
        "n_folds": args.n_folds,
        "n_bootstrap": args.n_bootstrap,
        "solver": args.solver,
        "Cs": args.Cs,
        "max_iter": args.max_iter,
        "class_weight": args.class_weight,
    }
    run_cv_pipeline(cfg)


if __name__ == "__main__":
    main()
