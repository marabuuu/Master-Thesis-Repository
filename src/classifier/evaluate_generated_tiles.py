#!/usr/bin/env python3
"""Evaluate a trained Basal-vs-LumA classifier on generated tile Virchow2 features.

Expects per-patient H5 files in a directory organised by subtype::

    generated_features_dir/
        Basal/
            TCGA-A2-A0ST.h5
            ...
        LumA/
            TCGA-XX-XXXX.h5
            ...

Each H5 file is produced by ``extract_virchow2_features.py`` and contains a
dataset ``features`` of shape ``(n_tiles, 1280)``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    roc_curve,
)

from .utils_subtype_data import canonical_patient_id, encode_labels, load_patient_h5_features


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _discover_generated_patients(
    features_dir: Path,
) -> List[Dict[str, Any]]:
    """Walk ``{features_dir}/{Basal,LumA}/*.h5`` and return per-patient records."""
    records: List[Dict[str, Any]] = []
    for subtype in ("Basal", "LumA"):
        subdir = features_dir / subtype
        if not subdir.is_dir():
            continue
        for h5 in sorted(subdir.glob("*.h5")):
            pid = canonical_patient_id(h5.stem)
            records.append({
                "patient_id": pid,
                "subtype": subtype,
                "h5_path": h5,
            })
    if not records:
        raise FileNotFoundError(
            f"No H5 files found under {{Basal,LumA}} subdirectories of {features_dir}"
        )
    return records


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_generated(
    model_path: str | Path,
    generated_features_dir: str | Path,
    output_dir: str | Path,
    cv_summary_path: Optional[str | Path] = None,
    max_tiles_per_patient: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run evaluation and save results."""
    model_path = Path(model_path)
    generated_features_dir = Path(generated_features_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = joblib.load(model_path)
    scaler = artifact["scaler"]
    clf = artifact["classifier"]
    threshold = float(artifact.get("threshold", 0.5))

    records = _discover_generated_patients(generated_features_dir)
    print(f"[GenEval] Found {len(records)} generated patient H5 files")

    rng = np.random.RandomState(seed)
    all_rows: List[Dict[str, Any]] = []
    patient_rows: List[Dict[str, Any]] = []

    for rec in records:
        pid, feats, _ = load_patient_h5_features(rec["h5_path"], rec["patient_id"])
        n = feats.shape[0]
        indices = np.arange(n)
        if max_tiles_per_patient is not None and n > max_tiles_per_patient:
            indices = rng.choice(indices, size=max_tiles_per_patient, replace=False)
        feats = feats[indices]

        x_scaled = scaler.transform(feats)
        p_pos = clf.predict_proba(x_scaled)[:, 1]
        y_true = 1 if rec["subtype"] == "Basal" else 0

        for i, (prob, idx) in enumerate(zip(p_pos, indices)):
            all_rows.append({
                "patient_id": pid,
                "subtype": rec["subtype"],
                "tile_index": int(idx),
                "y_true": y_true,
                "p_pos": float(prob),
            })

        mean_p = float(p_pos.mean())
        predicted = "Basal" if mean_p >= threshold else "LumA"
        patient_rows.append({
            "patient_id": pid,
            "subtype": rec["subtype"],
            "n_tiles": len(feats),
            "mean_p_basal": mean_p,
            "predicted_subtype": predicted,
            "correct": predicted == rec["subtype"],
        })

    tile_df = pd.DataFrame(all_rows)
    patient_df = pd.DataFrame(patient_rows)

    tile_df.to_parquet(output_dir / "tile_predictions.parquet", index=False)
    patient_df.to_parquet(output_dir / "per_patient_predictions.parquet", index=False)

    # Tile-level metrics
    y = tile_df["y_true"].to_numpy(dtype=np.int64)
    p = tile_df["p_pos"].to_numpy(dtype=np.float64)
    y_pred = (p >= threshold).astype(np.int64)
    has_both = len(np.unique(y)) > 1

    tile_metrics = {
        "n_tiles": int(len(y)),
        "roc_auc": float(roc_auc_score(y, p)) if has_both else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if has_both else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "precision_basal": float(precision_score(y, y_pred, pos_label=1, zero_division=0)),
        "recall_basal": float(recall_score(y, y_pred, pos_label=1, zero_division=0)),
        "precision_luma": float(precision_score(y, y_pred, pos_label=0, zero_division=0)),
        "recall_luma": float(recall_score(y, y_pred, pos_label=0, zero_division=0)),
        "confusion_matrix_normalized": confusion_matrix(y, y_pred, labels=[0, 1], normalize="true").tolist(),
    }

    # Patient-level metrics
    py = patient_df["subtype"].map({"LumA": 0, "Basal": 1}).to_numpy(dtype=np.int64)
    pp = patient_df["mean_p_basal"].to_numpy(dtype=np.float64)
    py_pred = (pp >= threshold).astype(np.int64)
    has_both_p = len(np.unique(py)) > 1

    pat_metrics = {
        "n_patients": int(len(patient_df)),
        "n_basal": int((py == 1).sum()),
        "n_luma": int((py == 0).sum()),
        "roc_auc": float(roc_auc_score(py, pp)) if has_both_p else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(py, py_pred)),
        "accuracy": float(patient_df["correct"].mean()),
    }

    results: Dict[str, Any] = {
        "threshold": threshold,
        "tile_metrics": tile_metrics,
        "patient_metrics": pat_metrics,
    }

    # Load CV summary for comparison if available
    if cv_summary_path and Path(cv_summary_path).exists():
        with open(cv_summary_path) as f:
            cv_summary = json.load(f)
        results["comparison_with_real_cv"] = {
            "real_tile_auroc": cv_summary["aggregate_tile_metrics"]["roc_auc"],
            "real_tile_auroc_ci": [
                cv_summary["bootstrap_ci"]["tile_roc_auc"]["ci_lower"],
                cv_summary["bootstrap_ci"]["tile_roc_auc"]["ci_upper"],
            ],
            "generated_tile_auroc": tile_metrics["roc_auc"],
            "real_patient_auroc": cv_summary["aggregate_patient_metrics"]["roc_auc"],
            "generated_patient_auroc": pat_metrics["roc_auc"],
        }

    with open(output_dir / "evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plots
    _plot_generated_results(tile_df, patient_df, results, threshold, output_dir / "plots")

    print(f"[GenEval] Tile AUROC: {tile_metrics['roc_auc']:.4f}")
    print(f"[GenEval] Patient accuracy: {pat_metrics['accuracy']:.4f} "
          f"({int(patient_df['correct'].sum())}/{len(patient_df)})")
    print(f"[GenEval] Results saved to: {output_dir}")
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_generated_results(
    tile_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    results: Dict[str, Any],
    threshold: float,
    plots_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[GenEval][WARN] matplotlib not available, skipping plots")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    # Probability histogram
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for subtype, color in [("LumA", "#1f77b4"), ("Basal", "#d62728")]:
        vals = tile_df.loc[tile_df["subtype"] == subtype, "p_pos"].to_numpy()
        ax.hist(vals, bins=50, alpha=0.6, color=color, label=subtype, density=True)
    ax.axvline(threshold, color="black", ls="--", lw=1, label=f"threshold={threshold:.2f}")
    ax.set_xlabel("P(Basal)")
    ax.set_ylabel("Density")
    ax.set_title("Generated Tiles — Predicted P(Basal) by Conditioning Subtype")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "generated_probability_histogram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ROC curve
    y = tile_df["y_true"].to_numpy(dtype=np.int64)
    p = tile_df["p_pos"].to_numpy(dtype=np.float64)
    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, p)
        auc_val = roc_auc_score(y, p)
        fig, ax = plt.subplots(figsize=(6, 5.5))
        ax.plot(fpr, tpr, color="#2ca02c", lw=2, label=f"Generated AUROC = {auc_val:.3f}")

        comp = results.get("comparison_with_real_cv")
        if comp:
            ci = comp.get("real_tile_auroc_ci", [])
            real_auc = comp["real_tile_auroc"]
            ci_str = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
            ax.axhline(y=real_auc, color="#1f77b4", ls=":", lw=1,
                       label=f"Real CV AUROC = {real_auc:.3f}{ci_str}")

        ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC — Generated Tiles (Basal vs LumA)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(plots_dir / "generated_roc.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Comparison bar chart
    comp = results.get("comparison_with_real_cv")
    if comp:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        labels = ["Tile AUROC", "Patient AUROC"]
        real_vals = [comp["real_tile_auroc"], comp["real_patient_auroc"]]
        gen_vals = [comp["generated_tile_auroc"], comp["generated_patient_auroc"]]
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, real_vals, w, label="Real (CV)", color="#1f77b4", alpha=0.8)
        ax.bar(x + w / 2, gen_vals, w, label="Generated", color="#2ca02c", alpha=0.8)
        ci = comp.get("real_tile_auroc_ci", [])
        if ci:
            ax.errorbar(x[0] - w / 2, real_vals[0],
                        yerr=[[real_vals[0] - ci[0]], [ci[1] - real_vals[0]]],
                        fmt="none", color="black", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.5, 1.0)
        ax.set_title("Real CV vs Generated — Subtype Separability")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "real_vs_generated_comparison.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Config-driven entry
# ---------------------------------------------------------------------------

def run_generated_eval_pipeline(cfg: Dict[str, Any]) -> None:
    """Config-driven entry point called from ``run_pipeline.py``."""
    model_path = cfg.get("model_path")
    if not model_path:
        raise ValueError("Missing 'generated_subtype_eval.model_path'")
    generated_features_dir = cfg.get("generated_features_dir")
    if not generated_features_dir:
        raise ValueError("Missing 'generated_subtype_eval.generated_features_dir'")
    output_dir = cfg.get("output_dir")
    if not output_dir:
        raise ValueError("Missing 'generated_subtype_eval.output_dir'")

    evaluate_generated(
        model_path=model_path,
        generated_features_dir=generated_features_dir,
        output_dir=output_dir,
        cv_summary_path=cfg.get("cv_summary_path"),
        max_tiles_per_patient=cfg.get("max_tiles_per_patient"),
        seed=int(cfg.get("seed", 42)),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate classifier on generated tiles")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--generated-features-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--cv-summary-path", type=str, default=None)
    p.add_argument("--max-tiles-per-patient", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_generated(
        model_path=args.model_path,
        generated_features_dir=args.generated_features_dir,
        output_dir=args.output_dir,
        cv_summary_path=args.cv_summary_path,
        max_tiles_per_patient=args.max_tiles_per_patient,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
