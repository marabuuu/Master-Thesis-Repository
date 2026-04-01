#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Latent Space Visualization Pipeline

This script produces comprehensive visualizations of the VAE latent space,
such as UMAP, t-SNE, PCA, Silhouette scores, dendrograms, and cosine distance heatmaps.
It relies on the `src.visualization` modules and uses Crameri colour maps.

Usage:
    python -m src.visualization.visualize_latents --config src/config.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict
import yaml
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add the project root to the python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from visualization import (
    build_embedding_matrix,
    prepare_labels,
    plot_umap,
    plot_tsne,
    plot_pca,
    plot_pca_variance,
    plot_umap_interactive,
    plot_cosine_distance_clustermap,
    plot_silhouette_per_group,
    plot_dendrogram,
    compute_silhouette,
    plot_kmeans_confusion,
    plot_countplot,
    plot_stacked_bar,
    plot_mosaic,
    setup_style,
)


def load_patient_splits(splits_json_path: str) -> Dict[str, str]:
    """Load patient splits from a JSON file.
    
    Expected format:
    {
        "train": {"patients": [...], "n_patients": ...},
        "val": {"patients": [...], "n_patients": ...},
        "test": {"patients": [...], "n_patients": ...}
    }
    
    Returns:
        dict: Mapping from patient_id → split_name ("train", "val", "test")
    """
    with open(splits_json_path, 'r') as f:
        splits_data = json.load(f)
    
    patient_to_split = {}
    for split_name in ["train", "val", "test"]:
        if split_name in splits_data:
            for patient_id in splits_data[split_name].get("patients", []):
                patient_to_split[patient_id] = split_name
    
    return patient_to_split


def load_latents_from_dir(latent_dir: str, patient_splits: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Load latent vectors and metadata directly from a flat folder of .h5 files.
    
    Args:
        latent_dir: Directory containing .h5 files
        patient_splits: Optional dict mapping patient_id → split_name from authoritative training metadata
    """
    rows = []
    h5_files = list(Path(latent_dir).glob("*.h5"))
    if not h5_files:
        raise ValueError(f"No .h5 files found in {latent_dir}")

    # Track which splits were found
    splits_found = {"train": 0, "val": 0, "test": 0, "unknown": 0}
    unmatched_patients = []

    for fp in tqdm(h5_files, desc="Loading latents"):
        with h5py.File(fp, "r") as f:
            # Load feature vector
            if "feats" in f:
                emb = np.array(f["feats"]).flatten()
            elif "z" in f:
                emb = np.array(f["z"]).flatten()
            else:
                emb = list(f.values())[0][()].flatten()
            
            # Clean NaN/Inf values
            if not np.isfinite(emb).all():
                emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Load attributes or fallback to filename
            pid = f.attrs.get("patient_id", fp.stem)
            if isinstance(pid, bytes):
                pid = pid.decode("utf-8")
                
            # Use patient_splits if provided (authoritative), else h5 attributes
            if patient_splits and pid in patient_splits:
                split = patient_splits[pid]
            else:
                split = f.attrs.get("split", "unknown")
                if isinstance(split, bytes):
                    split = split.decode("utf-8")
                if patient_splits and pid not in patient_splits:
                    unmatched_patients.append(pid)
            
            splits_found[split] = splits_found.get(split, 0) + 1
                
            rows.append({
                "patient_id": pid,
                "split": split,
                "embedding": emb,
                "file_path": str(fp)
            })
    
    # Print diagnostic info if test is missing
    if splits_found["test"] == 0 and patient_splits:
        print(f"⚠️  WARNING: No test patients found in h5 files!")
        print(f"   Splits found: train={splits_found['train']}, val={splits_found['val']}, test={splits_found['test']}")
        if unmatched_patients:
            print(f"   {len(unmatched_patients)} patient IDs from h5 files had no match in patient_splits.json")
            print(f"   Sample unmatched IDs: {unmatched_patients[:5]}")
    
    return pd.DataFrame(rows)


def run_visualizations(config: dict, verbose: bool = True):
    setup_style()

    latent_dir = config["latent_dir"]
    clinical_csv = config["csv_path"]
    out_dir = Path(config["output_dir"])
    label_col = config.get("label_col", "Majority_Subtype_mRNA")
    patient_col = config.get("patient_col", "Patient_ID")
    patient_splits_path = config.get("patient_splits_path")
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load patient splits if provided
    patient_splits = None
    if patient_splits_path and os.path.exists(patient_splits_path):
        if verbose:
            print(f"[Latents] Loading patient splits from {patient_splits_path}...")
        patient_splits = load_patient_splits(patient_splits_path)
        if verbose:
            print(f"[Latents] Loaded splits for {len(patient_splits)} patients")

    # 1. Load latents
    if verbose:
        print(f"[Latents] Loading from {latent_dir}...")
    df_emb = load_latents_from_dir(latent_dir, patient_splits=patient_splits)

    # 2. Load clinical data
    if verbose:
        print(f"[Latents] Loading clinical labels from {clinical_csv}...")
    df_clin = pd.read_csv(clinical_csv)
    # Ensure no leading/trailing spaces in columns
    df_clin.rename(columns=lambda c: c.strip(), inplace=True)

    if patient_col not in df_clin.columns:
        raise KeyError(f"Patient column '{patient_col}' not found in {clinical_csv}.")
    if label_col not in df_clin.columns:
        raise KeyError(f"Label column '{label_col}' not found in {clinical_csv}.")

    # 3. Merge
    df_all = df_emb.merge(df_clin[[patient_col, label_col]], left_on="patient_id", right_on=patient_col, how="inner")
    
    # Drop rows missing the target label
    df_all.dropna(subset=[label_col], inplace=True)
    
    if len(df_all) == 0:
        raise ValueError("Merged dataframe is empty! Check if patient IDs overlap between latents and clinical CSV.")
    
    if verbose:
        print(f"[Latents] Merged shape: {df_all.shape}. Building matrices...")

    X = build_embedding_matrix(df_all)
    y_subtype = prepare_labels(df_all, label_col)
    y_split = prepare_labels(df_all, "split")
    
    # Log split distribution
    if verbose:
        split_counts = pd.Series(y_split).value_counts()
        print(f"[Latents] Split distribution: {dict(split_counts)}")
        print(f"[Latents] Subtype distribution: {dict(pd.Series(y_subtype).value_counts())}")
    
    # Clean data: remove any rows with NaN or inf values
    # This is critical for sklearn/UMAP compatibility
    valid_mask = np.isfinite(X).all(axis=1)
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        if verbose:
            print(f"⚠️  Removing {n_invalid} samples with NaN/inf values")
        X = X[valid_mask]
        y_subtype = y_subtype[valid_mask]
        y_split = y_split[valid_mask]
        df_all = df_all[valid_mask]
    
    markers = {"train": "o", "test": "^", "val": "s", "unknown": "D"}

    # 4. Distribution Plots
    if verbose: print("[Latents] Plotting distributions...")
    plot_countplot(df_all, x=label_col, title=f"Samples per {label_col}", save_path=out_dir / "distribution_labels.png")
    plot_stacked_bar(df_all, group_col=label_col, stack_col="split", title="Split composition", save_path=out_dir / "distribution_splits.png")
    plot_mosaic(df_all, columns=[label_col, "split"], title="Mosaic", save_path=out_dir / "mosaic_splits.png")

    # 5. UMAP
    if verbose: print("[Latents] Computing UMAP...")
    X_umap, _ = plot_umap(X, y_subtype, style_labels=y_split, markers=markers, save_path=out_dir / "UMAP.png")
    df_all["UMAP1"] = X_umap[:, 0]
    df_all["UMAP2"] = X_umap[:, 1]

    # 6. t-SNE
    if verbose: print("[Latents] Computing t-SNE...")
    X_tsne, _ = plot_tsne(X, y_subtype, style_labels=y_split, markers=markers, save_path=out_dir / "tSNE.png")
    
    # 8. PCA
    if verbose: print("[Latents] Computing PCA...")
    X_pca, _, _ = plot_pca(X, y_subtype, style_labels=y_split, markers=markers, save_path=out_dir / "PCA.png")

    # 8.b PCA variance scree plot
    if verbose: print("[Latents] PCA variance scree plot...")
    try:
        plot_pca_variance(X, n_components=min(20, X.shape[1]), save_path=out_dir / "PCA_variance.png", show=False)
    except Exception as e:
        print(f"  ⚠️ PCA variance plot failed: {e}")

    # 8.c UMAP coloured by split (second view)
    if verbose: print("[Latents] UMAP coloured by split...")
    plot_umap(X, y_split, title="UMAP — coloured by data split", style_labels=y_subtype,
              markers={"LumA": "o", "LumB": "s", "Basal": "^", "Her2": "D", "Normal": "P"},
              save_path=out_dir / "UMAP_split.png", show=False)

    # 9. Cosine Distance Clustermap
    if verbose: print("[Latents] Plotting Cosine Clustermap...")
    n_sample = min(200, len(X))
    # Pass labels so we can see the subtype along the edges
    plot_cosine_distance_clustermap(X, labels=y_subtype, n_samples=n_sample, save_path=out_dir / "cosine_distance_matrix_clustermap.png")
    
    # 10. Silhouette
    if verbose: print("[Latents] Computing Silhouette scores...")
    try:
        sil_raw = compute_silhouette(X, y_subtype, metric="cosine")
        sil_umap = compute_silhouette(X_umap, y_subtype, metric="euclidean")
        if verbose:
            print(f"  Silhouette (raw 512-D, cosine):    {sil_raw:.3f}")
            print(f"  Silhouette (UMAP 2-D, euclidean):  {sil_umap:.3f}")

        plot_silhouette_per_group(X_umap, y_subtype, group_col_name=label_col, metric="euclidean", save_path=out_dir / "silhouette_per_subtype_umap.png")
    except Exception as e:
        print(f"  ⚠️ Silhouette failed: {e}")

    # 12. K-means confusion matrix on UMAP embedding
    if verbose: print("[Latents] K-means confusion matrix...")
    try:
        n_clusters = len(np.unique(y_subtype))
        plot_kmeans_confusion(X_umap, y_subtype, k=n_clusters,
                              save_path=out_dir / "kmeans_confusion_umap.png", show=False)
    except Exception as e:
        print(f"  ⚠️ K-means confusion failed: {e}")

    # 11. Dendrogram
    if verbose: print("[Latents] Plotting Dendrogram...")
    try:
        n_sample_dendro = min(150, len(X))
        plot_dendrogram(X, y_subtype, n_samples=n_sample_dendro, metric="cosine", method="ward", save_path=out_dir / "ward_dendrogram.png")
    except Exception as e:
        print(f"  ⚠️ Dendrogram failed: {e}")

    if verbose: print(f"[Latents] 🚀 All latent visualizations generated in {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        full_cfg = yaml.safe_load(f)

    if "visualize_latents" not in full_cfg:
        raise ValueError("Missing 'visualize_latents' key in config file.")

    run_visualizations(full_cfg["visualize_latents"])


if __name__ == "__main__":
    main()
