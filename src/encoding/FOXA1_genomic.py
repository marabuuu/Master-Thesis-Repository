"""FOXA1-only genomic feature extraction.

This module extracts the FOXA1 gene expression value for each sample and
creates per-sample H5 files where the feature vector is simply the FOXA1
value repeated 512 times, matching the shape (1, 512) expected by downstream
models.

This is useful for baseline/ablation studies comparing full genomic conditioning
against single-gene conditioning using FOXA1.

Usage (from config.yaml):
    python -m src.encoding.FOXA1_genomic --config src/config.yaml

Usage (command line):
    python src/encoding/FOXA1_genomic.py \\
        --csv-path /path/to/expression.csv \\
        --output-dir ./foxa1_output \\
        --patient-col Patient_ID
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, List

import h5py
import numpy as np
import pandas as pd
import yaml


def load_expression_data(
    csv_path: str,
    patient_col: str = "Patient_ID",
) -> Tuple[pd.DataFrame, list]:
    """Load gene expression matrix.

    Parameters
    ----------
    csv_path : str
        Path to CSV file with gene expression data.
    patient_col : str
        Name of patient ID column.

    Returns
    -------
    expr : pd.DataFrame
        Expression matrix (patients x genes). Index contains sample IDs.
    patient_ids : list
        Patient identifiers corresponding to rows.
    """
    df = pd.read_csv(csv_path)
    df.set_index(patient_col, inplace=True)
    patient_ids = df.index.tolist()
    return df, patient_ids


def extract_foxa1(
    expr: pd.DataFrame,
    patient_ids: list,
    output_dir: str,
    patient_col: str = "Patient_ID",
    feature_dim: int = 512,
) -> None:
    """Extract FOXA1 value and create per-sample H5 files.

    For each sample, the FOXA1 expression value is repeated `feature_dim` times
    to create a feature vector of shape (1, feature_dim), saved as dataset "feats"
    in an H5 file.

    Parameters
    ----------
    expr : pd.DataFrame
        Expression matrix (samples x genes). Index contains sample IDs.
    patient_ids : list
        List of sample identifiers (in same order as expr rows).
    output_dir : str
        Directory to save per-sample H5 files.
    patient_col : str
        Column name to use in CSV output (for consistency).
    feature_dim : int
        Dimension of the output feature vector (default 512).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if FOXA1 is in the expression matrix
    if "FOXA1" not in expr.columns:
        raise ValueError(
            f"FOXA1 not found in expression matrix. Available genes: {list(expr.columns[:10])}..."
        )

    # explicit typing to satisfy static analyzer
    # ensure we have a raw numpy array (pandas may return ExtensionArray)
    foxa1_values: np.ndarray = np.asarray(expr["FOXA1"].values)

    # Log FOXA1 value statistics
    print(f"\nFOXA1 expression value range:")
    min_val = np.min(foxa1_values).item()
    max_val = np.max(foxa1_values).item()
    mean_val = np.mean(foxa1_values).item()
    median_val = np.median(foxa1_values).item()
    print(f"  Min : {min_val:.6f}")
    print(f"  Max : {max_val:.6f}")
    print(f"  Mean: {mean_val:.6f}")
    print(f"  Median: {median_val:.6f}")
    print()

    # Check for duplicate sample IDs and warn if found
    if len(patient_ids) != len(set(patient_ids)):
        duplicates = [pid for pid in set(patient_ids) if patient_ids.count(pid) > 1]
        print(f"WARNING: Found duplicate sample IDs: {duplicates}")
        print(f"Each duplicate will be saved with a numeric suffix (-DX1, -DX2, etc.)")

    # Save per-sample H5 files
    print(f"Saving per-sample H5 files to {output_dir}")
    sample_id_counts = {}

    for idx, patient_id in enumerate(patient_ids):
        foxa1_val = foxa1_values[idx]

        # If sample ID appears multiple times, append -DX suffix (-DX1, -DX2, etc.)
        if patient_ids.count(patient_id) > 1:
            if patient_id not in sample_id_counts:
                sample_id_counts[patient_id] = 1
            else:
                sample_id_counts[patient_id] += 1
            h5_name = f"{patient_id}-DX{sample_id_counts[patient_id]}.h5"
        else:
            h5_name = f"{patient_id}.h5"

        # Create feature vector: FOXA1 value repeated feature_dim times
        # Shape: (1, feature_dim) as float32
        feature_vector = np.full((1, feature_dim), foxa1_val, dtype=np.float32)

        h5_path = os.path.join(output_dir, h5_name)
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("feats", data=feature_vector, compression="gzip")
            f.attrs["patient_id"] = patient_id
            f.attrs["sample_index"] = idx
            f.attrs["gene"] = "FOXA1"
            f.attrs["foxa1_value"] = float(foxa1_val)

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(patient_ids)} samples")

    print(f"Saved {len(patient_ids)} sample H5 files to {output_dir}")

    # Save summary CSV
    summary_df = pd.DataFrame({
        patient_col: patient_ids,
        "FOXA1": foxa1_values,
    })
    csv_path = os.path.join(output_dir, "FOXA1_values.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved FOXA1 summary to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract FOXA1 gene and create single-gene genomic features"
    )

    # Declare all arguments first, then load config defaults
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides command-line args)",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        help="Path to gene expression CSV (samples × genes)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for per-sample H5 output files",
    )
    parser.add_argument(
        "--patient-col",
        type=str,
        default="Patient_ID",
        help="Patient ID column name (default 'Patient_ID')",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=512,
        help="Dimension of output feature vector (default 512)",
    )

    # First pass: peek at --config so we can load YAML defaults
    temp_args, _ = parser.parse_known_args()

    if temp_args.config:
        try:
            with open(temp_args.config) as f:
                cfg = yaml.safe_load(f) or {}
            if isinstance(cfg, dict) and "encoding" in cfg:
                cfg = cfg["encoding"] or {}
            # Apply YAML values as defaults
            parser.set_defaults(**cfg)
        except Exception as e:
            print(f"Failed to load config {temp_args.config}: {e}")

    # Final parse using potentially updated defaults
    args = parser.parse_args()

    # Validate required arguments
    if not args.csv_path:
        parser.error("--csv-path is required")
    if not args.output_dir:
        parser.error("--output-dir is required")

    print(f"\n{'='*60}")
    print("FOXA1 GENOMIC FEATURE EXTRACTION")
    print(f"  Input CSV : {args.csv_path}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Feature dim: {args.feature_dim}")
    print(f"{'='*60}\n")

    # Load expression data
    print(f"Loading data from {args.csv_path}...")
    expr, patient_ids = load_expression_data(
        args.csv_path,
        patient_col=args.patient_col,
    )
    print(f"Loaded {expr.shape[0]} samples × {expr.shape[1]} genes")

    # Extract FOXA1 and create H5 files
    extract_foxa1(
        expr,
        patient_ids,
        args.output_dir,
        patient_col=args.patient_col,
        feature_dim=args.feature_dim,
    )

    print("\nFOXA1 feature extraction complete!")


if __name__ == "__main__":
    main()
