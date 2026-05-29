"""Gene subset selection module for dimensionality reduction.

This module implements a gene-subselection strategy to reduce high-dimensional
gene expression matrices (~19k genes) to a compact, biologically meaningful panel
(~1k-2k genes) that best separates PAM50 subtypes. The selected genes are then
preprocessed with log1p + z-score normalization to match VAE input requirements.

Gene selection strategy:
  1. Filter low-expressed genes (not expressed in >= min_nonzero samples)
  2. Perform OLS regression per gene with binary phenotype (Basal vs LumA)
  3. Score genes: |logFC| × -log₁₀(FDR)
  4. Select top N genes + force-include classic PAM50 markers
  5. Optional: prune highly correlated genes (ρ > threshold)
  6. Output: per-patient CSV and H5 files with normalized gene expressions
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Set
import h5py

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler

from preprocessing.utils import preprocess_log1p_zscore


def load_phenotype_data(
    csv_path: str,
    patient_col: str = "Patient_ID",
    subtype_col: str = "Majority_Subtype_mRNA",
    wanted_subtypes: Optional[list] = None,
) -> tuple[pd.DataFrame, np.ndarray, list]:
    """Load gene expression matrix and phenotype labels.

    Parameters
    ----------
    csv_path : str
        Path to CSV file with gene expression and subtype column.
    patient_col : str
        Name of patient ID column.
    subtype_col : str
        Name of subtype/phenotype column.
    wanted_subtypes : list, optional
        List of subtypes to keep (e.g. ['Basal', 'LumA']).
        If None, keep all.

    Returns
    -------
    df : pd.DataFrame
        Filtered expression matrix (patients x genes).
    y : np.array
        Binary phenotype vector (1 for first subtype, 0 for second).
    patient_ids : list
        Patient identifiers corresponding to rows.
    """
    df = pd.read_csv(csv_path)
    df.set_index(patient_col, inplace=True)
    patient_ids = df.index.tolist()

    if wanted_subtypes is not None:
        df = df.loc[df[subtype_col].isin(wanted_subtypes)].copy()
        patient_ids = df.index.tolist()

    # Binary encoding: first subtype = 1, second = 0
    if len(df[subtype_col].unique()) == 2:
        first_subtype = df[subtype_col].unique()[0]
        y = (df[subtype_col] == first_subtype).astype(int).values
    else:
        raise ValueError(
            f"Expected 2 unique subtypes but found {df[subtype_col].nunique()}"
        )

    expr = df.drop(columns=[subtype_col])
    return expr, y, patient_ids


def load_pam50_genes(pam50_path: str) -> Set[str]:
    """Load PAM50 gene list from file.

    Parameters
    ----------
    pam50_path : str
        Path to file containing PAM50 gene names (one per line).

    Returns
    -------
    set
        Set of PAM50 gene names.
    """
    if not os.path.exists(pam50_path):
        return set()
    pam50_genes = pd.read_csv(pam50_path, header=None).iloc[:, 0].astype(str).tolist()
    return set(pam50_genes)


def filter_low_expressed_genes(
    expr: pd.DataFrame, min_nonzero: int = 5
) -> pd.DataFrame:
    """Remove genes not expressed (>0) in at least min_nonzero samples.

    Parameters
    ----------
    expr : pd.DataFrame
        Expression matrix (samples x genes).
    min_nonzero : int
        Minimum number of non-zero samples required to keep a gene.

    Returns
    -------
    pd.DataFrame
        Filtered expression matrix.
    """
    mask = (expr > 0).sum(axis=0) >= min_nonzero
    return expr.loc[:, mask]


def ols_differential_expression(
    expr: pd.DataFrame, y: np.ndarray
) -> pd.DataFrame:
    """Fast "pseudo-DE" using OLS regression per gene.

    For each gene, fits: gene_expression ~ intercept + phenotype_label
    Returns log-fold-change (regression coefficient) and p-value.

    Parameters
    ----------
    expr : pd.DataFrame
        Expression matrix (samples x genes), typically log1p-transformed.
    y : np.array
        Binary phenotype labels (0/1).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: gene, logFC, pval, adj_p, score.
    """

    def ols_de(gene_series):
        """Return (logFC, p-value) from OLS: expr ~ intercept + label."""
        model = sm.OLS(gene_series.values, sm.add_constant(y)).fit()
        return model.params[1], model.pvalues[1]

    logfc, pval = zip(*[ols_de(expr[g]) for g in expr.columns])

    de_df = pd.DataFrame(
        {
            "gene": expr.columns,
            "logFC": np.asarray(logfc),
            "pval": np.asarray(pval),
        }
    )
    # FDR correction (Benjamini-Hochberg)
    de_df["adj_p"] = multipletests(de_df["pval"], method="fdr_bh")[1]
    # Score: |effect size| × significance
    de_df["score"] = np.abs(de_df["logFC"]) * -np.log10(
        de_df["adj_p"] + 1e-300
    )

    return de_df


def select_top_genes(
    de_df: pd.DataFrame,
    target_gene_number: int,
    pam50_genes: Optional[Set[str]] = None,
) -> list:
    """Select top-ranked genes and force-include PAM50 markers.

    Strategy: select top N genes by score, then replace lowest-scoring genes
    with PAM50 genes that aren't already included, keeping total <= target_gene_number.

    Parameters
    ----------
    de_df : pd.DataFrame
        Differential expression results with 'gene' and 'score' columns.
    target_gene_number : int
        Number of top genes to select.
    pam50_genes : set, optional
        PAM50 gene names to force-include.

    Returns
    -------
    list
        List of selected gene names (capped at target_gene_number).
    """
    if pam50_genes is None:
        pam50_genes = set()

    # sort by score and take top N
    sorted_genes = de_df.sort_values("score", ascending=False)
    top_genes = sorted_genes["gene"].tolist()[:target_gene_number]
    selected = set(top_genes)

    # identify PAM50 genes not yet in top N
    pam50_not_selected = pam50_genes - selected

    # if there are PAM50 genes to add and space available, replace
    # the lowest-scoring genes with them
    if pam50_not_selected and len(selected) < target_gene_number:
        n_to_add = min(len(pam50_not_selected), target_gene_number - len(selected))
        pam50_to_add = list(pam50_not_selected)[:n_to_add]
        selected.update(pam50_to_add)
    elif pam50_not_selected and len(selected) == target_gene_number:
        # if we're at capacity, replace lowest-scoring genes
        # with PAM50 genes (prioritize PAM50)
        selected_list = sorted_genes[sorted_genes["gene"].isin(selected)].copy()
        selected_list_sorted = selected_list.sort_values("score", ascending=True)
        
        n_to_replace = min(len(pam50_not_selected), len(selected_list_sorted))
        genes_to_remove = selected_list_sorted["gene"].tolist()[:n_to_replace]
        
        for gene in genes_to_remove:
            selected.discard(gene)
        
        pam50_to_add = list(pam50_not_selected)[:n_to_replace]
        selected.update(pam50_to_add)

    return list(selected)


def remove_correlated_genes(
    expr: pd.DataFrame,
    gene_list: list,
    corr_threshold: float = 0.90,
    protected_genes: Optional[Set[str]] = None,
) -> list:
    """Remove highly correlated genes to reduce redundancy.

    Parameters
    ----------
    expr : pd.DataFrame
        Expression matrix (samples x genes).
    gene_list : list
        List of genes to filter.
    corr_threshold : float
        Pearson correlation threshold above which genes are considered redundant.
    protected_genes : set, optional
        Genes that must never be dropped regardless of correlation (e.g. PAM50
        markers). A protected gene is always retained; if it is correlated with
        an already-kept non-protected gene, both are kept (the protected gene
        does not evict the earlier one, but it is never itself removed). The
        final list may therefore contain correlated pairs when both genes are
        protected, but the total count remains <= len(gene_list).

    Returns
    -------
    list
        Filtered gene list with correlated duplicates removed.
    """
    if protected_genes is None:
        protected_genes = set()

    X = expr[gene_list].values
    corr_mat = np.corrcoef(X, rowvar=False)

    keep_idx = []
    for i in range(len(gene_list)):
        if gene_list[i] in protected_genes:
            keep_idx.append(i)
        elif not any(np.abs(corr_mat[i, j]) > corr_threshold for j in keep_idx):
            keep_idx.append(i)

    return [gene_list[i] for i in keep_idx]


def select_genes(
    csv_path: str,
    target_gene_number: int = 1024,
    min_nonzero: int = 5,
    corr_threshold: Optional[float] = 0.90,
    force_pam50: bool = True,
    pam50_path: Optional[str] = None,
    patient_col: str = "Patient_ID",
    subtype_col: str = "Majority_Subtype_mRNA",
    wanted_subtypes: Optional[list] = None,
) -> list:
    """Main gene selection pipeline.

    Parameters
    ----------
    csv_path : str
        Path to gene expression CSV file.
    target_gene_number : int
        Number of top genes to select.
    min_nonzero : int
        Minimum samples with non-zero expression to keep a gene.
    corr_threshold : float or None
        Pearson correlation threshold for redundancy removal.
        If None, skip correlation filtering.
    force_pam50 : bool
        Whether to force-include PAM50 genes.
    pam50_path : str, optional
        Path to PAM50 gene list file.
    patient_col : str
        Patient ID column name.
    subtype_col : str
        Subtype column name.
    wanted_subtypes : list, optional
        Subtypes to keep (e.g. ['Basal', 'LumA']).

    Returns
    -------
    list
        Final set of selected gene names.
    """
    print(f"Loading data from {csv_path}")
    expr, y, _ = load_phenotype_data(
        csv_path,
        patient_col=patient_col,
        subtype_col=subtype_col,
        wanted_subtypes=wanted_subtypes,
    )

    print(f"Loaded {expr.shape[0]} samples × {expr.shape[1]} genes")

    pam50_genes = set()
    if force_pam50 and pam50_path:
        pam50_genes = load_pam50_genes(pam50_path)
        print(f"Loaded {len(pam50_genes)} PAM50 genes")

    print(f"Filtering genes: min_nonzero={min_nonzero}")
    expr = filter_low_expressed_genes(expr, min_nonzero=min_nonzero)
    print(f"After filtering: {expr.shape[1]} genes remain")

    print("Computing differential expression (OLS per gene)")
    de_df = ols_differential_expression(expr, y)

    print(f"Selecting top {target_gene_number} genes + PAM50 markers")
    selected = select_top_genes(de_df, target_gene_number, pam50_genes)
    print(f"After selection: {len(selected)} genes")

    if corr_threshold is not None and corr_threshold > 0:
        print(f"Removing correlated genes (threshold={corr_threshold})")
        selected = remove_correlated_genes(
            expr, selected, corr_threshold=corr_threshold,
            protected_genes=pam50_genes if force_pam50 else None,
        )
        print(f"After correlation filtering: {len(selected)} genes remain")

        # if the correlation step dropped us below the target size, try to
        # fill back up using the next-best genes that are still sufficiently
        # uncorrelated with the current set. this ensures the final list
        # has exactly `target_gene_number` entries whenever possible.
        if len(selected) < target_gene_number:
            print("Replenishing genes to reach target size after pruning...")
            ranked = de_df.sort_values("score", ascending=False)["gene"].tolist()
            for g in ranked:
                if len(selected) >= target_gene_number:
                    break
                if g in selected:
                    continue
                # check correlation between candidate and each already selected
                too_corr = False
                for h in selected:
                    r = np.corrcoef(expr[g], expr[h])[0, 1]
                    if abs(r) > corr_threshold:
                        too_corr = True
                        break
                if too_corr:
                    continue
                selected.append(g)
            print(f"After replenishment: {len(selected)} genes")
            if len(selected) < target_gene_number:
                print(f"WARNING: could not reach target of {target_gene_number} genes after correlation filtering (got {len(selected)})")

    return selected


def preprocess_and_save(
    expr: pd.DataFrame,
    patient_ids: list,
    selected_genes: list,
    output_dir: str,
    patient_col: str = "Patient_ID",
) -> None:
    """Apply log1p + z-score normalization and save per-patient H5 + combined CSV.

    This matches the VAE preprocessing pipeline (log1p_zscore).
    Outputs:
      - One CSV file with all patients (rows) and selected genes (columns)
      - One H5 file per sample with normalized expression.
        Each H5 uses dataset name ``feats`` of shape (1, num_genes) and
        dtype float32 (matching the format produced by the VAE pipeline).

    Parameters
    ----------
    expr : pd.DataFrame
        Expression matrix (patients x genes). Index contains sample IDs.
    patient_ids : list
        List of sample identifiers (in same order as expr rows).
    selected_genes : list
        List of selected gene names.
    output_dir : str
        Directory to save output files.
    patient_col : str
        Column name to use in CSV output.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check for duplicate sample IDs and warn if found
    if len(patient_ids) != len(set(patient_ids)):
        duplicates = [pid for pid in set(patient_ids) if patient_ids.count(pid) > 1]
        print(f"WARNING: Found duplicate sample IDs: {duplicates}")
        print(f"Each duplicate will be saved with a numeric suffix")

    # Apply log1p + z-score normalization (matches VAE preprocessing)
    print("Applying log1p + z-score normalization...")
    expr_selected = expr[selected_genes].copy()
    expr_normalized = preprocess_log1p_zscore(expr_selected)

    # Save combined CSV with all patients as rows
    print(f"Saving combined CSV with {len(patient_ids)} samples...")
    csv_data = expr_normalized.copy()
    csv_data.insert(0, patient_col, patient_ids)
    csv_path = os.path.join(output_dir, "gene_expression_normalized.csv")
    csv_data.to_csv(csv_path, index=False)
    print(f"Saved combined CSV to {csv_path}")

    # Save per-sample H5 files
    # Handle duplicate sample IDs by adding numeric suffix
    print(f"Saving per-sample H5 files to {output_dir}")
    sample_id_counts = {}
    
    for idx, patient_id in enumerate(patient_ids):
        patient_expr = expr_normalized.iloc[idx].values

        # If sample ID appears multiple times, append -DX suffix (-DX1, -DX2, etc.)
        if patient_ids.count(patient_id) > 1:
            if patient_id not in sample_id_counts:
                sample_id_counts[patient_id] = 1
            else:
                sample_id_counts[patient_id] += 1
            h5_name = f"{patient_id}-DX{sample_id_counts[patient_id]}.h5"
        else:
            h5_name = f"{patient_id}.h5"

        # H5 output (compact binary format)
        h5_path = os.path.join(output_dir, h5_name)
        # ensure shape and dtype match VAE features: (1, num_genes) float32
        # convert to plain ndarray to satisfy static type checkers
        patient_expr = np.asarray(patient_expr, dtype=np.float32)
        patient_expr = np.expand_dims(patient_expr, axis=0)  # make shape (1, N)

        with h5py.File(h5_path, "w") as f:
            # use key "feats" to be compatible with downstream loaders
            f.create_dataset("feats", data=patient_expr, compression="gzip")
            # note: we deliberately omit a separate "genes" field so that
            # the output file matches the vanilla VAE feature files exactly.
            f.attrs["patient_id"] = patient_id
            f.attrs["sample_index"] = idx

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(patient_ids)} samples")

    print(f"Saved {len(patient_ids)} sample H5 files to {output_dir}")


def load_all_patients(
    csv_path: str,
    patient_col: str = "Patient_ID",
    subtype_col: str = "Majority_Subtype_mRNA",
) -> tuple[pd.DataFrame, list]:
    """Load full expression matrix for ALL patients (no subtype filtering).

    Parameters
    ----------
    csv_path : str
        Path to CSV file with gene expression and subtype column.
    patient_col : str
        Name of patient ID column.
    subtype_col : str
        Name of subtype/phenotype column (will be dropped from expression).

    Returns
    -------
    expr : pd.DataFrame
        Full expression matrix (all patients x genes).
    patient_ids : list
        All patient identifiers.
    """
    df = pd.read_csv(csv_path)
    df.set_index(patient_col, inplace=True)
    patient_ids = df.index.tolist()
    expr = df.drop(columns=[subtype_col])
    return expr, patient_ids


def main():
    parser = argparse.ArgumentParser(
        description="Gene subset selection for dimensionality reduction"
    )

    # declare ALL arguments first, then load config defaults, then parse.
    # this avoids add_argument overwriting set_defaults values.
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
        help="Directory for per-patient output files",
    )
    parser.add_argument(
        "--gene-list-path",
        type=str,
        default=None,
        help="Path to save final gene list (one gene per line)",
    )
    parser.add_argument(
        "--target-genes",
        type=int,
        default=1024,
        help="Number of top genes to select (default 1024)",
    )
    parser.add_argument(
        "--min-nonzero",
        type=int,
        default=5,
        help="Min samples with non-zero expression to keep gene (default 5)",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.90,
        help="Pearson correlation threshold for redundancy removal (default 0.90, set to 0 to skip)",
    )
    parser.add_argument(
        "--pam50-path",
        type=str,
        default=None,
        help="Path to PAM50 gene list file",
    )
    parser.add_argument(
        "--force-pam50",
        action="store_true",
        default=True,
        help="Force-include PAM50 genes (default True)",
    )
    parser.add_argument(
        "--patient-col",
        type=str,
        default="Patient_ID",
        help="Patient ID column name (default 'Patient_ID')",
    )
    parser.add_argument(
        "--subtype-col",
        type=str,
        default="Majority_Subtype_mRNA",
        help="Subtype column name (default 'Majority_Subtype_mRNA')",
    )
    parser.add_argument(
        "--subtypes",
        type=str,
        nargs="+",
        default=["Basal", "LumA"],
        help="Subtypes used for DE gene selection (default Basal LumA)",
    )

    # first pass: peek at --config so we can load YAML defaults
    temp_args, _ = parser.parse_known_args()

    if temp_args.config:
        try:
            import yaml

            with open(temp_args.config) as f:
                cfg = yaml.safe_load(f) or {}
            if isinstance(cfg, dict) and "encoding" in cfg:
                cfg = cfg["encoding"] or {}
            # apply YAML values as defaults (AFTER add_argument, so they win)
            parser.set_defaults(**cfg)
        except Exception as e:
            print(f"Failed to load config {temp_args.config}: {e}")

    # final parse using the potentially updated defaults
    args = parser.parse_args()

    print(f"Configuration: target_genes={args.target_genes}, "
          f"subtypes={args.subtypes}, corr_threshold={args.corr_threshold}")

    # --- Phase 1: Gene selection (uses binary subtypes for DE) ---
    selected_genes = select_genes(
        csv_path=args.csv_path,
        target_gene_number=args.target_genes,
        min_nonzero=args.min_nonzero,
        corr_threshold=args.corr_threshold if args.corr_threshold > 0 else None,
        force_pam50=args.force_pam50,
        pam50_path=args.pam50_path,
        patient_col=args.patient_col,
        subtype_col=args.subtype_col,
        wanted_subtypes=args.subtypes,
    )

    # Save gene list
    if args.gene_list_path:
        os.makedirs(os.path.dirname(args.gene_list_path) or ".", exist_ok=True)
        pd.Series(selected_genes, name="gene").to_csv(
            args.gene_list_path, index=False, header=False
        )
        print(f"Saved gene list to {args.gene_list_path}")

    # --- Phase 2: Output for ALL patients (no subtype filtering) ---
    expr_all, all_patient_ids = load_all_patients(
        args.csv_path,
        patient_col=args.patient_col,
        subtype_col=args.subtype_col,
    )
    print(f"\n--- Patient statistics ---")
    print(f"Total patients in dataset: {len(all_patient_ids)}")

    preprocess_and_save(
        expr_all, all_patient_ids, selected_genes, args.output_dir, args.patient_col
    )

    print("\nGene reduction complete!")
    print(f"Selected {len(selected_genes)} genes")
    print(f"Output files in {args.output_dir}")


if __name__ == "__main__":
    main()
