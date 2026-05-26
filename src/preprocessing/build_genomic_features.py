"""
Genomic feature preprocessing pipeline for MoPaDi from-scratch training.

Produces per-patient H5 files (one 512-dim float32 gene-expression vector each)
that are consumed at training time by ZipTilesWithGenomicFeatures.

Pipeline steps
--------------
1. select_genes          — differential-expression ranking → 512-gene list
2. split_patients_stratified — 80/10/10 split, stratified by histological subtype,
                               keeping all samples of a patient in the same fold
3. fit_genomic_scaler    — log1p + StandardScaler fitted on training patients only
4. apply_genomic_scaler  — applies the fitted scaler to any patient subset
5. write_genomic_h5_files — writes one H5 per patient to output_dir
   (optional) compute_tile_counts / compute_balanced_tile_caps — suggests
             per-subtype tile caps for the training config

Each function is independently testable and has no side-effects beyond
explicit I/O arguments.

Entry point
-----------
    python -m src.preprocessing.build_genomic_features --config src/config.yaml

or via run_pipeline.py:
    python run_pipeline.py --config src/config.yaml --stage build_genomic_features
"""

from __future__ import annotations

import json
import logging
import os
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Reuse gene-selection logic from the encoding module (no code duplication).
from encoding.reduce_gene_number import select_genes as _select_genes_from_csv

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Gene selection
# ---------------------------------------------------------------------------

def select_genes(
    csv_path: str,
    target_gene_number: int = 512,
    min_nonzero: int = 5,
    corr_threshold: float = 0.90,
    force_pam50: bool = True,
    pam50_path: Optional[str] = None,
    patient_col: str = "Patient_ID",
    subtype_col: str = "Majority_Subtype_mRNA",
    wanted_subtypes: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> List[str]:
    """Select informative genes via OLS differential expression + FDR ranking.

    Thin wrapper around ``encoding.reduce_gene_number.select_genes`` that
    additionally saves the resulting gene list to ``output_dir/gene_list.txt``
    so that the selection is reproducible and auditable.

    Parameters
    ----------
    csv_path:
        Path to raw gene-expression CSV (patients × genes, with Patient_ID
        and subtype columns).
    target_gene_number:
        Number of genes to select.
    min_nonzero:
        Minimum samples expressing a gene (>0) for it to be considered.
    corr_threshold:
        Pearson correlation ceiling; genes more correlated than this with an
        already-selected gene are dropped.
    force_pam50:
        Whether to force-include PAM50 marker genes.
    pam50_path:
        Path to PAM50 gene list (one gene per line).  Required if
        ``force_pam50=True``.
    patient_col:
        Name of the patient-ID column in the CSV.
    subtype_col:
        Name of the subtype column in the CSV.
    wanted_subtypes:
        Two subtypes used for the binary OLS phenotype, e.g. ['Basal', 'LumA'].
        If None, all subtypes are used (only works if exactly 2 are present).
    output_dir:
        If given, saves ``gene_list.txt`` here.

    Returns
    -------
    List[str]
        Selected gene names in a deterministic order (sorted alphabetically).
    """
    genes: List[str] = _select_genes_from_csv(
        csv_path=csv_path,
        target_gene_number=target_gene_number,
        min_nonzero=min_nonzero,
        corr_threshold=corr_threshold,
        force_pam50=force_pam50,
        pam50_path=pam50_path,
        patient_col=patient_col,
        subtype_col=subtype_col,
        wanted_subtypes=wanted_subtypes,
    )
    genes = sorted(genes)   # deterministic order

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        gene_list_path = Path(output_dir) / "gene_list.txt"
        gene_list_path.write_text("\n".join(genes) + "\n")
        log.info("Saved gene list (%d genes) to %s", len(genes), gene_list_path)

    return genes


# ---------------------------------------------------------------------------
# 2. Patient splitting (stratified by subtype)
# ---------------------------------------------------------------------------

def split_patients_stratified(
    patient_ids: List[str],
    subtypes: List[str],
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> Dict[str, str]:
    """Assign each patient to train / val / test, stratified by subtype.

    Patients that appear multiple times in the CSV (multiple RNA-seq samples)
    are deduplicated before splitting so that all samples of one patient land
    in the same fold.

    Parameters
    ----------
    patient_ids:
        One entry per CSV row (may contain duplicates for multi-sample patients).
    subtypes:
        One subtype label per CSV row, aligned with ``patient_ids``.
    train_frac, val_frac, test_frac:
        Split fractions; must sum to 1.0.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, str]
        ``{patient_id: 'train' | 'val' | 'test'}``
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "train_frac + val_frac + test_frac must equal 1.0"

    # Deduplicate: keep one (subtype) entry per unique patient.
    # For a patient with multiple RNA-seq samples we keep the first subtype seen.
    seen: dict[str, str] = {}
    for pid, stype in zip(patient_ids, subtypes):
        if pid not in seen:
            seen[pid] = stype

    unique_patients = list(seen.keys())
    unique_subtypes = [seen[p] for p in unique_patients]

    # First cut: separate out test set, stratified by subtype.
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_frac, random_state=seed
    )
    trainval_idx, test_idx = next(
        sss_test.split(unique_patients, unique_subtypes)
    )

    trainval_patients = [unique_patients[i] for i in trainval_idx]
    trainval_subtypes = [unique_subtypes[i] for i in trainval_idx]
    test_patients = {unique_patients[i] for i in test_idx}

    # Second cut: separate val from train within the trainval pool.
    val_frac_of_trainval = val_frac / (train_frac + val_frac)
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac_of_trainval, random_state=seed
    )
    train_idx, val_idx = next(
        sss_val.split(trainval_patients, trainval_subtypes)
    )

    train_patients = {trainval_patients[i] for i in train_idx}
    val_patients = {trainval_patients[i] for i in val_idx}

    assignment: Dict[str, str] = {}
    for pid in unique_patients:
        if pid in train_patients:
            assignment[pid] = "train"
        elif pid in val_patients:
            assignment[pid] = "val"
        else:
            assignment[pid] = "test"

    counts = {s: sum(1 for v in assignment.values() if v == s)
              for s in ("train", "val", "test")}
    log.info(
        "Patient split — train: %d  val: %d  test: %d  (total: %d)",
        counts["train"], counts["val"], counts["test"], len(assignment),
    )
    return assignment


def save_patient_splits(
    splits: Dict[str, str],
    subtype_map: Dict[str, str],
    output_dir: str,
) -> str:
    """Persist the patient split and subtype map to ``patient_splits.json``.

    Parameters
    ----------
    splits:
        ``{patient_id: 'train'|'val'|'test'}`` from ``split_patients_stratified``.
    subtype_map:
        ``{patient_id: subtype_label}`` for each patient.
    output_dir:
        Directory to write the JSON file.

    Returns
    -------
    str
        Absolute path to the written JSON file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Dict] = {"train": {}, "val": {}, "test": {}}

    for pid, fold in splits.items():
        payload[fold][pid] = {"subtype": subtype_map.get(pid, "unknown")}

    for fold in payload:
        payload[fold]["_n_patients"] = len(
            [k for k in payload[fold] if not k.startswith("_")]
        )

    out_path = str(Path(output_dir) / "patient_splits.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    log.info("Saved patient splits to %s", out_path)
    return out_path


def load_patient_splits(path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load a previously saved patient_splits.json.

    Returns
    -------
    splits:
        ``{patient_id: 'train'|'val'|'test'}``
    subtype_map:
        ``{patient_id: subtype_label}``
    """
    with open(path) as f:
        payload = json.load(f)

    splits: Dict[str, str] = {}
    subtype_map: Dict[str, str] = {}
    for fold, entries in payload.items():
        if fold.startswith("_"):
            continue
        for pid, meta in entries.items():
            if pid.startswith("_"):
                continue
            splits[pid] = fold
            subtype_map[pid] = meta.get("subtype", "unknown")
    return splits, subtype_map


# ---------------------------------------------------------------------------
# 3. Normalization — fitted on training patients only
# ---------------------------------------------------------------------------

def fit_genomic_scaler(
    expr_df: pd.DataFrame,
    gene_list: List[str],
    train_patient_ids: List[str],
    output_dir: Optional[str] = None,
) -> Tuple[StandardScaler, np.ndarray]:
    """Fit a log1p + per-gene StandardScaler on training patients only.

    This is the scientifically correct approach: the z-score statistics
    describe the training distribution and are then frozen for val/test.
    Fitting on all patients would leak test-set statistics into normalization.

    Parameters
    ----------
    expr_df:
        Full expression DataFrame (all patients), index = Patient_ID.
    gene_list:
        Ordered list of gene names to use (must be columns of expr_df).
    train_patient_ids:
        Patient IDs belonging to the training split.
    output_dir:
        If given, saves ``scaler.json`` (per-gene mean/std) here so the same
        transform can be reproduced at inference time.

    Returns
    -------
    scaler:
        Fitted ``sklearn.preprocessing.StandardScaler``.
    gene_order:
        1-D array of gene names in the exact order the scaler expects.
    """
    missing = [g for g in gene_list if g not in expr_df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} genes from gene_list are not in the expression "
            f"DataFrame: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    train_mask = expr_df.index.isin(set(train_patient_ids))
    train_expr = expr_df.loc[train_mask, gene_list]
    if len(train_expr) == 0:
        raise ValueError("No training patients found in expr_df.")

    log.info(
        "Fitting scaler on %d training patients × %d genes",
        len(train_expr), len(gene_list),
    )

    log1p_train = np.log1p(train_expr.values.astype(np.float64))
    scaler = StandardScaler()
    scaler.fit(log1p_train)

    gene_order = np.array(gene_list)

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        scaler_path = Path(output_dir) / "scaler.json"
        scaler_data = {
            "genes": gene_list,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "n_train_patients": int(train_mask.sum()),
        }
        with open(scaler_path, "w") as f:
            json.dump(scaler_data, f, indent=2)
        log.info("Saved scaler stats to %s", scaler_path)

    return scaler, gene_order


def load_genomic_scaler(scaler_json_path: str) -> Tuple[StandardScaler, List[str]]:
    """Restore a previously fitted scaler from ``scaler.json``.

    Returns
    -------
    scaler:
        ``StandardScaler`` with ``mean_`` and ``scale_`` populated.
    gene_order:
        Ordered gene list that the scaler expects.
    """
    with open(scaler_json_path) as f:
        data = json.load(f)

    scaler = StandardScaler()
    scaler.mean_ = np.array(data["mean"], dtype=np.float64)
    scaler.scale_ = np.array(data["scale"], dtype=np.float64)
    scaler.n_features_in_ = len(data["genes"])
    scaler.n_samples_seen_ = data.get("n_train_patients", 0)
    return scaler, data["genes"]


# ---------------------------------------------------------------------------
# 4. Apply normalization (pure — no I/O)
# ---------------------------------------------------------------------------

def apply_genomic_scaler(
    expr_df: pd.DataFrame,
    gene_list: List[str],
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Apply log1p + the pre-fitted StandardScaler to a patient subset.

    Parameters
    ----------
    expr_df:
        Expression DataFrame (subset of patients); index = Patient_ID.
    gene_list:
        Genes in the exact order the scaler was fitted on.
    scaler:
        Fitted scaler returned by ``fit_genomic_scaler``.

    Returns
    -------
    pd.DataFrame
        Normalised DataFrame with the same index and ``gene_list`` as columns.
    """
    subset = expr_df[gene_list].values.astype(np.float64)
    log1p_vals = np.log1p(subset)
    scaled = scaler.transform(log1p_vals)
    return pd.DataFrame(scaled, index=expr_df.index, columns=gene_list)


# ---------------------------------------------------------------------------
# 5. Write per-patient H5 files
# ---------------------------------------------------------------------------

def write_genomic_h5_files(
    expr_normalized: pd.DataFrame,
    patient_splits: Dict[str, str],
    subtype_map: Dict[str, str],
    output_dir: str,
) -> Dict[str, str]:
    """Write one H5 file per patient row in ``expr_normalized``.

    H5 format
    ---------
    Dataset ``feats``: shape ``(512,)`` float32 — the normalized expression vector.
    Attributes: ``patient_id``, ``split``, ``subtype``.

    Multi-sample patients (same Patient_ID appearing more than once in the
    DataFrame) are disambiguated with a ``-DX1``, ``-DX2``, … suffix, matching
    the convention used by ``encoding.reduce_gene_number``.

    Parameters
    ----------
    expr_normalized:
        Normalised expression DataFrame produced by ``apply_genomic_scaler``.
        Index = Patient_ID (may repeat for multi-sample patients).
    patient_splits:
        ``{patient_id: 'train'|'val'|'test'}`` — used to populate H5 attributes.
    subtype_map:
        ``{patient_id: subtype_label}`` — used to populate H5 attributes.
    output_dir:
        Directory to write H5 files.

    Returns
    -------
    Dict[str, str]
        ``{patient_id_or_sample_id: h5_filepath}`` manifest.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Count how many times each patient ID appears (for -DX suffix logic).
    id_counts: Dict[str, int] = {}
    for pid in expr_normalized.index:
        id_counts[pid] = id_counts.get(pid, 0) + 1

    sample_counters: Dict[str, int] = {}
    manifest: Dict[str, str] = {}
    n_genes = expr_normalized.shape[1]

    for pid, row in zip(expr_normalized.index, expr_normalized.values):
        vec = row.astype(np.float32)

        # Determine filename: unique patients get bare name; duplicates get -DX suffix.
        if id_counts[pid] > 1:
            sample_counters[pid] = sample_counters.get(pid, 0) + 1
            h5_name = f"{pid}-DX{sample_counters[pid]}.h5"
            sample_key = f"{pid}-DX{sample_counters[pid]}"
        else:
            h5_name = f"{pid}.h5"
            sample_key = pid

        h5_path = str(Path(output_dir) / h5_name)

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("feats", data=vec, dtype="float32")
            f.attrs["patient_id"] = pid
            f.attrs["sample_key"] = sample_key
            f.attrs["split"] = patient_splits.get(pid, "unknown")
            f.attrs["subtype"] = subtype_map.get(pid, "unknown")
            f.attrs["n_genes"] = n_genes

        manifest[sample_key] = h5_path

    # Write manifest JSON alongside H5 files.
    manifest_path = Path(output_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    log.info(
        "Wrote %d H5 files to %s  (manifest: %s)",
        len(manifest), output_dir, manifest_path,
    )
    return manifest


# ---------------------------------------------------------------------------
# 6. Optional: compute tile counts and suggest balanced caps
# ---------------------------------------------------------------------------

def compute_tile_counts(
    zip_dir: str,
    patient_id_from_zipname: Optional[callable] = None,
) -> Dict[str, int]:
    """Count tiles per patient by scanning ZIP files (no extraction).

    Parameters
    ----------
    zip_dir:
        Directory containing ``*.zip`` tile archives.
    patient_id_from_zipname:
        Callable ``zipname → patient_id`` (3-token TCGA barcode).
        Defaults to: ``TCGA-XX-XXXX-01Z-00-DX1.UUID.zip → TCGA-XX-XXXX``.

    Returns
    -------
    Dict[str, int]
        ``{patient_id: n_tiles}``
    """
    if patient_id_from_zipname is None:
        def patient_id_from_zipname(name: str) -> str:
            barcode = Path(name).stem.split(".")[0]
            return "-".join(barcode.split("-")[:3])

    img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    counts: Dict[str, int] = {}

    for fname in Path(zip_dir).iterdir():
        if fname.suffix.lower() != ".zip":
            continue
        pid = patient_id_from_zipname(fname.name)
        try:
            with zipfile.ZipFile(fname, "r") as zf:
                n = sum(
                    1 for n_ in zf.namelist()
                    if Path(n_).suffix.lower() in img_exts
                )
        except Exception as exc:
            warnings.warn(f"Could not read {fname}: {exc}")
            n = 0
        counts[pid] = counts.get(pid, 0) + n

    return counts


def compute_balanced_tile_caps(
    tile_counts: Dict[str, int],
    subtype_map: Dict[str, str],
    split_patients: Optional[List[str]] = None,
    target_total_per_subtype: Optional[int] = None,
) -> Dict[str, Optional[int]]:
    """Suggest per-subtype max-tiles-per-patient to balance training.

    The target is chosen so that the smallest subtype contributes
    ``target_total_per_subtype`` tiles in total.  Larger subtypes are capped
    so they contribute the same total.  Subtypes that are already below the
    target are left uncapped (``None``).

    Parameters
    ----------
    tile_counts:
        ``{patient_id: n_tiles}`` from ``compute_tile_counts``.
    subtype_map:
        ``{patient_id: subtype_label}``.
    split_patients:
        If given, restrict to this patient subset (e.g. training patients).
    target_total_per_subtype:
        Total tiles to target per subtype.  If None, uses the total tile count
        of the smallest subtype as the target.

    Returns
    -------
    Dict[str, Optional[int]]
        ``{subtype: max_tiles_per_patient_or_None}``
    """
    patients = set(split_patients) if split_patients else set(tile_counts)

    # Group patients by subtype, collect their tile counts.
    subtype_tiles: Dict[str, List[int]] = {}
    for pid in patients:
        stype = subtype_map.get(pid)
        if stype is None:
            continue
        subtype_tiles.setdefault(stype, []).append(tile_counts.get(pid, 0))

    subtype_totals = {s: sum(v) for s, v in subtype_tiles.items()}

    if target_total_per_subtype is None:
        target_total_per_subtype = min(subtype_totals.values())

    caps: Dict[str, Optional[int]] = {}
    for stype, patients_tiles in subtype_tiles.items():
        n_patients = len(patients_tiles)
        current_total = subtype_totals[stype]
        if current_total <= target_total_per_subtype:
            caps[stype] = None   # no cap needed; already below target
        else:
            cap = max(1, target_total_per_subtype // n_patients)
            caps[stype] = cap

    log.info("Suggested tile caps per subtype (target=%d total):", target_total_per_subtype)
    for stype, cap in sorted(caps.items()):
        total = subtype_totals.get(stype, 0)
        n_p = len(subtype_tiles.get(stype, []))
        log.info("  %-12s  %d patients  %d tiles total  → cap=%s", stype, n_p, total, cap)

    return caps


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_build_genomic_features(cfg: Dict, verbose: bool = True) -> None:
    """Execute the full preprocessing pipeline from a config dict.

    Expected config keys (see ``src/config.yaml`` section
    ``mopadi_genomic_training``):

    .. code-block:: yaml

        csv_path: path/to/brca_gene_expression_with_subtypes.csv
        patient_col: Patient_ID
        subtype_col: Majority_Subtype_mRNA
        gene_selection:
            target_genes: 512
            min_nonzero: 5
            corr_threshold: 0.90
            force_pam50: true
            pam50_path: path/to/PAM50_gene_list.txt
            wanted_subtypes: [Basal, LumA]
        split:
            train: 0.80
            val: 0.10
            test: 0.10
            seed: 42
        output_dir: path/to/output
        zip_dir: path/to/BRCA-tumor-tiles-corrected  # optional, for tile-cap suggestion
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%H:%M:%S",
        )

    csv_path = cfg["csv_path"]
    patient_col = cfg.get("patient_col", "Patient_ID")
    subtype_col = cfg.get("subtype_col", "Majority_Subtype_mRNA")
    output_dir = cfg["output_dir"]
    gs_cfg = cfg.get("gene_selection", {})
    sp_cfg = cfg.get("split", {})
    genomic_h5_dir = str(Path(output_dir) / "genomic_h5")

    # ── Step 1: gene selection ───────────────────────────────────────────
    log.info("=== Step 1/5: Gene selection ===")
    gene_list = select_genes(
        csv_path=csv_path,
        target_gene_number=gs_cfg.get("target_genes", 512),
        min_nonzero=gs_cfg.get("min_nonzero", 5),
        corr_threshold=gs_cfg.get("corr_threshold", 0.90),
        force_pam50=gs_cfg.get("force_pam50", True),
        pam50_path=gs_cfg.get("pam50_path"),
        patient_col=patient_col,
        subtype_col=subtype_col,
        wanted_subtypes=gs_cfg.get("wanted_subtypes"),
        output_dir=output_dir,
    )
    log.info("Selected %d genes", len(gene_list))

    # ── Load full expression matrix for all patients ─────────────────────
    log.info("Loading expression CSV: %s", csv_path)
    expr_raw = pd.read_csv(csv_path)
    if patient_col not in expr_raw.columns:
        raise ValueError(f"Column '{patient_col}' not found in {csv_path}")

    # Build subtype_map BEFORE set_index to avoid the multi-row .loc pitfall:
    # when a Patient_ID appears more than once, expr_df.loc[pid, col] returns a
    # Series instead of a scalar, and str(Series) produces a multi-line garbage
    # string that becomes a unique "subtype" class — breaking StratifiedShuffleSplit.
    if subtype_col in expr_raw.columns:
        # Keep only the first occurrence per patient for the label lookup.
        subtype_map: Dict[str, str] = (
            expr_raw.drop_duplicates(subset=[patient_col])
            .set_index(patient_col)[subtype_col]
            .astype(str)
            .to_dict()
        )
    else:
        subtype_map = {}

    expr_all = expr_raw.set_index(patient_col)
    log.info("Loaded %d rows × %d columns", *expr_all.shape)

    patient_ids: List[str] = list(expr_all.index)
    subtypes: List[str] = [subtype_map.get(pid, "unknown") for pid in patient_ids]

    # ── Step 2: stratified patient splitting ────────────────────────────
    log.info("=== Step 2/5: Patient splitting ===")
    splits = split_patients_stratified(
        patient_ids=patient_ids,
        subtypes=subtypes,
        train_frac=sp_cfg.get("train", 0.80),
        val_frac=sp_cfg.get("val", 0.10),
        test_frac=sp_cfg.get("test", 0.10),
        seed=sp_cfg.get("seed", 42),
    )
    splits_path = save_patient_splits(splits, subtype_map, output_dir)
    log.info("Splits saved to %s", splits_path)

    # ── Step 3: fit scaler on training patients only ─────────────────────
    log.info("=== Step 3/5: Fitting scaler (train split only) ===")
    train_pids = [pid for pid, fold in splits.items() if fold == "train"]
    scaler, gene_order = fit_genomic_scaler(
        expr_df=expr_all,
        gene_list=gene_list,
        train_patient_ids=train_pids,
        output_dir=output_dir,
    )

    # ── Step 4: normalize all patients ──────────────────────────────────
    log.info("=== Step 4/5: Normalizing expression values ===")
    expr_normalized = apply_genomic_scaler(expr_all, list(gene_order), scaler)

    # ── Step 5: write H5 files ───────────────────────────────────────────
    log.info("=== Step 5/5: Writing H5 files to %s ===", genomic_h5_dir)
    manifest = write_genomic_h5_files(
        expr_normalized=expr_normalized,
        patient_splits=splits,
        subtype_map=subtype_map,
        output_dir=genomic_h5_dir,
    )
    log.info("Wrote %d H5 files", len(manifest))

    # ── Optional: tile count scan + balanced cap suggestion ──────────────
    zip_dir = cfg.get("zip_dir")
    if zip_dir and Path(zip_dir).is_dir():
        log.info("=== Optional: Computing tile counts for balance suggestion ===")
        tile_counts = compute_tile_counts(zip_dir)
        train_patients_set = list(train_pids)
        caps = compute_balanced_tile_caps(
            tile_counts=tile_counts,
            subtype_map=subtype_map,
            split_patients=train_patients_set,
        )
        caps_path = Path(output_dir) / "suggested_tile_caps.json"
        with open(caps_path, "w") as f:
            json.dump(caps, f, indent=2, sort_keys=True)
        log.info("Suggested tile caps written to %s", caps_path)

    log.info("=== Preprocessing complete ===")
    log.info("  Gene list:      %s/gene_list.txt", output_dir)
    log.info("  Patient splits: %s/patient_splits.json", output_dir)
    log.info("  Scaler stats:   %s/scaler.json", output_dir)
    log.info("  H5 files:       %s/  (%d files)", genomic_h5_dir, len(manifest))


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Build per-patient genomic H5 files for MoPaDi training."
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--section",
        default="build_genomic_features",
        help="Top-level config section to use (default: build_genomic_features)",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        full_cfg = yaml.safe_load(fh)

    section = full_cfg.get(args.section, {})
    if not section:
        raise SystemExit(f"No '{args.section}' section found in config.")

    run_build_genomic_features(section, verbose=True)
