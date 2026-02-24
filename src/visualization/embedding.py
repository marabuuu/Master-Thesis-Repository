"""
Embedding I/O & Preparation
============================

Helpers for loading VAE embeddings from HDF5 files, merging with clinical
metadata, and building clean NumPy matrices ready for downstream
dimensionality-reduction and clustering.

These utilities are intentionally **pipeline-agnostic**: they work on any
directory that follows the ``<root>/{train,test}/*.h5`` convention produced
by the encoding step.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional, Sequence, cast

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


# ===================================================================
#  Single-file loader
# ===================================================================


def load_embedding(
    h5_path: str | Path,
    dataset_keys: Sequence[str] = ("embedding", "latent", "z", "features"),
) -> np.ndarray:
    """Load a single embedding vector from an HDF5 file.

    The function probes *dataset_keys* in order, then falls back to the
    first ``h5py.Dataset`` it finds.  The result is **always** returned as
    a 1-D ``ndarray`` (singletons like ``(1, 512)`` are squeezed).

    Parameters
    ----------
    h5_path : str | Path
        Path to the ``.h5`` file.
    dataset_keys : sequence of str
        Dataset names to try, in priority order.

    Returns
    -------
    np.ndarray
        1-D embedding vector (e.g. shape ``(512,)``).

    Raises
    ------
    RuntimeError
        If the file contains no readable dataset.
    """
    with h5py.File(str(h5_path), "r") as f:
        for k in dataset_keys:
            if k in f:
                obj = f[k]
                if isinstance(obj, h5py.Dataset):
                    arr: np.ndarray = cast(h5py.Dataset, obj)[()]
                    return _normalise_embedding(arr)

        # Fallback – first dataset-like member
        first_key = next(
            (name for name in f.keys() if isinstance(f[name], h5py.Dataset)),
            None,
        )
        if first_key is None:
            raise RuntimeError(f"No dataset found in HDF5 file: {h5_path}")
        arr = cast(h5py.Dataset, f[first_key])[()]
        return _normalise_embedding(arr)


def _normalise_embedding(arr: np.ndarray) -> np.ndarray:
    """Squeeze / flatten to a 1-D vector."""
    arr = np.squeeze(arr)
    if arr.ndim > 1:
        arr = arr.ravel()
    return arr


# ===================================================================
#  Batch collection
# ===================================================================


def collect_embeddings(
    root_dir: str | Path,
    splits: Sequence[str] = ("train", "test"),
    expected_dim: int = 512,
    dataset_keys: Sequence[str] = ("embedding", "latent", "z", "features"),
    quiet: bool = False,
) -> pd.DataFrame:
    """Walk ``<root_dir>/<split>/*.h5`` and collect all embeddings.

    Parameters
    ----------
    root_dir : str | Path
        Top-level directory containing split sub-folders.
    splits : sequence of str
        Sub-folder names to scan (default ``("train", "test")``).
    expected_dim : int
        Expected embedding dimensionality – mismatches are logged as
        warnings but not rejected.
    dataset_keys : sequence of str
        Forwarded to :func:`load_embedding`.
    quiet : bool
        Suppress progress bars.

    Returns
    -------
    pd.DataFrame
        Columns: ``file_path``, ``encoded_id``, ``split``, ``embedding``.
    """
    rows: list[dict] = []
    for split in splits:
        pattern = os.path.join(str(root_dir), split, "*.h5")
        file_list = sorted(glob.glob(pattern))
        if not file_list and not quiet:
            print(f"⚠️  No .h5 files found in {os.path.join(str(root_dir), split)}")
        iterator = file_list if quiet else tqdm(file_list, desc=f"Loading {split}")
        for fp in iterator:
            encoded_id = Path(fp).stem
            emb = load_embedding(fp, dataset_keys=dataset_keys)
            if emb.shape[0] != expected_dim and not quiet:
                print(f"⚠️  Unexpected shape {emb.shape} in {fp}")
            rows.append(
                {
                    "file_path": fp,
                    "encoded_id": encoded_id,
                    "split": split,
                    "embedding": emb,
                }
            )
    return pd.DataFrame(rows)


# ===================================================================
#  Clinical metadata helpers
# ===================================================================


def read_clinical_data(
    clinical_path: str | Path,
    map_path: str | Path,
    patient_col: str = "PATIENT",
    subtype_col: str = "Majority_Subtype_mRNA",
    mapper_id_col: str = "unique_id",
    mapper_patient_col: str = "orig_patient",
) -> pd.DataFrame:
    """Read clinical CSV + ID-mapping CSV and return a tidy table.

    Parameters
    ----------
    clinical_path, map_path : str | Path
    patient_col : str
        Column in *clinical_path* with patient identifiers.
    subtype_col : str
        Column in *clinical_path* with subtype labels.
    mapper_id_col, mapper_patient_col : str
        Columns in *map_path* linking unique encoded IDs to patients.

    Returns
    -------
    pd.DataFrame
        Columns: ``PATIENT``, ``<subtype_col>``, ``split``, ``encoded_id``.
    """
    clinical = pd.read_csv(str(clinical_path))
    mapper = pd.read_csv(str(map_path))

    # normalise whitespace in column names
    clinical = clinical.rename(columns=lambda c: c.strip().replace(" ", "_"))
    mapper = mapper.rename(columns=lambda c: c.strip().replace(" ", "_"))

    merged = clinical.merge(
        mapper,
        left_on=patient_col,
        right_on=mapper_patient_col,
        how="left",
    )
    missing = merged[mapper_id_col].isna().sum()
    if missing:
        print(f"⚠️  {missing} rows could not be matched to an encoded_id")
    merged = merged.rename(columns={mapper_id_col: "encoded_id"})
    return merged[[patient_col, subtype_col, "split", "encoded_id"]]


# ===================================================================
#  Embedding matrix builder
# ===================================================================


def build_embedding_matrix(
    df: pd.DataFrame,
    embedding_col: str = "embedding",
    expected_dim: Optional[int] = 512,
) -> np.ndarray:
    """Stack the per-row embedding vectors into a dense 2-D ``ndarray``.

    Each entry is squeezed / flattened to 1-D first so that irregular
    shapes (e.g. ``(1, 512)``) are handled transparently.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *embedding_col* with array-like values.
    embedding_col : str
        Column name holding the embeddings.
    expected_dim : int | None
        If set, a warning is printed when the matrix width differs.

    Returns
    -------
    np.ndarray
        Shape ``(n_samples, dim)``.
    """
    emb_list = [_normalise_embedding(np.asarray(x)) for x in df[embedding_col]]
    X = np.stack(emb_list)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    if expected_dim is not None and X.shape[1] != expected_dim:
        print(
            f"⚠️  Embedding matrix width is {X.shape[1]}, expected {expected_dim}"
        )
    return X


def prepare_labels(
    df: pd.DataFrame,
    label_col: str,
    dtype: type = str,
) -> np.ndarray:
    """Extract a label column as a plain ``ndarray`` (avoids ExtensionArray issues)."""
    return df[label_col].to_numpy(dtype=dtype)
