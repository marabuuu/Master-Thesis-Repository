#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Virchow2 Feature UMAP
======================

Loads per-patient Virchow2 feature h5 files, mean-pools tile embeddings into
one vector per patient, runs UMAP, and plots the embedding coloured by PAM50
subtype — using the same colour palette, marker shapes, and seaborn style as
the genomic latent-space UMAP in ``visualize_latents``.

Colours:  ``build_label_palette(subtypes, cmap_name)``  — identical to both
          ``dataset_statistics`` (subtype pie) and ``visualize_latents``.
Markers:  train → ○   val → □   test → △   (same as ``visualize_latents``)

When ``patient_splits_path`` is provided the plot is **restricted** to the
patients that appear in that JSON (train + val + test combined), and split
membership is encoded as the marker shape.

Usage (via run_pipeline.py)
----------------------------
    python run_pipeline.py --config src/config.yaml --stage virchow2_umap

Config section (config.yaml)
------------------------------
    virchow2_umap:
      features_dir: /path/to/virchow2_h5_files
      csv_path: ../dataframes/brca_subtypes.csv
      patient_col: bcr_patient_barcode
      subtype_col: Majority_Subtype_mRNA
      patient_splits_path: null          # optional — restricts + annotates by split
      output_dir: ./experiments/virchow2_umap
      max_tiles_per_patient: 100         # null = all tiles
      n_neighbors: 15
      min_dist: 0.1
      metric: cosine
      seed: 42
      cmap_categorical: romaO
      figsize: [10, 8]
      dpi: 200
      point_size: 80
      alpha: 0.8
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False

try:
    import umap
    HAS_UMAP = True
except ImportError:  # pragma: no cover
    HAS_UMAP = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:  # pragma: no cover
    HAS_H5PY = False

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:  # pragma: no cover
    HAS_SNS = False

try:
    from visualization.core import (
        CATEGORICAL_CMAP,
        build_label_palette,
        save_figure,
        setup_style,
    )
except ImportError:
    from src.visualization.core import (  # type: ignore[import-not-found]
        CATEGORICAL_CMAP,
        build_label_palette,
        save_figure,
        setup_style,
    )

# Reuse h5 reading utilities from the classifier
try:
    from classifier.utils_subtype_data import (
        canonical_patient_id,
        _pick_feature_array,
    )
except ImportError:
    try:
        from src.classifier.utils_subtype_data import (  # type: ignore[import-not-found]
            canonical_patient_id,
            _pick_feature_array,
        )
    except ImportError:
        def canonical_patient_id(name: str) -> str:  # type: ignore[misc]
            stem = Path(str(name)).stem.upper()
            for sep in ("_", "."):
                stem = stem.replace(sep, "-")
            while "--" in stem:
                stem = stem.replace("--", "-")
            parts = stem.split("-")
            if len(parts) >= 3 and parts[0].startswith("TCGA"):
                return "-".join(parts[:3])
            return stem

        def _pick_feature_array(h5f) -> np.ndarray:  # type: ignore[misc]
            for key in h5f.keys():
                arr = h5f[key][()]
                if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] > 100:
                    return arr.astype(np.float32)
            raise ValueError("No feature array found in H5 file")


# Split marker shapes — identical to visualize_latents
_SPLIT_MARKERS: Dict[str, str] = {"train": "o", "val": "s", "test": "^", "unknown": "D"}


# ---------------------------------------------------------------------------
# Split JSON loading
# ---------------------------------------------------------------------------

def load_patient_splits(splits_path: str | Path) -> Dict[str, str]:
    """Return ``{patient_id: split_name}`` from the patient splits JSON."""
    with open(splits_path) as f:
        raw = json.load(f)

    pid_to_split: Dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        if split_name not in raw:
            continue
        entry = raw[split_name]
        if isinstance(entry, dict):
            # Two formats: {"patients": [...]} or {pid: ..., pid: ...}
            patients = entry.get("patients", None)
            if patients is None:
                patients = list(entry.keys())
        else:
            patients = entry  # plain list
        for pid in patients:
            pid_to_split[canonical_patient_id(pid)] = split_name
    return pid_to_split


# ---------------------------------------------------------------------------
# H5 feature loading
# ---------------------------------------------------------------------------

def _load_patient_mean_feature(
    h5_path: Path,
    max_tiles: Optional[int],
    rng: np.random.RandomState,
) -> Tuple[str, np.ndarray]:
    pid = canonical_patient_id(h5_path.stem)
    with h5py.File(h5_path, "r") as h5f:
        feats = _pick_feature_array(h5f)  # (n_tiles, feat_dim)

    if max_tiles is not None and feats.shape[0] > max_tiles:
        idx = rng.choice(feats.shape[0], max_tiles, replace=False)
        feats = feats[idx]

    return pid, feats.mean(axis=0).astype(np.float32)


def load_all_features(
    features_dir: Path,
    subtype_map: Dict[str, str],
    pid_to_split: Optional[Dict[str, str]],
    max_tiles_per_patient: Optional[int],
    seed: int,
    verbose: bool,
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """
    Load h5 files, mean-pool tiles, join with subtype + split info.

    When ``pid_to_split`` is provided only patients present in that mapping
    are included (i.e. the same patient selection as the genomic UMAP).

    Returns
    -------
    embeddings  : ndarray (N, feat_dim)
    patient_ids : list[str]
    subtypes    : list[str]
    splits      : list[str]  — "train" / "val" / "test" / "unknown"
    """
    h5_files = sorted(features_dir.glob("*.h5")) + sorted(features_dir.glob("*.hdf5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5/.hdf5 files found in: {features_dir}")

    rng = np.random.RandomState(seed)
    vecs: List[np.ndarray] = []
    patient_ids: List[str] = []
    subtypes: List[str] = []
    splits: List[str] = []
    skipped_subtype = 0
    skipped_split = 0

    for h5_path in h5_files:
        pid, mean_vec = _load_patient_mean_feature(h5_path, max_tiles_per_patient, rng)

        # Split filter: when a splits map is given, skip patients not in it
        if pid_to_split is not None:
            if pid not in pid_to_split:
                skipped_split += 1
                continue
            split = pid_to_split[pid]
        else:
            split = "unknown"

        subtype = subtype_map.get(pid)
        if subtype is None:
            skipped_subtype += 1
            continue

        vecs.append(mean_vec)
        patient_ids.append(pid)
        subtypes.append(subtype)
        splits.append(split)

    if not vecs:
        raise RuntimeError(
            f"No patients matched.  "
            f"Checked {len(h5_files)} h5 files; "
            f"{skipped_split} had no split entry, {skipped_subtype} had no subtype."
        )

    if verbose:
        print(
            f"[Virchow2UMAP] Loaded {len(vecs):,} patients  "
            f"(skipped: {skipped_split} no-split, {skipped_subtype} no-subtype)"
        )

    return np.stack(vecs, axis=0), patient_ids, subtypes, splits


# ---------------------------------------------------------------------------
# Plot — mirrors plot_projection in latent_space.py
# ---------------------------------------------------------------------------

def plot_umap_scatter(
    embedding: np.ndarray,
    subtypes: List[str],
    splits: List[str],
    palette: Dict[str, str],
    output_dir: Path,
    figsize: Tuple[int, int] = (10, 8),
    point_size: float = 80,
    alpha: float = 0.8,
    dpi: int = 200,
) -> None:
    assert HAS_MPL and HAS_SNS

    tmp = pd.DataFrame({
        "UMAP 1":  embedding[:, 0],
        "UMAP 2":  embedding[:, 1],
        "Subtype": subtypes,
        "Split":   splits,
    })

    has_multiple_splits = tmp["Split"].nunique() > 1

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=tmp,
        x="UMAP 1",
        y="UMAP 2",
        hue="Subtype",
        style="Split" if has_multiple_splits else None,
        palette=palette,
        markers=_SPLIT_MARKERS if has_multiple_splits else None,
        s=point_size,
        alpha=alpha,
        edgecolor="k",
        linewidth=0.3,
        ax=ax,
    )

    ax.set_title("Virchow2 Feature UMAP — PAM50 Subtypes", fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False,
              fontsize=9, title_fontsize=10)

    fig.tight_layout()
    save_figure(fig, output_dir / "virchow2_umap.png", dpi=dpi)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_virchow2_umap(cfg: dict, verbose: bool = True) -> None:
    """Run the Virchow2 UMAP visualization stage."""
    if not HAS_MPL:
        raise ImportError("matplotlib is required.  pip install matplotlib")
    if not HAS_UMAP:
        raise ImportError("umap-learn is required.  pip install umap-learn")
    if not HAS_H5PY:
        raise ImportError("h5py is required.  pip install h5py")
    if not HAS_SNS:
        raise ImportError("seaborn is required.  pip install seaborn")

    setup_style()

    features_dir = Path(cfg["features_dir"])
    if not features_dir.exists():
        raise FileNotFoundError(f"features_dir not found: {features_dir}")

    csv_path           = cfg["csv_path"]
    patient_col        = cfg.get("patient_col", "Patient_ID")
    subtype_col        = cfg.get("subtype_col", "Majority_Subtype_mRNA")
    splits_path        = cfg.get("patient_splits_path")
    output_dir         = Path(cfg.get("output_dir", "./experiments/virchow2_umap"))
    max_tiles          = cfg.get("max_tiles_per_patient", 100)
    n_neighbors        = int(cfg.get("n_neighbors", 15))
    min_dist           = float(cfg.get("min_dist", 0.1))
    metric             = cfg.get("metric", "cosine")
    seed               = int(cfg.get("seed", 42))
    cmap_name          = cfg.get("cmap_categorical", CATEGORICAL_CMAP)
    figsize_cfg        = cfg.get("figsize", [10, 8])
    figsize            = (int(figsize_cfg[0]), int(figsize_cfg[1]))
    dpi                = int(cfg.get("dpi", 200))
    point_size         = float(cfg.get("point_size", 80))
    alpha              = float(cfg.get("alpha", 0.8))

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Patient splits ────────────────────────────────────────────────────────
    pid_to_split: Optional[Dict[str, str]] = None
    if splits_path and Path(splits_path).exists():
        pid_to_split = load_patient_splits(splits_path)
        if verbose:
            from collections import Counter
            split_counts = Counter(pid_to_split.values())
            print(
                f"[Virchow2UMAP] Splits loaded from {Path(splits_path).name}: "
                + ", ".join(f"{s}={n}" for s, n in sorted(split_counts.items()))
            )
    elif splits_path:
        print(f"[Virchow2UMAP][WARN] patient_splits_path not found: {splits_path} — using all patients")

    # ── Subtype map ───────────────────────────────────────────────────────────
    meta_df = pd.read_csv(csv_path, low_memory=False)
    meta_df[patient_col] = (
        meta_df[patient_col].astype(str).str.strip().apply(canonical_patient_id)
    )
    meta_df = meta_df.drop_duplicates(subset=[patient_col])
    subtype_map: Dict[str, str] = {
        pid: sub
        for pid, sub in zip(meta_df[patient_col], meta_df[subtype_col].astype(str).str.strip())
        if sub and sub.lower() not in {"nan", ""}
    }
    if verbose:
        print(f"[Virchow2UMAP] Subtype map: {len(subtype_map):,} patients from {Path(csv_path).name}")

    # ── Load features ─────────────────────────────────────────────────────────
    embeddings, patient_ids, subtypes, splits = load_all_features(
        features_dir, subtype_map, pid_to_split, max_tiles, seed, verbose
    )

    if verbose:
        from collections import Counter
        print("[Virchow2UMAP] Subtype counts:")
        for sub, cnt in sorted(Counter(subtypes).items()):
            print(f"  {sub}: {cnt}")
        if pid_to_split is not None:
            print("[Virchow2UMAP] Split counts:")
            for spl, cnt in sorted(Counter(splits).items()):
                print(f"  {spl}: {cnt}")

    # ── UMAP ──────────────────────────────────────────────────────────────────
    if verbose:
        print(
            f"[Virchow2UMAP] Running UMAP  "
            f"n={embeddings.shape[0]}, dim={embeddings.shape[1]}, "
            f"n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}"
        )

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=2,
        random_state=seed,
        verbose=verbose,
    )
    umap_embedding = reducer.fit_transform(embeddings)

    # Save coordinates for downstream use
    coords_df = pd.DataFrame({
        "patient_id": patient_ids,
        "subtype":    subtypes,
        "split":      splits,
        "umap_1":     umap_embedding[:, 0],
        "umap_2":     umap_embedding[:, 1],
    })
    coords_path = output_dir / "virchow2_umap_coords.csv"
    coords_df.to_csv(coords_path, index=False)
    if verbose:
        print(f"[Virchow2UMAP] Saved coordinates → {coords_path.name}")

    # ── Palette — same build as dataset_statistics and visualize_latents ──────
    palette = build_label_palette(np.array(subtypes), cmap_name=cmap_name)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_umap_scatter(
        umap_embedding, subtypes, splits, palette, output_dir,
        figsize=figsize, point_size=point_size, alpha=alpha, dpi=dpi,
    )

    if verbose:
        print(f"[Virchow2UMAP] Done. Output → {output_dir}")
