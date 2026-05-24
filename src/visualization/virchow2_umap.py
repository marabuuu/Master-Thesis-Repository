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

def _load_patient_feats(
    h5_path: Path,
    max_tiles: Optional[int],
    rng: np.random.RandomState,
) -> Tuple[str, np.ndarray]:
    """Return (patient_id, feats) where feats is shape (n_sampled, feat_dim)."""
    pid = canonical_patient_id(h5_path.stem)
    with h5py.File(h5_path, "r") as h5f:
        feats = _pick_feature_array(h5f)  # (n_tiles, feat_dim)

    if max_tiles is not None and feats.shape[0] > max_tiles:
        idx = rng.choice(feats.shape[0], max_tiles, replace=False)
        feats = feats[idx]

    return pid, feats.astype(np.float32)


def _iter_h5_files(
    features_dir: Path,
    subtype_map: Dict[str, str],
    pid_to_split: Optional[Dict[str, str]],
    max_tiles_per_patient: Optional[int],
    rng: np.random.RandomState,
    verbose: bool,
) -> Tuple[List[np.ndarray], List[str], List[str], List[str]]:
    """Shared h5 iteration used by both patient-level and tile-level loaders."""
    h5_files = sorted(features_dir.glob("*.h5")) + sorted(features_dir.glob("*.hdf5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5/.hdf5 files found in: {features_dir}")

    feat_blocks: List[np.ndarray] = []
    patient_ids: List[str] = []
    subtypes: List[str] = []
    splits: List[str] = []
    skipped_subtype = 0
    skipped_split = 0

    for h5_path in h5_files:
        pid, feats = _load_patient_feats(h5_path, max_tiles_per_patient, rng)

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

        feat_blocks.append(feats)
        patient_ids.append(pid)
        subtypes.append(subtype)
        splits.append(split)

    if not feat_blocks:
        raise RuntimeError(
            f"No patients matched.  "
            f"Checked {len(h5_files)} h5 files; "
            f"{skipped_split} had no split entry, {skipped_subtype} had no subtype."
        )

    if verbose:
        print(
            f"[Virchow2UMAP] Loaded {len(feat_blocks):,} patients  "
            f"(skipped: {skipped_split} no-split, {skipped_subtype} no-subtype)"
        )

    return feat_blocks, patient_ids, subtypes, splits


def load_patient_level_features(
    features_dir: Path,
    subtype_map: Dict[str, str],
    pid_to_split: Optional[Dict[str, str]],
    max_tiles_per_patient: Optional[int],
    seed: int,
    verbose: bool,
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """One mean-pooled vector per patient.  Returns (embeddings, pids, subtypes, splits)."""
    rng = np.random.RandomState(seed)
    feat_blocks, patient_ids, subtypes, splits = _iter_h5_files(
        features_dir, subtype_map, pid_to_split, max_tiles_per_patient, rng, verbose
    )
    embeddings = np.stack([b.mean(axis=0) for b in feat_blocks], axis=0)
    return embeddings, patient_ids, subtypes, splits


def load_tile_level_features(
    features_dir: Path,
    subtype_map: Dict[str, str],
    pid_to_split: Optional[Dict[str, str]],
    max_tiles_per_patient: Optional[int],
    seed: int,
    verbose: bool,
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """One row per tile.  Returns (embeddings, pids, subtypes, splits) — all tile-length."""
    rng = np.random.RandomState(seed)
    feat_blocks, patient_ids, subtypes, splits = _iter_h5_files(
        features_dir, subtype_map, pid_to_split, max_tiles_per_patient, rng, verbose
    )
    # Expand metadata to one entry per tile
    tile_pids: List[str] = []
    tile_subtypes: List[str] = []
    tile_splits: List[str] = []
    for feats, pid, sub, spl in zip(feat_blocks, patient_ids, subtypes, splits):
        n = feats.shape[0]
        tile_pids.extend([pid] * n)
        tile_subtypes.extend([sub] * n)
        tile_splits.extend([spl] * n)

    embeddings = np.concatenate(feat_blocks, axis=0)
    if verbose:
        print(f"[Virchow2UMAP] Total tiles for UMAP: {embeddings.shape[0]:,}")
    return embeddings, tile_pids, tile_subtypes, tile_splits


# ---------------------------------------------------------------------------
# Plot — mirrors plot_projection in latent_space.py
# ---------------------------------------------------------------------------

def plot_umap_scatter(
    embedding: np.ndarray,
    subtypes: List[str],
    splits: List[str],
    palette: Dict[str, str],
    output_dir: Path,
    mode: str = "patient",
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

    unit = "tiles" if mode == "tile" else "patients"
    title = f"Virchow2 Feature UMAP — PAM50 Subtypes ({len(tmp):,} {unit})"

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
        rasterized=mode == "tile",
        ax=ax,
    )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False,
              fontsize=9, title_fontsize=10)

    fig.tight_layout()
    fname = f"virchow2_umap_{mode}.png"
    save_figure(fig, output_dir / fname, dpi=dpi)


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
    mode               = cfg.get("mode", "patient")  # "patient" or "tile"
    wanted_subtypes    = cfg.get("subtypes", None)   # None = all subtypes
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
    if mode == "tile":
        embeddings, patient_ids, subtypes, splits = load_tile_level_features(
            features_dir, subtype_map, pid_to_split, max_tiles, seed, verbose
        )
    else:
        embeddings, patient_ids, subtypes, splits = load_patient_level_features(
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

    # ── Subtype filter ────────────────────────────────────────────────────────
    if wanted_subtypes:
        wanted_set = set(wanted_subtypes)
        mask = [s in wanted_set for s in subtypes]
        if not any(mask):
            raise RuntimeError(
                f"No patients found for subtypes {wanted_subtypes}. "
                f"Available: {sorted(set(subtypes))}"
            )
        embeddings  = embeddings[np.array(mask)]
        patient_ids = [p for p, m in zip(patient_ids, mask) if m]
        subtypes    = [s for s, m in zip(subtypes,    mask) if m]
        splits      = [sp for sp, m in zip(splits,    mask) if m]
        if verbose:
            from collections import Counter
            print(f"[Virchow2UMAP] Filtered to {wanted_subtypes}:")
            for sub, cnt in sorted(Counter(subtypes).items()):
                print(f"  {sub}: {cnt}")

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
    umap_embedding: np.ndarray = np.asarray(reducer.fit_transform(embeddings))

    # Save coordinates for downstream use
    coords_df = pd.DataFrame({
        "patient_id": patient_ids,
        "subtype":    subtypes,
        "split":      splits,
        "umap_1":     umap_embedding[:, 0],
        "umap_2":     umap_embedding[:, 1],
    })
    coords_path = output_dir / f"virchow2_umap_{mode}_coords.csv"
    coords_df.to_csv(coords_path, index=False)
    if verbose:
        print(f"[Virchow2UMAP] Saved coordinates → {coords_path.name}")

    # ── Palette — same build as dataset_statistics and visualize_latents ──────
    palette = build_label_palette(np.array(subtypes), cmap_name=cmap_name)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_umap_scatter(
        umap_embedding, subtypes, splits, palette, output_dir,
        mode=mode, figsize=figsize, point_size=point_size, alpha=alpha, dpi=dpi,
    )

    if verbose:
        print(f"[Virchow2UMAP] Done. Output → {output_dir}")
