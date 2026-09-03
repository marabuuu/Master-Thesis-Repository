#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Perturbation Plausibility Check
================================

Before pushing a single-gene manipulation (``manipulate_tiles.py``) toward a
full "knockout", we need to know whether the perturbed conditioning vector
still looks like a real patient's genomic profile or has drifted outside the
distribution the model was ever trained on. A dramatic tile change at an
implausible input is evidence of out-of-distribution model behavior, not of
simulated gene-knockout biology.

Two checks against the real training cohort
(``experiments/20260528_genomic_features/genomic_h5`` + ``scaler.json``):

  1. Per-gene marginal check: where does the perturbed gene's z-score fall
     relative to the empirical distribution of real patients for that gene,
     and what real (raw, delogged) expression value does it correspond to?
     "Complete knockout" is defined precisely here as the z-score for which
     the delogged expression is exactly 0.

  2. Joint/manifold check: project the full 512-gene perturbed vector
     alongside the real cohort into (a) PCA space and (b) a jointly-fit UMAP
     embedding, and compute a whitened-PCA Mahalanobis-style distance from
     the real-patient centroid, benchmarked against the distribution of
     real-patient-to-centroid distances (i.e. "is this perturbation more
     of an outlier than any real patient already is?").

Usage:
    python -m src.reconstruction.perturbation_plausibility \\
        --h5-dir experiments/20260528_genomic_features/genomic_h5 \\
        --scaler experiments/20260528_genomic_features/scaler.json \\
        --gene-list experiments/20260528_genomic_features/gene_list.txt \\
        --patient-id TCGA-E9-A1NF --gene POSTN \\
        --deltas -1 -2 -3 -4 -4.98 -6 -8 \\
        --output-dir experiments/20260607_brca_pam50_cfg_v2_256/perturbation_plausibility
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_gene_index(gene_list_path: Path) -> dict[str, int]:
    genes = [l.strip() for l in gene_list_path.read_text().splitlines() if l.strip()]
    return {g: i for i, g in enumerate(genes)}


def load_feats(patient_id: str, h5_dir: Path) -> np.ndarray:
    for suffix in ("", "-DX1"):
        p = h5_dir / f"{patient_id}{suffix}.h5"
        if p.exists():
            with h5py.File(p, "r") as f:
                return np.asarray(f["feats"][:], dtype=np.float64).squeeze()
    raise FileNotFoundError(f"No H5 file for {patient_id} in {h5_dir}")


def load_real_cohort(h5_dir: Path) -> tuple[list[str], np.ndarray]:
    """Load every patient's z-scored genomic feature vector in ``h5_dir``."""
    paths = sorted(h5_dir.glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"No .h5 files found in {h5_dir}")
    ids, feats = [], []
    for p in paths:
        with h5py.File(p, "r") as f:
            feats.append(np.asarray(f["feats"][:], dtype=np.float64).squeeze())
        ids.append(p.stem)
    return ids, np.stack(feats, axis=0)


def load_scaler(scaler_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    d = json.loads(scaler_path.read_text())
    return d["genes"], np.asarray(d["mean"], dtype=np.float64), np.asarray(d["scale"], dtype=np.float64)


# ── Per-gene marginal check ─────────────────────────────────────────────────

def gene_marginal_check(
    gene: str,
    gene_idx: int,
    real_feats: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    query_deltas: list[float],
    patient_z: float,
) -> dict[str, Any]:
    real_z = real_feats[:, gene_idx]
    real_expr = np.expm1(mean[gene_idx] + real_z * scale[gene_idx])

    knockout_z = float((0.0 - mean[gene_idx]) / scale[gene_idx])  # delogged expr == 0

    rows = []
    for delta in query_deltas:
        z = patient_z + delta
        expr = float(np.expm1(mean[gene_idx] + z * scale[gene_idx]))
        pct_rank = float((real_z < z).mean() * 100.0)  # percentile of real cohort below this z
        rows.append({
            "delta_std": delta,
            "z_score": z,
            "real_expression": expr,
            "percentile_of_real_cohort": pct_rank,
            "beyond_observed_range": bool(z < real_z.min() or z > real_z.max()),
        })

    return {
        "gene": gene,
        "patient_baseline_z": patient_z,
        "real_cohort_z_min": float(real_z.min()),
        "real_cohort_z_max": float(real_z.max()),
        "real_cohort_z_p01": float(np.percentile(real_z, 1)),
        "real_cohort_z_p99": float(np.percentile(real_z, 99)),
        "knockout_z_delogged_expr_zero": knockout_z,
        "queries": rows,
    }


# ── Joint / manifold check ──────────────────────────────────────────────────

def manifold_check(
    real_feats: np.ndarray,
    query_feats: np.ndarray,
    query_labels: list[str],
    n_pca_components: int = 50,
) -> dict[str, Any]:
    from sklearn.decomposition import PCA

    n_comp = min(n_pca_components, real_feats.shape[0] - 1, real_feats.shape[1])
    pca_white = PCA(n_components=n_comp, whiten=True, random_state=42).fit(real_feats)

    real_scores = pca_white.transform(real_feats)
    query_scores = pca_white.transform(query_feats)

    real_dist = np.linalg.norm(real_scores, axis=1)  # Mahalanobis-style distance to centroid
    query_dist = np.linalg.norm(query_scores, axis=1)

    real_dist_pcts = {p: float(np.percentile(real_dist, p)) for p in (50, 90, 95, 99, 100)}

    rows = []
    for label, dist in zip(query_labels, query_dist):
        pct_rank = float((real_dist < dist).mean() * 100.0)
        rows.append({
            "label": label,
            "mahalanobis_dist_whitened_pca": float(dist),
            "percentile_vs_real_cohort_self_distances": pct_rank,
            "more_outlying_than_real_p99": bool(dist > real_dist_pcts[99]),
        })

    # Leave-one-out nearest-real-neighbor distance (raw z-scored space) as a
    # simpler, model-free complement to the whitened-PCA Mahalanobis distance.
    from scipy.spatial.distance import cdist

    real_nn = cdist(real_feats, real_feats)
    np.fill_diagonal(real_nn, np.inf)
    real_nn_dist = real_nn.min(axis=1)
    real_nn_pcts = {p: float(np.percentile(real_nn_dist, p)) for p in (50, 90, 95, 99, 100)}

    query_nn = cdist(query_feats, real_feats).min(axis=1)
    for row, dist in zip(rows, query_nn):
        row["nearest_real_patient_dist_raw"] = float(dist)
        row["nn_dist_percentile_vs_real_self_distances"] = float((real_nn_dist < dist).mean() * 100.0)

    return {
        "n_pca_components": n_comp,
        "real_cohort_mahalanobis_percentiles": real_dist_pcts,
        "real_cohort_nn_dist_percentiles": real_nn_pcts,
        "queries": rows,
        # Kept for plotting
        "_real_dist": real_dist,
        "_pca2_real": PCA(n_components=2, random_state=42).fit(real_feats).transform(real_feats),
    }


# ── Class-mean-conditioned trajectory ───────────────────────────────────────

def compute_low_expression_centroid(
    real_feats: np.ndarray,
    gene_idx: int,
    percentile: float = 10.0,
) -> tuple[np.ndarray, float]:
    """Average feature vector of real patients in the bottom `percentile`% for one gene."""
    gene_z = real_feats[:, gene_idx]
    threshold = np.percentile(gene_z, percentile)
    mask = gene_z <= threshold
    centroid = real_feats[mask].mean(axis=0)
    return centroid, float(centroid[gene_idx])


def build_classmean_trajectory(
    patient_feats: np.ndarray,
    centroid_feats: np.ndarray,
    gene_idx: int,
    patient_z: float,
    centroid_z: float,
    deltas: list[float],
) -> tuple[np.ndarray, list[float]]:
    """Interpolate patient -> low-expression-class centroid, matched to the same
    target gene z-scores as the naive single-gene deltas, so the two trajectories
    are directly comparable step-for-step. alpha=1 reaches the class centroid
    exactly; alpha>1 means that delta pushes further than the class mean itself.
    """
    direction = centroid_feats - patient_feats
    rows, alphas = [], []
    for delta in deltas:
        target_z = patient_z + delta
        alpha = (target_z - patient_z) / (centroid_z - patient_z)
        rows.append(patient_feats + alpha * direction)
        alphas.append(alpha)
    return np.stack(rows, axis=0), alphas


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_manifold(
    real_feats: np.ndarray,
    real_gene_z: np.ndarray,
    gene_name: str,
    trajectories: list[dict[str, Any]],
    output_path: Path,
    suptitle: str | None = None,
) -> None:
    """trajectories: list of {"name": str, "feats": (n,512) array, "labels": list[str],
    "marker": str, "cmap": str} — each plotted as its own connected, colored path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import umap

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
    })

    all_query = np.concatenate([t["feats"] for t in trajectories], axis=0)
    combined = np.concatenate([real_feats, all_query], axis=0)
    n_real = real_feats.shape[0]

    pca2 = PCA(n_components=2, random_state=42).fit(real_feats)
    pca_combined = pca2.transform(combined)

    log.info("Fitting joint UMAP over real cohort + trajectories (n=%d)...", combined.shape[0])
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    umap_combined = reducer.fit_transform(combined)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, emb, title in zip(
        axes, [pca_combined, umap_combined], ["PCA (fit on real cohort)", "UMAP (joint fit: real + trajectories)"]
    ):
        real_pts = emb[:n_real]
        sc = ax.scatter(
            real_pts[:, 0], real_pts[:, 1], s=10, c=real_gene_z, cmap="coolwarm",
            alpha=0.75, zorder=1, vmin=np.percentile(real_gene_z, 2), vmax=np.percentile(real_gene_z, 98),
        )
        offset = n_real
        for traj in trajectories:
            n_t = traj["feats"].shape[0]
            pts = emb[offset:offset + n_t]
            offset += n_t
            cmap = plt.cm.get_cmap(traj["cmap"])
            colors = [cmap(i / max(1, n_t - 1)) for i in range(n_t)]
            ax.plot(pts[:, 0], pts[:, 1], "-", color="0.3", lw=1.2, zorder=2)
            for pt, color in zip(pts, colors):
                ax.scatter(*pt, s=90, marker=traj["marker"], color=color,
                           edgecolors="black", linewidths=0.8, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if ax is axes[-1]:
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f"real patient {gene_name} z-score")

    from matplotlib.lines import Line2D

    legend_handles = []
    for traj in trajectories:
        n_t = traj["feats"].shape[0]
        cmap = plt.cm.get_cmap(traj["cmap"])
        mid_color = cmap(0.7)
        legend_handles.append(Line2D(
            [0], [0], marker=traj["marker"], color="none", markerfacecolor=mid_color,
            markeredgecolor="black", markersize=11, label=traj["name"],
        ))
    fig.legend(
        handles=legend_handles, loc="center left", bbox_to_anchor=(1.15, 0.5),
        fontsize=14, frameon=True, title="Trajectory\n(light→dark = 0→−σ delta)",
        title_fontsize=13,
    )

    fig.suptitle(suptitle or f"{gene_name} perturbation: naive single-gene edit vs. class-mean-conditioned edit", fontsize=19)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved -> %s", output_path)


# ── Main ─────────────────────────────────────────────────────────────────────

def run(
    h5_dir: Path,
    scaler_path: Path,
    gene_list_path: Path,
    patient_id: str,
    gene: str,
    deltas: list[float],
    output_dir: Path,
    plot_title: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    gene_index = load_gene_index(gene_list_path)
    if gene not in gene_index:
        raise ValueError(f"Gene '{gene}' not found in gene list")
    g_idx = gene_index[gene]

    scaler_genes, mean, scale = load_scaler(scaler_path)
    assert scaler_genes == list(gene_index.keys()), "scaler.json gene order does not match gene_list.txt"

    log.info("Loading real cohort from %s ...", h5_dir)
    real_ids, real_feats = load_real_cohort(h5_dir)
    log.info("Loaded %d real patients, %d genes", *real_feats.shape)

    patient_feats = load_feats(patient_id, h5_dir)
    patient_z = float(patient_feats[g_idx])

    # ── 1. Per-gene marginal check ──
    marginal = gene_marginal_check(gene, g_idx, real_feats, mean, scale, deltas, patient_z)
    log.info(
        "%s: patient baseline z=%.2f | real cohort observed z range [%.2f, %.2f] | "
        "knockout (expr=0) at z=%.2f",
        gene, patient_z, marginal["real_cohort_z_min"], marginal["real_cohort_z_max"],
        marginal["knockout_z_delogged_expr_zero"],
    )
    for row in marginal["queries"]:
        flag = " <-- BEYOND OBSERVED RANGE" if row["beyond_observed_range"] else ""
        log.info(
            "  delta=%.2f -> z=%.2f, real_expr=%.3f, percentile=%.1f%%%s",
            row["delta_std"], row["z_score"], row["real_expression"],
            row["percentile_of_real_cohort"], flag,
        )

    # ── 2. Joint/manifold check: naive single-gene edit ──
    deltas_full = [0.0] + list(deltas)
    naive_labels = ["baseline (delta=0)"] + [f"delta={d:+.2f}" for d in deltas]
    naive_feats = np.stack(
        [patient_feats] + [
            (lambda f: (f.__setitem__(g_idx, patient_z + d) or f))(patient_feats.copy())
            for d in deltas
        ],
        axis=0,
    )

    naive_manifold = manifold_check(real_feats, naive_feats, naive_labels)
    log.info("Naive edit — whitened-PCA Mahalanobis percentiles (real cohort): %s", naive_manifold["real_cohort_mahalanobis_percentiles"])
    for row in naive_manifold["queries"]:
        outlier_flag = " <-- MORE OUTLYING THAN 99% OF REAL PATIENTS" if row["more_outlying_than_real_p99"] else ""
        log.info(
            "  [naive] %s: Mahalanobis=%.2f (percentile=%.1f%%), NN-dist=%.2f (percentile=%.1f%%)%s",
            row["label"], row["mahalanobis_dist_whitened_pca"],
            row["percentile_vs_real_cohort_self_distances"],
            row["nearest_real_patient_dist_raw"], row["nn_dist_percentile_vs_real_self_distances"],
            outlier_flag,
        )

    # ── 3. Class-mean-conditioned edit: move the WHOLE vector toward the average
    #    of real patients who naturally have low expression of `gene`, instead of
    #    editing one dimension in isolation on top of this patient's other 511
    #    genes (which never move, however extreme the target percentile is).
    low_pct = 10.0
    centroid_feats, centroid_z = compute_low_expression_centroid(real_feats, g_idx, percentile=low_pct)
    log.info(
        "Low-%.0f%% %s class centroid: z=%.2f (n=%d real patients) vs. patient baseline z=%.2f",
        low_pct, gene, centroid_z, int((real_feats[:, g_idx] <= np.percentile(real_feats[:, g_idx], low_pct)).sum()), patient_z,
    )

    classmean_feats, classmean_alphas = build_classmean_trajectory(
        patient_feats, centroid_feats, g_idx, patient_z, centroid_z, deltas_full,
    )
    classmean_labels = [f"classmean delta={d:+.2f} (α={a:.2f})" for d, a in zip(deltas_full, classmean_alphas)]

    classmean_manifold = manifold_check(real_feats, classmean_feats, classmean_labels)
    log.info("Class-mean edit — whitened-PCA Mahalanobis percentiles (real cohort): %s", classmean_manifold["real_cohort_mahalanobis_percentiles"])
    for row, alpha in zip(classmean_manifold["queries"], classmean_alphas):
        outlier_flag = " <-- MORE OUTLYING THAN 99% OF REAL PATIENTS" if row["more_outlying_than_real_p99"] else ""
        log.info(
            "  [classmean] %s: alpha=%.2f, Mahalanobis=%.2f (percentile=%.1f%%), NN-dist=%.2f (percentile=%.1f%%)%s",
            row["label"], alpha, row["mahalanobis_dist_whitened_pca"],
            row["percentile_vs_real_cohort_self_distances"],
            row["nearest_real_patient_dist_raw"], row["nn_dist_percentile_vs_real_self_distances"],
            outlier_flag,
        )

    summary = {
        "patient_id": patient_id,
        "gene": gene,
        "deltas": deltas,
        "gene_marginal_check": marginal,
        "naive_manifold_check": {k: v for k, v in naive_manifold.items() if not k.startswith("_")},
        "classmean_manifold_check": {
            "low_expression_percentile": low_pct,
            "centroid_gene_z": centroid_z,
            "alphas": classmean_alphas,
            **{k: v for k, v in classmean_manifold.items() if not k.startswith("_")},
        },
    }
    with open(output_dir / "plausibility_summary_classmean.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved -> %s", output_dir / "plausibility_summary_classmean.json")

    plot_manifold(
        real_feats, real_feats[:, g_idx], gene,
        trajectories=[
            {"name": "naive (single-gene edit)", "feats": naive_feats, "labels": naive_labels, "marker": "o", "cmap": "plasma"},
            {"name": f"class-mean-conditioned (bottom {low_pct:.0f}% {gene})", "feats": classmean_feats, "labels": classmean_labels, "marker": "s", "cmap": "viridis"},
        ],
        output_path=output_dir / "manifold_plot_classmean.png",
        suptitle=plot_title,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5-dir", type=Path, required=True)
    p.add_argument("--scaler", type=Path, required=True)
    p.add_argument("--gene-list", type=Path, required=True)
    p.add_argument("--patient-id", type=str, required=True)
    p.add_argument("--gene", type=str, required=True)
    p.add_argument("--deltas", type=float, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--plot-title", type=str, default=None)
    args = p.parse_args()

    run(
        h5_dir=args.h5_dir, scaler_path=args.scaler, gene_list_path=args.gene_list,
        patient_id=args.patient_id, gene=args.gene, deltas=args.deltas,
        output_dir=args.output_dir, plot_title=args.plot_title,
    )


if __name__ == "__main__":
    main()
