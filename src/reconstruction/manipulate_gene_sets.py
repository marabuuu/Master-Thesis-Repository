"""Generate paired manipulation panels for curated gene SETS (mechanistic programs)
instead of a single gene.

Each gene set (e.g. "proliferation", "stromal_ecm", "immune") is shifted as one
coordinated module: delta_std is the TARGET TOTAL Euclidean-norm displacement of
the conditioning vector, split evenly across all member genes
(delta_per_gene = delta_std / sqrt(n_genes)). This keeps the perturbation
magnitude comparable across sets of different size — and comparable to a
single-gene ±Nσ shift, e.g. the POSTN pilot in manipulate_tiles.py — instead of
compounding with set size, which would push the conditioning vector far off the
training manifold for large sets.

Produces one panel PNG (+ optional metrics JSON) per gene set, reusing the same
model-loading / sampling / metrics machinery as manipulate_tiles.py. Panels are
meant to be stacked manually (e.g. in PowerPoint) into one multi-row figure.

Usage:
    python -m src.reconstruction.manipulate_gene_sets --config src/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.reconstruction.manipulate_tiles import (
    load_gene_index,
    load_feats,
    sample_tile,
    _tensor_to_rgb,
    _load_model,
    compute_manipulation_metrics,
    save_panel,
)
from src.model_training.checkpoint import resolve_checkpoint as _resolve_checkpoint

log = logging.getLogger(__name__)


def perturb_feats_by_set(
    feats: torch.Tensor,
    gene_indices: list[int],
    target_delta_std: float,
) -> tuple[torch.Tensor, float]:
    """Shift a coordinated gene set by a fixed TOTAL Euclidean-norm displacement.

    Splits target_delta_std evenly across all member genes (same sign,
    magnitude target_delta_std / sqrt(N) each) so ||feats_pert - feats|| stays
    equal to |target_delta_std| regardless of set size N, matching the
    magnitude convention of a single-gene ±Nσ shift instead of compounding.
    """
    n = len(gene_indices)
    per_gene_delta = target_delta_std / math.sqrt(n)
    out = feats.clone()
    for idx in gene_indices:
        out[idx] = out[idx] + per_gene_delta
    achieved_norm = float(torch.norm(out - feats).item())
    return out, achieved_norm


def run_gene_set_manipulation(cfg: dict[str, Any]) -> None:
    from src.reconstruction.utils import _ensure_mopadi_import_path
    _ensure_mopadi_import_path()

    run_dir = Path(cfg["run_dir"])
    h5_dir = Path(cfg["genomic_h5_dir"])
    gene_list_path = Path(cfg["gene_list_path"])
    output_dir = Path(cfg["output_dir"])

    patient_id = cfg["patient_id"]
    n_tiles = int(cfg.get("n_tiles", 10))
    guidance_scale = float(cfg.get("guidance_scale", 5.0))
    n_steps = int(cfg.get("steps", 20))
    seed = int(cfg.get("seed", 42))
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    gene_sets: list[dict] = cfg.get("gene_sets", [])
    if not gene_sets:
        raise ValueError("No gene_sets defined in config")

    gene_index = load_gene_index(gene_list_path)

    log.info("Loading checkpoint …")
    ckpt_path = _resolve_checkpoint(run_dir)
    model = _load_model(run_dir, ckpt_path).to(device)

    feats_orig = load_feats(patient_id, h5_dir)
    img_size = model.conf.img_size

    # Same noise per column across all gene sets (seed * 10000 + i, matching
    # manipulation_panel.png), except columns listed in col_seed_overrides
    # (1-indexed) which get an explicit seed instead — used to pin specific
    # columns to the exact same tile crop as manipulation_panel.png regardless
    # of the base `seed` above.
    col_seed_overrides: dict[int, int] = {int(k): int(v) for k, v in cfg.get("col_seed_overrides", {}).items()}
    noises = []
    for i in range(n_tiles):
        g = torch.Generator()
        if (i + 1) in col_seed_overrides:
            override_seed = col_seed_overrides[i + 1]
            g.manual_seed(override_seed)
            log.info("column %d: override seed=%d", i + 1, override_seed)
        else:
            g.manual_seed(seed * 10000 + i)
        noises.append(torch.randn(1, 3, img_size, img_size, generator=g))

    summary: dict[str, Any] = {}

    for gs in gene_sets:
        name = gs["name"]
        genes = gs["genes"]
        target_delta_std = float(gs["delta_std"])

        missing = [g for g in genes if g not in gene_index]
        if missing:
            raise ValueError(f"Gene set '{name}': genes not found in gene list: {missing}")
        gene_indices = [gene_index[g] for g in genes]

        feats_pert, achieved_norm = perturb_feats_by_set(feats_orig, gene_indices, target_delta_std)
        log.info(
            "Gene set '%s': %d genes, target Δ=%.2f → per-gene Δ=%.3f, achieved ||Δfeats||=%.3f",
            name, len(genes), target_delta_std,
            target_delta_std / math.sqrt(len(genes)), achieved_norm,
        )

        original_tiles: list[np.ndarray] = []
        perturbed_tiles: list[np.ndarray] = []
        log.info("Sampling %d tile pairs for '%s' …", n_tiles, name)
        with torch.no_grad():
            for noise in noises:
                original_tiles.append(_tensor_to_rgb(
                    sample_tile(model, feats_orig, noise, guidance_scale, n_steps, device)))
                perturbed_tiles.append(_tensor_to_rgb(
                    sample_tile(model, feats_pert, noise, guidance_scale, n_steps, device)))

        metrics = None
        if cfg.get("compute_metrics", False):
            feature_models = cfg.get("feature_models", ["virchow2"])
            metrics = compute_manipulation_metrics(
                original_tiles, perturbed_tiles,
                feature_models=feature_models,
                device=device,
            )
            metrics["target_delta_std"] = target_delta_std
            metrics["achieved_feats_norm"] = achieved_norm
            metrics["n_genes"] = len(genes)
            metrics["genes"] = genes

        sign = "+" if target_delta_std >= 0 else ""
        title = (f"{name.replace('_', ' ').title()} {sign}{target_delta_std:.1f}σ "
                 f"({len(genes)} genes)  —  patient {patient_id}")
        output_path = output_dir / f"manipulation_panel_{name}.png"

        save_panel(
            original_tiles, perturbed_tiles,
            title=title,
            row_labels=("original", f"{name}\n{sign}{target_delta_std:.1f}σ"),
            output_path=output_path,
            metrics=metrics,
        )

        if metrics:
            metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
            metrics_path.write_text(json.dumps(metrics, indent=2))
            log.info("Metrics saved → %s", metrics_path)
            summary[name] = {k: v for k, v in metrics.items() if not k.endswith("_per_tile")}

    if summary:
        summary_path = output_dir / "manipulation_gene_sets_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        log.info("Summary saved → %s", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config.yaml with a 'gene_set_manipulation' section")
    parser.add_argument("--compute-metrics", action="store_true",
                        help="Force-enable MS-SSIM and feature cosine similarity")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)["gene_set_manipulation"]
    if args.compute_metrics:
        cfg["compute_metrics"] = True

    run_gene_set_manipulation(cfg)


if __name__ == "__main__":
    main()
