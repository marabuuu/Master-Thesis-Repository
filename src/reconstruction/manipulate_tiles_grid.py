"""Generate a multi-delta manipulation grid: one row per perturbation strength.

Companion to ``manipulate_tiles.py`` (single delta, 2-row original/perturbed
panel) and ``perturbation_plausibility.py`` (manifold/OOD check, no images).
This script renders the same delta grid used in the plausibility check as
actual tiles, using the SAME per-column noise seeds as
``manipulation_panel.png`` (``seed * 10000 + i``), so the top row (delta=0)
is pixel-for-pixel the same sampling procedure as that panel's "original"
row, and every row after it is directly comparable to the corresponding
point in ``manifold_plot.png``.

Usage:
    python -m src.reconstruction.manipulate_tiles_grid --config src/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.reconstruction.manipulate_tiles import (
    load_gene_index,
    load_feats,
    perturb_feats,
    sample_tile,
    _tensor_to_rgb,
    _load_model,
)
from src.model_training.checkpoint import resolve_checkpoint as _resolve_checkpoint

log = logging.getLogger(__name__)


def run_grid(cfg: dict) -> None:
    from src.reconstruction.utils import _ensure_mopadi_import_path
    _ensure_mopadi_import_path()

    run_dir = Path(cfg["run_dir"])
    h5_dir = Path(cfg["genomic_h5_dir"])
    gene_list_path = Path(cfg["gene_list_path"])
    output_path = Path(cfg["output_path"])

    patient_id = cfg["patient_id"]
    gene = cfg["gene"]
    deltas: list[float] = cfg["deltas"]
    n_tiles = int(cfg.get("n_tiles", 10))
    guidance_scale = float(cfg.get("guidance_scale", 5.0))
    n_steps = int(cfg.get("steps", 20))
    seed = int(cfg.get("seed", 42))
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    gene_index = load_gene_index(gene_list_path)
    if gene not in gene_index:
        raise ValueError(f"Gene '{gene}' not found in gene list")
    g_idx = gene_index[gene]

    log.info("Loading checkpoint …")
    ckpt_path = _resolve_checkpoint(run_dir, explicit=cfg.get("checkpoint"))
    log.info("Using checkpoint: %s", ckpt_path)
    model = _load_model(run_dir, ckpt_path).to(device)

    feats_orig = load_feats(patient_id, h5_dir)
    baseline_z = float(feats_orig[g_idx])

    # Same noise per column as manipulation_panel.png, reused across every row,
    # except columns listed in col_seed_overrides (1-indexed) which get their
    # own independent seed instead of the seed*10000+i formula.
    col_seed_overrides: dict[int, int] = cfg.get("col_seed_overrides", {})
    img_size = model.conf.img_size
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

    rows: list[list[np.ndarray]] = []
    row_labels: list[str] = []
    with torch.no_grad():
        for delta in deltas:
            feats = perturb_feats(feats_orig, [(g_idx, delta)]) if delta != 0.0 else feats_orig
            log.info("delta=%+.2f -> %s z=%.3f", delta, gene, baseline_z + delta)
            tiles = [
                _tensor_to_rgb(sample_tile(model, feats, noise, guidance_scale, n_steps, device))
                for noise in noises
            ]
            rows.append(tiles)
            row_labels.append("baseline" if delta == 0.0 else f"{gene} {delta:+.2f}σ")

    save_grid(rows, row_labels, patient_id, f"{gene} [{ckpt_path.name}]", output_path, title=cfg.get("title"))


def save_grid(
    rows: list[list[np.ndarray]],
    row_labels: list[str],
    patient_id: str,
    gene: str,
    output_path: Path,
    title: str | None = None,
) -> None:
    n_rows = len(rows)
    n_cols = len(rows[0])
    h, w = rows[0][0].shape[:2]
    dpi = 150
    fig_w = n_cols * w / dpi + 1.1
    fig_h = n_rows * h / dpi + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        hspace=0.04, wspace=0.02,
        left=0.09, right=0.99, top=0.94, bottom=0.01,
    )

    for r, tiles in enumerate(rows):
        for c, tile in enumerate(tiles):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(tile)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    for r, label in enumerate(row_labels):
        row_top = 0.94 - r / n_rows * 0.93
        row_bot = 0.94 - (r + 1) / n_rows * 0.93
        fig.text(
            0.03, (row_top + row_bot) / 2, label,
            va="center", ha="center", fontsize=9, rotation=90,
            color="#333333" if r == 0 else "#aa2222",
            transform=fig.transFigure,
        )

    fig.suptitle(title or f"{gene} perturbation grid — patient {patient_id} (same noise per column)", fontsize=11, y=0.985)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Grid saved -> %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--deltas", type=float, nargs="+", default=None)
    parser.add_argument("--n-tiles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit checkpoint filename (e.g. epoch=272-step=414000.ckpt) instead of auto-selected best")
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--col-seed", type=str, action="append", default=[],
                        help="Override noise seed for one column, format COL=SEED (1-indexed), e.g. --col-seed 4=777")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import yaml
    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)["gene_manipulation"]

    def _resolve(value: str) -> str:
        normalized = value[2:] if value.startswith("./") else value
        if normalized.startswith(("data/", "dataframes/", "experiments/")):
            return str((Path(__file__).resolve().parents[3] / normalized))
        return value

    for key in ("run_dir", "genomic_h5_dir", "gene_list_path", "output_path"):
        base_cfg[key] = _resolve(base_cfg[key])

    manipulation = base_cfg["manipulations"][0]
    cfg = {
        "run_dir": base_cfg["run_dir"],
        "genomic_h5_dir": base_cfg["genomic_h5_dir"],
        "gene_list_path": base_cfg["gene_list_path"],
        "patient_id": base_cfg["patient_id"],
        "gene": manipulation["gene"],
        "deltas": args.deltas or [0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.56, -8.0],
        "n_tiles": args.n_tiles or base_cfg.get("n_tiles", 10),
        "guidance_scale": args.guidance_scale if args.guidance_scale is not None else base_cfg.get("guidance_scale", 5.0),
        "steps": base_cfg.get("steps", 20),
        "seed": args.seed if args.seed is not None else base_cfg.get("seed", 42),
        "checkpoint": args.checkpoint,
        "col_seed_overrides": {
            int(k): int(v) for k, v in (pair.split("=", 1) for pair in args.col_seed)
        },
        "title": args.title,
        "output_path": args.output or str(
            Path(base_cfg["output_path"]).with_name(f"manipulation_grid_{manipulation['gene']}.png")
        ),
    }

    run_grid(cfg)


if __name__ == "__main__":
    main()
