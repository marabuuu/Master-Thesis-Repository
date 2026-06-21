"""Generate a paired manipulation panel: original vs. gene-perturbed conditioning.

Loads a single patient's conditioning vector, shifts one or more gene dimensions
by delta standard deviations, and runs DDIM sampling with the SAME noise seed for
both conditions so that any visual difference is attributable solely to the
conditioning change. Saves one panel PNG (top row = original, bottom row = perturbed).
No intermediate tile files are written.

Usage:
    python -m src.reconstruction.manipulate_tiles --config src/config.yaml
    python -m src.reconstruction.manipulate_tiles \\
        --run-dir experiments/20260607_brca_pam50_cfg_v2_256/gda \\
        --h5-dir experiments/20260528_genomic_features/genomic_h5 \\
        --gene-list experiments/20260528_genomic_features/gene_list.txt \\
        --patient-id TCGA-E9-A1NF --gene POSTN --delta -3.0 \\
        --output experiments/20260607_brca_pam50_cfg_v2_256/panel_POSTN.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

log = logging.getLogger(__name__)


# ── Checkpoint / model loading ────────────────────────────────────────────────

from src.model_training.checkpoint import (
    load_config_from_run as _load_config,
    resolve_checkpoint as _resolve_checkpoint,
    load_model as _load_model_raw,
)


def _load_model(run_dir: Path, ckpt_path: Path):
    """Load CfgBackboneLitModel from a run directory and checkpoint."""
    conf = _load_config(run_dir)
    return _load_model_raw(conf, ckpt_path)


# ── Feature helpers ───────────────────────────────────────────────────────────

def load_gene_index(gene_list_path: Path) -> dict[str, int]:
    genes = [l.strip() for l in gene_list_path.read_text().splitlines() if l.strip()]
    return {g: i for i, g in enumerate(genes)}


def load_feats(patient_id: str, h5_dir: Path) -> torch.Tensor:
    for suffix in ("", "-DX1"):
        p = h5_dir / f"{patient_id}{suffix}.h5"
        if p.exists():
            with h5py.File(p, "r") as f:
                return torch.from_numpy(np.asarray(f["feats"][:], dtype=np.float32).squeeze())
    raise FileNotFoundError(f"No H5 file for {patient_id} in {h5_dir}")


def perturb_feats(feats: torch.Tensor, perturbations: list[tuple[int, float]]) -> torch.Tensor:
    """Apply one or more (gene_index, delta_std) shifts to the feature vector."""
    out = feats.clone()
    for idx, delta in perturbations:
        out[idx] = out[idx] + delta
    return out


# ── Sampling ─────────────────────────────────────────────────────────────────

def sample_tile(
    model,
    feats: torch.Tensor,
    noise: torch.Tensor,
    guidance_scale: float,
    n_steps: int,
    device: torch.device,
) -> torch.Tensor:
    import torch.nn as nn
    from mopadi.diffusion.base import DummyReturn

    backbone = model.ema_model
    sampler = model.conf._make_diffusion_conf(n_steps).make_sampler()
    normalize = getattr(model.conf, "normalize_feats", False)
    zeros = torch.zeros(1, model.conf.feat_dim, device=device, dtype=torch.float32)
    f = feats.to(device=device, dtype=torch.float32).view(1, -1)
    if normalize:
        f = F.normalize(f, p=2, dim=-1)

    class _CFG(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._bb = backbone

        def forward(self, x, t, **kw):
            t_s = sampler._scale_timesteps(t)
            eps_null = self._bb.forward(x=x, t=t_s, x_start=None, cond=zeros).pred
            eps_cond = self._bb.forward(x=x, t=t_s, x_start=None, cond=f).pred
            return DummyReturn(pred=eps_null + guidance_scale * (eps_cond - eps_null))

    return sampler.sample(
        model=_CFG(), shape=noise.shape,
        noise=noise.to(device), model_kwargs={}, progress=False,
    )


def _tensor_to_rgb(x: torch.Tensor) -> np.ndarray:
    if x.ndim == 4:
        x = x[0]
    x = ((x.cpu().float() + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
    return x.permute(1, 2, 0).numpy()


# ── Panel ─────────────────────────────────────────────────────────────────────

def save_panel(
    original_tiles: list[np.ndarray],
    perturbed_tiles: list[np.ndarray],
    title: str,
    row_labels: tuple[str, str],
    output_path: Path,
    metrics: dict[str, Any] | None = None,
) -> None:
    n = len(original_tiles)
    h, w = original_tiles[0].shape[:2]
    dpi = 150
    fig_w = n * w / dpi + 0.6
    fig_h = 2 * h / dpi + 0.9
    if metrics:
        fig_h += 0.35

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    top = 0.88 if not metrics else 0.85
    bottom = 0.01 if not metrics else 0.06
    gs = gridspec.GridSpec(2, n, figure=fig,
                           hspace=0.04, wspace=0.02,
                           left=0.05, right=0.99,
                           top=top, bottom=bottom)

    for col in range(n):
        for row, tiles in enumerate([original_tiles, perturbed_tiles]):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(tiles[col])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    label_kw = dict(va="center", ha="center", fontsize=9, rotation=90,
                    transform=fig.transFigure)
    row_center_top = (top + (top + bottom) / 2) / 2
    row_center_bot = (bottom + (top + bottom) / 2) / 2
    fig.text(0.022, row_center_top, row_labels[0], color="#333333", **label_kw)
    fig.text(0.022, row_center_bot, row_labels[1], color="#aa2222", **label_kw)

    fig.suptitle(title, fontsize=10, y=0.97)

    if metrics:
        parts = []
        if "ms_ssim_mean" in metrics:
            parts.append(f"MS-SSIM: {metrics['ms_ssim_mean']:.4f}")
        for model_name in sorted(k.replace("_cosine_sim_mean", "")
                                  for k in metrics if k.endswith("_cosine_sim_mean")):
            val = metrics[f"{model_name}_cosine_sim_mean"]
            parts.append(f"cos-sim ({model_name}): {val:.4f}")
        if parts:
            fig.text(0.5, 0.015, "    ".join(parts),
                     ha="center", va="bottom", fontsize=8, color="#555555",
                     transform=fig.transFigure)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Panel saved → %s", output_path)


# ── Metrics ──────────────────────────────────────────────────────────────────

def _compute_ms_ssim(
    original_tiles: list[np.ndarray],
    perturbed_tiles: list[np.ndarray],
) -> dict[str, Any]:
    """Compute MS-SSIM between each original/perturbed tile pair."""
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

    msssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=255.0)
    per_tile: list[float] = []
    for orig, pert in zip(original_tiles, perturbed_tiles):
        t_orig = torch.from_numpy(orig).permute(2, 0, 1).unsqueeze(0).float()
        t_pert = torch.from_numpy(pert).permute(2, 0, 1).unsqueeze(0).float()
        val = msssim(t_pert, t_orig).item()
        per_tile.append(val)
        msssim.reset()
    return {
        "ms_ssim_per_tile": per_tile,
        "ms_ssim_mean": float(np.mean(per_tile)),
        "ms_ssim_std": float(np.std(per_tile)),
    }


def _compute_feature_cosine_similarity(
    original_tiles: list[np.ndarray],
    perturbed_tiles: list[np.ndarray],
    model_name: str,
    device: torch.device,
) -> dict[str, Any]:
    """Extract features with a ViT encoder and compute cosine similarity per pair."""
    from PIL import Image

    if model_name == "virchow2":
        from src.classifier.extract_virchow2_features import _Virchow2Extractor
        extractor = _Virchow2Extractor(device=str(device))
        encode_fn = extractor.encode_batch
    else:
        raise ValueError(f"Unsupported feature model: {model_name}")

    per_tile: list[float] = []
    for orig, pert in zip(original_tiles, perturbed_tiles):
        img_orig = Image.fromarray(orig)
        img_pert = Image.fromarray(pert)
        feat_orig = encode_fn([img_orig])  # (1, D)
        feat_pert = encode_fn([img_pert])  # (1, D)
        cos = float(F.cosine_similarity(
            torch.from_numpy(feat_orig),
            torch.from_numpy(feat_pert),
        ).item())
        per_tile.append(cos)

    return {
        f"{model_name}_cosine_sim_per_tile": per_tile,
        f"{model_name}_cosine_sim_mean": float(np.mean(per_tile)),
        f"{model_name}_cosine_sim_std": float(np.std(per_tile)),
    }


def compute_manipulation_metrics(
    original_tiles: list[np.ndarray],
    perturbed_tiles: list[np.ndarray],
    feature_models: list[str] | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Compute MS-SSIM and feature cosine similarity for tile pairs."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("Computing MS-SSIM …")
    metrics = _compute_ms_ssim(original_tiles, perturbed_tiles)
    log.info("  MS-SSIM mean: %.4f (±%.4f)", metrics["ms_ssim_mean"], metrics["ms_ssim_std"])

    for model_name in (feature_models or []):
        log.info("Computing %s cosine similarity …", model_name)
        feat_metrics = _compute_feature_cosine_similarity(
            original_tiles, perturbed_tiles, model_name, device,
        )
        metrics.update(feat_metrics)
        key = f"{model_name}_cosine_sim_mean"
        log.info("  %s: %.4f (±%.4f)", key, feat_metrics[key],
                 feat_metrics[f"{model_name}_cosine_sim_std"])

    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────

def run_manipulation(cfg: dict[str, Any]) -> None:
    from src.reconstruction.utils import _ensure_mopadi_import_path
    _ensure_mopadi_import_path()

    run_dir = Path(cfg["run_dir"])
    h5_dir = Path(cfg["genomic_h5_dir"])
    gene_list_path = Path(cfg["gene_list_path"])
    output_path = Path(cfg["output_path"])

    patient_id = cfg["patient_id"]
    n_tiles = int(cfg.get("n_tiles", 10))
    guidance_scale = float(cfg.get("guidance_scale", 5.0))
    n_steps = int(cfg.get("steps", 20))
    seed = int(cfg.get("seed", 42))
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Each entry: {gene: str, delta_std: float}
    manipulations: list[dict] = cfg.get("manipulations", [])
    if not manipulations:
        raise ValueError("No manipulations defined in config")

    gene_index = load_gene_index(gene_list_path)
    perturbations: list[tuple[int, float]] = []
    for m in manipulations:
        gene = m["gene"]
        if gene not in gene_index:
            raise ValueError(f"Gene '{gene}' not found in gene list")
        perturbations.append((gene_index[gene], float(m["delta_std"])))

    log.info("Loading checkpoint …")
    ckpt_path = _resolve_checkpoint(run_dir)
    model = _load_model(run_dir, ckpt_path).to(device)

    feats_orig = load_feats(patient_id, h5_dir)
    feats_pert = perturb_feats(feats_orig, perturbations)

    # Log original vs perturbed values for each gene
    for (idx, delta), m in zip(perturbations, manipulations):
        log.info("  %s [%d]: %.3f → %.3f (Δ=%.1f)",
                 m["gene"], idx, feats_orig[idx].item(),
                 feats_pert[idx].item(), delta)

    original_tiles: list[np.ndarray] = []
    perturbed_tiles: list[np.ndarray] = []

    log.info("Sampling %d tile pairs …", n_tiles)
    img_size = model.conf.img_size
    with torch.no_grad():
        for i in range(n_tiles):
            g = torch.Generator()
            g.manual_seed(seed * 10000 + i)
            noise = torch.randn(1, 3, img_size, img_size, generator=g)
            original_tiles.append(_tensor_to_rgb(
                sample_tile(model, feats_orig, noise, guidance_scale, n_steps, device)))
            perturbed_tiles.append(_tensor_to_rgb(
                sample_tile(model, feats_pert, noise, guidance_scale, n_steps, device)))

    # Compute metrics if requested
    metrics = None
    if cfg.get("compute_metrics", False):
        feature_models = cfg.get("feature_models", ["virchow2"])
        metrics = compute_manipulation_metrics(
            original_tiles, perturbed_tiles,
            feature_models=feature_models,
            device=device,
        )
        import json
        metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
        metrics_path.write_text(json.dumps(metrics, indent=2))
        log.info("Metrics saved → %s", metrics_path)

    # Build title and row labels
    gene_parts = []
    for m in manipulations:
        sign = "+" if m["delta_std"] >= 0 else ""
        gene_parts.append(f"{m['gene']} {sign}{m['delta_std']:.0f}σ")
    manip_str = ",  ".join(gene_parts)
    title = f"{manip_str}  —  patient {patient_id}"

    perturb_label = "\n".join(
        f"{m['gene']} {'+'if m['delta_std']>=0 else ''}{m['delta_std']:.0f}σ"
        for m in manipulations
    )

    save_panel(
        original_tiles, perturbed_tiles,
        title=title,
        row_labels=("original", perturb_label),
        output_path=output_path,
        metrics=metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str)
    parser.add_argument("--run-dir", type=str)
    parser.add_argument("--h5-dir", type=str)
    parser.add_argument("--gene-list", type=str)
    parser.add_argument("--patient-id", type=str)
    parser.add_argument("--gene", type=str, action="append", dest="genes")
    parser.add_argument("--delta", type=float, action="append", dest="deltas")
    parser.add_argument("--output", type=str, dest="output_path")
    parser.add_argument("--n-tiles", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--compute-metrics", action="store_true",
                        help="Compute MS-SSIM and feature cosine similarity")
    parser.add_argument("--feature-models", type=str, nargs="+", default=["virchow2"],
                        help="Feature models for cosine similarity (default: virchow2)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)["gene_manipulation"]
        if args.compute_metrics:
            cfg["compute_metrics"] = True
            cfg.setdefault("feature_models", args.feature_models)
    else:
        genes = args.genes or []
        deltas = args.deltas or []
        if len(genes) != len(deltas):
            parser.error("Number of --gene and --delta arguments must match")
        cfg = {
            "run_dir": args.run_dir,
            "genomic_h5_dir": args.h5_dir,
            "gene_list_path": args.gene_list,
            "patient_id": args.patient_id,
            "output_path": args.output_path or "manipulation_panel.png",
            "n_tiles": args.n_tiles,
            "guidance_scale": args.guidance_scale,
            "steps": args.steps,
            "seed": args.seed,
            "device": args.device,
            "manipulations": [{"gene": g, "delta_std": d} for g, d in zip(genes, deltas)],
            "compute_metrics": args.compute_metrics,
            "feature_models": args.feature_models,
        }

    run_manipulation(cfg)


if __name__ == "__main__":
    main()
