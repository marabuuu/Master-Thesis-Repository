#!/usr/bin/env python3
"""
GDA v13 CFG Amplification Experiment.

Generates histology tiles from noise at increasing guidance scales to test
whether the genomic adapter produces subtype-differentiated morphology when
the guidance signal is amplified at inference.

CFG formula (GDA-specific):
    ε_guided = (ε_backbone + Δε_null) + scale × (Δε_own − Δε_null)

The unconditional base is (ε_backbone + Δε_null), not ε_backbone alone.
In practice the backbone learned near-zero outputs (the adapter absorbed the full
denoising signal), so Δε_null ≈ actual noise prediction and must not be omitted.

The backbone always receives cond=zeros (trained that way).  Only the adapter
delta is amplified — scale=1 means the adapter runs normally, scale=50 means
the genomic residual is 50× amplified.

For each Basal / LumA test patient:
  1. Draw N Gaussian noise seeds (reused across scales — only conditioning differs).
  2. Generate tiles at each --scale.
  3. Save image contact-sheet (rows = patients, columns = scales).
  4. Extract Virchow2 features + apply linear classifier → AUROC per scale.
  5. Write results JSON + bar-chart.

Usage:
    python -m src.evaluation.gda_cfg_eval \\
        --config src/config.yaml \\
        --checkpoint experiments/20260517_gda_v13/gda/autoenc/last-v3.ckpt \\
        --output experiments/20260525_gda_cfg_eval \\
        --scales 1 5 10 30 50 \\
        --n-tiles 8 --max-patients 4 --save-tiles
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import joblib
import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision.utils import make_grid

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# MoPaDi path setup
# ---------------------------------------------------------------------------

def _add_mopadi_to_path() -> None:
    mopadi_src = (_REPO_ROOT.parent / "mopadi" / "src").resolve()
    if mopadi_src.exists() and str(mopadi_src) not in sys.path:
        sys.path.insert(0, str(mopadi_src))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_gda_v13(ckpt_path: str, gda_cfg: dict, device: torch.device):
    """Load GDA v13 EMA modules from a Lightning checkpoint.

    Returns (backbone, adapter, genomic_encoder, null_token, sampler, conf).
    All modules are in eval mode on *device*.
    """
    _add_mopadi_to_path()

    from src.genomic_adapter.run_training import _build_config
    from src.genomic_adapter.model import GDALitModel

    conf = _build_config(gda_cfg)

    # Instantiate the model structure (no data loading needed)
    model = GDALitModel(conf)

    log.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        log.warning("Missing keys: %d (first 5: %s)", len(missing), missing[:5])
    if unexpected:
        log.warning("Unexpected keys: %d", len(unexpected))

    # EMA modules — these are what we use at inference
    backbone = model.ema_model.to(device).eval()
    adapter = model._ema_adapter.to(device).eval()
    enc = model._ema_genomic_encoder.to(device).eval()
    null_tok = model._ema_null_token.to(device)  # (n_tokens, token_dim)

    n_backbone = sum(p.numel() for p in backbone.parameters()) // 1_000_000
    n_adapter = sum(p.numel() for p in adapter.parameters()) // 1_000_000
    log.info("Loaded: backbone=%dM  adapter=%dM  encoder params=%d",
             n_backbone, n_adapter, sum(p.numel() for p in enc.parameters()))

    sampler = conf._make_diffusion_conf(conf.T_eval).make_sampler()

    _diagnose_loaded_modules(backbone, adapter, enc, conf, device)

    return backbone, adapter, enc, null_tok, sampler, conf


@torch.no_grad()
def _diagnose_loaded_modules(backbone, adapter, enc, conf, device) -> None:
    """Print per-component health stats immediately after loading.

    Checks:
    1. Backbone final-conv weight magnitude (near-zero → zero_module init, not trained).
    2. Backbone eps norm for t=0/500/999 with fixed dummy noise — if all three are
       identical the backbone is ignoring the timestep (collapsed or untrained).
    3. Adapter guidance_delta = ||Δε_own − Δε_null||² for the same timesteps.
       Near-zero means the adapter isn't using its genomic tokens.
    """
    log.info("── Model diagnostics ──────────────────────────────────────────")

    # 1. Final-conv weight magnitude (backbone)
    out_conv = backbone.out[-1]  # zero_module conv in BeatGANsUNetModel
    w_mag = out_conv.weight.abs().mean().item()
    log.info("backbone.out[-1].weight |mean| = %.6f  (near-zero → not trained)", w_mag)

    feat_dim = conf.feat_dim
    img_size = conf.img_size
    B = 1
    x_dummy = torch.randn(B, 3, img_size, img_size, device=device)
    zeros_cond = torch.zeros(B, feat_dim, device=device)

    # Dummy genomic feats (random, L2-normalised by encoder)
    g_raw = torch.randn(B, feat_dim, device=device)
    g_tokens = enc(g_raw)                                        # (1, n, d)
    null_tok = torch.zeros(
        1, conf.adapter_n_tokens, conf.adapter_token_dim, device=device
    )

    # 2. Backbone eps vs timestep
    eps_norms = {}
    for t_val in (0, 500, 999):
        t_tensor = torch.tensor([t_val], device=device, dtype=torch.long)
        eps = backbone.forward(x=x_dummy, t=t_tensor, x_start=None, cond=zeros_cond).pred
        eps_norms[t_val] = eps.norm().item()
    log.info("backbone eps |norm|  t=0: %.4f  t=500: %.4f  t=999: %.4f",
             eps_norms[0], eps_norms[500], eps_norms[999])

    identical = abs(eps_norms[0] - eps_norms[500]) < 1e-5 and \
                abs(eps_norms[0] - eps_norms[999]) < 1e-5
    if identical:
        log.warning("  ⚠ backbone norms are IDENTICAL — backbone ignores t "
                    "(likely collapsed to near-zero; adapter must carry everything)")
    elif w_mag < 1e-5:
        log.warning("  ⚠ backbone final-conv weights are near-zero "
                    "(ema_model not trained or not loaded from checkpoint)")

    # 3. Adapter guidance delta vs timestep
    for t_val in (0, 250, 500, 750, 999):
        t_tensor = torch.tensor([t_val], device=device, dtype=torch.long)
        xt_dummy = torch.randn(B, 3, img_size, img_size, device=device)
        d_own  = adapter(xt_dummy, t_tensor, g_tokens)
        d_null = adapter(xt_dummy, t_tensor, null_tok)
        delta = (d_own - d_null).pow(2).mean().item()
        log.info("  adapter guidance_delta  t=%-4d  Δε²=%.6f", t_val, delta)

    log.info("───────────────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# GDA CFG sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_gda(
    backbone: torch.nn.Module,
    adapter: torch.nn.Module,
    enc: torch.nn.Module,
    null_tok: torch.Tensor,
    sampler,
    genomic_feats: torch.Tensor,   # (1, feat_dim)
    guidance_scale: float,
    noise: torch.Tensor,            # (1, 3, H, W) — reused across scales
    feat_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Run one GDA CFG denoising pass.  Returns (1, 3, H, W) in [-1, 1]."""
    B = noise.shape[0]
    genomic_feats = genomic_feats.to(device, dtype=torch.float32)
    g_tokens = enc(genomic_feats)                                  # (1, n, d)
    null_exp = null_tok.unsqueeze(0).expand(B, -1, -1)            # (1, n, d)
    zeros_cond = torch.zeros(B, feat_dim, device=device, dtype=torch.float32)

    class _Model(torch.nn.Module):
        def __init__(self_):
            super().__init__()
            # Register backbone so sampler can resolve device via model.parameters()
            self_._backbone = backbone
            self_._adapter = adapter

        def forward(self_, x, t, **kw):
            from mopadi.diffusion.base import DummyReturn
            t_sc = sampler._scale_timesteps(t)
            eps_base = backbone.forward(x=x, t=t_sc, x_start=None, cond=zeros_cond).pred
            d_own  = adapter(x, t, g_tokens)
            d_null = adapter(x, t, null_exp)
            # Correct CFG: unconditional base = eps_base + d_null
            eps_uncond = eps_base + d_null
            eps_cond   = eps_base + d_own
            return DummyReturn(pred=eps_uncond + guidance_scale * (eps_cond - eps_uncond))

    out = sampler.sample(
        model=_Model(),
        shape=noise.shape,
        noise=noise,
        model_kwargs={},  # bypass model_type=None crash in sampler.sample()
        progress=False,
    )
    return out  # (1, 3, H, W)


# ---------------------------------------------------------------------------
# Data helpers (shared with cfg_tile_eval.py)
# ---------------------------------------------------------------------------

def _load_genomic_feats(genomic_h5_dir: Path, patient_id: str) -> Optional[np.ndarray]:
    h5_path = genomic_h5_dir / f"{patient_id}.h5"
    if not h5_path.exists():
        return None
    with h5py.File(h5_path) as f:
        return f["feats"][()].astype(np.float32)


def _iter_test_tiles(
    zip_dir: Path,
    patient_id: str,
    n_tiles: int,
    img_size: int,
    rng: np.random.RandomState,
) -> List[Tuple[torch.Tensor, str]]:
    import io, zipfile
    from torchvision import transforms

    zip_path = next(zip_dir.glob(f"{patient_id}*.zip"), None)
    if zip_path is None:
        zip_path = next(zip_dir.glob(f"**/{patient_id}*.zip"), None)
    if zip_path is None:
        log.warning("No zip found for %s", patient_id)
        return []

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    tiles = []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
            chosen = rng.choice(names, size=min(n_tiles, len(names)), replace=False)
            for name in chosen:
                with zf.open(name) as fh:
                    img = Image.open(io.BytesIO(fh.read())).convert("RGB")
                tiles.append((tf(img).unsqueeze(0), Path(name).stem))
    except Exception as exc:
        log.warning("Error reading %s: %s", zip_path, exc)
    return tiles


# ---------------------------------------------------------------------------
# Virchow2 + classification (optional)
# ---------------------------------------------------------------------------

def _make_virchow2_extractor(device: torch.device):
    from src.classifier.extract_virchow2_features import _Virchow2Extractor
    return _Virchow2Extractor(device=str(device))


def extract_virchow2_features(images: List[torch.Tensor], extractor) -> np.ndarray:
    pil_imgs = []
    for t in images:
        arr = t.squeeze(0).cpu()
        arr = ((arr + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        pil_imgs.append(Image.fromarray(arr.permute(1, 2, 0).numpy()))
    all_feats = []
    for i in range(0, len(pil_imgs), 16):
        all_feats.append(extractor.encode_batch(pil_imgs[i:i + 16]))
    return np.concatenate(all_feats) if all_feats else np.zeros((0, 1280), np.float32)


def _classify(feats: np.ndarray, clf_bundle: dict) -> Tuple[np.ndarray, np.ndarray]:
    fs = clf_bundle["scaler"].transform(feats)
    proba = clf_bundle["classifier"].predict_proba(fs)[:, 1]
    labels = (proba >= clf_bundle["threshold"]).astype(int)
    return labels, proba


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).cpu()
    arr = ((arr + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(arr.permute(1, 2, 0).numpy())


def _save_contact_sheet(
    tiles_per_scale: Dict[str, List[torch.Tensor]],
    path: Path,
    nrow_per_scale: int = 4,
) -> None:
    """One block of tiles per scale, stacked vertically."""
    import torchvision.transforms.functional as TF

    rows = []
    for scale_name, tensors in tiles_per_scale.items():
        if not tensors:
            continue
        grid = make_grid(
            [t.squeeze(0) for t in tensors[:nrow_per_scale]],
            nrow=nrow_per_scale, normalize=True, value_range=(-1, 1), padding=4,
        )
        rows.append(grid)

    if not rows:
        return

    max_w = max(r.shape[2] for r in rows)
    padded = [
        torch.nn.functional.pad(r, (0, max_w - r.shape[2], 0, 0))
        for r in rows
    ]
    full = torch.cat(padded, dim=1)  # (3, H_total, W)
    img = Image.fromarray(
        (full.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    log.info("Saved contact sheet → %s", path)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    gda_ckpt: str,
    gda_cfg: dict,
    splits_path: str,
    genomic_h5_dir: str,
    zip_dir: str,
    clf_path: Optional[str],
    output_dir: str,
    scales: Tuple[float, ...] = (1.0, 5.0, 10.0, 30.0, 50.0),
    subtypes: Tuple[str, ...] = ("Basal", "LumA"),
    n_tiles_per_patient: int = 8,
    max_patients_per_subtype: Optional[int] = 4,
    device_str: str = "cuda",
    seed: int = 42,
    save_tiles: bool = False,
    skip_virchow: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(seed)

    log.info("Output: %s  Device: %s  Scales: %s", output_dir, device, scales)

    # ── Load model ──────────────────────────────────────────────────────────
    backbone, adapter, enc, null_tok, sampler, conf = load_gda_v13(
        gda_ckpt, gda_cfg, device
    )
    img_size = conf.img_size
    feat_dim = conf.feat_dim

    # ── Classifier (optional) ───────────────────────────────────────────────
    clf_bundle = None
    if clf_path and not skip_virchow:
        clf_bundle = joblib.load(clf_path)
        log.info("Classifier loaded. label_map=%s", clf_bundle["label_mapping"])
        log.info("Loading Virchow2 extractor…")
        v2_extractor = _make_virchow2_extractor(device)

    # ── Patient lists ────────────────────────────────────────────────────────
    with open(splits_path) as fh:
        splits = json.load(fh)

    def _pick(subtype: str) -> Dict:
        pats = {pid: m for pid, m in splits["test"].items()
                if isinstance(m, dict) and m.get("subtype") == subtype}
        if max_patients_per_subtype:
            chosen = rng.choice(sorted(pats), size=min(max_patients_per_subtype, len(pats)), replace=False)
            return {pid: pats[pid] for pid in chosen}
        return pats

    test_patients = {}
    for st in subtypes:
        test_patients.update(_pick(st))
    log.info("Test patients: %d", len(test_patients))

    genomic_h5_dir = Path(genomic_h5_dir)
    zip_dir_p = Path(zip_dir)

    # Condition names for each scale
    cond_names = [f"own_s{s:.0f}" for s in scales]

    # Accumulators for AUROC (only if classifier available)
    results: Dict[str, Dict] = {c: {"labels": [], "probas": []} for c in cond_names}
    meta_rows: List[Dict] = []

    sheets_dir = output_dir / "contact_sheets"
    tiles_dir  = output_dir / "tiles"
    feats_dir  = output_dir / "features"
    sheets_dir.mkdir(exist_ok=True)
    feats_dir.mkdir(exist_ok=True)
    if save_tiles:
        tiles_dir.mkdir(exist_ok=True)

    for pid, meta in test_patients.items():
        subtype = meta["subtype"]
        true_label = clf_bundle["label_mapping"][subtype] if clf_bundle else -1

        feats = _load_genomic_feats(genomic_h5_dir, pid)
        if feats is None:
            log.warning("No genomic feats for %s, skipping.", pid)
            continue

        tile_list = _iter_test_tiles(zip_dir_p, pid, n_tiles_per_patient, img_size, rng)
        if not tile_list:
            log.warning("No tiles for %s, skipping.", pid)
            continue

        log.info("Patient %s (%s): %d tiles", pid, subtype, len(tile_list))

        feats_t = torch.tensor(feats).unsqueeze(0).to(device)

        # Per-patient accumulators for contact sheet (one per tile × scale)
        tiles_per_scale: Dict[str, List[torch.Tensor]] = {c: [] for c in cond_names}
        pid_virchow: Dict[str, List[np.ndarray]] = {c: [] for c in cond_names}
        pid_tile_names: List[str] = []

        for tile_idx, (_, tile_name) in enumerate(tile_list):
            pid_tile_names.append(tile_name)

            # Fixed noise — reused across all scales so differences = conditioning only
            tile_seed = int(rng.randint(0, 2**31))
            g = torch.Generator(device=device)
            g.manual_seed(tile_seed)
            noise = torch.randn(1, 3, img_size, img_size, device=device, generator=g)

            for cond_name, scale in zip(cond_names, scales):
                recon = generate_gda(
                    backbone, adapter, enc, null_tok, sampler,
                    feats_t, scale, noise, feat_dim, device,
                )  # (1, 3, H, W)

                tiles_per_scale[cond_name].append(recon.cpu())

                if clf_bundle is not None:
                    v2 = extract_virchow2_features([recon.cpu()], v2_extractor)
                    _, p_basal = _classify(v2, clf_bundle)
                    results[cond_name]["labels"].append(true_label)
                    results[cond_name]["probas"].append(float(p_basal[0]))
                    pid_virchow[cond_name].append(v2[0])

                    meta_rows.append({
                        "patient_id": pid, "subtype": subtype,
                        "true_label": true_label, "tile_name": tile_name,
                        "tile_idx": tile_idx, "condition": cond_name,
                        "scale": scale, "p_basal": float(p_basal[0]),
                    })

                if save_tiles:
                    p = tiles_dir / cond_name / f"{pid}_{tile_name}.jpg"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    _tensor_to_pil(recon).save(p, quality=92)

        # Per-patient contact sheet: rows = scales, cols = tiles
        _save_contact_sheet(tiles_per_scale, sheets_dir / f"{pid}_{subtype}.png",
                            nrow_per_scale=len(tile_list))

        # Per-patient h5 feature dump
        if clf_bundle is not None:
            h5_path = feats_dir / f"{pid}.h5"
            with h5py.File(h5_path, "w") as hf:
                hf.attrs["subtype"] = subtype
                hf.attrs["true_label"] = true_label
                dt = h5py.string_dtype()
                hf.create_dataset("tile_names",
                                  data=np.array(pid_tile_names, dtype=object), dtype=dt)
                for cond_name, flist in pid_virchow.items():
                    if flist:
                        hf.create_dataset(cond_name, data=np.stack(flist), compression="gzip")

    # ── Metrics + plots ──────────────────────────────────────────────────────
    if clf_bundle is not None and meta_rows:
        meta_csv = output_dir / "gda_cfg_eval_metadata.csv"
        with open(meta_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(meta_rows[0].keys()))
            writer.writeheader()
            writer.writerows(meta_rows)
        log.info("Saved metadata CSV → %s", meta_csv)

        metrics = {}
        for cond_name in cond_names:
            labels = results[cond_name]["labels"]
            probas = results[cond_name]["probas"]
            if len(set(labels)) < 2 or not labels:
                log.warning("%s: not enough data for AUROC", cond_name)
                continue
            auroc = float(roc_auc_score(labels, probas))
            preds = [int(p >= clf_bundle["threshold"]) for p in probas]
            acc   = float(np.mean([p == l for p, l in zip(preds, labels)]))
            metrics[cond_name] = {"auroc": auroc, "accuracy": acc, "n_tiles": len(labels)}
            log.info("%-12s  AUROC=%.3f  ACC=%.3f  n=%d", cond_name, auroc, acc, len(labels))

        with open(output_dir / "gda_cfg_eval_metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)

        _plot_results(metrics, output_dir, scales)

    log.info("Done. Results in %s", output_dir)


def _plot_results(metrics: dict, output_dir: Path, scales: Tuple[float, ...]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds  = list(metrics.keys())
    aurocs = [metrics[c]["auroc"] for c in conds]
    accs   = [metrics[c]["accuracy"] for c in conds]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(conds))

    axes[0].bar(x, aurocs, color="steelblue", alpha=0.85)
    axes[0].axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="random")
    axes[0].set_xticks(x); axes[0].set_xticklabels(conds, rotation=30, ha="right")
    axes[0].set_ylabel("AUROC (Basal vs LumA)"); axes[0].set_ylim(0, 1)
    axes[0].set_title("GDA v13 — AUROC by guidance scale"); axes[0].legend()

    axes[1].bar(x, accs, color="tomato", alpha=0.85)
    axes[1].axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="chance")
    axes[1].set_xticks(x); axes[1].set_xticklabels(conds, rotation=30, ha="right")
    axes[1].set_ylabel("Accuracy"); axes[1].set_ylim(0, 1)
    axes[1].set_title("GDA v13 — Accuracy by guidance scale"); axes[1].legend()

    fig.tight_layout()
    out_path = output_dir / "gda_cfg_eval_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved plot → %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config",          required=True, help="Path to src/config.yaml")
    parser.add_argument("--config-section",  default="genomic_adapter_training",
                        help="Top-level config.yaml section to use "
                             "(e.g. poc_breast_vs_liver_gda for the PoC run)")
    parser.add_argument("--checkpoint",      default=None,
                        help="GDA checkpoint path (default: last.ckpt in experiment dir)")
    parser.add_argument("--output",          required=True, help="Output directory")
    parser.add_argument("--scales",          type=float, nargs="+", default=[1.0, 5.0, 10.0, 30.0, 50.0])
    parser.add_argument("--subtypes",        nargs="+", default=["Basal", "LumA"],
                        help="Subtype labels to evaluate, e.g. --subtypes TCGA-BRCA TCGA-LIHC")
    parser.add_argument("--n-tiles",         type=int, default=8)
    parser.add_argument("--max-patients",    type=int, default=4,
                        help="Max patients per subtype")
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--save-tiles",      action="store_true")
    parser.add_argument("--skip-virchow",    action="store_true",
                        help="Skip Virchow2 extraction (visual inspection only)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = Path(args.config).resolve()
    repo_root = _REPO_ROOT

    with open(config_path) as fh:
        full_cfg = yaml.safe_load(fh)

    gda_cfg = full_cfg[args.config_section]

    # Resolve relative paths in gda_cfg
    exp_root = (repo_root.parent / "experiments").resolve()
    data_root = (repo_root.parent / "data").resolve()

    def _resolve(raw: str) -> str:
        p = Path(raw)
        if p.is_absolute():
            return str(p)
        s = raw[2:] if raw.startswith("./") else raw
        if s.startswith("experiments/"):
            return str((repo_root.parent / s).resolve())
        if s.startswith("../data/") or s.startswith("../") or s.startswith("data/"):
            return str((repo_root.parent / s.lstrip("../")).resolve())
        return str((repo_root / s).resolve())

    for k, v in list(gda_cfg.items()):
        if isinstance(v, str) and ("/" in v or v.startswith(".")):
            gda_cfg[k] = _resolve(v)

    # Checkpoint: use latest if not specified.
    # WARNING: do NOT blindly use last.ckpt — check TensorBoard for loss/val
    # divergence before using a checkpoint.  A healthy checkpoint has
    # loss/val ≈ stable/decreasing; a diverged run can have loss/val jump
    # by 10× in the final steps (e.g. v10: 0.035 → 0.516).
    if args.checkpoint:
        ckpt_path = str(Path(args.checkpoint).resolve())
    else:
        base_dir_key = args.config_section
        exp_dir = Path(_resolve(full_cfg[base_dir_key]["base_dir"]))
        ckpt_dir = exp_dir / "gda" / "autoenc"
        # Prefer best-val checkpoint over last.ckpt (avoids using diverged runs)
        candidates = sorted(
            [p for p in ckpt_dir.glob("*.ckpt") if "last" not in p.name],
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            candidates = sorted(ckpt_dir.glob("last*.ckpt"),
                                key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")
        ckpt_path = str(candidates[-1])
        log.info("Auto-selected checkpoint (most recent non-last): %s", ckpt_path)
        log.info("Auto-selected checkpoint: %s", ckpt_path)

    clf_path = "/mnt/bulk-saturn/maralampert/genhist/experiments/20260326_cross_subtype_classifier/train/subtype_linear_model.joblib"

    run_evaluation(
        gda_ckpt=ckpt_path,
        gda_cfg=gda_cfg,
        splits_path=gda_cfg["patient_splits_path"],
        genomic_h5_dir=gda_cfg["genomic_feature_dir"],
        zip_dir=gda_cfg["zip_dir"],
        clf_path=clf_path,
        output_dir=args.output,
        scales=tuple(args.scales),
        subtypes=tuple(args.subtypes),
        n_tiles_per_patient=args.n_tiles,
        max_patients_per_subtype=args.max_patients,
        device_str=args.device,
        seed=args.seed,
        save_tiles=args.save_tiles,
        skip_virchow=args.skip_virchow,
    )


if __name__ == "__main__":
    main()
