#!/usr/bin/env python3
"""
CFG Tile Evaluation — v11 genomic conditioning quality test.

For each Basal / LumA test patient:
  1. Load N real tiles from their zip file.
  2. DDIM-invert each tile to noise x_T (preserving content).
  3. Decode with CFG at several scales using:
       (a) own genomic features        → self-reconstruction
       (b) null features (zeros)       → unconditioned baseline
       (c) own features + CFG s=S     → amplified own conditioning
       (d) swapped features + CFG s=S → counterfactual (Basal↔LumA)
  4. Extract Virchow2 (1280-d) features from every generated tile.
  5. Apply the pre-trained Basal-vs-LumA linear classifier.
  6. Report AUROC + accuracy per condition; save JSON + bar plot.

Usage (on the DGX cluster):
    python -m src.evaluation.cfg_tile_eval \\
        --config src/config.yaml \\
        --output experiments/20260515_cfg_eval_v11
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import h5py
import joblib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _add_mopadi_to_path(repo_root: Path) -> None:
    mopadi_src = (repo_root.parent / "mopadi" / "src").resolve()
    if mopadi_src.exists() and str(mopadi_src) not in sys.path:
        sys.path.insert(0, str(mopadi_src))


def load_v11_model(ckpt_path: str, v11_cfg: dict, repo_root: Path, device: torch.device):
    """
    Load the v11 GenomicCaLitModel EMA weights into a bare BeatGANsAutoencModel.

    Returns (ema_model, sampler_50step, conf)
    """
    _add_mopadi_to_path(repo_root)

    from src.cross_attention.genomic_config import GenomicCaConfig
    from src.cross_attention.run_training import _build_ca_config

    conf = _build_ca_config(v11_cfg)
    conf.make_model_conf()

    # Build model shell (identical architecture to what was trained)
    model_conf = conf.make_model_conf()
    ema_model = model_conf.make_model()

    # Load EMA weights from PL checkpoint
    log.info("Loading v11 EMA weights from: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    ema_sd = {k[len("ema_model."):]: v for k, v in sd.items() if k.startswith("ema_model.")}
    missing, unexpected = ema_model.load_state_dict(ema_sd, strict=False)
    if missing:
        log.warning("Missing keys in EMA model: %d", len(missing))
    if unexpected:
        log.warning("Unexpected keys in EMA model: %d", len(unexpected))

    ema_model = ema_model.to(device).eval()
    log.info("EMA model loaded (%d M params).", sum(p.numel() for p in ema_model.parameters()) // 1_000_000)

    # Build fast DDIM sampler (50 steps)
    sampler = conf._make_diffusion_conf(T=50).make_sampler()

    return ema_model, sampler, conf


# ---------------------------------------------------------------------------
# CFG wrapper
# ---------------------------------------------------------------------------

class _CFGWrapper(torch.nn.Module):
    """Classifier-free guidance: ε_guided = ε_null + s·(ε_cond − ε_null)."""

    def __init__(self, model: torch.nn.Module, scale: float) -> None:
        super().__init__()
        self.model = model
        self.scale = scale

    def forward(self, x, t, cond=None, **kwargs):
        out_cond = self.model(x, t, cond=cond, **kwargs)
        out_null = self.model(x, t, cond=torch.zeros_like(cond), **kwargs)
        guided = out_null.pred + self.scale * (out_cond.pred - out_null.pred)
        return SimpleNamespace(pred=guided)


# ---------------------------------------------------------------------------
# DDIM invert + decode
# ---------------------------------------------------------------------------

@torch.no_grad()
def invert_and_decode(
    ema_model: torch.nn.Module,
    sampler,
    img_tensor: torch.Tensor,
    cond: torch.Tensor,
    guidance_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """DDIM invert img → x_T, then decode with optional CFG.

    img_tensor: (1, 3, H, W) in [-1, 1]
    cond:       (1, feat_dim)
    Returns: (1, 3, H, W) in [-1, 1]
    """
    img_tensor = img_tensor.to(device)
    cond = cond.to(device)

    # Inversion (no CFG — faithful content encoding)
    inv_out = sampler.ddim_reverse_sample_loop(
        model=ema_model,
        x=img_tensor,
        clip_denoised=True,
        model_kwargs={"cond": cond},
        eta=0.0,
        device=device,
    )
    x_T = inv_out["sample"]

    # Decode with or without CFG
    decode_model = _CFGWrapper(ema_model, scale=guidance_scale) if guidance_scale > 1.0 else ema_model

    prog = sampler.ddim_sample_loop_progressive(
        model=decode_model,
        shape=img_tensor.shape,
        noise=x_T,
        model_kwargs={"cond": cond},
        device=device,
        progress=False,
        eta=0.0,
    )
    recon = None
    for out in prog:
        recon = out["sample"]
    return recon


# ---------------------------------------------------------------------------
# Data loading helpers
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
) -> List[torch.Tensor]:
    """Load up to n_tiles from a patient's zip file, return list of (1,3,H,W) tensors."""
    import zipfile, io
    from torchvision import transforms

    zip_path = next(zip_dir.glob(f"{patient_id}*.zip"), None)
    if zip_path is None:
        zip_path = next(zip_dir.glob(f"**/{patient_id}*.zip"), None)
    if zip_path is None:
        log.warning("No zip found for %s", patient_id)
        return []

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    tensors = []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
            chosen = rng.choice(names, size=min(n_tiles, len(names)), replace=False)
            for name in chosen:
                with zf.open(name) as fh:
                    img = Image.open(io.BytesIO(fh.read())).convert("RGB")
                img_t = transform(img).unsqueeze(0)  # (1, 3, H, W)
                tensors.append(img_t)
    except Exception as exc:
        log.warning("Error reading %s: %s", zip_path, exc)
    return tensors


# ---------------------------------------------------------------------------
# Virchow2 extraction
# ---------------------------------------------------------------------------

def extract_virchow2_features(
    images: List[torch.Tensor],
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """
    images: list of (1, 3, H, W) tensors in [-1, 1]
    Returns: (N, 1280) float32 array
    """
    from src.classifier.extract_virchow2_features import _Virchow2Extractor
    extractor = _Virchow2Extractor(device=str(device))

    # Convert [-1, 1] tensors → PIL RGB images (Virchow2 expects standard RGB)
    pil_imgs = []
    for t in images:
        arr = t.squeeze(0).cpu()
        arr = ((arr + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        pil_imgs.append(Image.fromarray(arr.permute(1, 2, 0).numpy()))

    all_feats = []
    for i in range(0, len(pil_imgs), batch_size):
        batch = pil_imgs[i: i + batch_size]
        all_feats.append(extractor.encode_batch(batch))
    return np.concatenate(all_feats, axis=0) if all_feats else np.zeros((0, 1280), dtype=np.float32)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(feats: np.ndarray, clf_bundle: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    feats: (N, 1280)
    Returns (labels_pred, proba_basal)
    """
    scaler = clf_bundle["scaler"]
    clf = clf_bundle["classifier"]
    threshold = clf_bundle["threshold"]
    feats_scaled = scaler.transform(feats)
    proba = clf.predict_proba(feats_scaled)[:, 1]  # P(Basal)
    labels = (proba >= threshold).astype(int)
    return labels, proba


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    v11_ckpt: str,
    v11_cfg: dict,
    splits_path: str,
    genomic_h5_dir: str,
    zip_dir: str,
    clf_path: str,
    output_dir: str,
    repo_root: Path,
    n_tiles_per_patient: int = 20,
    cfg_scales: Tuple[float, ...] = (1.0, 3.0, 5.0, 10.0),
    device_str: str = "cuda",
    seed: int = 42,
    max_patients_per_subtype: Optional[int] = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(seed)

    log.info("Output dir: %s", output_dir)
    log.info("Device: %s", device)

    # Load model
    ema_model, sampler, conf = load_v11_model(v11_ckpt, v11_cfg, repo_root, device)

    # Load classifier
    clf_bundle = joblib.load(clf_path)
    label_map = clf_bundle["label_mapping"]  # {'LumA': 0, 'Basal': 1}
    log.info("Classifier loaded. label_map=%s", label_map)

    # Load test split — Basal and LumA only
    with open(splits_path) as fh:
        splits = json.load(fh)

    basal_patients = {pid: meta for pid, meta in splits["test"].items()
                      if isinstance(meta, dict) and meta.get("subtype") == "Basal"}
    luma_patients  = {pid: meta for pid, meta in splits["test"].items()
                      if isinstance(meta, dict) and meta.get("subtype") == "LumA"}

    if max_patients_per_subtype is not None:
        chosen_basal = rng.choice(sorted(basal_patients), size=min(max_patients_per_subtype, len(basal_patients)), replace=False)
        chosen_luma  = rng.choice(sorted(luma_patients),  size=min(max_patients_per_subtype, len(luma_patients)),  replace=False)
        test_patients = {pid: basal_patients[pid] for pid in chosen_basal}
        test_patients.update({pid: luma_patients[pid] for pid in chosen_luma})
    else:
        test_patients = {**basal_patients, **luma_patients}

    log.info("Test patients: %d Basal + %d LumA",
             sum(1 for m in test_patients.values() if m["subtype"] == "Basal"),
             sum(1 for m in test_patients.values() if m["subtype"] == "LumA"))

    genomic_h5_dir = Path(genomic_h5_dir)
    zip_dir = Path(zip_dir)
    img_size = v11_cfg.get("img_size", 512)

    # Build mean genomic vector per subtype (for counterfactual generation)
    basal_feats_list, luma_feats_list = [], []
    for pid, meta in test_patients.items():
        feats = _load_genomic_feats(genomic_h5_dir, pid)
        if feats is None:
            continue
        if meta["subtype"] == "Basal":
            basal_feats_list.append(feats)
        else:
            luma_feats_list.append(feats)

    mean_basal_feats = np.mean(basal_feats_list, axis=0)  # (512,)
    mean_luma_feats  = np.mean(luma_feats_list,  axis=0)

    # Conditions to evaluate:
    # key → (use_own_feats, cfg_scale, swap_to_subtype_or_None)
    conditions = {"null_s1": ("null", 1.0, None)}
    for s in cfg_scales:
        conditions[f"own_s{s:.0f}"] = ("own", s, None)
    for s in cfg_scales:
        if s > 1.0:
            conditions[f"cf_s{s:.0f}"] = ("counterfactual", s, None)

    # Accumulate results: per condition → list of (true_label, p_basal)
    results: Dict[str, Tuple[List[int], List[float]]] = {c: ([], []) for c in conditions}

    for pid, meta in test_patients.items():
        subtype = meta["subtype"]
        true_label = label_map[subtype]  # 0=LumA, 1=Basal

        # Load genomic features
        own_feats = _load_genomic_feats(genomic_h5_dir, pid)
        if own_feats is None:
            log.warning("No genomic feats for %s, skipping.", pid)
            continue

        # Counterfactual feats: Basal gets LumA mean, LumA gets Basal mean
        cf_feats = mean_luma_feats if subtype == "Basal" else mean_basal_feats

        # Load tiles
        tiles = _iter_test_tiles(zip_dir, pid, n_tiles_per_patient, img_size, rng)
        if not tiles:
            log.warning("No tiles for %s, skipping.", pid)
            continue

        log.info("Patient %s (%s): %d tiles", pid, subtype, len(tiles))

        for tile_t in tiles:
            own_t   = torch.tensor(own_feats).unsqueeze(0)   # (1, 512)
            null_t  = torch.zeros_like(own_t)
            cf_t    = torch.tensor(cf_feats).unsqueeze(0)

            for cond_name, (feat_mode, scale, _) in conditions.items():
                if feat_mode == "null":
                    cond_t = null_t
                elif feat_mode == "own":
                    cond_t = own_t
                else:  # counterfactual
                    cond_t = cf_t

                recon = invert_and_decode(ema_model, sampler, tile_t, cond_t, scale, device)
                # Extract Virchow2 from generated tile
                v2_feats = extract_virchow2_features([recon.cpu()], device)  # (1, 1280)
                _, p_basal = classify(v2_feats, clf_bundle)

                results[cond_name][0].append(true_label)
                results[cond_name][1].append(float(p_basal[0]))

    # Compute and save metrics
    metrics = {}
    for cond_name, (labels, probas) in results.items():
        if len(set(labels)) < 2 or not labels:
            log.warning("Condition %s: insufficient data for AUROC", cond_name)
            continue
        auroc = float(roc_auc_score(labels, probas))
        threshold = clf_bundle["threshold"]
        preds = [int(p >= threshold) for p in probas]
        acc = float(np.mean([p == l for p, l in zip(preds, labels)]))
        metrics[cond_name] = {"auroc": auroc, "accuracy": acc, "n_tiles": len(labels)}
        log.info("%-15s  AUROC=%.3f  ACC=%.3f  n=%d", cond_name, auroc, acc, len(labels))

    metrics_path = output_dir / "cfg_eval_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("Saved metrics → %s", metrics_path)

    _plot_results(metrics, output_dir)


def _plot_results(metrics: dict, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = list(metrics.keys())
    aurocs = [metrics[c]["auroc"] for c in conds]
    accs   = [metrics[c]["accuracy"] for c in conds]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(conds))
    axes[0].bar(x, aurocs, color="steelblue", alpha=0.85)
    axes[0].axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="random")
    axes[0].set_xticks(x); axes[0].set_xticklabels(conds, rotation=30, ha="right")
    axes[0].set_ylabel("AUROC (Basal vs LumA)"); axes[0].set_ylim(0, 1)
    axes[0].set_title("Classifier AUROC by CFG condition"); axes[0].legend()

    axes[1].bar(x, accs, color="tomato", alpha=0.85)
    axes[1].axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="chance")
    axes[1].set_xticks(x); axes[1].set_xticklabels(conds, rotation=30, ha="right")
    axes[1].set_ylabel("Accuracy"); axes[1].set_ylim(0, 1)
    axes[1].set_title("Classifier Accuracy by CFG condition"); axes[1].legend()

    fig.tight_layout()
    out_path = output_dir / "cfg_eval_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Saved plot → %s", out_path)
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config",    required=True, help="Path to src/config.yaml")
    parser.add_argument("--output",    required=True, help="Output directory")
    parser.add_argument("--n-tiles",          type=int, default=20, help="Tiles per patient")
    parser.add_argument("--max-patients",     type=int, default=None,
                        help="Max patients per subtype to evaluate (None = all test patients)")
    parser.add_argument("--scales",    type=float, nargs="+", default=[1.0, 3.0, 5.0, 10.0])
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s", datefmt="%H:%M:%S")

    config_path = Path(args.config).resolve()
    repo_root   = config_path.parent.parent if config_path.name == "config.yaml" else config_path.parent
    if not (repo_root / "run_pipeline.py").exists():
        repo_root = repo_root.parent

    with open(config_path) as fh:
        full_cfg = yaml.safe_load(fh)

    v11_cfg = full_cfg["mopadi_genomic_ca"]

    def _resolve(raw: str) -> str:
        from pathlib import Path as P
        p = P(raw)
        if p.is_absolute():
            return str(p)
        normalized = raw[2:] if raw.startswith("./") else raw
        for prefix in ("experiments/", "data/", "dataframes/"):
            if normalized.startswith(prefix):
                return str((repo_root.parent / normalized).resolve())
        return str((repo_root / normalized).resolve())

    # Resolve all paths in v11 config
    for k, v in v11_cfg.items():
        if isinstance(v, str) and ("/" in v or v.startswith(".")):
            v11_cfg[k] = _resolve(v)

    v11_ckpt    = str(Path(v11_cfg["logdir"]) / "autoenc" / "last.ckpt")
    splits_path = v11_cfg["patient_splits_path"]
    genomic_h5  = v11_cfg["genomic_feature_dir"]
    zip_dir     = v11_cfg["zip_dir"]
    clf_path    = "/mnt/bulk-saturn/maralampert/genhist/experiments/20260326_cross_subtype_classifier/train/subtype_linear_model.joblib"

    run_evaluation(
        v11_ckpt=v11_ckpt,
        v11_cfg=v11_cfg,
        splits_path=splits_path,
        genomic_h5_dir=genomic_h5,
        zip_dir=zip_dir,
        clf_path=clf_path,
        output_dir=args.output,
        repo_root=repo_root,
        n_tiles_per_patient=args.n_tiles,
        cfg_scales=tuple(args.scales),
        device_str=args.device,
        seed=args.seed,
        max_patients_per_subtype=args.max_patients,
    )


if __name__ == "__main__":
    main()
