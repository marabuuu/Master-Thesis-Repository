#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FID evaluation for the PoC BRCA vs LIHC orthogonal-conditioning experiment.

Generates 5 000 BRCA tiles and 5 000 LIHC tiles from test-set patients using
the CfgBackboneLitModel with orthogonal binary conditioning, then computes the
2×2 Fréchet Distance matrix:

    rows = {real BRCA,  real LIHC}
    cols = {gen  BRCA,  gen  LIHC}

    FD(real_BRCA, gen_BRCA)   FD(real_BRCA, gen_LIHC)
    FD(real_LIHC, gen_BRCA)   FD(real_LIHC, gen_LIHC)

Diagonal  = within-class fidelity (lower = better).
Off-diag  = cross-class separation (higher = more conditioning signal).

Generated images are saved as per-patient ZIPs under:
    <output-dir>/generated/TCGA-BRCA/<patient_id>.zip
    <output-dir>/generated/TCGA-LIHC/<patient_id>.zip

Real tiles are sampled from the source zip directory and saved under:
    <output-dir>/real/TCGA-BRCA/<patient_id>.zip
    <output-dir>/real/TCGA-LIHC/<patient_id>.zip

Usage
-----
# Full run (5000 tiles per cohort, batched on GPU):
python -m src.poc_experiment.fid_evaluation \\
    --run-dir experiments/20260603_poc_128_orthogonal/gda \\
    --checkpoint experiments/20260603_poc_128_orthogonal/gda/autoenc/epoch=8-step=109375.ckpt \\
    --patient-splits experiments/20260524_poc_breast_vs_liver_genomic_features/patient_splits.json \\
    --tiles-dir ../data/PoC-BRCA-LIHC-tumor-tiles-128 \\
    --output-dir experiments/20260603_poc_128_orthogonal/fid_evaluation \\
    --n-tiles 5000 \\
    --guidance-scale 5.0 \\
    --gen-batch-size 16 \\
    --steps 20

# Quick smoke-test:
python -m src.poc_experiment.fid_evaluation \\
    --run-dir experiments/20260603_poc_128_orthogonal/gda \\
    --patient-splits experiments/20260524_poc_breast_vs_liver_genomic_features/patient_splits.json \\
    --tiles-dir ../data/PoC-BRCA-LIHC-tumor-tiles-128 \\
    --output-dir experiments/20260603_poc_128_orthogonal/fid_evaluation_test \\
    --n-tiles 100 \\
    --test
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Path helpers
# ─────────────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]       
_WORKSPACE_ROOT = _REPO_ROOT.parent                      


def _resolve(p: str, must_exist: bool = True) -> Path:
    path = Path(p)
    if path.is_absolute():
        resolved = path
    else:
        ws_cand = (_WORKSPACE_ROOT / p).resolve()
        cwd_cand = path.resolve()
        if ws_cand.exists():
            resolved = ws_cand
        elif cwd_cand.exists():
            resolved = cwd_cand
        else:
            resolved = ws_cand
    if must_exist and not resolved.exists():
        raise FileNotFoundError(
            f"Path not found: {p}\n"
            f"  tried workspace: {(_WORKSPACE_ROOT / p).resolve()}\n"
            f"  tried cwd:       {Path(p).resolve()}"
        )
    return resolved


def _ensure_mopadi_path() -> None:
    candidates = [
        _WORKSPACE_ROOT / "mopadi" / "src",
        _REPO_ROOT / "mopadi" / "src",
    ]
    for c in candidates:
        if (c / "mopadi").exists():
            s = str(c)
            if s not in sys.path:
                sys.path.insert(0, s)
            break


_ensure_mopadi_path()


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(run_dir: Path):
    """Load GDAConfig from hparams.yaml."""
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("PyYAML required") from e
    from .config import GDAConfig
    from mopadi.configs.choices import ModelName

    hparams_path = run_dir / "hparams.yaml"
    if not hparams_path.exists():
        raise FileNotFoundError(f"hparams.yaml not found in {run_dir}")
    with hparams_path.open() as f:
        load_fn = getattr(yaml, "unsafe_load", yaml.full_load)
        raw = load_fn(f)
    conf = GDAConfig.from_dict(raw)
    if getattr(conf, "model_name", None) is None:
        conf.model_name = ModelName.beatgans_autoenc
    return conf


def _resolve_best_checkpoint(run_dir: Path, explicit: Optional[str]) -> Path:
    """Return the checkpoint with the lowest val loss, or the explicitly given one."""
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = _resolve(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {explicit}")
        return p

    autoenc_dir = run_dir / "autoenc"
    ckpts = sorted(autoenc_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {autoenc_dir}")

    def _score(ckpt_path: Path) -> float:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception:
            return float("inf")
        best_scores = []
        for v in ckpt.get("callbacks", {}).values():
            if isinstance(v, dict):
                s = v.get("best_model_score")
                if s is not None:
                    try:
                        best_scores.append(float(s))
                    except (TypeError, ValueError):
                        pass
        return min(best_scores) if best_scores else float("inf")

    best = min(ckpts, key=_score)
    logger.info(f"Auto-selected checkpoint: {best.name} (score={_score(best):.4f})")
    return best


def load_model(conf, ckpt_path: Path, device: torch.device):
    from .cfg_model import CfgBackboneLitModel

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    conf.backbone_ckpt_path = None  # skip warm-start load
    model = CfgBackboneLitModel(conf)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.info(f"  {len(missing)} missing keys (expected for new params)")
    if unexpected:
        logger.warning(f"  {len(unexpected)} unexpected keys in checkpoint")
    model.eval()
    model.to(device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Orthogonal conditioning vectors
# ─────────────────────────────────────────────────────────────────────────────

COHORTS = ("TCGA-BRCA", "TCGA-LIHC")


def build_conditioning_vectors(feat_dim: int, device: torch.device, normalize: bool = True) -> Dict[str, torch.Tensor]:
    """Build the same orthogonal vectors the dataset uses during training.

    normalize=True  → unit-norm (training default for most runs)
    normalize=False → raw binary codes, norm=sqrt(feat_dim/2) ≈ 16 for feat_dim=512
                      (matches runs trained with normalize_feats=False)
    """
    import torch.nn.functional as F

    brca = torch.zeros(feat_dim, dtype=torch.float32)
    lihc = torch.zeros(feat_dim, dtype=torch.float32)
    brca[1::2] = 1.0
    lihc[0::2] = 1.0
    if normalize:
        brca = F.normalize(brca, p=2, dim=-1)
        lihc = F.normalize(lihc, p=2, dim=-1)
    brca = brca.to(device)
    lihc = lihc.to(device)
    dot = float(torch.dot(brca, lihc).item())
    logger.info(f"Orthogonal codes (normalize={normalize}): BRCA norm={float(brca.norm()):.4f}, LIHC norm={float(lihc.norm()):.4f}, dot={dot:.4f}")
    return {"TCGA-BRCA": brca, "TCGA-LIHC": lihc}


# ─────────────────────────────────────────────────────────────────────────────
#  Patient splits
# ─────────────────────────────────────────────────────────────────────────────

def load_test_patients(splits_path: Path) -> Dict[str, List[str]]:
    """Return {cohort: [patient_id, ...]} for the test split."""
    with open(splits_path) as f:
        payload = json.load(f)

    test_block = payload.get("test", {})
    by_cohort: Dict[str, List[str]] = {c: [] for c in COHORTS}
    for pid, meta in test_block.items():
        if pid.startswith("_"):
            continue
        cohort = meta.get("subtype") if isinstance(meta, dict) else None
        if cohort in by_cohort:
            by_cohort[cohort].append(pid)

    for c in COHORTS:
        by_cohort[c] = sorted(by_cohort[c])
        logger.info(f"  {c}: {len(by_cohort[c])} test patients")
    return by_cohort


# ─────────────────────────────────────────────────────────────────────────────
#  Tile generation
# ─────────────────────────────────────────────────────────────────────────────

def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) float tensor in [-1, 1] → (H, W, 3) uint8."""
    arr = t.detach().cpu().clamp(-1.0, 1.0)
    arr = ((arr + 1.0) / 2.0 * 255.0).to(torch.uint8)
    return arr.permute(1, 2, 0).numpy()


@torch.inference_mode()
def generate_batch(
    model,
    cond_vec: torch.Tensor,
    batch_size: int,
    guidance_scale: float,
    n_steps: int,
    device: torch.device,
    seed: Optional[int] = None,
    use_noise_cond: bool = False,
) -> List[np.ndarray]:
    """Generate ``batch_size`` tiles conditioned on ``cond_vec`` (1-d unit vector).

    When ``use_noise_cond=True`` a fresh random unit-sphere vector is sampled
    per tile (matches conditioning_type="noise" training exactly).

    Returns a list of (H, W, 3) uint8 numpy arrays.
    """
    import torch.nn.functional as _NF
    from mopadi.diffusion.base import DummyReturn

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    backbone = model.ema_model
    sampler = model.conf._make_diffusion_conf(n_steps).make_sampler()
    img_size = model.conf.img_size
    feat_dim = model.conf.feat_dim

    if use_noise_cond:
        # Fresh unit-sphere random vector per tile — matches noise-run training exactly.
        # Seeded above, so the noise tensor and cond vectors are jointly reproducible.
        raw = torch.randn(batch_size, feat_dim, device=device)
        cond = _NF.normalize(raw, p=2, dim=-1)
    else:
        cond  = cond_vec.view(1, -1).expand(batch_size, -1).to(device)
    zeros = torch.zeros(batch_size, feat_dim, device=device, dtype=torch.float32)
    noise = torch.randn(batch_size, 3, img_size, img_size, device=device)

    if guidance_scale == 0.0:
        class _Null(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._bb = backbone  # register so .parameters() is non-empty
            def forward(self, x, t, **_):
                t_sc = sampler._scale_timesteps(t)
                return DummyReturn(pred=backbone.forward(x=x, t=t_sc, x_start=None, cond=zeros).pred)
        wrapped = _Null()
    else:
        _scale = guidance_scale
        class _CFG(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._bb = backbone  # register so .parameters() is non-empty
            def forward(self, x, t, **_):
                t_sc = sampler._scale_timesteps(t)
                eps_null = backbone.forward(x=x, t=t_sc, x_start=None, cond=zeros).pred
                eps_cond = backbone.forward(x=x, t=t_sc, x_start=None, cond=cond).pred
                return DummyReturn(pred=eps_null + _scale * (eps_cond - eps_null))
        wrapped = _CFG()

    samples = sampler.sample(
        model=wrapped,
        shape=noise.shape,
        noise=noise,
        model_kwargs={},
        progress=False,
    )  # (B, 3, H, W) in [-1, 1]

    return [_tensor_to_uint8(samples[i]) for i in range(samples.shape[0])]


# ─────────────────────────────────────────────────────────────────────────────
#  ZIP helpers  (shared with fid_subtype_evaluation)
# ─────────────────────────────────────────────────────────────────────────────

_IMG_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _save_images_to_zip(images: List[np.ndarray], zip_path: Path, patient_id: str) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for idx, img in enumerate(images):
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="PNG")
            zf.writestr(f"{patient_id}_{idx:05d}.png", buf.getvalue())


def _sample_real_tiles(src_zip: Path, n_tiles: int, rng: np.random.Generator) -> List[np.ndarray]:
    with zipfile.ZipFile(src_zip, "r") as zf:
        members = [m for m in zf.namelist() if Path(m).suffix.lower() in _IMG_SUFFIXES]
        if not members:
            return []
        chosen = rng.choice(members, size=min(n_tiles, len(members)), replace=False)
        imgs = []
        for name in chosen:
            with zf.open(name) as fh:
                imgs.append(np.array(Image.open(io.BytesIO(fh.read())).convert("RGB")))
    return imgs


def _iter_zip_images(zip_path: Path) -> List[Image.Image]:
    imgs = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if Path(name).suffix.lower() in _IMG_SUFFIXES:
                with zf.open(name) as fh:
                    imgs.append(Image.open(io.BytesIO(fh.read())).convert("RGB"))
    return imgs


# ─────────────────────────────────────────────────────────────────────────────
#  Generate tiles for one cohort
# ─────────────────────────────────────────────────────────────────────────────

def _load_patient_h5_feat(h5_dir: Path, pid: str, device: torch.device) -> Optional[torch.Tensor]:
    """Load a patient's genomic feature vector from an H5 file."""
    import h5py
    h5_path = h5_dir / f"{pid}.h5"
    if not h5_path.exists():
        return None
    with h5py.File(h5_path, "r") as f:
        feats = torch.from_numpy(f["feats"][:]).float().to(device)
    return feats


def generate_cohort_tiles(
    model,
    cond_vec: torch.Tensor,
    patient_ids: List[str],
    n_tiles_total: int,
    out_dir: Path,
    device: torch.device,
    guidance_scale: float,
    gen_batch_size: int,
    n_steps: int,
    seed: int,
    skip_existing: bool = True,
    genomic_h5_dir: Optional[Path] = None,
    normalize_feats: bool = False,
    use_noise_cond: bool = False,
) -> None:
    """Generate and save per-patient ZIPs for one cohort.

    Distributes ``n_tiles_total`` evenly across patients (ceiling division).
    If ``genomic_h5_dir`` is given, each patient's own H5 feature vector is
    used as conditioning instead of the shared ``cond_vec`` (needed for
    conditioning_type="real" RNA models).
    ``normalize_feats`` must match the training config: when True, H5 features
    are L2-normalized to unit norm before conditioning, matching _resolve_feats.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_patients = len(patient_ids)
    n_per_patient = max(1, math.ceil(n_tiles_total / n_patients))
    logger.info(
        f"  {n_patients} patients × {n_per_patient} tiles ≈ "
        f"{n_patients * n_per_patient} tiles  (target {n_tiles_total})"
    )

    for p_idx, pid in enumerate(patient_ids):
        out_zip = out_dir / f"{pid}.zip"
        if skip_existing and out_zip.exists():
            logger.info(f"  [{p_idx+1}/{n_patients}] {pid}: skip (exists)")
            continue

        # Per-patient conditioning for RNA models; fall back to shared cond_vec
        if genomic_h5_dir is not None:
            patient_cond = _load_patient_h5_feat(genomic_h5_dir, pid, device)
            if patient_cond is None:
                logger.warning(f"  [{p_idx+1}/{n_patients}] {pid}: H5 not found, skipping")
                continue
            if normalize_feats:
                import torch.nn.functional as _F
                patient_cond = _F.normalize(patient_cond, p=2, dim=-1)
        else:
            patient_cond = cond_vec

        logger.info(f"  [{p_idx+1}/{n_patients}] {pid}: generating {n_per_patient} tiles…")
        imgs: List[np.ndarray] = []
        remaining = n_per_patient
        batch_idx = 0
        while remaining > 0:
            bs = min(gen_batch_size, remaining)
            tile_seed = (seed + p_idx * 100_000 + batch_idx * 10_000) % (2 ** 32)
            batch_imgs = generate_batch(
                model=model,
                cond_vec=patient_cond,
                batch_size=bs,
                guidance_scale=guidance_scale,
                n_steps=n_steps,
                device=device,
                seed=tile_seed,
                use_noise_cond=use_noise_cond,
            )
            imgs.extend(batch_imgs)
            remaining -= bs
            batch_idx += 1

        _save_images_to_zip(imgs[:n_per_patient], out_zip, pid)
        logger.info(f"    → saved {min(len(imgs), n_per_patient)} tiles to {out_zip.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  Collect real tiles
# ─────────────────────────────────────────────────────────────────────────────

def collect_real_cohort_tiles(
    patient_ids: List[str],
    tiles_dir: Path,
    n_tiles_total: int,
    out_dir: Path,
    seed: int,
    skip_existing: bool = True,
) -> None:
    """Sample real tiles from source ZIPs and save per-patient ZIPs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_patients = len(patient_ids)
    n_per_patient = max(1, math.ceil(n_tiles_total / n_patients))

    found = 0
    for p_idx, pid in enumerate(patient_ids):
        out_zip = out_dir / f"{pid}.zip"
        if skip_existing and out_zip.exists():
            found += 1
            continue

        # Match source ZIP: pid appears in the filename (case-insensitive).
        candidates = [
            p for p in tiles_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".zip"
            and pid.upper() in p.name.upper()
        ]
        if not candidates:
            logger.warning(f"  [{p_idx+1}/{n_patients}] {pid}: no source ZIP found")
            continue

        imgs = _sample_real_tiles(candidates[0], n_per_patient, rng)
        if not imgs:
            logger.warning(f"  [{p_idx+1}/{n_patients}] {pid}: no tiles in {candidates[0].name}")
            continue

        _save_images_to_zip(imgs, out_zip, pid)
        found += 1

    logger.info(f"  Real tiles ready for {found}/{n_patients} patients → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  InceptionV3 feature extraction  (mirrors fid_subtype_evaluation)
# ─────────────────────────────────────────────────────────────────────────────

class _InceptionV3Extractor:
    def __init__(self, device: str = "cpu") -> None:
        import torchvision.transforms as T
        from pytorch_fid.inception import InceptionV3

        logger.info("Loading InceptionV3 (pytorch-fid TF GAN weights)…")
        # InceptionV3 from pytorch-fid uses the original TF GAN checkpoint
        # (pt_inception-2015-12-05-6726825d.pth) and applies 2*x-1 normalisation
        # internally (normalize_input=True, default), so we feed [0, 1] images.
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx], normalize_input=True, resize_input=False)
        model.eval().to(device)
        self.model = model
        self.device = device
        # ToTensor gives [0, 1]; resize/crop here, InceptionV3 normalises to [-1,1]
        self.transform = T.Compose([
            T.Resize(299),
            T.CenterCrop(299),
            T.ToTensor(),
        ])
        logger.info(f"InceptionV3 ready on '{device}'")

    @torch.inference_mode()
    def encode_batch(self, pil_images: List[Image.Image]) -> np.ndarray:
        tensors: List[torch.Tensor] = [self.transform(img) for img in pil_images]  # type: ignore[misc]
        batch = torch.stack(tensors).to(self.device)
        out = self.model(batch)[0]  # list of outputs; [0] = Pool3 (B, 2048, 1, 1)
        return out.squeeze(3).squeeze(2).float().cpu().numpy()


def extract_inception_features(
    tile_dir: Path,
    extractor: _InceptionV3Extractor,
    batch_size: int = 32,
    label: str = "",
) -> np.ndarray:
    zip_files = sorted(tile_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No ZIP files in {tile_dir}")

    all_features: List[np.ndarray] = []
    batch: List[Image.Image] = []
    n_tiles = 0

    for zp in zip_files:
        for img in _iter_zip_images(zp):
            batch.append(img)
            if len(batch) == batch_size:
                all_features.append(extractor.encode_batch(batch))
                n_tiles += len(batch)
                batch = []
    if batch:
        all_features.append(extractor.encode_batch(batch))
        n_tiles += len(batch)

    if not all_features:
        raise ValueError(f"No tiles found under {tile_dir}")

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    logger.info(f"  {label}: {n_tiles} tiles → features {features.shape}")
    return features


# ─────────────────────────────────────────────────────────────────────────────
#  Fréchet Distance
# ─────────────────────────────────────────────────────────────────────────────

def _compute_statistics(features: np.ndarray, eps: float = 1e-6):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False) + eps * np.eye(features.shape[1])
    return mu, sigma


def _frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            logger.warning("sqrtm: large imaginary component; using real part")
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


# ─────────────────────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(
    run_dir: str,
    patient_splits_path: str,
    tiles_dir: str,
    output_dir: str,
    checkpoint: Optional[str] = None,
    n_tiles: int = 5_000,
    guidance_scale: float = 5.0,
    gen_batch_size: int = 16,
    n_steps: int = 20,
    inception_batch_size: int = 32,
    device: Optional[str] = None,
    seed: int = 42,
    skip_generate: bool = False,
    skip_real: bool = False,
    skip_features: bool = False,
    test_mode: bool = False,
    n_tiles_test: int = 50,
) -> Dict:
    run_dir_p    = _resolve(run_dir)
    splits_path  = _resolve(patient_splits_path)
    tiles_dir_p  = _resolve(tiles_dir)
    out_dir      = (_WORKSPACE_ROOT / output_dir
                    if not Path(output_dir).is_absolute()
                    else Path(output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    effective_n = n_tiles_test if test_mode else n_tiles
    if test_mode:
        logger.info(f"TEST MODE: {effective_n} tiles per cohort")
    else:
        logger.info(f"FULL MODE: {effective_n} tiles per cohort")

    # ── Patient splits ────────────────────────────────────────────────────────
    test_patients = load_test_patients(splits_path)

    # ── Load model ────────────────────────────────────────────────────────────
    if not skip_generate:
        ckpt_path = _resolve_best_checkpoint(run_dir_p, checkpoint)
        logger.info(f"Loading model from {ckpt_path.name}…")
        conf = _load_config(run_dir_p)
        model = load_model(conf, ckpt_path, device_obj)
        logger.info(f"  img_size={conf.img_size}  feat_dim={conf.feat_dim}")

        normalize_feats = getattr(conf, "normalize_feats", True)
        conditioning_type = getattr(conf, "conditioning_type", "one_hot")

        # Build cohort-level conditioning vectors that match training distribution
        use_noise_cond = False
        if conditioning_type == "zeros":
            zero_vec = torch.zeros(conf.feat_dim, device=device_obj)
            cond_vecs = {c: zero_vec for c in COHORTS}
            logger.info("zeros model: using all-zero conditioning vectors at inference")
        elif conditioning_type == "noise":
            # cond_vecs unused — per-tile fresh randn is generated inside generate_batch
            cond_vecs = {c: torch.zeros(conf.feat_dim, device=device_obj) for c in COHORTS}
            use_noise_cond = True
            logger.info("noise model: using per-tile fresh unit-sphere conditioning at inference")
        else:
            cond_vecs = build_conditioning_vectors(conf.feat_dim, device_obj, normalize=normalize_feats)

        # For RNA models, use per-patient H5 features instead of cohort-level codes
        genomic_h5_dir: Optional[Path] = None
        if conditioning_type == "real":
            h5_dir_str = getattr(conf, "genomic_feature_dir", None)
            if h5_dir_str:
                genomic_h5_dir = Path(h5_dir_str)
                logger.info(f"RNA model: using per-patient H5 features from {genomic_h5_dir}")
            else:
                logger.warning("conditioning_type=real but no genomic_feature_dir in conf; using cohort codes")

        # Generate tiles per cohort
        for cohort in COHORTS:
            gen_dir = out_dir / "generated" / cohort
            logger.info(f"Generating {cohort} tiles → {gen_dir}")
            generate_cohort_tiles(
                model=model,
                cond_vec=cond_vecs[cohort],
                patient_ids=test_patients[cohort],
                n_tiles_total=effective_n,
                out_dir=gen_dir,
                device=device_obj,
                guidance_scale=guidance_scale,
                gen_batch_size=gen_batch_size,
                n_steps=n_steps,
                seed=seed,
                skip_existing=True,
                genomic_h5_dir=genomic_h5_dir,
                normalize_feats=normalize_feats,
                use_noise_cond=use_noise_cond,
            )
    else:
        logger.info("--skip-generate: skipping tile generation")

    # ── Collect real tiles ────────────────────────────────────────────────────
    if not skip_real:
        for cohort in COHORTS:
            real_dir = out_dir / "real" / cohort
            logger.info(f"Collecting real {cohort} tiles → {real_dir}")
            collect_real_cohort_tiles(
                patient_ids=test_patients[cohort],
                tiles_dir=tiles_dir_p,
                n_tiles_total=effective_n,
                out_dir=real_dir,
                seed=seed,
                skip_existing=True,
            )
    else:
        logger.info("--skip-real: skipping real tile collection")

    # ── Extract Inception features ────────────────────────────────────────────
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    distributions = {
        f"real_{c}": out_dir / "real" / c for c in COHORTS
    }
    distributions.update({
        f"gen_{c}": out_dir / "generated" / c for c in COHORTS
    })

    features: Dict[str, np.ndarray] = {}

    if not skip_features:
        extractor = _InceptionV3Extractor(device=device)
        for key, tile_dir in distributions.items():
            npz_path = feat_dir / f"inception_{key}.npz"
            if npz_path.exists():
                logger.info(f"  {key}: loading cached features from {npz_path.name}")
                features[key] = np.load(npz_path)["features"]
            else:
                logger.info(f"  Extracting features for {key}…")
                feats = extract_inception_features(
                    tile_dir, extractor, batch_size=inception_batch_size, label=key
                )
                np.savez_compressed(npz_path, features=feats)
                features[key] = feats
                logger.info(f"    saved → {npz_path.name}")
    else:
        logger.info("--skip-features: loading cached .npz files")
        for key in distributions:
            npz_path = feat_dir / f"inception_{key}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(
                    f"Feature file not found: {npz_path}. Run without --skip-features first."
                )
            features[key] = np.load(npz_path)["features"]

    # ── FD matrix ─────────────────────────────────────────────────────────────
    logger.info("Computing Fréchet Distances…")
    stats: Dict = {}
    for key, feats in features.items():
        stats[key] = _compute_statistics(feats)

    rows = [f"real_{c}" for c in COHORTS]
    cols = [f"gen_{c}"  for c in COHORTS]
    fd_matrix: Dict[str, float] = {}
    for rk in rows:
        for ck in cols:
            mu1, s1 = stats[rk]
            mu2, s2 = stats[ck]
            fd = _frechet_distance(mu1, s1, mu2, s2)
            label = f"{rk}_vs_{ck}"
            fd_matrix[label] = fd
            logger.info(f"  FD({rk}, {ck}) = {fd:.2f}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "fd_matrix": fd_matrix,
        "tile_counts": {k: int(v.shape[0]) for k, v in features.items()},
        "cohorts": list(COHORTS),
        "rows": rows,
        "cols": cols,
        "n_tiles_target": effective_n,
        "guidance_scale": guidance_scale,
        "n_steps": n_steps,
        "test_mode": test_mode,
        "checkpoint": checkpoint,
    }
    results_path = out_dir / "fd_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {results_path}")

    # ── Pretty-print summary ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("FD MATRIX  (rows=real, cols=generated)")
    logger.info(f"{'':20s}  {'gen_BRCA':>12s}  {'gen_LIHC':>12s}")
    for rk in rows:
        short = rk.replace("real_TCGA-", "real_")
        vals = "  ".join(f"{fd_matrix[f'{rk}_vs_{ck}']:12.2f}" for ck in cols)
        logger.info(f"  {short:20s}  {vals}")
    logger.info("=" * 60)
    logger.info("Diagonal (within-class FID, lower=better):")
    for c in COHORTS:
        fd = fd_matrix[f"real_{c}_vs_gen_{c}"]
        logger.info(f"  {c}: {fd:.2f}")
    logger.info("Off-diagonal (cross-class FID, higher=more conditioning signal):")
    logger.info(f"  real_BRCA vs gen_LIHC: {fd_matrix['real_TCGA-BRCA_vs_gen_TCGA-LIHC']:.2f}")
    logger.info(f"  real_LIHC vs gen_BRCA: {fd_matrix['real_TCGA-LIHC_vs_gen_TCGA-BRCA']:.2f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="FID evaluation for PoC BRCA vs LIHC experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", required=True,
                   help="GDA run directory (contains hparams.yaml and autoenc/ checkpoints)")
    p.add_argument("--checkpoint", default=None,
                   help="Explicit checkpoint path; auto-selects best by val loss if omitted")
    p.add_argument("--patient-splits", required=True, dest="patient_splits",
                   help="Path to patient_splits.json")
    p.add_argument("--tiles-dir", required=True,
                   help="Directory with per-patient ZIP files of real tiles")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for generated ZIPs, features, and results")
    p.add_argument("--n-tiles", type=int, default=5_000,
                   help="Target tiles per cohort for the full run")
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--gen-batch-size", type=int, default=16,
                   help="Images generated per GPU call (tune for VRAM)")
    p.add_argument("--steps", type=int, default=20, help="DDIM sampling steps")
    p.add_argument("--inception-batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-generate", action="store_true",
                   help="Skip generation (use existing ZIPs in output-dir/generated/)")
    p.add_argument("--skip-real", action="store_true",
                   help="Skip real tile collection (use existing ZIPs in output-dir/real/)")
    p.add_argument("--skip-features", action="store_true",
                   help="Skip Inception feature extraction (use cached .npz files)")
    p.add_argument("--test", action="store_true",
                   help="Smoke-test: generate --n-tiles-test tiles per cohort")
    p.add_argument("--n-tiles-test", type=int, default=50)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    run(
        run_dir=args.run_dir,
        patient_splits_path=args.patient_splits,
        tiles_dir=args.tiles_dir,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        n_tiles=args.n_tiles,
        guidance_scale=args.guidance_scale,
        gen_batch_size=args.gen_batch_size,
        n_steps=args.steps,
        inception_batch_size=args.inception_batch_size,
        device=args.device,
        seed=args.seed,
        skip_generate=args.skip_generate,
        skip_real=args.skip_real,
        skip_features=args.skip_features,
        test_mode=args.test,
        n_tiles_test=args.n_tiles_test,
    )


if __name__ == "__main__":
    main()
