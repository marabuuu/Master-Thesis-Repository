"""Generate a BRCA vs LIHC tile panel for a PoC conditioning run.

Left:  N tiles with BRCA conditioning.
Right: N tiles with LIHC conditioning (same initial noise per column).

Keeping the base seed fixed across runs makes panels directly comparable:
column i always starts from the same noise tensor; only the conditioning differs.

Conditioning vectors per run type
----------------------------------
zeros / noise / one_hot  → orthogonal binary codes (same codes used in FID eval)
real (RNA-seq)           → mean test-patient feature vector per cohort, L2-normalised

Usage
-----
Run from Master-Thesis-Repository/ with the venv active:

    python -m src.visualization.poc_panel \\
        --run-dir experiments/20260607_poc_128_rna_norm_30M/gda \\
        --out experiments/20260607_poc_128_rna_norm_30M/panel.png

Options
-------
    --run-dir         GDA run dir (contains hparams.yaml + autoenc/ checkpoints)
    --out             Output PNG path (default: <experiment_dir>/panel.png)
    --n-tiles         Tiles per cohort side (default: 3)
    --seed            Base noise seed; column i uses seed+i (default: 1000)
    --guidance-scale  CFG scale (default: 1.0)
    --steps           DDIM steps (default: 20)
    --device          torch device (auto-detected if omitted)
    --checkpoint      Explicit checkpoint path (auto-selects last.ckpt otherwise)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WS_ROOT = _REPO_ROOT.parent
sys.path.insert(0, str(_REPO_ROOT))

_MOPADI_SRC = _WS_ROOT / "mopadi" / "src"
if (_MOPADI_SRC / "mopadi").exists():
    _s = str(_MOPADI_SRC)
    if _s not in sys.path:
        sys.path.insert(0, _s)


# ─── conditioning vectors ────────────────────────────────────────────────────

def _build_cond_vecs(
    conf,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Return {cohort: cond_tensor} for BRCA and LIHC.

    Non-RNA runs: orthogonal binary codes (matching FID evaluation).
    RNA runs:     mean test-patient feature vector per cohort (L2-normalised).
    """
    import torch.nn.functional as F
    from src.evaluation.poc_fid import build_conditioning_vectors, load_test_patients

    conditioning_type: str = getattr(conf, "conditioning_type", "one_hot")
    normalize_feats: bool = getattr(conf, "normalize_feats", True)

    if conditioning_type == "real":
        h5_dir_str: Optional[str] = getattr(conf, "genomic_feature_dir", None)
        if not h5_dir_str:
            raise ValueError(
                "conditioning_type=real but genomic_feature_dir not set in hparams.yaml"
            )
        h5_dir = Path(h5_dir_str)
        splits_path = Path(conf.patient_splits_path)

        test_patients = load_test_patients(splits_path)

        import h5py
        cohort_feats: dict[str, list[torch.Tensor]] = {
            "TCGA-BRCA": [],
            "TCGA-LIHC": [],
        }
        for cohort, pids in test_patients.items():
            for pid in pids:
                h5_path = h5_dir / f"{pid}.h5"
                if h5_path.exists():
                    with h5py.File(h5_path, "r") as fh:
                        feat = torch.from_numpy(fh["feats"][:]).float()
                    cohort_feats[cohort].append(feat)

        vecs: dict[str, torch.Tensor] = {}
        for cohort, feats in cohort_feats.items():
            if not feats:
                raise ValueError(f"No H5 files found for {cohort} in {h5_dir}")
            mean_feat = torch.stack(feats).mean(dim=0)
            if normalize_feats:
                mean_feat = F.normalize(mean_feat, p=2, dim=-1)
            vecs[cohort] = mean_feat.to(device)
        return vecs

    elif conditioning_type == "zeros":
        zero_vec = torch.zeros(conf.feat_dim, device=device)
        return {"TCGA-BRCA": zero_vec, "TCGA-LIHC": zero_vec}

    elif conditioning_type == "noise":
        # Per-tile random vectors are generated inside generate_batch; return placeholder.
        zero_vec = torch.zeros(conf.feat_dim, device=device)
        return {"TCGA-BRCA": zero_vec, "TCGA-LIHC": zero_vec}

    else:
        # Orthogonal binary codes — same as FID evaluation for one_hot runs.
        return build_conditioning_vectors(conf.feat_dim, device, normalize=normalize_feats)


# ─── panel assembly ──────────────────────────────────────────────────────────

def _build_panel_canvas(
    brca_tiles: list[np.ndarray],
    lihc_tiles: list[np.ndarray],
    gap_inner: int = 2,
    gap_center: int = 8,
    bg: int = 255,
) -> np.ndarray:
    """Assemble [BRCA ... | gap_center | ... LIHC] canvas (H×W×3 uint8)."""
    n = len(brca_tiles)
    assert len(lihc_tiles) == n
    h, w = brca_tiles[0].shape[:2]

    half_w = n * w + (n - 1) * gap_inner
    total_w = 2 * half_w + gap_center
    canvas = np.full((h, total_w, 3), bg, dtype=np.uint8)

    for i, tile in enumerate(brca_tiles):
        x = i * (w + gap_inner)
        canvas[:, x : x + w] = tile

    offset = half_w + gap_center
    for i, tile in enumerate(lihc_tiles):
        x = offset + i * (w + gap_inner)
        canvas[:, x : x + w] = tile

    return canvas


def make_panel(
    run_dir: Path,
    out_path: Path,
    n_tiles: int = 3,
    base_seed: int = 1000,
    guidance_scale: float = 1.0,
    n_steps: int = 20,
    device: Optional[torch.device] = None,
    checkpoint: Optional[str] = None,
) -> None:
    from src.evaluation.poc_fid import _load_config, _resolve_best_checkpoint, load_model, generate_batch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading config from {run_dir} …")
    conf = _load_config(run_dir)

    ckpt_path = _resolve_best_checkpoint(run_dir, checkpoint)
    print(f"Loading checkpoint {ckpt_path.name} …")
    model = load_model(conf, ckpt_path, device)

    conditioning_type: str = getattr(conf, "conditioning_type", "one_hot")
    print(f"Building conditioning vectors (type={conditioning_type}) …")
    cond_vecs = _build_cond_vecs(conf, device)
    brca_vec = cond_vecs["TCGA-BRCA"]
    lihc_vec = cond_vecs["TCGA-LIHC"]
    use_noise_cond = conditioning_type == "noise"

    cohorts = [("TCGA-BRCA", brca_vec), ("TCGA-LIHC", lihc_vec)]
    all_tiles: dict[str, list[np.ndarray]] = {}

    for cohort, cond_vec in cohorts:
        tiles: list[np.ndarray] = []
        for i in range(n_tiles):
            seed = base_seed + i
            print(f"  {cohort} tile {i + 1}/{n_tiles} (seed={seed}) …", flush=True)
            batch = generate_batch(
                model=model,
                cond_vec=cond_vec,
                batch_size=1,
                guidance_scale=guidance_scale,
                n_steps=n_steps,
                device=device,
                seed=seed,
                use_noise_cond=use_noise_cond,
            )
            tiles.append(batch[0])
        all_tiles[cohort] = tiles

    canvas = _build_panel_canvas(
        all_tiles["TCGA-BRCA"],
        all_tiles["TCGA-LIHC"],
        gap_inner=2,
        gap_center=8,
    )

    # ── figure with minimal labels ─────────────────────────────────────────
    tile_h, tile_w = canvas.shape[:2][0], all_tiles["TCGA-BRCA"][0].shape[1]
    n = n_tiles
    half_w_px = n * tile_w + (n - 1) * 2
    label_height_in = 0.22
    fig_w_in = canvas.shape[1] / 128
    fig_h_in = canvas.shape[0] / 128 + label_height_in

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=300)
    ax.imshow(canvas, interpolation="nearest", aspect="equal")
    ax.axis("off")

    # BRCA label (centred over left half, in axes fraction)
    brca_centre = (half_w_px / 2) / canvas.shape[1]
    lihc_centre = (half_w_px + 8 + half_w_px / 2) / canvas.shape[1]
    label_y = 1.01  # just above the image

    ax.text(brca_centre, label_y, "TCGA-BRCA",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7, fontfamily="sans-serif", color="0.2")
    ax.text(lihc_centre, label_y, "TCGA-LIHC",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7, fontfamily="sans-serif", color="0.2")

    # thin vertical separator between halves
    sep_x = (half_w_px + 4) / canvas.shape[1]  # midpoint of the 8-px gap
    ax.axvline(x=sep_x * canvas.shape[1], color="#cccccc", linewidth=0.5, ymin=0, ymax=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Generate BRCA vs LIHC panel for a PoC run")
    p.add_argument("--run-dir", type=Path, required=True,
                   help="GDA run dir (contains hparams.yaml + autoenc/ checkpoints)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG path (default: <experiment_dir>/panel.png)")
    p.add_argument("--n-tiles", type=int, default=3,
                   help="Tiles per cohort side (default: 3)")
    p.add_argument("--seed", type=int, default=1000,
                   help="Base noise seed; column i uses seed+i (default: 1000)")
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Explicit checkpoint path (auto-selects last.ckpt if omitted)")
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_absolute():
        run_dir = (_WS_ROOT / args.run_dir).resolve()

    if args.out is None:
        experiment_dir = run_dir.parent
        out_path = experiment_dir / "panel.png"
    else:
        out_path = args.out.resolve()

    device = torch.device(args.device) if args.device else None

    make_panel(
        run_dir=run_dir,
        out_path=out_path,
        n_tiles=args.n_tiles,
        base_seed=args.seed,
        guidance_scale=args.guidance_scale,
        n_steps=args.steps,
        device=device,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
