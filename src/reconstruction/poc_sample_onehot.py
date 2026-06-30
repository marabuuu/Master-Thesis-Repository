"""Generate small diagnostic samples for the one-hot conditioning signal.

Produces paired unconditional/CFG-guided images for two fixed one-hot vectors
(BRCA -> e1, LIHC -> e2). The script reuses existing helpers in
`sample_generated_tiles.py` to resolve checkpoints and load the trained model.

Example:
    python -m src.reconstruction.poc_sample_onehot \
        --run-dir experiments/20260603_poc_128_1hot/gda \
        --checkpoint best.ckpt --out-dir /tmp/onehot_test --n-per-class 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from src.reconstruction.poc_sample_tiles import (
    _build_config,
    _resolve_checkpoint,
    _load_model,
    _sample_cfg_backbone_pair,
    _save_pair_image,
)
from src.model_training.dataset import _make_orthogonal_binary_codes


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample one-hot conditioned pairs")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-per-class", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    # Default output directory lives inside the run directory so samples are
    # colocated with the experiment artifacts.
    out_dir = (args.out_dir or (run_dir / "onehot_test")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    conf = _build_config(run_dir)
    ckpt = _resolve_checkpoint(run_dir, args.checkpoint)
    model, _ = _load_model(conf, ckpt)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()

    feat_dim = conf.feat_dim
    normalize_feats = getattr(conf, "normalize_feats", True)
    codes = _make_orthogonal_binary_codes(feat_dim, normalize=normalize_feats)
    e1 = codes["TCGA-BRCA"]
    e2 = codes["TCGA-LIHC"]
    print(
        f"Using orthogonal synthetic codes for sampling: dot={torch.dot(e1, e2).item():.1f}, "
        f"BRCA_norm={e1.norm().item():.1f}, LIHC_norm={e2.norm().item():.1f}"
    )

    def save_for_vector(vec: torch.Tensor, name_prefix: str, count: int) -> None:
        for i in range(count):
            uncond, cond = _sample_cfg_backbone_pair(model, vec, args.guidance_scale, args.steps, device)
            out_pair = out_dir / f"{name_prefix}__{i:02d}__pair.png"
            _save_pair_image(uncond, cond, out_pair)

    save_for_vector(e1, "BRCA_e1", args.n_per_class)
    save_for_vector(e2, "LIHC_e2", args.n_per_class)

    print(f"Wrote {2 * args.n_per_class} paired images to {out_dir}")


if __name__ == "__main__":
    main()
