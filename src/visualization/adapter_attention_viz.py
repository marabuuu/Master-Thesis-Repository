"""
Cross-attention visualizations for GDA v13.

Three views, in increasing scientific importance:

  Idea 2 — Token gradient importance  (works on current checkpoint, no code change needed)
      ∂‖Δε‖/∂g_tokens  gradient norm per token, grouped by subtype.
      If all 8 tokens have similar importance → encoder collapse or adapter not using tokens.
      If some tokens dominate → adapter has learned to route subtype signal through specific tokens.

  Idea 1 — Spatial attention heatmaps  (requires adapter.capture_attention())
      For a chosen CA layer, overlays the per-token attention weight map on the tile.
      Shows which spatial regions of the histology attend most to each genomic token.

  Idea 3 — Per-subtype attention profiles  (requires adapter.capture_attention())
      Mean attention weight per token per subtype, across multiple CA layers.
      The most thesis-relevant plot: if profiles differ across subtypes the adapter
      has learned subtype-specific routing through the cross-attention.

Usage
-----
    # Idea 2 only (works on any checkpoint right now):
    python -m src.visualization.adapter_attention_viz \\
        --checkpoint experiments/20260526_poc_brca_lihc_gda_v2/gda/autoenc/last.ckpt \\
        --hparams   experiments/20260526_poc_brca_lihc_gda_v2/gda/hparams.yaml \\
        --out       experiments/20260526_poc_brca_lihc_gda_v2/attention_viz/ \\
        --idea 2

    # All three ideas:
    python -m src.visualization.adapter_attention_viz \\
        --checkpoint experiments/20260526_poc_brca_lihc_gda_v2/gda/autoenc/last.ckpt \\
        --hparams   experiments/20260526_poc_brca_lihc_gda_v2/gda/hparams.yaml \\
        --out       experiments/20260526_poc_brca_lihc_gda_v2/attention_viz/ \\
        --idea all

    # Spatial heatmaps at full resolution (ca0) instead of bottleneck (mid_ca):
    python -m src.visualization.adapter_attention_viz ... --idea 1 --attn_layer ca0
"""

from __future__ import annotations

import argparse
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Subtype colours
# Known entries cover PAM50 (for BRCA subtype runs) and common tissue labels.
# Any other subtype gets an automatically assigned colour from the fallback
# palette so the plots remain readable regardless of the label vocabulary.
# ---------------------------------------------------------------------------

SUBTYPE_COLORS: dict[str, str] = {
    # PAM50 subtypes
    "Basal":   "#E74C3C",
    "Her2":    "#9B59B6",
    "LumA":    "#3498DB",
    "LumB":    "#1ABC9C",
    "Normal":  "#2ECC71",
    # Tissue-type labels — bare and TCGA-prefixed forms (e.g. BRCA vs LIHC PoC runs)
    "BRCA":      "#E74C3C",
    "LIHC":      "#3498DB",
    "TCGA-BRCA": "#E74C3C",
    "TCGA-LIHC": "#3498DB",
    "unknown": "#95A5A6",
}

_FALLBACK_PALETTE: list[str] = [
    "#E67E22", "#8E44AD", "#16A085", "#C0392B", "#2980B9",
    "#27AE60", "#D35400", "#7F8C8D", "#F39C12", "#1A5276",
]

_fallback_cache: dict[str, str] = {}


def _subtype_color(name: str) -> str:
    """Return the colour for *name*, falling back to a stable auto-assigned colour."""
    if name in SUBTYPE_COLORS:
        return SUBTYPE_COLORS[name]
    if name not in _fallback_cache:
        _fallback_cache[name] = _FALLBACK_PALETTE[len(_fallback_cache) % len(_FALLBACK_PALETTE)]
    return _fallback_cache[name]

# ---------------------------------------------------------------------------
# Model + data loading
# ---------------------------------------------------------------------------

def _load_model(ckpt_path: str, hparams_path: str, device: str):
    """Load GDALitModel from a Lightning checkpoint and a hparams.yaml."""
    import yaml

    try:
        from src.genomic_adapter.config import GDAConfig
        from src.genomic_adapter.model import GDALitModel
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.genomic_adapter.config import GDAConfig
        from src.genomic_adapter.model import GDALitModel

    with open(hparams_path) as f:
        hp = yaml.load(f, Loader=yaml.FullLoader)

    conf = GDAConfig.from_dict(hp)
    # model_name is an enum not serialised by Lightning's hparams.yaml — set it explicitly.
    if conf.model_name is None:
        from mopadi.configs.choices import ModelName
        conf.model_name = ModelName.beatgans_autoenc
    conf.make_model_conf()

    model = GDALitModel(conf)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        warnings.warn(
            f"{len(missing)} missing keys when loading checkpoint "
            f"(likely EMA buffers — safe to ignore): {missing[:4]}..."
        )

    return model.to(device).eval(), conf


def _load_tiles_per_subtype(
    conf,
    n_per_subtype: int,
    split: str,
) -> Dict[str, List[dict]]:
    """Return up to n_per_subtype items per subtype from the dataset."""
    from src.genomic_adapter.dataset import ZipTilesWithGenomicFeatures

    dataset = ZipTilesWithGenomicFeatures(
        zip_dir=conf.zip_dir,
        genomic_h5_dir=conf.genomic_feature_dir,
        patient_splits_path=conf.patient_splits_path,
        split=split,
        do_normalize=True,
        do_resize=False,
        img_size=conf.img_size,
    )

    # Determine every subtype that actually has tiles in this split so the
    # early-exit below doesn't fire after the first subtype fills up.
    # _genomic_cache keys = patients with both tiles and H5 files in this split.
    expected_subtypes = {
        dataset._subtype_map.get(pid, "unknown")
        for pid in dataset._genomic_cache
    }

    rng = np.random.default_rng(seed=0)
    indices = rng.permutation(len(dataset)).tolist()

    by_subtype: Dict[str, List[dict]] = defaultdict(list)
    for idx in indices:
        item = dataset[idx]
        sub = item.get("subtype", "unknown")
        if len(by_subtype[sub]) < n_per_subtype:
            by_subtype[sub].append(item)
        if (
            expected_subtypes <= by_subtype.keys()
            and all(len(v) >= n_per_subtype for v in by_subtype.values())
        ):
            break

    return dict(by_subtype)


def _batch(items: List[dict], device: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    imgs   = torch.stack([it["img"]  for it in items]).to(device)           # (B, 3, H, W)
    feats  = torch.stack([it["feat"] for it in items]).to(device, dtype=torch.float32)  # (B, D)
    subs   = [it.get("subtype", "unknown") for it in items]
    return imgs, feats, subs


def _q_sample(imgs: torch.Tensor, t_val: int, T: int = 1000) -> torch.Tensor:
    """Add noise at a fixed timestep using the linear DDPM schedule."""
    betas        = torch.linspace(1e-4, 0.02, T, dtype=torch.float32, device=imgs.device)
    ac           = torch.cumprod(1.0 - betas, dim=0)
    sqrt_ac      = ac[t_val].sqrt()
    sqrt_onemac  = (1.0 - ac[t_val]).sqrt()
    return sqrt_ac * imgs + sqrt_onemac * torch.randn_like(imgs)


# ---------------------------------------------------------------------------
# Idea 2 — Token gradient importance
# ---------------------------------------------------------------------------

def plot_token_gradient_importance(
    model,
    tiles_per_subtype: Dict[str, List[dict]],
    t_val: int = 500,
    device: str = "cuda",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Compute ∂‖Δε‖/∂g_tokens and plot the L2-norm of the gradient per token.

    A flat bar chart (all tokens equal) means the adapter is not differentially
    routing genomic information through individual tokens.
    Variation across tokens — especially subtype-specific variation — indicates
    the encoder has learned to pack subtype signal into specific token slots.
    """
    importance_rows: List[np.ndarray] = []
    subtype_labels:  List[str]        = []

    for subtype, items in sorted(tiles_per_subtype.items()):
        imgs, feats, subs = _batch(items, device)
        B = imgs.shape[0]

        x_t = _q_sample(imgs.detach(), t_val)
        t   = torch.full((B,), t_val, device=device, dtype=torch.long)

        model.zero_grad()
        with torch.enable_grad():
            g_tokens = model.genomic_encoder(feats)   # (B, n_tokens, token_dim)
            g_tokens.retain_grad()
            delta_eps = model.adapter(x_t, t, g_tokens)
            delta_eps.flatten(1).norm(dim=1).sum().backward()

        # (B, n_tokens, token_dim) → (B, n_tokens)
        imp = g_tokens.grad.detach().norm(dim=-1).cpu().numpy()
        importance_rows.append(imp)
        subtype_labels.extend(subs)

    importance = np.concatenate(importance_rows, axis=0)   # (N, n_tokens)
    n_tokens   = importance.shape[1]
    tok_x      = np.arange(n_tokens)
    tok_labels = [f"T{i}" for i in range(n_tokens)]

    subtypes_sorted = sorted(set(subtype_labels))
    n_sub = len(subtypes_sorted)

    fig, axes = plt.subplots(1, n_sub + 1, figsize=(3.0 * (n_sub + 1), 3.5), sharey=True)
    ymax = importance.max() * 1.15

    for ax, sub in zip(axes, subtypes_sorted):
        mask  = np.array([s == sub for s in subtype_labels])
        vals  = importance[mask].mean(axis=0)
        ax.bar(tok_labels, vals, color=_subtype_color(sub), width=0.7)
        ax.set_title(sub, fontsize=10)
        ax.set_ylim(0, ymax)
        ax.tick_params(axis="x", labelsize=8)

    axes[-1].bar(tok_labels, importance.mean(axis=0), color="#888", width=0.7)
    axes[-1].set_title("All", fontsize=10)
    axes[-1].tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("‖∂‖Δε‖/∂token‖", fontsize=9)

    fig.suptitle(f"Token Gradient Importance  (t = {t_val})", fontsize=12)
    fig.tight_layout()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")

    return fig


# ---------------------------------------------------------------------------
# Idea 1 — Spatial attention heatmaps
# ---------------------------------------------------------------------------

def plot_spatial_attention_heatmaps(
    model,
    tiles_per_subtype: Dict[str, List[dict]],
    layer: str = "mid_ca",
    t_val: int = 500,
    device: str = "cuda",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """
    For one representative tile per subtype, overlay the per-token
    attention weight map from a chosen CA layer on the original tile.

    Rows = subtypes.  Columns = original tile + one column per token.

    Recommended layers (img_size=512):
      mid_ca : 128×128 spatial — good balance of resolution and memory (~2 MB)
      ca0    : 512×512 spatial — full resolution, highest memory (~134 MB per tile)
    """
    n_tokens       = model.conf.adapter_n_tokens
    subtypes_sorted = sorted(tiles_per_subtype.keys())
    n_rows, n_cols = len(subtypes_sorted), n_tokens + 1   # +1 for original

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.0 * n_cols, 2.2 * n_rows),
        squeeze=False,
    )

    for row, subtype in enumerate(subtypes_sorted):
        item = tiles_per_subtype[subtype][0]   # one tile per subtype
        imgs, feats, _ = _batch([item], device)

        x_t = _q_sample(imgs.detach(), t_val)
        t   = torch.full((1,), t_val, device=device, dtype=torch.long)

        with torch.no_grad():
            g_tokens = model.genomic_encoder(feats)
            with model.adapter.capture_attention():
                model.adapter(x_t, t, g_tokens)
            weights_dict = model.adapter.get_attn_weights()

        if layer not in weights_dict:
            raise ValueError(
                f"Layer '{layer}' not captured. "
                f"Available: {list(weights_dict.keys())}"
            )

        # w: (1, n_heads, HW, n_tokens) → mean heads → (HW, n_tokens)
        w   = weights_dict[layer][0].mean(dim=0).cpu().numpy()   # (HW, n_tokens)
        HW  = w.shape[0]
        H   = W = int(math.isqrt(HW))
        assert H * W == HW, f"Spatial dim {HW} is not square"
        w_spatial = w.reshape(H, W, n_tokens)                    # (H, W, n_tokens)

        img_np = imgs[0].cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * 0.5 + 0.5).clip(0.0, 1.0)            # [-1,1] → [0,1]
        tile_H, tile_W = img_np.shape[:2]

        # Original tile column
        axes[row, 0].imshow(img_np)
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("Original", fontsize=8)
        axes[row, 0].set_ylabel(subtype, fontsize=8, rotation=0, labelpad=44, va="center")

        # Per-token heatmap columns
        for tok in range(n_tokens):
            hmap = w_spatial[:, :, tok]
            hmap_up = F.interpolate(
                torch.from_numpy(hmap)[None, None].float(),
                size=(tile_H, tile_W),
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()

            ax = axes[row, tok + 1]
            ax.imshow(img_np, alpha=0.55)
            ax.imshow(hmap_up, cmap="inferno", alpha=0.55,
                      vmin=hmap_up.min(), vmax=hmap_up.max())
            ax.axis("off")
            if row == 0:
                ax.set_title(f"T{tok}", fontsize=8)

    fig.suptitle(
        f"Spatial Attention Heatmaps  (layer = {layer},  t = {t_val})",
        fontsize=11,
    )
    fig.tight_layout()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Saved: {out_path}")

    return fig


# ---------------------------------------------------------------------------
# Idea 3 — Per-subtype attention profiles
# ---------------------------------------------------------------------------

def plot_subtype_attention_profiles(
    model,
    tiles_per_subtype: Dict[str, List[dict]],
    layers: Optional[List[str]] = None,
    t_val: int = 500,
    device: str = "cuda",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """
    For each CA layer, plot the mean attention weight per genomic token broken
    down by subtype.

    This is the most thesis-relevant plot:
      • Identical profiles across subtypes → adapter is NOT learning subtype-specific
        routing (genomic tokens are treated the same regardless of subtype).
      • Different profiles → the adapter is routing different information through
        different tokens depending on the sample's subtype.

    Layers default to the two encoder levels + bottleneck + first decoder level,
    which give a coarse-to-fine view of where subtype information enters.
    """
    if layers is None:
        layers = ["ca0", "ca2", "mid_ca", "ca_dec1"]

    n_tokens        = model.conf.adapter_n_tokens
    subtypes_sorted = sorted(tiles_per_subtype.keys())

    # profiles[subtype][layer] = mean attention weight vector, shape (n_tokens,)
    profiles: Dict[str, Dict[str, np.ndarray]] = {s: {} for s in subtypes_sorted}

    for subtype, items in sorted(tiles_per_subtype.items()):
        imgs, feats, _ = _batch(items, device)
        B = imgs.shape[0]

        x_t = _q_sample(imgs.detach(), t_val)
        t   = torch.full((B,), t_val, device=device, dtype=torch.long)

        with torch.no_grad():
            g_tokens = model.genomic_encoder(feats)
            with model.adapter.capture_attention():
                model.adapter(x_t, t, g_tokens)
            weights_dict = model.adapter.get_attn_weights()

        for lyr in layers:
            if lyr not in weights_dict:
                continue
            # (B, n_heads, HW, n_tokens) → mean over heads & spatial → (B, n_tokens)
            w = weights_dict[lyr].mean(dim=(1, 2)).cpu().numpy()   # (B, n_tokens)
            profiles[subtype][lyr] = w.mean(axis=0)               # (n_tokens,)

    avail_layers = [lyr for lyr in layers if any(lyr in profiles[s] for s in subtypes_sorted)]
    n_layers     = len(avail_layers)

    fig, axes = plt.subplots(1, n_layers, figsize=(4.0 * n_layers, 3.5))
    if n_layers == 1:
        axes = [axes]

    tok_x  = np.arange(n_tokens)
    width  = 0.8 / len(subtypes_sorted)

    for ax, lyr in zip(axes, avail_layers):
        for sub_idx, sub in enumerate(subtypes_sorted):
            if lyr not in profiles[sub]:
                continue
            offset = (sub_idx - len(subtypes_sorted) / 2 + 0.5) * width
            ax.bar(
                tok_x + offset,
                profiles[sub][lyr],
                width=width,
                color=_subtype_color(sub),
                label=sub,
            )
        ax.set_xticks(tok_x)
        ax.set_xticklabels([f"T{i}" for i in range(n_tokens)], fontsize=8)
        ax.set_title(lyr, fontsize=10)
        ax.set_ylabel("Mean attention weight", fontsize=9)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SUBTYPE_COLORS.get(s, "#aaa"))
        for s in subtypes_sorted
    ]
    fig.legend(
        handles, subtypes_sorted,
        loc="lower center", ncol=len(subtypes_sorted),
        bbox_to_anchor=(0.5, -0.08), fontsize=9,
    )
    fig.suptitle(f"Per-Subtype Attention Profiles  (t={t_val})", fontsize=12)
    fig.tight_layout()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_REPO_ROOT       = Path(__file__).resolve().parents[2]   # .../Master-Thesis-Repository
_EXP_BASE        = _REPO_ROOT.parent / "experiments"     # .../genhist/experiments
_RUN             = "20260526_poc_brca_lihc_gda_v2"

_DEFAULT_CKPT    = str(_EXP_BASE / _RUN / "gda/autoenc/last.ckpt")
_DEFAULT_HPARAMS = str(_EXP_BASE / _RUN / "gda/hparams.yaml")
_DEFAULT_OUT     = str(_EXP_BASE / _RUN / "attention_viz")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GDA cross-attention visualizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint",  default=_DEFAULT_CKPT)
    parser.add_argument("--hparams",     default=_DEFAULT_HPARAMS)
    parser.add_argument("--out",         default=_DEFAULT_OUT)
    parser.add_argument(
        "--idea",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which visualization to produce. '2' works on any checkpoint; "
             "'1' and '3' require the attention-capture code in adapter.py.",
    )
    parser.add_argument("--n_tiles",    type=int,   default=4,
                        help="Tiles per subtype")
    parser.add_argument("--t",          type=int,   default=500,
                        help="Diffusion timestep for forward pass (0–999)")
    parser.add_argument("--attn_layer", default="mid_ca",
                        help="CA layer for Idea 1 heatmaps "
                             "(ca0/ca1/ca2/mid_ca/ca_dec2/ca_dec1)")
    parser.add_argument("--split",      default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading model from  {args.checkpoint}")
    model, conf = _load_model(args.checkpoint, args.hparams, args.device)

    print(f"[2/4] Loading {args.n_tiles} tiles/subtype from {args.split} split")
    tiles = _load_tiles_per_subtype(conf, n_per_subtype=args.n_tiles, split=args.split)
    print(f"      Subtypes found: {sorted(tiles.keys())}")

    run_1   = args.idea in ("1", "all")
    run_2   = args.idea in ("2", "all")
    run_3   = args.idea in ("3", "all")

    if run_2:
        print("[3/4] Idea 2 — token gradient importance")
        plot_token_gradient_importance(
            model, tiles,
            t_val=args.t, device=args.device,
            out_path=out / "token_gradient_importance.png",
        )

    if run_1:
        print(f"[3/4] Idea 1 — spatial attention heatmaps  (layer={args.attn_layer})")
        plot_spatial_attention_heatmaps(
            model, tiles,
            layer=args.attn_layer,
            t_val=args.t, device=args.device,
            out_path=out / f"spatial_attention_heatmaps_{args.attn_layer}.png",
        )

    if run_3:
        print("[3/4] Idea 3 — per-subtype attention profiles")
        plot_subtype_attention_profiles(
            model, tiles,
            t_val=args.t, device=args.device,
            out_path=out / "subtype_attention_profiles.png",
        )

    print(f"[4/4] Done. Figures saved to {out}/")


if __name__ == "__main__":
    main()
