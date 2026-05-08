#!/usr/bin/env python
"""
Inspect AdaGN conditioning strength in a GenomicDiffusion checkpoint.

For each ResBlock the model has:
  emb_layers     : time embedding   → scale + shift  (2 * C_out output)
  cond_emb_layers: genomic embedding → scale only    (C_out output)

We compare their Frobenius norms layer-by-layer to check whether the
genomic pathway is learning anything relative to the time pathway.

We also compare across multiple checkpoints to see if the norms grow over
training — flat norms would indicate the genomic pathway is not receiving
gradient signal.

Usage:
    cd Master-Thesis-Repository
    python -m src.statistics.inspect_conditioning --config src/config.yaml
    python -m src.statistics.inspect_conditioning --ckpt /path/to/checkpoint.ckpt
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml


# ── helpers ───────────────────────────────────────────────────────────────────

def _frob(tensor: torch.Tensor) -> float:
    return tensor.float().norm().item()


def load_conditioning_norms(ckpt_path: Path, model_prefix: str = "model") -> dict:
    """
    Load a checkpoint and return per-ResBlock Frobenius norms for:
      - cond_emb_layers (genomic)
      - emb_layers      (time)

    Returns dict with keys: 'genomic', 'time', 'layer_names'
    """
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = sd.get("state_dict", sd)

    genomic_norms = {}   # key → norm
    time_norms    = {}

    for k, v in state.items():
        if not k.startswith(model_prefix + "."):
            continue
        # only weight tensors (skip biases for the norm comparison)
        if not k.endswith(".weight"):
            continue
        if "cond_emb_layers" in k:
            # strip prefix and ".1.weight" suffix → ResBlock path
            block_path = k[len(model_prefix) + 1:].replace(".cond_emb_layers.1.weight", "")
            genomic_norms[block_path] = _frob(v)
        elif "emb_layers" in k and "cond" not in k:
            block_path = k[len(model_prefix) + 1:].replace(".emb_layers.1.weight", "")
            time_norms[block_path] = _frob(v)

    # Align keys
    common = sorted(set(genomic_norms) & set(time_norms))
    return {
        "layer_names": common,
        "genomic":     [genomic_norms[k] for k in common],
        "time":        [time_norms[k]    for k in common],
    }


def _layer_group(name: str) -> str:
    """Classify a ResBlock path into a human-readable group."""
    if "input_blocks" in name:
        return f"enc.{name.split('.')[1]:>2}"
    elif "middle_block" in name:
        return "middle"
    elif "output_blocks" in name:
        return f"dec.{name.split('.')[1]:>2}"
    return name


def _print_summary(norms: dict, label: str = "") -> None:
    g = np.array(norms["genomic"])
    t = np.array(norms["time"])
    ratio = g / (t + 1e-12)
    print(f"\n── {label} ──────────────────────────────────────────────")
    print(f"  ResBlocks with cond_emb_layers : {len(g)}")
    print(f"  Genomic norm   : mean={g.mean():.4f}  std={g.std():.4f}  "
          f"min={g.min():.4f}  max={g.max():.4f}")
    print(f"  Time norm      : mean={t.mean():.4f}  std={t.std():.4f}  "
          f"min={t.min():.4f}  max={t.max():.4f}")
    print(f"  Genomic/Time   : mean={ratio.mean():.3f}  "
          f"min={ratio.min():.3f}  max={ratio.max():.3f}")
    top5 = np.argsort(ratio)[-5:][::-1]
    print(f"  Top-5 blocks by ratio:")
    for i in top5:
        print(f"    {norms['layer_names'][i]:55s}  ratio={ratio[i]:.3f}  "
              f"genomic={g[i]:.4f}  time={t[i]:.4f}")


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_conditioning_strength(
    ckpt_norms: list[tuple[str, dict]],
    output_dir: Path,
    show: bool = False,
) -> None:
    """
    Parameters
    ----------
    ckpt_norms : list of (label, norms_dict) pairs, one per checkpoint
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── figure layout ─────────────────────────────────────────────────────────
    n_ckpts = len(ckpt_norms)
    fig = plt.figure(figsize=(16, 4 + 4 * n_ckpts))
    gs = fig.add_gridspec(n_ckpts + 1, 2, hspace=0.45, wspace=0.35)

    colors = plt.cm.tab10(np.linspace(0, 0.6, max(n_ckpts, 2)))

    # ── per-checkpoint: bar charts of layer-wise norms ────────────────────────
    for row, (label, norms) in enumerate(ckpt_norms):
        g = np.array(norms["genomic"])
        t = np.array(norms["time"])
        x = np.arange(len(g))

        # Left: absolute norms
        ax = fig.add_subplot(gs[row, 0])
        ax.bar(x - 0.2, t, width=0.4, label="time emb", color="steelblue", alpha=0.85)
        ax.bar(x + 0.2, g, width=0.4, label="genomic emb", color="tomato", alpha=0.85)
        ax.set_title(f"{label} — per-block ‖W‖_F", fontsize=9)
        ax.set_xlabel("ResBlock index"); ax.set_ylabel("Frobenius norm")
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", labelsize=6)

        # Right: ratio genomic/time
        ax2 = fig.add_subplot(gs[row, 1])
        ratio = g / (t + 1e-12)
        bar_colors = ["tomato" if r > 0.5 else "salmon" for r in ratio]
        ax2.bar(x, ratio, color=bar_colors, alpha=0.85)
        ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8,
                    label="ratio = 1 (equal strength)")
        ax2.axhline(ratio.mean(), color="darkred", linestyle=":", linewidth=1.2,
                    label=f"mean = {ratio.mean():.3f}")
        ax2.set_title(f"{label} — genomic/time ratio", fontsize=9)
        ax2.set_xlabel("ResBlock index"); ax2.set_ylabel("Ratio")
        ax2.legend(fontsize=7)
        ax2.tick_params(axis="x", labelsize=6)

    # ── bottom row: norm growth over training ─────────────────────────────────
    if n_ckpts > 1:
        labels   = [l for l, _ in ckpt_norms]
        g_means  = [np.mean(n["genomic"]) for _, n in ckpt_norms]
        t_means  = [np.mean(n["time"])    for _, n in ckpt_norms]
        ratios   = [gm / (tm + 1e-12) for gm, tm in zip(g_means, t_means)]

        ax3 = fig.add_subplot(gs[n_ckpts, 0])
        ax3.plot(labels, g_means, "o-", color="tomato", linewidth=2, markersize=7,
                 label="genomic ‖W‖_F mean")
        ax3.plot(labels, t_means, "s--", color="steelblue", linewidth=2, markersize=7,
                 label="time ‖W‖_F mean")
        ax3.set_title("Mean norm growth over training")
        ax3.set_ylabel("Mean Frobenius norm")
        ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[n_ckpts, 1])
        ax4.plot(labels, ratios, "o-", color="darkred", linewidth=2, markersize=7)
        ax4.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax4.set_title("Mean genomic/time ratio over training")
        ax4.set_ylabel("Ratio")
        ax4.grid(True, alpha=0.3)

    out_path = output_dir / "conditioning_strength.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    if show:
        plt.show()
    plt.close()


def plot_layer_groups(norms: dict, label: str, output_dir: Path, show: bool = False) -> None:
    """Group ResBlocks into encoder / middle / decoder and show per-group ratio."""
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for name, g, t in zip(norms["layer_names"], norms["genomic"], norms["time"]):
        grp = _layer_group(name)
        groups[grp].append(g / (t + 1e-12))

    grp_names  = sorted(groups.keys())
    grp_means  = [np.mean(groups[g]) for g in grp_names]
    grp_stds   = [np.std(groups[g])  for g in grp_names]

    fig, ax = plt.subplots(figsize=(max(8, len(grp_names) * 0.5), 4))
    x = np.arange(len(grp_names))
    ax.bar(x, grp_means, yerr=grp_stds, color="tomato", alpha=0.8,
           capsize=3, error_kw={"linewidth": 0.8})
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(grp_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Genomic / Time norm ratio")
    ax.set_title(f"{label} — conditioning ratio by network group")
    ax.grid(True, axis="y", alpha=0.3)

    out_path = output_dir / "conditioning_by_group.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    if show:
        plt.show()
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, help="Path to config.yaml")
    parser.add_argument("--ckpt",   type=str, nargs="+", help="Checkpoint path(s)")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--model",  type=str, default="model",
                        help="State-dict prefix for the model (default: 'model'; "
                             "use 'ema_model' for the EMA copy)")
    parser.add_argument("--show",   action="store_true")
    args = parser.parse_args()

    ckpt_paths = []
    output_dir = None

    if args.config:
        config_path = Path(args.config).resolve()
        repo_root   = config_path.parent
        if not (repo_root / "run_pipeline.py").exists():
            repo_root = repo_root.parent
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        stats = cfg.get("training_stats", {})

        def _resolve(raw: str) -> Path:
            p = Path(raw)
            if p.is_absolute():
                return p
            normalized = raw[2:] if raw.startswith("./") else raw
            for prefix in ("experiments/", "data/", "dataframes/"):
                if normalized.startswith(prefix):
                    return (repo_root.parent / normalized).resolve()
            return (repo_root / normalized).resolve()

        if stats.get("logdir"):
            ckpt_dir = _resolve(stats["logdir"]) / "genomic_training" / "autoenc"
            if ckpt_dir.exists():
                found = sorted(ckpt_dir.glob("best-composite-step*.ckpt"),
                               key=lambda p: int(p.stem.split("step")[-1]))
                ckpt_paths = found
        if stats.get("output_dir"):
            output_dir = _resolve(stats["output_dir"])

    if args.ckpt:
        ckpt_paths = [Path(p) for p in args.ckpt]
    if args.output:
        output_dir = Path(args.output)
    if output_dir is None:
        output_dir = Path(".")

    if not ckpt_paths:
        parser.error("No checkpoints found. Provide --ckpt or --config with training_stats.logdir")

    print(f"Analysing {len(ckpt_paths)} checkpoint(s):")
    for p in ckpt_paths:
        print(f"  {p.name}")

    ckpt_norms = []
    for ckpt_path in ckpt_paths:
        label = ckpt_path.stem.replace("best-composite-", "")
        norms = load_conditioning_norms(ckpt_path, model_prefix=args.model)
        _print_summary(norms, label=f"{label}  [{args.model}]")
        ckpt_norms.append((label, norms))

    plot_conditioning_strength(ckpt_norms, output_dir, show=args.show)
    # Detailed group plot for the latest checkpoint
    plot_layer_groups(ckpt_norms[-1][1], ckpt_norms[-1][0], output_dir, show=args.show)


if __name__ == "__main__":
    main()
