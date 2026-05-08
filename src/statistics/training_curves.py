#!/usr/bin/env python
"""Plot training statistics from a TensorBoard events file.

Usage:
    python -m src.statistics.training_curves --config src/config.yaml
    python -m src.statistics.training_curves --logdir experiments/my_run --output out/
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ── data loading ──────────────────────────────────────────────────────────────

def load_scalars(logdir: Path) -> dict[str, dict]:
    """Load all scalar series from the newest events file under logdir."""
    event_files = sorted(logdir.rglob("events.out.tfevents.*"),
                         key=lambda p: p.stat().st_mtime)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {logdir}")

    ea = EventAccumulator(str(event_files[-1].parent))
    ea.Reload()

    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = {
            "steps":  [e.step  for e in events],
            "values": [e.value for e in events],
        }
    return data


# ── helpers ───────────────────────────────────────────────────────────────────

def _smooth(values, window: int):
    if len(values) < window * 2:
        return None, None
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    return smoothed, window - 1  # offset into original steps array


def _print_summary(data: dict[str, dict]) -> None:
    print("\n── Training Statistics ──────────────────────────────")
    for tag, series in sorted(data.items()):
        values = series["values"]
        steps  = series["steps"]
        if not values:
            continue
        print(f"  {tag:30s}  n={len(values):6d}  "
              f"first={values[0]:.6f}  last={values[-1]:.6f}  "
              f"min={min(values):.6f}  max={max(values):.6f}  "
              f"step_range=[{steps[0]}, {steps[-1]}]")
    print("─────────────────────────────────────────────────────\n")


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_training_stats(data: dict[str, dict], output_dir: Path, show: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    has_val      = "loss/val"         in data
    has_shuffled = "loss/val_shuffled" in data
    has_gap      = "cond/gap"          in data
    lr_tag       = next((t for t in data if "lr" in t.lower()), None)

    n_cols = 2
    n_rows = 1 + (1 if has_val or has_shuffled else 0) + (1 if has_gap or lr_tag else 0)
    n_rows = max(n_rows, 2)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows),
                             squeeze=False)
    fig.suptitle("Training Statistics", fontsize=13, y=1.01)

    # ── row 0, col 0 : training loss (linear scale) ──────────────────────────
    ax = axes[0][0]
    if "loss" in data:
        steps  = np.array(data["loss"]["steps"])
        values = np.array(data["loss"]["values"])
        ax.plot(steps, values, alpha=0.25, color="steelblue", linewidth=0.6)
        window = max(10, len(values) // 50)
        sm, off = _smooth(values, window)
        if sm is not None:
            ax.plot(steps[off:], sm, color="steelblue", linewidth=1.8,
                    label=f"smooth (w={window})")
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.set_title("Training Loss (linear)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── row 0, col 1 : training loss (log scale) ─────────────────────────────
    ax = axes[0][1]
    if "loss" in data:
        steps  = np.array(data["loss"]["steps"])
        values = np.array(data["loss"]["values"])
        ax.plot(steps, values, alpha=0.25, color="steelblue", linewidth=0.6)
        if sm is not None:
            ax.plot(steps[off:], sm, color="steelblue", linewidth=1.8,
                    label=f"smooth (w={window})")
        ax.set_yscale("log")
        ax.set_xlabel("Step"); ax.set_ylabel("Loss (log)")
        ax.set_title("Training Loss (log scale)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    # ── row 1, col 0 : val loss ───────────────────────────────────────────────
    if has_val or has_shuffled:
        ax = axes[1][0]
        if has_val:
            s, v = data["loss/val"]["steps"], data["loss/val"]["values"]
            ax.plot(s, v, "o-", color="tomato", linewidth=2, markersize=7,
                    label="val loss")
            for xi, yi in zip(s, v):
                ax.annotate(f"{yi:.3f}", (xi, yi),
                            textcoords="offset points", xytext=(4, 4), fontsize=7)
        if has_shuffled:
            s, v = data["loss/val_shuffled"]["steps"], data["loss/val_shuffled"]["values"]
            ax.plot(s, v, "s--", color="orange", linewidth=1.5, markersize=5,
                    label="val loss (shuffled cond.)")
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.set_title("Validation Loss")
        ax.legend(); ax.grid(True, alpha=0.3)

        # ── row 1, col 1 : val loss overlaid on train loss ───────────────────
        ax = axes[1][1]
        if "loss" in data:
            steps  = np.array(data["loss"]["steps"])
            values = np.array(data["loss"]["values"])
            if sm is not None:
                ax.plot(steps[off:], sm, color="steelblue", linewidth=1.5,
                        alpha=0.7, label="train (smoothed)")
        if has_val:
            s, v = data["loss/val"]["steps"], data["loss/val"]["values"]
            ax.plot(s, v, "o-", color="tomato", linewidth=2, markersize=7,
                    label="val loss")
        ax.set_yscale("log")
        ax.set_xlabel("Step"); ax.set_ylabel("Loss (log)")
        ax.set_title("Train vs Val Loss (log)")
        ax.legend(); ax.grid(True, alpha=0.3, which="both")

    # ── row 2, col 0 : learning rate ─────────────────────────────────────────
    row2 = 1 + (1 if has_val or has_shuffled else 0)
    if row2 < n_rows:
        ax = axes[row2][0]
        if lr_tag:
            s, v = data[lr_tag]["steps"], data[lr_tag]["values"]
            ax.plot(s, v, color="seagreen", linewidth=1.8)
            ax.set_xlabel("Step"); ax.set_ylabel("LR")
            ax.set_title(f"Learning Rate  ({lr_tag})")
            ax.grid(True, alpha=0.3)
        else:
            ax.set_visible(False)

        # ── row 2, col 1 : cond/gap ──────────────────────────────────────────
        ax = axes[row2][1]
        if has_gap:
            s, v = data["cond/gap"]["steps"], data["cond/gap"]["values"]
            ax.plot(s, v, "o-", color="mediumpurple", linewidth=2, markersize=7)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
            for xi, yi in zip(s, v):
                ax.annotate(f"{yi:.2e}", (xi, yi),
                            textcoords="offset points", xytext=(4, 4), fontsize=7)
            ax.set_xlabel("Step"); ax.set_ylabel("Gap")
            ax.set_title("Conditioning Gap  (val − val_shuffled)")
            ax.grid(True, alpha=0.3)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    out_path = output_dir / "training_stats.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    if show:
        plt.show()
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config",  type=str, help="Path to config.yaml")
    parser.add_argument("--logdir",  type=str, help="Experiment directory")
    parser.add_argument("--output",  type=str, help="Output directory")
    parser.add_argument("--show",    action="store_true", help="Display plot")
    args = parser.parse_args()

    logdir = output_dir = None

    if args.config:
        config_path = Path(args.config).resolve()
        # repo_root is the directory containing run_pipeline.py (one level up from src/)
        repo_root = config_path.parent
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
            # experiments/, data/, dataframes/ live next to the repo root
            for prefix in ("experiments/", "data/", "dataframes/"):
                if normalized.startswith(prefix):
                    return (repo_root.parent / normalized).resolve()
            return (repo_root / normalized).resolve()

        if stats.get("logdir"):
            logdir = _resolve(stats["logdir"])
        if stats.get("output_dir"):
            output_dir = _resolve(stats["output_dir"])

    if args.logdir:
        logdir = Path(args.logdir).resolve()
    if args.output:
        output_dir = Path(args.output).resolve()

    if logdir is None:
        parser.error("Provide --logdir or --config with training_stats.logdir")
    if output_dir is None:
        output_dir = logdir / "training_stats"

    print(f"Loading events from: {logdir}")
    data = load_scalars(logdir)

    _print_summary(data)
    plot_training_stats(data, output_dir, show=args.show)


if __name__ == "__main__":
    main()
