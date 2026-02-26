#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Curves Visualization

Plot training loss curves from:
1. Checkpoint files (.pt) - extracts 'epoch' and 'loss' fields
2. Log files (.log) - parses training output
3. Manual data input

Usage:
    # From checkpoints directory
    python training_curves.py --checkpoint-dir ./experiments/training_run/
    
    # From log file
    python training_curves.py --log-file ./slurm/logs/12345.out
    
    # Compare multiple runs
    python training_curves.py --checkpoint-dir ./run1 ./run2 --labels "Run 1" "Run 2"
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

if TYPE_CHECKING:
    import torch

# Import torch at runtime inside functions that need it. This keeps the script
# usable for log-only plotting when PyTorch isn't installed while still
# informing static type checkers about `torch`.


def parse_checkpoints(checkpoint_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
    """
    Extract training history from checkpoint files.
    
    Returns dict with keys like 'loss', 'mean', 'var', 'diversity'
    mapping to list of (epoch, value) tuples.
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Import torch at runtime to allow the module to be imported without
    # PyTorch when only plotting logs. Static type checkers still see `torch`
    # from the `TYPE_CHECKING` import above.
    try:
        import torch  # type: ignore
    except Exception:
        raise ImportError("PyTorch is required to parse checkpoint files")
    
    # Find all checkpoint files
    ckpt_files = sorted(checkpoint_dir.glob("*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .pt files found in {checkpoint_dir}")
    
    history = {}
    
    for ckpt_path in ckpt_files:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            epoch = ckpt.get("epoch", None)
            
            if epoch is None:
                # Try to extract from filename like "epoch010.pt"
                match = re.search(r'epoch(\d+)', ckpt_path.stem)
                if match:
                    epoch = int(match.group(1))
                else:
                    continue
            
            # Extract loss values
            if "loss" in ckpt:
                if "loss" not in history:
                    history["loss"] = []
                history["loss"].append((epoch, float(ckpt["loss"])))
            
            # For projection head checkpoints with component losses
            for key in ["loss_mean", "loss_var", "loss_diversity"]:
                if key in ckpt:
                    if key not in history:
                        history[key] = []
                    history[key].append((epoch, float(ckpt[key])))
                    
        except Exception as e:
            print(f"Warning: Could not parse {ckpt_path}: {e}")
            continue
    
    # Sort by epoch
    for key in history:
        history[key] = sorted(history[key], key=lambda x: x[0])
    
    return history


def parse_log_file(log_path: Path) -> Dict[str, List[Tuple[int, float]]]:
    """
    Parse training log file for loss values.
    
    Supports formats like:
    - "Epoch 5/50 | Time: 12.3s | Loss: 0.028057"
    - "Losses: total=0.026, mean=0.012, var=0.008, diversity=0.006"
    """
    log_path = Path(log_path)
    
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    
    history = {"loss": [], "mean": [], "var": [], "diversity": []}
    
    with open(log_path, "r") as f:
        content = f.read()
    
    # Pattern for diffusion fine-tuning logs
    # "Epoch 5/50 | Time: 12.3s | Loss: 0.028057"
    epoch_pattern = re.compile(
        r"Epoch\s+(\d+)/\d+.*?Loss:\s*([\d.]+)",
        re.IGNORECASE
    )
    
    for match in epoch_pattern.finditer(content):
        epoch = int(match.group(1))
        loss = float(match.group(2))
        history["loss"].append((epoch, loss))
    
    # Pattern for projection head logs
    # "Losses: total=0.026, mean=0.012, var=0.008, diversity=0.006"
    component_pattern = re.compile(
        r"Epoch\s+(\d+)/\d+.*?"
        r"total=([\d.]+).*?mean=([\d.]+).*?var=([\d.]+).*?diversity=([\d.]+)",
        re.IGNORECASE | re.DOTALL
    )
    
    for match in component_pattern.finditer(content):
        epoch = int(match.group(1))
        history["loss"].append((epoch, float(match.group(2))))
        history["mean"].append((epoch, float(match.group(3))))
        history["var"].append((epoch, float(match.group(4))))
        history["diversity"].append((epoch, float(match.group(5))))
    # Fallback parser (line-oriented) to support tqdm/progressbar logging
    # Example lines:
    # Epoch 1: 100%|...| 2581/2581 [24:51<00:00, ..., loss_step=0.045, loss_epoch=0.0384]
    curr_epoch: Optional[int] = None
    for line in content.splitlines():
        # Update current epoch when present on the line
        m_epoch = re.search(r"Epoch\s+(\d+)", line, re.IGNORECASE)
        if m_epoch:
            try:
                curr_epoch = int(m_epoch.group(1))
            except Exception:
                curr_epoch = None

        # Prefer explicit loss_epoch (final epoch-level loss)
        if curr_epoch is not None:
            m_le = re.search(r"loss_epoch\s*=\s*([0-9]*\.?[0-9]+)", line, re.IGNORECASE)
            if m_le:
                try:
                    history["loss"].append((curr_epoch, float(m_le.group(1))))
                    continue
                except Exception:
                    pass

            # Fallback to loss_step if loss_epoch not present
            m_ls = re.search(r"loss_step\s*=\s*([0-9]*\.?[0-9]+)", line, re.IGNORECASE)
            if m_ls:
                try:
                    history["loss"].append((curr_epoch, float(m_ls.group(1))))
                    continue
                except Exception:
                    pass

            # Generic patterns like "Loss: 0.028057" or "loss=0.03"
            m_generic = re.search(r"(?:Loss|loss)[:=]\s*([0-9]*\.?[0-9]+)", line)
            if m_generic:
                try:
                    history["loss"].append((curr_epoch, float(m_generic.group(1))))
                except Exception:
                    pass
    
    # Remove empty keys and sort
    history = {k: sorted(v, key=lambda x: x[0]) for k, v in history.items() if v}
    
    return history


def plot_training_curves(
    histories: List[Dict[str, List[Tuple[int, float]]]],
    labels: Optional[List[str]] = None,
    title: str = "Training Curves",
    output_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 6),
) -> Figure:
    """
    Plot training curves from one or more training runs.
    
    Args:
        histories: List of history dicts from parse_checkpoints or parse_log_file
        labels: Names for each run (for legend)
        title: Plot title
        output_path: Save figure to this path (if provided)
        show: Display the plot interactively
        figsize: Figure size (width, height)
    
    Returns:
        matplotlib Figure object
    """
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(histories))]
    
    # Determine which metrics are present
    all_metrics = set()
    for h in histories:
        all_metrics.update(h.keys())

    # Collapse multiple entries for the same epoch: keep the last reported value
    def _collapse_entries(metric_list: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        d: dict[int, float] = {}
        for epoch, val in metric_list:
            try:
                d[int(epoch)] = float(val)
            except Exception:
                continue
        return sorted(d.items())

    for h in histories:
        for metric in list(h.keys()):
            h[metric] = _collapse_entries(h[metric])
    
    # Create subplot layout
    n_metrics = len(all_metrics)
    if n_metrics == 1:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes = [axes]
    else:
        ncols = min(2, n_metrics)
        nrows = (n_metrics + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows // 2))
        axes = axes.flatten() if n_metrics > 1 else [axes]
    
    # Color palette
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(histories)))
    
    # Plot each metric
    for idx, metric in enumerate(sorted(all_metrics)):
        ax = axes[idx]
        
        for run_idx, (history, label, color) in enumerate(zip(histories, labels, colors)):
            if metric not in history:
                continue
            
            epochs, values = zip(*history[metric])
            ax.plot(epochs, values, marker='o', markersize=3, 
                    color=color, label=label, linewidth=1.5, alpha=0.8)
            
            # Mark minimum
            min_idx = np.argmin(values)
            ax.scatter([epochs[min_idx]], [values[min_idx]], 
                       color=color, s=100, marker='*', zorder=5,
                       edgecolors='black', linewidth=0.5)
            ax.annotate(f'{values[min_idx]:.4f}', 
                        (epochs[min_idx], values[min_idx]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=8, color=color)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} over Training')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Use log scale if values span multiple orders of magnitude
        all_vals = []
        for h in histories:
            if metric in h:
                all_vals.extend([v for _, v in h[metric]])
        if all_vals and max(all_vals) / (min(all_vals) + 1e-10) > 100:
            ax.set_yscale('log')
    
    # Hide unused subplots
    for idx in range(len(all_metrics), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    else:
        # Also auto-save a default file so users know where plots are stored
        default_out = "training_curves.png"
        fig.savefig(default_out, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {default_out} (default)")
    
    if show:
        plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot training curves from checkpoints or log files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--checkpoint-dir", type=str, nargs="+",
                        help="Directory(ies) containing .pt checkpoint files")
    parser.add_argument("--log-file", type=str, nargs="+",
                        help="Log file(s) to parse for training metrics")
    parser.add_argument("--labels", type=str, nargs="+",
                        help="Labels for each run (for legend)")
    parser.add_argument("--output", "-o", type=str,
                        help="Output file path for the plot (e.g., training_curves.png)")
    parser.add_argument("--title", type=str, default="Training Curves",
                        help="Title for the plot")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display the plot (only save)")
    parser.add_argument("--figsize", type=int, nargs=2, default=[12, 6],
                        help="Figure size (width height)")
    
    args = parser.parse_args()
    
    if not args.checkpoint_dir and not args.log_file:
        parser.error("Either --checkpoint-dir or --log-file is required")
    
    histories = []
    source_names = []
    
    # Parse checkpoints
    if args.checkpoint_dir:
        for ckpt_dir in args.checkpoint_dir:
            try:
                history = parse_checkpoints(Path(ckpt_dir))
                if not history:
                    print(f"Warning: no metrics parsed from checkpoints in {ckpt_dir}")
                else:
                    histories.append(history)
                    source_names.append(Path(ckpt_dir).name)
                    print(f"Parsed {len(history.get('loss', []))} epochs from {ckpt_dir}")
            except Exception as e:
                print(f"Error parsing {ckpt_dir}: {e}")
    
    # Parse log files
    if args.log_file:
        for log_path in args.log_file:
            try:
                history = parse_log_file(Path(log_path))
                if not history:
                    print(f"Warning: no metrics parsed from log file {log_path}. Check log format or patterns in parse_log_file().")
                else:
                    histories.append(history)
                    source_names.append(Path(log_path).stem)
                    print(f"Parsed {len(history.get('loss', []))} epochs from {log_path}")
            except Exception as e:
                print(f"Error parsing {log_path}: {e}")
    
    if not histories:
        print("No training data found!")
        return
    
    # Use provided labels or auto-generate
    labels = args.labels if args.labels else source_names
    
    # Plot
    plot_training_curves(
        histories=histories,
        labels=labels,
        title=args.title,
        output_path=args.output,
        show=not args.no_show,
        figsize=tuple(args.figsize),
    )


if __name__ == "__main__":
    main()
