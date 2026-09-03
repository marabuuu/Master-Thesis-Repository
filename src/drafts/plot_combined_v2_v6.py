"""Combined training plot: v2 (baseline) + v6 (pairwise Δε loss).

Merges TFEvents from two runs that share a step axis (v6 resumes from v2's
last checkpoint at sample 576 000).  A vertical line marks the resume point
and the activation of the unsupervised pairwise Δε loss + L2 token norm.

Layout (4 rows × 2 cols):
  Row 1: Training & Val Loss (linear)  |  Training & Val Loss (log)
  Row 2: Guidance Delta                |  Genomic Token Diversity
  Row 3: Genomic vs Null Token Dist    |  LR Schedule
  Row 4: Pairwise Δε metric            |  Pairwise Δε loss (training)

Usage
-----
    python src/visualization/plot_combined_v2_v6.py --no-show
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from src.visualization.training_plots import load_gda_tfevents, setup_style  # noqa: E402

LOGDIR_V2 = "../experiments/20260526_poc_brca_lihc_gda_v2/gda"
LOGDIR_V6 = "../experiments/20260527_poc_brca_lihc_gda_v6/gda"
DEFAULT_OUT = "../experiments/20260527_poc_brca_lihc_gda_v6/combined_v2_v6_diagnostics.png"
RESUME_SAMPLE = 576_000  # sample count where v6 starts


def _merge(a: dict, b: dict) -> dict:
    out = {}
    all_tags = set(a) | set(b)
    for tag in all_tags:
        sa, va = a.get(tag, ([], []))
        sb, vb = b.get(tag, ([], []))
        out[tag] = (sa + sb, va + vb)
    return out


def _smooth(vals, alpha=0.05):
    s = []
    ema = None
    for v in vals:
        ema = v if ema is None else alpha * v + (1 - alpha) * ema
        s.append(ema)
    return s


def plot_combined(logdir_v2, logdir_v6, out_path, show=True):
    import matplotlib.pyplot as plt

    setup_style()

    print(f"Loading v2: {logdir_v2}")
    data_v2 = load_gda_tfevents(logdir_v2)
    print(f"Loading v6: {logdir_v6}")
    data_v6 = load_gda_tfevents(logdir_v6)
    data = _merge(data_v2, data_v6)

    M = 1e6  # scale to millions of samples

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle(
        "GDA PoC — BRCA vs LIHC   (v2 baseline → v6: pairwise Δε + L2 token norm)",
        fontsize=13, fontweight="bold",
    )

    resume_M = RESUME_SAMPLE / M

    def _vline(ax, label="_nolegend_"):
        ax.axvline(resume_M, color="#e63946", lw=1.4, ls="--", alpha=0.8, label=label)

    def _pairs_sorted(tag):
        steps, vals = data.get(tag, ([], []))
        if not steps:
            return [], []
        p = sorted(zip(steps, vals))
        return [x[0] / M for x in p], [x[1] for x in p]

    def _xlabel(ax):
        ax.set_xlabel("Samples seen (millions)")

    # ── Row 1 left: Training & Validation Loss (linear) ───────────────
    ax = axes[0, 0]
    xs, ys = _pairs_sorted("loss/train")
    xs_v, ys_v = _pairs_sorted("loss/val")
    if xs:
        ax.plot(xs, _smooth(ys, alpha=0.03), color="#264653", lw=1.2, label="train loss (EMA)")
    if xs_v:
        ax.scatter(xs_v, ys_v, color="#e76f51", s=25, zorder=5, label="val loss")
        best_idx = int(min(range(len(ys_v)), key=lambda i: ys_v[i]))
        ax.annotate(
            f"best {ys_v[best_idx]:.4f}",
            xy=(xs_v[best_idx], ys_v[best_idx]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=7, color="#e76f51",
        )
    _vline(ax, label="v6: pairwise Δε + L2 norm")
    ax.set_ylabel("MSE loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(fontsize=8)

    # ── Row 1 right: Training & Validation Loss (log scale) ───────────
    ax = axes[0, 1]
    xs, ys = _pairs_sorted("loss/train")
    xs_v, ys_v = _pairs_sorted("loss/val")
    if xs:
        ax.plot(xs, _smooth(ys, alpha=0.03), color="#264653", lw=1.2, label="train loss (EMA)")
    if xs_v:
        ax.scatter(xs_v, ys_v, color="#e76f51", s=25, zorder=5, label="val loss")
    _vline(ax)
    ax.set_yscale("log")
    ax.set_ylabel("MSE loss (log)")
    ax.set_title("Training & Validation Loss (log scale)")
    ax.legend(fontsize=8)

    # ── Row 2 left: Guidance Delta ─────────────────────────────────────
    ax = axes[1, 0]
    xs, ys = _pairs_sorted("cond/guidance_delta")
    if xs:
        ax.plot(xs, ys, color="#2a9d8f", lw=1.4, marker="o", ms=4,
                label="E[‖Δε_own − Δε_null‖²]")
    _vline(ax)
    ax.set_ylabel("guidance_delta")
    ax.set_title("Guidance Delta — real vs null token response")
    ax.legend(fontsize=8)

    # ── Row 2 right: Genomic Token Diversity ──────────────────────────
    ax = axes[1, 1]
    xs, ys = _pairs_sorted("cond/g_token_diversity")
    if xs:
        ax.plot(xs, ys, color="#f4a261", lw=1.4, marker="o", ms=4,
                label="g_token_diversity (MSE across patients)")
    _vline(ax)
    ax.set_ylabel("token diversity")
    ax.set_title("Genomic Token Diversity — patient-level variation")
    ax.legend(fontsize=8)

    # ── Row 3 left: Genomic vs Null Token Distance ────────────────────
    ax = axes[2, 0]
    xs, ys = _pairs_sorted("cond/g_vs_null_dist")
    if xs:
        ax.plot(xs, ys, color="#457b9d", lw=1.4, marker="o", ms=4,
                label="g_vs_null_dist (MSE real vs null)")
    _vline(ax)
    ax.set_ylabel("g_vs_null_dist")
    ax.set_title("Genomic vs Null Token Distance")
    ax.legend(fontsize=8)

    # ── Row 3 right: LR Schedule ──────────────────────────────────────
    ax = axes[2, 1]
    xs_b, ys_b = _pairs_sorted("lr-AdamW")
    xs_a, ys_a = _pairs_sorted("lr-AdamW-1")
    if xs_b:
        ax.plot(xs_b, ys_b, color="#264653", lw=1.2, label="backbone lr (AdamW)")
    if xs_a:
        ax.plot(xs_a, ys_a, color="#e76f51", lw=1.2, label="adapter lr (AdamW-1)")
    _vline(ax)
    ax.set_ylabel("learning rate")
    ax.set_title("Learning Rate Schedule")
    ax.legend(fontsize=8)

    # ── Row 4 left: Pairwise Δε metric (v6 only) ─────────────────────
    ax = axes[3, 0]
    xs, ys = _pairs_sorted("cond/pairwise_delta")
    if xs:
        ax.plot(xs, _smooth(ys, alpha=0.1), color="#8338ec", lw=1.4,
                label="E[‖Δε_own − Δε_perm‖²]")
    _vline(ax)
    _xlabel(ax)
    ax.set_ylabel("pairwise_delta")
    ax.set_title("Pairwise Δε — patient-specific conditioning (unsupervised)")
    ax.legend(fontsize=8)

    # ── Row 4 right: Pairwise Δε loss (training, v6 only) ────────────
    ax = axes[3, 1]
    xs, ys = _pairs_sorted("cond/pairwise_delta_loss")
    if xs:
        ax.plot(xs, _smooth(ys, alpha=0.05), color="#6d6875", lw=1.2,
                label="pairwise_delta_loss (per-step)")
    else:
        # Fallback: guidance_delta per training step
        xs, ys = _pairs_sorted("cond/guidance_delta_train")
        if xs:
            ax.plot(xs, _smooth(ys, alpha=0.05), color="#6d6875", lw=1.2,
                    label="guidance_delta_train (per-step)")
    _vline(ax)
    _xlabel(ax)
    ax.set_ylabel("pairwise_delta_loss")
    ax.set_title("Pairwise Δε Loss (per training step)")
    ax.legend(fontsize=8)

    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir-v2", default=LOGDIR_V2)
    parser.add_argument("--logdir-v6", default=LOGDIR_V6)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    plot_combined(args.logdir_v2, args.logdir_v6, args.out, show=not args.no_show)
