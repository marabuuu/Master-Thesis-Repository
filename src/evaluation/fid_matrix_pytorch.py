"""Compute a real×generated FID matrix using the official pytorch_fid library.

Expects an eval directory with this structure:
    <eval_dir>/
        real/
            Basal/   ← zip files containing PNG/JPG tiles
            LumA/
            ...
        generated/
            Basal/
            LumA/
            ...

Subtypes are discovered automatically from subdirectory names.
Zips are extracted to a temporary directory; nothing is written back.

Outputs (saved into <eval_dir>/):
    fid_matrix_official.json   — FID values and tile counts
    fid_matrix_official.png    — heatmap (rows=real, cols=generated)

Example:
    python -m src.evaluation.fid_matrix_pytorch \\
        --eval-dir experiments/20260407_gene_token_cross_attention_training/fid_evaluation_5k
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _extract_zips_to_dir(zip_dir: Path, out_dir: Path) -> int:
    """Extract all PNGs/JPGs from zip files in zip_dir into out_dir. Returns tile count."""
    count = 0
    for zip_path in sorted(zip_dir.glob("*.zip")):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for name in zf.namelist():
                    if name.lower().endswith((".png", ".jpg", ".jpeg")):
                        data = zf.read(name)
                        out_path = out_dir / f"{zip_path.stem}__{Path(name).name}"
                        out_path.write_bytes(data)
                        count += 1
        except zipfile.BadZipFile:
            print(f"  [warn] skipping corrupt zip: {zip_path.name}")
    return count


def _compute_fid(path_a: Path, path_b: Path, device: str, batch_size: int = 64) -> float:
    from pytorch_fid.fid_score import calculate_fid_given_paths
    return calculate_fid_given_paths(
        [str(path_a), str(path_b)],
        batch_size=batch_size,
        device=device,
        dims=2048,
        num_workers=4,
    )


def _plot_fid_matrix(matrix: np.ndarray, rows: list[str], cols: list[str], out_path: Path) -> None:
    from cmcrameri import cm as crameri_cm

    fig, ax = plt.subplots(figsize=(max(5, len(cols) * 2.5), max(4, len(rows) * 2.0)))
    im = ax.imshow(matrix, cmap=crameri_cm.oslo)
    cbar = plt.colorbar(im, ax=ax, label="FID")
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("FID", fontsize=22)

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels([c.replace("gen_", "gen\n") for c in cols], fontsize=21)
    ax.set_yticklabels([r.replace("real_", "real\n") for r in rows], fontsize=21)
    ax.set_xlabel("Generated", fontsize=22)
    ax.set_ylabel("Real", fontsize=22)

    valid = matrix[~np.isnan(matrix)]
    vmin, vmax = valid.min(), valid.max()
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = matrix[i, j]
            if not np.isnan(v):
                norm_val = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                # oslo: dark at low values, near-white at high → white text on dark, black on light
                text_color = "white" if norm_val < 0.5 else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=21, fontweight="normal", color=text_color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def run(eval_dir: Path, device: str | None = None, batch_size: int = 64) -> dict:
    """Compute the real×generated FID matrix for *eval_dir* and save results.

    Expects ``eval_dir/real/<Subtype>/`` and ``eval_dir/generated/<Subtype>/``
    containing per-patient zip files.  Writes ``fid_matrix_official.json`` and
    ``fid_matrix_official.png`` into *eval_dir* and returns the results dict.
    """
    eval_dir = eval_dir.resolve()
    real_root = eval_dir / "real"
    gen_root  = eval_dir / "generated"

    if not real_root.exists() or not gen_root.exists():
        raise FileNotFoundError(f"Expected real/ and generated/ inside {eval_dir}")

    subtypes = sorted(d.name for d in real_root.iterdir() if d.is_dir())
    print(f"Subtypes found: {subtypes}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with tempfile.TemporaryDirectory(prefix="fid_matrix_") as tmp:
        tmp_path = Path(tmp)

        tile_counts: dict[str, int] = {}
        group_dirs: dict[str, Path] = {}
        for split, root in [("real", real_root), ("gen", gen_root)]:
            for subtype in subtypes:
                key = f"{split}_{subtype}"
                src = root / subtype
                if not src.exists():
                    print(f"  [warn] missing: {src}")
                    continue
                dst = tmp_path / key
                dst.mkdir()
                print(f"Extracting {key} ...", flush=True)
                n = _extract_zips_to_dir(src, dst)
                print(f"  → {n} tiles")
                tile_counts[key] = n
                group_dirs[key] = dst

        fid_matrix: dict[str, float] = {}
        rows = [f"real_{s}" for s in subtypes]
        cols = [f"gen_{s}"  for s in subtypes]

        for row in rows:
            for col in cols:
                if row not in group_dirs or col not in group_dirs:
                    print(f"Skipping {row} vs {col} (missing data)")
                    continue
                key = f"{row}_vs_{col}"
                print(f"Computing FID: {row} vs {col} ...", flush=True)
                fid = _compute_fid(group_dirs[row], group_dirs[col], device, batch_size)
                fid_matrix[key] = fid
                print(f"  → FID = {fid:.2f}")

    results = {
        "fid_matrix": fid_matrix,
        "tile_counts": tile_counts,
        "subtypes": subtypes,
        "rows": rows,
        "cols": cols,
    }
    out_json = eval_dir / "fid_matrix_official.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_json}")

    matrix = np.array([[fid_matrix.get(f"{r}_vs_{c}", float("nan"))
                         for c in cols] for r in rows])
    out_png = eval_dir / "fid_matrix_official.png"
    _plot_fid_matrix(matrix, rows, cols, out_png)
    print(f"Saved: {out_png}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="FID matrix via pytorch_fid")
    parser.add_argument("--eval-dir", type=Path, default=None,
                        help="Root dir containing real/ and generated/ subdirs of zips")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip FID computation; re-plot from existing fid_matrix_official.json")
    parser.add_argument("--json", type=Path, default=None,
                        help="Plot directly from any FID/FD results JSON (auto-detects format)")
    args = parser.parse_args()

    # --json: plot from an arbitrary results file, regardless of key naming
    if args.json:
        json_path = args.json.resolve()
        if not json_path.exists():
            raise FileNotFoundError(f"No JSON found at {json_path}")
        results = json.loads(json_path.read_text())
        fid_map = results.get("fid_matrix") or results.get("fd_matrix")
        rows = results["rows"]
        cols = results["cols"]
        matrix = np.array([[fid_map.get(f"{r}_vs_{c}", float("nan"))
                             for c in cols] for r in rows])
        out_png = json_path.with_name("fid_matrix_official.png")
        _plot_fid_matrix(matrix, rows, cols, out_png)
        print(f"Saved: {out_png}")
        return

    if args.eval_dir is None:
        raise ValueError("--eval-dir is required unless --json is given")

    eval_dir = args.eval_dir.resolve()

    if args.plot_only:
        json_path = eval_dir / "fid_matrix_official.json"
        if not json_path.exists():
            raise FileNotFoundError(f"No JSON found at {json_path}")
        results = json.loads(json_path.read_text())
        fid_matrix = results["fid_matrix"]
        rows, cols = results["rows"], results["cols"]
        matrix = np.array([[fid_matrix.get(f"{r}_vs_{c}", float("nan"))
                             for c in cols] for r in rows])
        out_png = eval_dir / "fid_matrix_official.png"
        _plot_fid_matrix(matrix, rows, cols, out_png)
        print(f"Saved: {out_png}")
        return

    real_root = eval_dir / "real"
    gen_root  = eval_dir / "generated"

    if not real_root.exists() or not gen_root.exists():
        raise FileNotFoundError(f"Expected real/ and generated/ inside {eval_dir}")

    subtypes = sorted(d.name for d in real_root.iterdir() if d.is_dir())
    print(f"Subtypes found: {subtypes}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with tempfile.TemporaryDirectory(prefix="fid_matrix_") as tmp:
        tmp_path = Path(tmp)

        # Extract all groups
        tile_counts: dict[str, int] = {}
        group_dirs: dict[str, Path] = {}
        for split, root in [("real", real_root), ("gen", gen_root)]:
            for subtype in subtypes:
                key = f"{split}_{subtype}"
                src = root / subtype
                if not src.exists():
                    print(f"  [warn] missing: {src}")
                    continue
                dst = tmp_path / key
                dst.mkdir()
                print(f"Extracting {key} ...", flush=True)
                n = _extract_zips_to_dir(src, dst)
                print(f"  → {n} tiles")
                tile_counts[key] = n
                group_dirs[key] = dst

        # Compute real×generated FID matrix
        fid_matrix: dict[str, float] = {}
        rows = [f"real_{s}" for s in subtypes]
        cols = [f"gen_{s}"  for s in subtypes]

        for row in rows:
            for col in cols:
                if row not in group_dirs or col not in group_dirs:
                    print(f"Skipping {row} vs {col} (missing data)")
                    continue
                key = f"{row}_vs_{col}"
                print(f"Computing FID: {row} vs {col} ...", flush=True)
                fid = _compute_fid(group_dirs[row], group_dirs[col], device, args.batch_size)
                fid_matrix[key] = fid
                print(f"  → FID = {fid:.2f}")

    # Save JSON
    results = {
        "fid_matrix": fid_matrix,
        "tile_counts": tile_counts,
        "subtypes": subtypes,
        "rows": rows,
        "cols": cols,
    }
    out_json = eval_dir / "fid_matrix_official.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_json}")

    # Plot heatmap
    matrix = np.array([[fid_matrix.get(f"{r}_vs_{c}", float("nan"))
                         for c in cols] for r in rows])

    out_png = eval_dir / "fid_matrix_official.png"
    _plot_fid_matrix(matrix, rows, cols, out_png)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
