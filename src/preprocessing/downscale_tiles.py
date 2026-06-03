"""Downscale PNG tiles inside per-patient zip archives.

Reads every zip from *input_dir*, resizes each PNG to *target_size* × *target_size*
pixels, and writes a new zip with identical internal structure to *output_dir*.
The original zips are never modified.

Non-PNG entries (coords JSON sidecars, etc.) are copied verbatim so the archive
structure stays intact.

Usage (YAML config):
    python run_pipeline.py --config src/config.yaml --stage downscale_tiles

Usage (standalone):
    python -m src.preprocessing.downscale_tiles --config src/config.yaml
    python -m src.preprocessing.downscale_tiles \\
        --input-dir ../data/BRCA-tumor-tiles-final \\
        --output-dir ../data/BRCA-tumor-tiles-128 \\
        --target-size 128
"""

import argparse
import io
import logging
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

RESAMPLE = Image.Resampling.LANCZOS


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _process_zip(args):
    """Downscale all PNGs in one zip and write to output_dir. Returns (name, n_tiles)."""
    src_path, dst_path, target_size, compression = args

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    n_tiles = 0
    with zipfile.ZipFile(src_path, "r") as src_zip:
        members = src_zip.infolist()
        with zipfile.ZipFile(dst_path, "w", compression=compression) as dst_zip:
            for info in members:
                raw = src_zip.read(info.filename)
                if info.filename.lower().endswith(".png"):
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    img = img.resize((target_size, target_size), RESAMPLE)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG", optimize=False)
                    dst_zip.writestr(info, buf.getvalue())
                    n_tiles += 1
                else:
                    dst_zip.writestr(info, raw)

    return Path(src_path).name, n_tiles


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_downscale_tiles(cfg: dict, verbose: bool = True) -> None:
    input_dir = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])
    target_size = int(cfg.get("target_size", 128))
    num_workers = int(cfg.get("num_workers", 4))
    compression_name = cfg.get("compression", "deflate").upper()

    compression = {
        "DEFLATE": zipfile.ZIP_DEFLATED,
        "STORED": zipfile.ZIP_STORED,
        "BZIP2": zipfile.ZIP_BZIP2,
    }.get(compression_name, zipfile.ZIP_DEFLATED)

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")

    zip_files = sorted(input_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No .zip files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Downscaling %d zip archives: %s → %s (target %dpx, %d workers)",
        len(zip_files), input_dir, output_dir, target_size, num_workers,
    )

    tasks = [
        (str(zp), str(output_dir / zp.name), target_size, compression)
        for zp in zip_files
    ]

    total_tiles = 0
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_process_zip, t): t[0] for t in tasks}
        with tqdm(total=len(tasks), desc="Downscaling zips", disable=not verbose) as pbar:
            for fut in as_completed(futures):
                src = futures[fut]
                try:
                    name, n = fut.result()
                    total_tiles += n
                except Exception as exc:
                    log.error("Failed to process %s: %s", src, exc)
                pbar.update(1)

    log.info("Done. Processed %d tiles across %d zips → %s", total_tiles, len(zip_files), output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Downscale tiles inside zip archives")
    parser.add_argument("--config", type=str, help="Path to config.yaml")
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--compression", type=str, default="deflate",
                        choices=["deflate", "stored", "bzip2"])
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            full_cfg = yaml.safe_load(f)
        cfg = full_cfg.get("downscale_tiles", {})
        # Resolve relative paths the same way run_pipeline.py does
        repo_root = Path(args.config).resolve().parent
        if not (repo_root / "run_pipeline.py").exists():
            repo_root = repo_root.parent
        for key in ("input_dir", "output_dir"):
            if key in cfg and not Path(cfg[key]).is_absolute():
                v = cfg[key]
                normalized = v[2:] if v.startswith("./") else v
                if normalized.startswith(("data/", "dataframes/", "experiments/")):
                    cfg[key] = str((repo_root.parent / normalized).resolve())
                else:
                    cfg[key] = str((repo_root / v).resolve())
    else:
        if not args.input_dir or not args.output_dir:
            parser.error("--input-dir and --output-dir are required without --config")
        cfg = {
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "target_size": args.target_size,
            "num_workers": args.num_workers,
            "compression": args.compression,
        }

    run_downscale_tiles(cfg, verbose=True)


if __name__ == "__main__":
    main()
