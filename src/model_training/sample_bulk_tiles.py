"""Generate bulk conditioned tiles per subtype, collect real tiles, and compute the FID matrix.

Outputs per-patient ZIPs under <output>/generated/<Subtype>/ and <output>/real/<Subtype>/.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import torch

from src.evaluation.poc_fid import (
    _load_config,
    _resolve_best_checkpoint,
    load_model,
    generate_cohort_tiles,
    collect_real_cohort_tiles,
)
from src.evaluation.fid_matrix_pytorch import run as run_fid_matrix

log = logging.getLogger(__name__)


def _load_test_patients_by_subtype(
    splits_path: Path,
    subtypes: List[str],
    genomic_dir: Path | None = None,
    tiles_dir: Path | None = None,
) -> dict[str, List[str]]:
    with open(splits_path) as f:
        splits = json.load(f)
    test = splits.get("test", {})
    by_subtype: dict[str, List[str]] = {s: [] for s in subtypes}
    skipped_no_h5: list[str] = []
    skipped_no_tiles: list[str] = []
    for pid, meta in test.items():
        if isinstance(meta, dict):
            subtype = meta.get("subtype", "")
            if subtype in by_subtype:
                if genomic_dir is not None and not (genomic_dir / f"{pid}.h5").exists():
                    skipped_no_h5.append(pid)
                    continue
                if tiles_dir is not None:
                    has_tiles = any(
                        p.is_file() and p.suffix.lower() == ".zip" and pid.upper() in p.name.upper()
                        for p in tiles_dir.iterdir()
                    )
                    if not has_tiles:
                        skipped_no_tiles.append(pid)
                        continue
                by_subtype[subtype].append(pid)
    if skipped_no_h5:
        log.warning("Skipped %d patients without genomic H5: %s", len(skipped_no_h5), skipped_no_h5)
    if skipped_no_tiles:
        log.warning("Skipped %d patients without tile ZIPs: %s", len(skipped_no_tiles), skipped_no_tiles)
    for s in subtypes:
        by_subtype[s] = sorted(by_subtype[s])
        log.info("%s: %d test patients", s, len(by_subtype[s]))
    return by_subtype


def run(
    run_dir: Path,
    splits_path: Path,
    genomic_dir: Path,
    tiles_dir: Path,
    output_dir: Path,
    checkpoint: str | None,
    subtypes: List[str],
    n_per_subtype: int,
    guidance_scale: float,
    n_steps: int,
    batch_size: int,
    device_str: str,
    seed: int,
    skip_generate: bool,
    skip_real: bool,
    skip_fid: bool,
    fid_batch_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    patients = _load_test_patients_by_subtype(
        splits_path, subtypes, genomic_dir=genomic_dir, tiles_dir=tiles_dir,
    )

    # -- Generate tiles -------------------------------------------------------
    if not skip_generate:
        log.info("Device: %s  |  guidance_scale=%.1f  steps=%d  batch=%d",
                 device, guidance_scale, n_steps, batch_size)

        conf = _load_config(run_dir)
        ckpt_path = _resolve_best_checkpoint(run_dir, checkpoint)
        log.info("Checkpoint: %s", ckpt_path)
        model = load_model(conf, ckpt_path, device)

        normalize_feats = getattr(conf, "normalize_feats", True)
        log.info("img_size=%d  feat_dim=%d  normalize_feats=%s",
                 conf.img_size, conf.feat_dim, normalize_feats)

        null_vec = torch.zeros(conf.feat_dim, device=device)

        for subtype in subtypes:
            pids = patients[subtype]
            if not pids:
                log.warning("No test patients for %s, skipping", subtype)
                continue
            gen_dir = output_dir / "generated" / subtype
            log.info("Generating %d tiles for %s -> %s", n_per_subtype, subtype, gen_dir)
            generate_cohort_tiles(
                model=model,
                cond_vec=null_vec,       # overridden per-patient by genomic_h5_dir
                patient_ids=pids,
                n_tiles_total=n_per_subtype,
                out_dir=gen_dir,
                device=device,
                guidance_scale=guidance_scale,
                gen_batch_size=batch_size,
                n_steps=n_steps,
                seed=seed,
                skip_existing=True,
                genomic_h5_dir=genomic_dir,
                normalize_feats=normalize_feats,
            )
    else:
        log.info("--skip-generate: skipping tile generation")

    # -- Collect real tiles ---------------------------------------------------
    if not skip_real:
        for subtype in subtypes:
            pids = patients[subtype]
            if not pids:
                continue
            real_dir = output_dir / "real" / subtype
            log.info("Collecting real tiles for %s -> %s", subtype, real_dir)
            collect_real_cohort_tiles(
                patient_ids=pids,
                tiles_dir=tiles_dir,
                n_tiles_total=n_per_subtype,
                out_dir=real_dir,
                seed=seed,
                skip_existing=True,
            )
    else:
        log.info("--skip-real: skipping real tile collection")

    # -- FID matrix -----------------------------------------------------------
    if not skip_fid:
        log.info("Computing FID matrix -> %s", output_dir)
        run_fid_matrix(
            eval_dir=output_dir,
            device=device_str if torch.cuda.is_available() else "cpu",
            batch_size=fid_batch_size,
        )
    else:
        log.info("--skip-fid: skipping FID computation")

    log.info("Done. Output: %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir",        type=Path, required=True,
                        help="gda/ run directory (contains hparams.yaml and autoenc/)")
    parser.add_argument("--checkpoint",     type=str, default=None,
                        help="Checkpoint filename or path (default: auto-select best val loss)")
    parser.add_argument("--splits",         type=Path, required=True,
                        help="patient_splits.json")
    parser.add_argument("--genomic-dir",    type=Path, required=True,
                        help="Directory with per-patient .h5 genomic feature files")
    parser.add_argument("--tiles-dir",      type=Path, required=True,
                        help="Directory with per-patient source zip files of real tiles")
    parser.add_argument("--output",         type=Path, required=True,
                        help="Output root; zips under <output>/generated/<Subtype>/ and real/<Subtype>/")
    parser.add_argument("--subtypes",       nargs="+", default=["Basal", "LumA"])
    parser.add_argument("--n-per-subtype",  type=int, default=10000)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--steps",          type=int, default=20)
    parser.add_argument("--batch-size",     type=int, default=16,
                        help="Tiles per forward pass during generation")
    parser.add_argument("--fid-batch-size", type=int, default=64,
                        help="Batch size for InceptionV3 feature extraction")
    parser.add_argument("--device",         type=str, default="cuda")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--skip-generate",  action="store_true")
    parser.add_argument("--skip-real",      action="store_true")
    parser.add_argument("--skip-fid",       action="store_true")
    parser.add_argument("--verbose",        action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log.setLevel(logging.INFO)

    run(
        run_dir=args.run_dir.resolve(),
        splits_path=args.splits.resolve(),
        genomic_dir=args.genomic_dir.resolve(),
        tiles_dir=args.tiles_dir.resolve(),
        output_dir=args.output.resolve(),
        checkpoint=args.checkpoint,
        subtypes=args.subtypes,
        n_per_subtype=args.n_per_subtype,
        guidance_scale=args.guidance_scale,
        n_steps=args.steps,
        batch_size=args.batch_size,
        device_str=args.device,
        seed=args.seed,
        skip_generate=args.skip_generate,
        skip_real=args.skip_real,
        skip_fid=args.skip_fid,
        fid_batch_size=args.fid_batch_size,
    )


if __name__ == "__main__":
    main()
