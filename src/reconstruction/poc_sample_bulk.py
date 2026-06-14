"""Generate bulk conditioned tiles for PAM50 subtype evaluation, then compute
the real×generated FID matrix and save the heatmap.

Generates N tiles per subtype (default 10 000) for test-set patients and saves
them as per-patient ZIP files, matching the layout used by fid_evaluation.py:

    <output-dir>/generated/<Subtype>/<patient_id>.zip
        → {patient_id}_00000.png, {patient_id}_00001.png, …

Real tiles are sampled from the source zip directory and saved under:

    <output-dir>/real/<Subtype>/<patient_id>.zip

After generation and real-tile collection the FID matrix is computed:

    <output-dir>/fid_matrix_official.json
    <output-dir>/fid_matrix_official.png   ← real×generated heatmap

Usage:
    python -m src.reconstruction.poc_sample_bulk \\
        --run-dir     experiments/20260607_brca_pam50_cfg_v2_256/gda \\
        --splits      experiments/20260528_genomic_features/patient_splits.json \\
        --genomic-dir experiments/20260528_genomic_features/genomic_h5 \\
        --tiles-dir   ../data/BRCA-tumor-tiles-final \\
        --output      experiments/20260607_brca_pam50_cfg_v2_256/generated_tiles_cfg1 \\
        --subtypes    Basal LumA \\
        --n-per-subtype 10000 \\
        --guidance-scale 1.0 \\
        --batch-size  16 \\
        --steps       20
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


def _load_test_patients_by_subtype(splits_path: Path, subtypes: List[str]) -> dict[str, List[str]]:
    with open(splits_path) as f:
        splits = json.load(f)
    test = splits.get("test", {})
    by_subtype: dict[str, List[str]] = {s: [] for s in subtypes}
    for pid, meta in test.items():
        if isinstance(meta, dict):
            subtype = meta.get("subtype", "")
            if subtype in by_subtype:
                by_subtype[subtype].append(pid)
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

    patients = _load_test_patients_by_subtype(splits_path, subtypes)

    # ── Generate tiles ────────────────────────────────────────────────────────
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
            log.info("Generating %d tiles for %s → %s", n_per_subtype, subtype, gen_dir)
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

    # ── Collect real tiles ────────────────────────────────────────────────────
    if not skip_real:
        for subtype in subtypes:
            pids = patients[subtype]
            if not pids:
                continue
            real_dir = output_dir / "real" / subtype
            log.info("Collecting real tiles for %s → %s", subtype, real_dir)
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

    # ── FID matrix ────────────────────────────────────────────────────────────
    if not skip_fid:
        log.info("Computing FID matrix → %s", output_dir)
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
