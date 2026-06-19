"""Top up generated tiles to reach a target count per subtype.

After removing invalid patient ZIPs (patients with genomics but no histology),
this script generates exactly the missing tiles and saves them as supplementary
per-patient ZIPs (e.g., TCGA-XXX_topup_0.zip) that the FID pipeline picks up
automatically.

Usage:
    python -m src.poc_experiment.topup_tiles \
        --run-dir     experiments/20260607_brca_pam50_cfg_v2_256/gda \
        --splits      experiments/20260528_genomic_features/patient_splits.json \
        --genomic-dir experiments/20260528_genomic_features/genomic_h5 \
        --tiles-dir   ../data/BRCA-tumor-tiles-256 \
        --output      experiments/20260607_brca_pam50_cfg_v2_256/generated_tiles_cfg1 \
        --subtypes    Basal LumA \
        --target      10000 \
        --guidance-scale 1.0 \
        --steps       20 \
        --batch-size  16
"""

from __future__ import annotations

import argparse
import logging
import math
import zipfile
from pathlib import Path
from typing import List

import torch

from src.evaluation.poc_fid import (
    _load_config,
    _resolve_best_checkpoint,
    load_model,
    generate_batch,
    _save_images_to_zip,
    _load_patient_h5_feat,
)
from src.evaluation.fid_matrix_pytorch import run as run_fid_matrix
from src.poc_experiment.sample_bulk_tiles import _load_test_patients_by_subtype

log = logging.getLogger(__name__)

_IMG_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _count_tiles_in_dir(d: Path) -> int:
    total = 0
    for zp in d.glob("*.zip"):
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                total += sum(1 for n in zf.namelist() if Path(n).suffix.lower() in _IMG_SUFFIXES)
        except zipfile.BadZipFile:
            pass
    return total


def run(
    run_dir: Path,
    splits_path: Path,
    genomic_dir: Path,
    tiles_dir: Path,
    output_dir: Path,
    checkpoint: str | None,
    subtypes: List[str],
    target: int,
    guidance_scale: float,
    n_steps: int,
    batch_size: int,
    device_str: str,
    seed: int,
    skip_fid: bool,
    fid_batch_size: int,
) -> None:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    patients = _load_test_patients_by_subtype(
        splits_path, subtypes, genomic_dir=genomic_dir, tiles_dir=tiles_dir,
    )

    needs_generation = False
    for subtype in subtypes:
        gen_dir = output_dir / "generated" / subtype
        if not gen_dir.exists():
            log.warning("generated/%s does not exist, nothing to top up", subtype)
            continue
        current = _count_tiles_in_dir(gen_dir)
        deficit = target - current
        log.info("%s: %d current tiles, target %d, deficit %d", subtype, current, target, deficit)
        if deficit > 0:
            needs_generation = True

    if not needs_generation:
        log.info("All subtypes already at or above target — nothing to generate")
    else:
        conf = _load_config(run_dir)
        ckpt_path = _resolve_best_checkpoint(run_dir, checkpoint)
        log.info("Checkpoint: %s", ckpt_path)
        model = load_model(conf, ckpt_path, device)
        normalize_feats = getattr(conf, "normalize_feats", True)
        log.info("img_size=%d  feat_dim=%d  normalize_feats=%s",
                 conf.img_size, conf.feat_dim, normalize_feats)

        for subtype in subtypes:
            gen_dir = output_dir / "generated" / subtype
            if not gen_dir.exists():
                continue
            current = _count_tiles_in_dir(gen_dir)
            deficit = target - current
            if deficit <= 0:
                log.info("%s: already at %d tiles (target %d), skipping", subtype, current, target)
                continue

            pids = patients[subtype]
            if not pids:
                log.warning("No valid patients for %s", subtype)
                continue

            n_per_patient = max(1, math.ceil(deficit / len(pids)))
            n_patients_needed = math.ceil(deficit / n_per_patient)
            selected = pids[:n_patients_needed]

            log.info("%s: generating %d top-up tiles across %d patients (%d each)",
                     subtype, deficit, len(selected), n_per_patient)

            remaining_deficit = deficit
            for p_idx, pid in enumerate(selected):
                n_this = min(n_per_patient, remaining_deficit)
                if n_this <= 0:
                    break

                out_zip = gen_dir / f"{pid}_topup_0.zip"
                suffix = 0
                while out_zip.exists():
                    suffix += 1
                    out_zip = gen_dir / f"{pid}_topup_{suffix}.zip"

                patient_cond = _load_patient_h5_feat(genomic_dir, pid, device)
                if patient_cond is None:
                    log.warning("  %s: H5 not found, skipping", pid)
                    continue
                if normalize_feats:
                    import torch.nn.functional as _F
                    patient_cond = _F.normalize(patient_cond, p=2, dim=-1)

                log.info("  [%d/%d] %s: generating %d top-up tiles",
                         p_idx + 1, len(selected), pid, n_this)

                imgs = []
                rem = n_this
                batch_idx = 0
                while rem > 0:
                    bs = min(batch_size, rem)
                    tile_seed = (seed + 99_000_000 + p_idx * 100_000 + batch_idx * 10_000) % (2 ** 32)
                    batch_imgs = generate_batch(
                        model=model,
                        cond_vec=patient_cond,
                        batch_size=bs,
                        guidance_scale=guidance_scale,
                        n_steps=n_steps,
                        device=device,
                        seed=tile_seed,
                    )
                    imgs.extend(batch_imgs)
                    rem -= bs
                    batch_idx += 1

                _save_images_to_zip(imgs[:n_this], out_zip, f"{pid}_topup")
                remaining_deficit -= n_this
                log.info("    -> saved %d tiles to %s", n_this, out_zip.name)

            final = _count_tiles_in_dir(gen_dir)
            log.info("%s: final count = %d tiles", subtype, final)

    if not skip_fid:
        log.info("Recomputing FID matrix -> %s", output_dir)
        run_fid_matrix(
            eval_dir=output_dir,
            device=device_str if torch.cuda.is_available() else "cpu",
            batch_size=fid_batch_size,
        )
    else:
        log.info("--skip-fid: skipping FID recomputation")

    log.info("Done. Output: %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir",        type=Path, required=True)
    parser.add_argument("--checkpoint",     type=str, default=None)
    parser.add_argument("--splits",         type=Path, required=True)
    parser.add_argument("--genomic-dir",    type=Path, required=True)
    parser.add_argument("--tiles-dir",      type=Path, required=True)
    parser.add_argument("--output",         type=Path, required=True)
    parser.add_argument("--subtypes",       nargs="+", default=["Basal", "LumA"])
    parser.add_argument("--target",         type=int, default=10000,
                        help="Target tile count per subtype")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--steps",          type=int, default=20)
    parser.add_argument("--batch-size",     type=int, default=16)
    parser.add_argument("--fid-batch-size", type=int, default=64)
    parser.add_argument("--device",         type=str, default="cuda")
    parser.add_argument("--seed",           type=int, default=42)
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
        target=args.target,
        guidance_scale=args.guidance_scale,
        n_steps=args.steps,
        batch_size=args.batch_size,
        device_str=args.device,
        seed=args.seed,
        skip_fid=args.skip_fid,
        fid_batch_size=args.fid_batch_size,
    )


if __name__ == "__main__":
    main()
