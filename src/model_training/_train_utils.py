"""Training utilities for the CFG backbone (config building, cohort validation)."""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict

from mopadi.configs.choices import ModelName

from .config import GDAConfig

log = logging.getLogger(__name__)


def _validate_cohort_coverage(
    patient_splits_path: str,
    zip_dir: str,
    drop_threshold: float = 0.30,
    check_cohort: str = "TCGA-LIHC",
) -> None:
    """Raise RuntimeError if too many patients from *check_cohort* lack valid zips."""
    with open(patient_splits_path) as f:
        splits_data = json.load(f)

    expected_pids: set = set()
    for fold, entries in splits_data.items():
        if fold.startswith("_") or not isinstance(entries, dict):
            continue
        for pid, meta in entries.items():
            if pid.startswith("_") or not isinstance(meta, dict):
                continue
            if meta.get("subtype") == check_cohort:
                expected_pids.add(pid)

    if not expected_pids:
        log.debug("_validate_cohort_coverage: no '%s' patients found — skipping.", check_cohort)
        return

    zip_dir_path = Path(zip_dir)
    valid_pids: set = set()
    invalid: list = []

    for pid in expected_pids:
        matches = list(zip_dir_path.glob(f"{pid}*.zip"))
        if not matches:
            invalid.append((pid, "no zip file found"))
            continue
        zip_path = matches[0]
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                if names:
                    with zf.open(names[0]) as _f:
                        _f.read(256)
            valid_pids.add(pid)
        except Exception as exc:
            invalid.append((pid, str(exc)))

    n_expected = len(expected_pids)
    n_valid = len(valid_pids)
    n_dropped = n_expected - n_valid
    drop_frac = n_dropped / n_expected if n_expected > 0 else 0.0

    if invalid:
        log.warning(
            "%d %s patients lack valid zips (showing first 10): %s",
            len(invalid), check_cohort,
            [(pid, err) for pid, err in invalid[:10]],
        )

    log.info(
        "Cohort coverage check [%s]: %d/%d patients have valid zips (%.1f%% dropped)",
        check_cohort, n_valid, n_expected, drop_frac * 100,
    )

    if drop_frac > drop_threshold:
        raise RuntimeError(
            f"\n[CFG Startup] Too many {check_cohort} patients are missing or have corrupt zip archives!\n"
            f"  Expected : {n_expected} patients (from patient_splits.json)\n"
            f"  Valid    : {n_valid}\n"
            f"  Dropped  : {n_dropped} ({drop_frac:.1%})\n"
            f"  Threshold: {drop_threshold:.0%}\n"
            f"  First bad entries: {invalid[:5]}\n"
            f"  → Check {zip_dir} for missing or corrupt zip archives.\n"
        )


def _build_config(cfg: Dict[str, Any]) -> GDAConfig:
    conditioning_type = cfg.get("conditioning_type", "real")
    for key in ("zip_dir", "patient_splits_path"):
        if not cfg.get(key):
            raise ValueError(f"Missing required key '{key}' in config.")
    if not Path(cfg["patient_splits_path"]).exists():
        raise FileNotFoundError(f"patient_splits_path not found: {cfg['patient_splits_path']}")
    if conditioning_type == "real":
        if not cfg.get("genomic_feature_dir"):
            raise ValueError("genomic_feature_dir is required when conditioning_type='real'")
        if not Path(cfg["genomic_feature_dir"]).is_dir():
            raise FileNotFoundError(f"genomic_feature_dir not found: {cfg['genomic_feature_dir']}")

    model_cfg = cfg.get("model", {})

    def _get(key, default=None):
        return cfg.get(key, model_cfg.get(key, default))

    img_size = int(_get("img_size", 256))
    net_ch = int(_get("net_ch", 128))
    style_ch = int(_get("style_ch", 512))
    feat_dim = style_ch

    raw_mult = _get("net_ch_mult")
    net_ch_mult = tuple(raw_mult) if raw_mult else (1, 1, 2, 2, 4, 4)
    raw_attn = _get("net_attn")
    net_attn = tuple(raw_attn) if raw_attn else (16,)

    base_dir = cfg.get("output_dir", cfg.get("base_dir", "experiments/cfg_v1"))

    fields = dict(
        name="gda",
        base_dir=base_dir,
        seed=int(_get("seed", 42)),
        model_name=ModelName.beatgans_autoenc,
        diffusion_type="beatgans",
        img_size=img_size,
        net_ch=net_ch,
        net_ch_mult=net_ch_mult,
        net_attn=net_attn,
        net_num_res_blocks=int(_get("net_num_res_blocks", 2)),
        net_beatgans_gradient_checkpoint=bool(_get("net_beatgans_gradient_checkpoint", False)),
        style_ch=style_ch,
        feat_dim=feat_dim,
        net_beatgans_embed_channels=style_ch,
        net_beatgans_resnet_two_cond=True,
        T=int(_get("T", 1000)),
        T_eval=int(_get("T_eval", 20)),
        batch_size=int(_get("batch_size", 16)),
        lr=float(_get("backbone_lr", _get("lr", 1e-4))),
        ema_decay=float(_get("ema_decay", 0.9999)),
        grad_clip=float(_get("grad_clip", 1.0)),
        fp16=bool(_get("fp16", False)),
        accum_batches=int(_get("accumulate_grad_batches", 1)),
        num_workers=int(_get("num_workers", 4)),
        total_samples=int(_get("total_samples", 200_000_000)),
        steps_per_epoch=int(_get("steps_per_epoch", 5_000)),
        save_every_samples=int(_get("save_every_samples", 200_000)),
        reconstruct_every_samples=int(_get("reconstruct_every_samples", 50_000)),
        sample_every_samples=int(_get("sample_every_samples", 250_000)),
        sample_size=int(_get("sample_size", 16)),
        zip_dir=cfg["zip_dir"],
        genomic_feature_dir=cfg.get("genomic_feature_dir"),
        patient_splits_path=cfg["patient_splits_path"],
        conditioning_type=conditioning_type,
        max_tiles_by_subtype=cfg.get("max_tiles_by_subtype"),
        tile_sampling_seed=int(cfg.get("tile_sampling_seed", 42)),
        do_normalize=bool(cfg.get("do_normalize", True)),
        do_resize=bool(cfg.get("do_resize", False)),
        val_limit_batches=int(cfg.get("limit_val_batches", 100)),
        # GDA-specific fields kept for GDAConfig compatibility (unused by CFG)
        adapter_base_ch=int(_get("adapter_base_ch", 64)),
        adapter_n_tokens=int(_get("adapter_n_tokens", 8)),
        adapter_token_dim=int(_get("adapter_token_dim", 256)),
        adapter_t_dim=int(_get("adapter_t_dim", 256)),
        adapter_n_heads=int(_get("adapter_n_heads", 4)),
        normalize_feats=bool(_get("normalize_feats", True)),
        val_swap_basal_luma=bool(_get("val_swap_basal_luma", False)),
        cfg_dropout=float(_get("cfg_dropout", 0.30)),
        backbone_lr=float(_get("backbone_lr", 1e-4)),
        adapter_lr=float(_get("adapter_lr", 3e-4)),
        freeze_backbone=bool(_get("freeze_backbone", False)),
        backbone_ckpt_path=str(_get("backbone_ckpt_path", "")),
        reinit_adapter=bool(_get("reinit_adapter", False)),
        delta_encouragement_weight=float(_get("delta_encouragement_weight", 0.0)),
        genomic_recon_weight=float(_get("genomic_recon_weight", 0.0)),
    )

    return GDAConfig(**fields)  # type: ignore[arg-type]
