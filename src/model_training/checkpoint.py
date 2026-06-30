"""Checkpoint loading utilities for CfgBackboneLitModel runs."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import torch

log = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file, accepting pickled objects for hparams.yaml compat."""
    import yaml

    with path.open("r", encoding="utf-8") as f:
        load_fn = getattr(yaml, "unsafe_load", None) or yaml.full_load
        payload = load_fn(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def load_config_from_run(run_dir: Path):
    """Load GDAConfig from a run directory's hparams.yaml."""
    from mopadi.configs.choices import ModelName
    from .config import GDAConfig

    hparams_path = run_dir / "hparams.yaml"
    if not hparams_path.exists():
        raise FileNotFoundError(f"hparams.yaml not found in {run_dir}")
    conf = GDAConfig.from_dict(_load_yaml(hparams_path))
    if getattr(conf, "model_name", None) is None:
        conf.model_name = ModelName.beatgans_autoenc
    return conf


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.item())
    return None


def _iter_checkpoint_scores(payload: dict[str, Any]) -> Iterable[float]:
    """Walk checkpoint metadata to find best_model_score values."""
    def walk(obj: Any) -> Iterable[float]:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "best_model_score":
                    score = _to_float(value)
                    if score is not None:
                        yield score
                elif key == "best_k_models" and isinstance(value, dict):
                    for score in value.values():
                        score_f = _to_float(score)
                        if score_f is not None:
                            yield score_f
                else:
                    yield from walk(value)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                yield from walk(item)

    callbacks = payload.get("callbacks", payload)
    yield from walk(callbacks)


def resolve_checkpoint(run_dir: Path, explicit: Optional[str] = None) -> Path:
    """Find the best checkpoint in a run directory.

    Checks explicit path first, then best.ckpt, then selects by lowest
    validation loss from checkpoint metadata, falling back to highest step.
    """
    if explicit:
        ckpt = Path(explicit)
        if not ckpt.is_absolute():
            direct = run_dir / ckpt
            under_autoenc = run_dir / "autoenc" / ckpt
            if direct.exists():
                ckpt = direct
            elif under_autoenc.exists():
                ckpt = under_autoenc
            else:
                raise FileNotFoundError(f"checkpoint not found: {direct} or {under_autoenc}")
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        return ckpt

    autoenc_dir = run_dir / "autoenc"
    for name in ("best.ckpt", "best_model.ckpt", "best_model_path.ckpt"):
        candidate = autoenc_dir / name
        if candidate.exists():
            return candidate

    ckpts = sorted(autoenc_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {autoenc_dir}")

    scored: list[tuple[float, Path]] = []
    fallback: list[tuple[int, Path]] = []
    for ckpt in ckpts:
        try:
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        except Exception as exc:
            log.warning("Could not inspect checkpoint %s: %s", ckpt.name, exc)
            fallback.append((0, ckpt))
            continue
        scores = list(_iter_checkpoint_scores(payload))
        if scores:
            scored.append((min(scores), ckpt))
            continue
        step = int(payload.get("global_step", 0))
        if step == 0:
            m = re.search(r"step[=_]?(\d+)", ckpt.stem)
            if m:
                step = int(m.group(1))
        fallback.append((step, ckpt))

    if scored:
        best = min(scored, key=lambda item: item[0])
        log.info("Auto-selected checkpoint: %s (score=%.4f)", best[1].name, best[0])
        return best[1]
    if fallback:
        return max(fallback, key=lambda item: item[0])[1]
    return ckpts[-1]


def load_model(conf, ckpt_path: Path, device: Optional[torch.device] = None):
    """Load a CfgBackboneLitModel from a checkpoint.

    Clears backbone_ckpt_path to skip the warm-start load (the full state dict
    is loaded directly from ckpt_path).
    """
    from .cfg_model import CfgBackboneLitModel

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)

    conf.backbone_ckpt_path = None
    model = CfgBackboneLitModel(conf)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.info("Loaded %s: %d missing keys", ckpt_path.name, len(missing))
    if unexpected:
        log.warning("Loaded %s: %d unexpected keys", ckpt_path.name, len(unexpected))
    model.eval()
    if device is not None:
        model.to(device)
    return model
