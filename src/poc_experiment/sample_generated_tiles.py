"""Generate diffusion samples for BRCA and LIHC patients (CFG backbone only).

For each selected patient, writes a side-by-side pair:
- unconditional / null-conditioned output
- conditioned output using the real patient genomic vector

Usage:
    python -m src.poc_experiment.sample_generated_tiles \\
        --run-dir experiments/20260601_poc_brca_lihc_cfg_v2_dgx/gda \\
        --subtypes BRCA LIHC --n-per-subtype 2
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required to read hparams.yaml") from exc

from .config import GDAConfig
from .dataset import (
    ZipTilesWithGenomicFeatures,
    patient_id_from_tile_path,
    _build_genomic_cache,
    _load_splits_and_subtypes,
)

log = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        load_fn = getattr(yaml, "unsafe_load", None) or yaml.full_load
        payload = load_fn(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def _to_float(value: Any) -> float | None:
    import torch
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.item())
    return None


def _iter_checkpoint_scores(payload: dict[str, Any]) -> Iterable[float]:
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


def _resolve_checkpoint(run_dir: Path, explicit_ckpt: str | None = None) -> Path:
    if explicit_ckpt:
        ckpt = Path(explicit_ckpt)
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
    for candidate_name in ("best.ckpt", "best_model.ckpt", "best_model_path.ckpt"):
        candidate = autoenc_dir / candidate_name
        if candidate.exists():
            return candidate

    ckpts = sorted(autoenc_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {autoenc_dir}")

    scored: list[tuple[float, Path]] = []
    fallback: list[tuple[int, Path]] = []
    for ckpt in ckpts:
        try:
            import torch
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        except Exception as exc:
            log.warning("Could not inspect checkpoint %s: %s", ckpt.name, exc)
            fallback.append((0, ckpt))
            continue
        scores = list(_iter_checkpoint_scores(payload))
        if scores:
            scored.append((min(scores), ckpt))
            continue
        fallback.append((int(payload.get("global_step", 0)), ckpt))

    if scored:
        return min(scored, key=lambda item: item[0])[1]
    if fallback:
        return max(fallback, key=lambda item: item[0])[1]
    return ckpts[-1]


def _build_config(run_dir: Path) -> GDAConfig:
    from mopadi.configs.choices import ModelName
    hparams_path = run_dir / "hparams.yaml"
    if not hparams_path.exists():
        raise FileNotFoundError(f"Missing hparams.yaml in {run_dir}")
    conf = GDAConfig.from_dict(_load_yaml(hparams_path))
    if getattr(conf, "model_name", None) is None:
        conf.model_name = ModelName.beatgans_autoenc
    return conf


def _load_model(conf: GDAConfig, ckpt_path: Path):
    import torch
    from .cfg_model import CfgBackboneLitModel

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint structure in {ckpt_path}")

    # backbone_ckpt_path in hparams.yaml points to the warm-start predecessor run.
    # We are loading a full checkpoint state dict below, so the warm-start load in
    # __init__ would be wasted (and slow — the file is ~20 GB).  Clear it first.
    conf.backbone_ckpt_path = None
    model = CfgBackboneLitModel(conf)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.info("Loaded %s with %d missing keys", ckpt_path.name, len(missing))
    if unexpected:
        log.info("Loaded %s with %d unexpected keys", ckpt_path.name, len(unexpected))
    model.eval()
    return model, "cfg_backbone"


def _normalise_subtype(value: str) -> str:
    return value.strip().lower()


def _match_subtype(subtype: str, wanted: Sequence[str]) -> bool:
    subtype_n = _normalise_subtype(subtype)
    return any(_normalise_subtype(token) in subtype_n for token in wanted)


def _select_patients(
    dataset: Any,
    wanted_subtypes: Sequence[str],
    n_per_subtype: int,
    seed: int,
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    by_subtype: dict[str, list[str]] = {}
    for pid, subtype in dataset._subtype_map.items():
        if _match_subtype(subtype, wanted_subtypes):
            by_subtype.setdefault(subtype, []).append(pid)

    selected: list[tuple[str, str]] = []
    for subtype, pids in sorted(by_subtype.items()):
        chosen = pids if len(pids) <= n_per_subtype else rng.sample(pids, n_per_subtype)
        for pid in chosen:
            selected.append((pid, subtype))
    return selected


def _sample_cfg_backbone_pair(model, feats, guidance_scale, n_steps, device):
    import torch
    import torch.nn as nn
    from mopadi.diffusion.base import DummyReturn

    backbone = model.ema_model
    sampler = model.conf._make_diffusion_conf(n_steps).make_sampler()
    zeros = torch.zeros(1, model.conf.feat_dim, device=device, dtype=torch.float32)
    feats = feats.to(device=device, dtype=torch.float32).view(1, -1)
    img_size = model.conf.img_size
    noise = torch.randn(1, 3, img_size, img_size, device=device)

    def make_model(scale: float) -> nn.Module:
        class _Wrapped(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._bb = backbone
            def forward(self, x, t, **kw):
                t_sc = sampler._scale_timesteps(t)
                eps_null = backbone.forward(x=x, t=t_sc, x_start=None, cond=zeros).pred
                if scale == 0.0:
                    return DummyReturn(pred=eps_null)
                eps_cond = backbone.forward(x=x, t=t_sc, x_start=None, cond=feats).pred
                return DummyReturn(pred=eps_null + scale * (eps_cond - eps_null))
        return _Wrapped()

    uncond = sampler.sample(model=make_model(0.0), shape=noise.shape, noise=noise, model_kwargs={}, progress=False)
    cond   = sampler.sample(model=make_model(guidance_scale), shape=noise.shape, noise=noise, model_kwargs={}, progress=False)
    return uncond, cond


def _save_pair_image(uncond, cond, out_path: Path) -> None:
    import torch
    from torchvision.utils import make_grid, save_image
    pair = torch.cat([uncond, cond], dim=0)
    grid = make_grid(pair, nrow=2, normalize=True, value_range=(-1, 1), padding=4)
    save_image(grid, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate conditional diffusion samples.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-per-subtype", type=int, default=2)
    parser.add_argument("--n-tiles", type=int, default=1, help="Tiles to sample per patient")
    parser.add_argument("--subtypes", nargs="*", default=["BRCA", "LIHC"])
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "generated_tiles")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading hparams from {run_dir / 'hparams.yaml'}", flush=True)
    conf = _build_config(run_dir)
    print(f"[2/4] resolving checkpoint", flush=True)
    ckpt_path = _resolve_checkpoint(run_dir, args.checkpoint)
    print(f"[3/4] loading model {ckpt_path.name}", flush=True)
    model, model_kind = _load_model(conf, ckpt_path)

    import torch
    from torchvision.utils import save_image

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    model = model.to(device)
    model.eval()

    print(f"[4/4] loading genomic features and selecting patients on {device}", flush=True)

    splits_raw, subtype_map = _load_splits_and_subtypes(conf.patient_splits_path)
    if args.split == "all":
        eligible = set(splits_raw.keys())
    else:
        eligible = {pid for pid, fold in splits_raw.items() if fold == args.split}

    wanted_norm = [s.strip().lower() for s in args.subtypes]
    candidates = {
        pid for pid in eligible
        if any(tok in subtype_map.get(pid, "").lower() for tok in wanted_norm)
    }

    conditioning_type = getattr(conf, "conditioning_type", "real")
    feat_dim = conf.feat_dim

    if conditioning_type == "real":
        genomic_cache, missing = _build_genomic_cache(candidates, conf.genomic_feature_dir)
        if missing:
            print(f"  warning: {len(missing)} patients have no H5 file, skipping them", flush=True)
    else:
        # Synthetic conditioning: build the same fixed vectors the dataset would return.
        import torch
        from .dataset import _ONEHOT_COHORT_INDEX
        genomic_cache = {}
        for pid in candidates:
            subtype = subtype_map.get(pid, "unknown")
            if conditioning_type == "zeros":
                genomic_cache[pid] = torch.zeros(feat_dim)
            elif conditioning_type == "noise":
                v = torch.randn(feat_dim)
                genomic_cache[pid] = v / v.norm().clamp(min=1e-8)
            elif conditioning_type == "one_hot":
                if subtype not in _ONEHOT_COHORT_INDEX:
                    raise KeyError(f"one_hot: unknown subtype '{subtype}'")
                v = torch.zeros(feat_dim)
                v[_ONEHOT_COHORT_INDEX[subtype]] = 1.0
                genomic_cache[pid] = v

    class _FakeDataset:
        def __init__(self):
            self._subtype_map = {pid: st for pid, st in subtype_map.items() if pid in eligible}
    fake_ds = _FakeDataset()

    selected = _select_patients(fake_ds, args.subtypes, args.n_per_subtype, args.seed)
    selected = [(pid, st) for pid, st in selected if pid in genomic_cache]
    if not selected:
        raise RuntimeError(f"No patients matched subtypes {args.subtypes!r} in split '{args.split}'")

    rng = random.Random(args.seed)
    manifest: list[dict[str, Any]] = []

    for index, (pid, subtype) in enumerate(selected):
        print(f"sampling {index + 1}/{len(selected)}: {subtype} {pid} ({args.n_tiles} tile(s))", flush=True)
        import torch.nn.functional as F
        feats = genomic_cache[pid].clone().to(dtype=torch.float32)
        feats = F.normalize(feats, p=2, dim=-1)   # must match training_step normalisation
        safe_subtype = subtype.replace("/", "_")
        safe_pid = pid.replace("/", "_")

        for tile_idx in range(args.n_tiles):
            noise_seed = args.seed + index * 1000 + tile_idx
            torch.manual_seed(noise_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(noise_seed)

            uncond, cond = _sample_cfg_backbone_pair(model, feats, args.guidance_scale, args.steps, device)

            suffix = f"__{tile_idx:02d}" if args.n_tiles > 1 else ""
            pair_path   = output_dir / f"{safe_subtype}__{safe_pid}{suffix}__pair.png"
            uncond_path = output_dir / f"{safe_subtype}__{safe_pid}{suffix}__uncond.png"
            cond_path   = output_dir / f"{safe_subtype}__{safe_pid}{suffix}__cond.png"

            _save_pair_image(uncond, cond, pair_path)
            save_image(uncond.clamp(-1, 1), uncond_path, normalize=True, value_range=(-1, 1))
            save_image(cond.clamp(-1, 1), cond_path, normalize=True, value_range=(-1, 1))

            manifest.append({
                "patient_id": pid, "subtype": subtype, "tile_idx": tile_idx,
                "checkpoint": str(ckpt_path), "model_kind": model_kind,
                "noise_seed": noise_seed,
                "pair_path": str(pair_path), "uncond_path": str(uncond_path), "cond_path": str(cond_path),
            })

    import enum
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, enum.Enum):
            return obj.value
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({
            "run_dir": str(run_dir), "checkpoint": str(ckpt_path),
            "model_kind": model_kind, "guidance_scale": args.guidance_scale,
            "steps": args.steps, "seed": args.seed, "split": args.split,
            "subtypes": list(args.subtypes), "selected": manifest,
            "config": asdict(conf),
        }, f, indent=2, default=_json_default)

    print(f"Wrote {len(manifest)} patient pairs to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
