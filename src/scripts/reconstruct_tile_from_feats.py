#!/usr/bin/env python3
"""
Standalone helper to reconstruct image tiles from per-sample H5 feature files
using a MoPaDi diffusion checkpoint. This script is written so you can copy
it into another repository and run it there; pass `--mopadi-root` if mopadi
is not on `PYTHONPATH`.

Example usage:
python src/scripts/reconstruct_tile_from_feats.py \
  --ckpt /data/horse/ws/mala059b-rna2wsi/models/TCGA-BRCA/diffusion_split.ckpt \
  --feat /data/horse/ws/mala059b-rna2wsi/512_vae/mopadi_features/train/TCGA-5L-AAT1.h5 \
  --hparams /data/horse/ws/mala059b-rna2wsi/models/TCGA-BRCA/models--KatherLab--MoPaDi/snapshots/5d8e775e24473c5d8f4c0c57fd5c865c3c2a4aab/brca_512_model/autoenc_hparams.yaml \
  --out-dir /data/horse/ws/mala059b-rna2wsi/diffusion_output \
  --device cuda \
  --img-size 224 \
  --noisy-seed 42

The script expects each per-sample HDF5 file to contain a dataset named
`feats` or `feat` (1-D or 2-D where the first row is used). If your files
store features under a different key, pass `--feat-key`.
"""
import argparse
import os
import sys
import glob
import h5py
import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from dataclasses import dataclass
import typing
import yaml


def add_mopadi_to_path(mopadi_root):
    if mopadi_root:
        mopadi_root = os.path.abspath(mopadi_root)
        if mopadi_root not in sys.path:
            sys.path.insert(0, mopadi_root)


def load_feature_from_h5(path, feat_key='feats', expected_dim: int = None):
    """Load feature array from H5 and return a 1-D vector.

    If the dataset contains a spatial map (e.g. CxHxW or HxWxC), this
    function will collapse the spatial dimensions by global averaging to
    produce a vector of length `C`. If `expected_dim` is provided the
    code will try to detect which axis corresponds to channels.
    """
    with h5py.File(path, 'r') as f:
        if feat_key in f:
            arr = np.array(f[feat_key])
        else:
            # try fallback keys
            if 'feat' in f:
                arr = np.array(f['feat'])
            else:
                # try first dataset
                keys = list(f.keys())
                if not keys:
                    raise KeyError(f"No datasets found in H5 file {path}")
                arr = np.array(f[keys[0]])

    # Convert to float32 for model compatibility
    arr = arr.astype(np.float32)

    # If already a vector, return as-is
    if arr.ndim == 1:
        return arr

    # If 2-D, treat as (N, C) or (C, N). Prefer returning a single row
    if arr.ndim == 2:
        # If one dimension equals expected_dim, collapse the other
        if expected_dim is not None:
            if arr.shape[0] == expected_dim:
                return arr.mean(axis=1)
            if arr.shape[1] == expected_dim:
                return arr.mean(axis=0)
        # fallback: return first row
        return arr[0].ravel()

    # For 3-D or higher (spatial maps), try to identify channel axis
    # using expected_dim if available, otherwise assume channel-first.
    if expected_dim is not None:
        for axis in range(arr.ndim):
            if arr.shape[axis] == expected_dim:
                # move channel axis to front then average over remaining axes
                a = np.moveaxis(arr, axis, 0)
                vec = a.mean(axis=tuple(range(1, a.ndim)))
                return vec

    # Default: assume channel is first axis (C, H, W, ...)
    if arr.shape[0] <= arr.size and arr.ndim >= 3:
        vec = arr.mean(axis=tuple(range(1, arr.ndim)))
        return vec

    # As a last resort, flatten to 1-D and return
    return arr.ravel()


def make_train_conf_default(img_size=224):
    # Lazy import to keep script lightweight until mopadi is available
    from mopadi.configs.config import TrainConfig
    from mopadi.configs.choices import ModelName

    conf = TrainConfig()
    # typical autoencoder model name used in mopadi training
    conf.model_name = ModelName.beatgans_autoenc
    conf.img_size = img_size
    conf.sample_size = img_size
    conf.base_dir = os.getcwd()
    # Compatibility: some mopadi versions expect `net_num_res_blocks`
    # while others name it `net_num_input_res_blocks`. Provide a
    # sensible fallback so this helper works across versions.
    if not hasattr(conf, 'net_num_res_blocks'):
        value = getattr(conf, 'net_num_input_res_blocks', 2)
        setattr(typing.cast(typing.Any, conf), 'net_num_res_blocks', value)
    # Always set channel_mult using setattr to avoid type errors
    setattr(typing.cast(typing.Any, conf), 'channel_mult', [1, 2, 4, 8])
    # Ensure numeric residual block counts exist (different mopadi versions use
    # `num_input_res_blocks` / `num_res_blocks` or `net_num_input_res_blocks` / `net_num_res_blocks`).
    num_input_rb = getattr(conf, 'num_input_res_blocks', None)
    if num_input_rb is None:
        num_input_rb = getattr(conf, 'num_input_res_blocks', None) or getattr(conf, 'num_res_blocks', None)
        num_input_rb = num_input_rb or getattr(conf, 'net_num_input_res_blocks', None) or getattr(conf, 'net_num_res_blocks', None) or 2
        setattr(typing.cast(typing.Any, conf), 'num_input_res_blocks', int(num_input_rb))
    if not hasattr(conf, 'num_res_blocks') or getattr(conf, 'num_res_blocks', None) is None:
        setattr(typing.cast(typing.Any, conf), 'num_res_blocks', int(getattr(conf, 'num_input_res_blocks', 2)))
    # Ensure attention_resolutions exists (list of ints or empty means no attention)
    if not hasattr(conf, 'attention_resolutions') or getattr(conf, 'attention_resolutions', None) is None:
        setattr(typing.cast(typing.Any, conf), 'attention_resolutions', [])
    # Monkeypatch TrainConfig.make_model_conf so the returned model
    # config contains required fields even if the mopadi version
    # populating it leaves them unset.
    try:
        orig_make_model_conf = TrainConfig.make_model_conf

        def _make_model_conf_with_defaults(self):
            model_conf = orig_make_model_conf(self)
            # ensure channel_mult present on model_conf
            if getattr(model_conf, 'channel_mult', None) is None:
                setattr(typing.cast(typing.Any, model_conf), 'channel_mult', getattr(self, 'channel_mult', [1, 2, 4, 8]))
            # ensure net_num_res_blocks present on model_conf
            if not hasattr(model_conf, 'net_num_res_blocks'):
                setattr(typing.cast(typing.Any, model_conf), 'net_num_res_blocks', getattr(self, 'net_num_res_blocks', getattr(self, 'net_num_input_res_blocks', 2)))
            # ensure numeric residual block counts exist on model_conf
            if getattr(model_conf, 'num_input_res_blocks', None) is None:
                nij = getattr(self, 'num_input_res_blocks', None) or getattr(self, 'num_res_blocks', None)
                nij = nij or getattr(self, 'net_num_input_res_blocks', None) or getattr(self, 'net_num_res_blocks', None) or 2
                setattr(typing.cast(typing.Any, model_conf), 'num_input_res_blocks', int(nij))
            if getattr(model_conf, 'num_res_blocks', None) is None:
                setattr(typing.cast(typing.Any, model_conf), 'num_res_blocks', int(getattr(model_conf, 'num_input_res_blocks', 2)))
            # ensure attention_resolutions exists on model_conf
            if getattr(model_conf, 'attention_resolutions', None) is None:
                setattr(typing.cast(typing.Any, model_conf), 'attention_resolutions', getattr(self, 'attention_resolutions', []))
            return model_conf

        TrainConfig.make_model_conf = _make_model_conf_with_defaults
    except Exception:
        # If monkeypatching fails, we still proceed — model construction
        # will raise an informative error which the user can report.
        pass
    return conf


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mopadi-root', default=None,
                   help='Path to mopadi repo root (so `import mopadi` works)')
    p.add_argument('--ckpt', required=True, help='Path to diffusion/autoenc checkpoint (.ckpt)')
    p.add_argument('--feat', required=True,
                   help='Path to single H5 feature file or to a directory (will process *.h5)')
    p.add_argument('--feat-key', default='feats', help='HDF5 dataset key for features')
    p.add_argument('--out-dir', default='reconstructions', help='Directory to save reconstructed PNGs')
    p.add_argument('--device', default='cuda', help='Torch device (cuda or cpu)')
    p.add_argument('--img-size', type=int, default=224, help='Image size (must match model)')
    p.add_argument('--hparams', default=None, help='Path to training hparams YAML (optional)')
    p.add_argument('--noisy-seed', type=int, default=None, help='Seed for initial noise (optional)')
    p.add_argument('--batch-size', type=int, default=1, help='Batch size for sampling')
    args = p.parse_args()

    add_mopadi_to_path(args.mopadi_root)

    # Import mopadi pieces
    try:
        from mopadi.train_diff_autoenc import LitModel
    except Exception as e:
        print('Failed to import mopadi. Did you pass --mopadi-root or install mopadi?')
        raise

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')

    # Build a minimal config and instantiate Lightning model (weights will be loaded non-strict)
    conf = make_train_conf_default(img_size=args.img_size)
    # Ensure some TrainConfig fields exist that newer/older mopadi versions
    # may expect; provide safe defaults so the script can instantiate
    # the Lightning model without requiring a full training config.
    if not hasattr(conf, 'load_pretrained_autoenc'):
        setattr(typing.cast(typing.Any, conf), 'load_pretrained_autoenc', False)
    if not hasattr(conf, 'data_dirs'):
        setattr(typing.cast(typing.Any, conf), 'data_dirs', [])
    if not hasattr(conf, 'feature_dirs'):
        setattr(typing.cast(typing.Any, conf), 'feature_dirs', [])
    if not hasattr(conf, 'do_resize'):
        setattr(typing.cast(typing.Any, conf), 'do_resize', False)
    if not hasattr(conf, 'do_normalize'):
        setattr(typing.cast(typing.Any, conf), 'do_normalize', False)
    # If a training hparams YAML is supplied, overlay its top-level keys onto conf
    if args.hparams is not None:
        hpath = os.path.expanduser(args.hparams)
        if not os.path.isfile(hpath):
            raise FileNotFoundError(f'Hparams YAML not found: {hpath}')
        # Read as text first so we can retry with a different loader if needed
        with open(hpath, 'r') as fh:
            text = fh.read()
        try:
            h = yaml.safe_load(text)
        except yaml.constructor.ConstructorError:
            # Some hparams YAML files contain Python-specific tags
            # (e.g. !!python/tuple). `safe_load` rejects those, so
            # fall back to `FullLoader` which understands those tags.
            h = yaml.load(text, Loader=yaml.FullLoader)
        if not isinstance(h, dict):
            raise RuntimeError(f'Invalid hparams YAML format: {hpath}')
        for k, v in h.items():
            try:
                setattr(typing.cast(typing.Any, conf), k, v)
            except Exception:
                # best-effort: ignore keys that cannot be set
                pass
    lit = LitModel(conf)

    # Load checkpoint
    print(f'Loading checkpoint: {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = None
    # common lightning / training checkpoint layouts
    for key in ('state_dict', 'state_dict_ema', 'model_state_dict'):
        if key in ckpt:
            state_dict = ckpt[key]
            break
    if state_dict is None:
        # maybe the checkpoint *is* a state_dict
        if isinstance(ckpt, dict) and any(k.startswith('model') or k.startswith('state_dict') for k in ckpt.keys()):
            # try common key
            state_dict = ckpt.get('state_dict', ckpt)
        else:
            # fallback: try loading entire ckpt into lit
            try:
                lit.load_state_dict(ckpt, strict=False)
                print('Loaded checkpoint as raw state dict into model (strict=False)')
            except Exception:
                raise RuntimeError('Could not find state_dict in checkpoint; please provide a Lightning checkpoint or a model state_dict.')

    if state_dict is not None:
        try:
            lit.load_state_dict(state_dict, strict=False)
            print('Loaded `state_dict` into model (strict=False)')
        except Exception as e:
            print('Warning: loading state dict with strict=False failed; attempting direct assignment where possible')

    lit.eval()
    lit.to(device)

    # prefer ema model for sampling if available
    model = getattr(lit, 'ema_model', None) or lit.model
    sampler = getattr(lit, 'eval_sampler', None) or getattr(lit, 'sampler')
    if sampler is None:
        raise RuntimeError('Sampler not available on the loaded LitModel')

    os.makedirs(args.out_dir, exist_ok=True)

    # gather feature files
    feat_path = args.feat
    if os.path.isdir(feat_path):
        files = sorted(glob.glob(os.path.join(feat_path, '*.h5')))
    else:
        files = [feat_path]

    if not files:
        raise FileNotFoundError(f'No H5 feature files found at {feat_path}')

    # Optional normalization if lit has conds_mean / conds_std registered
    conds_mean = getattr(lit, 'conds_mean', None)
    conds_std = getattr(lit, 'conds_std', None)
    if conds_mean is not None and conds_std is not None:
        print('Using conds_mean and conds_std from loaded model for feature normalization')
        conds_mean = conds_mean.to(device)
        conds_std = conds_std.to(device)

    for i in range(0, len(files), args.batch_size):
        batch_files = files[i:i+args.batch_size]
        feats = []
        names = []
        for fpath in batch_files:
            feat = load_feature_from_h5(fpath, feat_key=args.feat_key, expected_dim=getattr(conf, 'feat_dim', None))
            feats.append(torch.from_numpy(feat))
            names.append(os.path.splitext(os.path.basename(fpath))[0])

        cond = torch.stack([f.float() for f in feats], dim=0)
        cond = cond.to(device)

        if conds_mean is not None and conds_std is not None:
            # try to broadcast
            cond = (cond - conds_mean) / (conds_std + 1e-8)

        # sample noise x_T. Use the model `conf.img_size` (possibly from hparams)
        # to avoid mismatches with the trained model. Warn if user-provided
        # `--img-size` differs from the loaded config.
        b = cond.shape[0]
        img_size = getattr(conf, 'img_size', args.img_size)
        if args.img_size != img_size:
            print(f"Warning: requested --img-size={args.img_size} differs from model conf.img_size={img_size}; using conf.img_size")
        if args.noisy_seed is not None:
            rng = torch.Generator(device=device)
            rng.manual_seed(args.noisy_seed)
            x_T = torch.randn(b, 3, img_size, img_size, device=device, generator=rng)
        else:
            x_T = torch.randn(b, 3, img_size, img_size, device=device)

        with torch.no_grad():
            out = sampler.sample(model=model, noise=x_T, cond=cond, x_start=None)

        # `out` should be a tensor in [-1, 1]
        if isinstance(out, dict) and 'sample' in out:
            samples = out['sample']
        else:
            samples = out

        # clamp and scale to [0,1]
        samples = (samples + 1.0) / 2.0
        samples = samples.clamp(0, 1)

        for s, name in zip(samples, names):
            # save using torchvision
            path = os.path.join(args.out_dir, f'{name}.png')
            save_image(s, path)
            print(f'Saved {path}')

    print('Done')

if __name__ == '__main__':
    main()
