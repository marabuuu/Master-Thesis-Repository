#!/usr/bin/env python3
"""
Fine-tune / train a small adapter that maps genomic feature vectors to the
MoPaDi diffusion model's conditioning space so the decoder can accept genomic
features for reconstruction.

This script implements a simple encode->decode reconstruction training loop:
- For each (image, genomic_feature) pair:
  - Encode the image to noise x_T using `ImageEncoder.encode_to_noise(image, neutral_feat, T=encode_steps)`
    where `neutral_feat` is a zero vector (so encoder focuses on image).
  - Compute `cond = adapter(genomic_feature)` (adapter is an MLP mapping genomic_dim -> cond_dim).
  - Decode: `recon = ImageEncoder.decode_image(x_T, cond, T=decode_steps)`
  - Compute reconstruction loss (MSE) between `recon` and the original image (normalized space).
  - Backpropagate to adapter (and optionally a small portion of the decoder if `--finetune-decoder`).

Inputs/assumptions:
- A CSV file mapping image_path,feature_h5 columns is provided. Each row should
  point to one image and the HDF5 file that contains the genomic vector for
  that sample (feature_key default 'feats'). The HDF5 file can contain a
  vector or matrix (in which case it's averaged across rows to a vector).
- A pretrained diffusion/autoencoder checkpoint (MoPaDi) is available.

Example usage:
  python src/scripts/finetune_decoder_with_genomic.py \
    --pairs-csv data/pairs.csv \
    --diffusion-ckpt /path/to/diffusion.ckpt \
    --out-dir /path/to/outdir --epochs 10 --batch-size 4 --lr 1e-4

"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Tuple

import h5py
import zipfile
from io import BytesIO
try:
    import yaml
except Exception:
    yaml = None
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ImageFeatureDataset(Dataset):
    """Dataset reading either a CSV with `image_path,feature_h5` columns, or
    building pairs automatically when provided with a features root directory
    and a clinical CSV mapping patient ids to splits.

    When `images_zip_dir` is provided, the dataset will try to find a zip file
    whose filename contains the patient id and will pick the first PNG/JPEG
    inside the zip as the tile to load.
    """

    def __init__(self, pairs_csv: str = None, image_root: str = None, feature_key: str = 'feats',
                 transform=None, auto_build_from=None, images_zip_dir: str = None, use_split: str = 'train', clinical_csv: str = None):
        # auto_build_from: path to a directory containing train/ and test/ subdirs
        # and a clinical CSV (we try to discover the clinical CSV in that dir)
        self.rows = []
        self.feature_key = feature_key
        self.transform = transform
        self.images_zip_dir = images_zip_dir

        if pairs_csv is not None:
            with open(pairs_csv, 'r') as fh:
                reader = csv.DictReader(fh)
                for r in reader:
                    img = r.get('image_path')
                    feat = r.get('feature_h5')
                    if image_root and not os.path.isabs(img):
                        img = os.path.join(image_root, img)
                    if not os.path.exists(img):
                        raise FileNotFoundError(f"Image not found: {img}")
                    if not os.path.exists(feat):
                        raise FileNotFoundError(f"Feature file not found: {feat}")
                    self.rows.append((('file', img), feat))
            return

        if auto_build_from is None:
            raise RuntimeError('Either pairs_csv or auto_build_from must be provided')

        # determine clinical CSV: prefer provided clinical_csv, otherwise discover
        if clinical_csv is None:
            cand_csvs = [
                os.path.join(auto_build_from, 'clinical_table.csv'),
                os.path.join(auto_build_from, 'clinical.csv'),
                os.path.join(auto_build_from, 'clinical_table.csv'),
            ]
            clinical_csv = None
            for p in cand_csvs:
                if os.path.exists(p):
                    clinical_csv = p
                    break
            # fallback: search for any csv in auto_build_from
            if clinical_csv is None:
                for f in os.listdir(auto_build_from):
                    if f.lower().endswith('.csv'):
                        clinical_csv = os.path.join(auto_build_from, f)
                        break
        if clinical_csv is None or not os.path.exists(clinical_csv):
            raise RuntimeError(f'Could not find clinical CSV; please provide pairs_csv or a clinical CSV (searched {auto_build_from})')

        # read clinical CSV and expect a patient id column and a split column
        with open(clinical_csv, 'r') as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        # heuristics for id and split column names
        cols = rows[0].keys()
        id_col = None
        split_col = None
        for c in cols:
            lc = c.lower()
            if lc in ('patient_id', 'patientid', 'patient', 'id'):
                id_col = c
            if lc in ('split', 'set'):
                split_col = c
        if id_col is None:
            # try first column
            id_col = list(cols)[0]
        if split_col is None:
            # if no split column, assume all belong to 'train'
            split_col = None

        # build pairs by looking for feature files under auto_build_from/{split}/{patient_id}.h5
        for r in rows:
            pid = r.get(id_col)
            row_split = r.get(split_col) if split_col is not None else 'train'
            if use_split is not None and row_split.lower() != use_split.lower():
                continue
            feat_path = os.path.join(auto_build_from, row_split, f"{pid}.h5")
            if not os.path.exists(feat_path):
                # also try upper/lower variants
                if os.path.exists(os.path.join(auto_build_from, row_split, f"{pid.upper()}.h5")):
                    feat_path = os.path.join(auto_build_from, row_split, f"{pid.upper()}.h5")
                elif os.path.exists(os.path.join(auto_build_from, row_split, f"{pid.lower()}.h5")):
                    feat_path = os.path.join(auto_build_from, row_split, f"{pid.lower()}.h5")
                else:
                    # skip missing feature files
                    continue

            # find an image for this patient: look into images_zip_dir for a zip file containing pid
            img_source = None
            if images_zip_dir is not None and os.path.isdir(images_zip_dir):
                for fname in os.listdir(images_zip_dir):
                    if pid in fname and fname.lower().endswith('.zip'):
                        zip_path = os.path.join(images_zip_dir, fname)
                        # open zip and find first png/jpg
                        try:
                            with zipfile.ZipFile(zip_path, 'r') as zf:
                                members = [m for m in zf.namelist() if m.lower().endswith(('.png', '.jpg', '.jpeg'))]
                                if len(members) > 0:
                                    img_source = ('zip', zip_path, members[0])
                                    break
                        except Exception:
                            continue

            # if no image found, skip sample
            if img_source is None:
                continue

            self.rows.append((img_source, feat_path))

    def __len__(self):
        return len(self.rows)

    def _load_image_from_source(self, src):
        typ = src[0]
        if typ == 'file':
            _, path = src
            img = Image.open(path).convert('RGB')
            return img
        elif typ == 'zip':
            _, zip_path, inner = src
            with zipfile.ZipFile(zip_path, 'r') as zf:
                with zf.open(inner) as f:
                    data = f.read()
                    img = Image.open(BytesIO(data)).convert('RGB')
                    return img
        else:
            raise RuntimeError('Unknown image source type')

    def __getitem__(self, idx):
        img_src, feat_h5 = self.rows[idx]
        img = self._load_image_from_source(img_src)
        if self.transform is not None:
            img_t = self.transform(img)
        else:
            img_t = transforms.ToTensor()(img)
        # load feature vector
        with h5py.File(feat_h5, 'r') as fh:
            if self.feature_key in fh:
                arr = np.array(fh[self.feature_key])
            elif 'features' in fh:
                arr = np.array(fh['features'])
            else:
                raise RuntimeError(f"No '{self.feature_key}' or 'features' dataset in {feat_h5}")
        arr = arr.astype(np.float32)
        if arr.ndim == 2:
            vec = arr.mean(axis=0)
        elif arr.ndim == 1:
            vec = arr
        else:
            raise RuntimeError(f"Unsupported feature shape: {arr.shape} in {feat_h5}")
        feat_t = torch.from_numpy(vec).float()
        # return image tensor, feature tensor, and img_source description
        return img_t, feat_t, img_src


class AdapterMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, nlayers: int = 2):
        super().__init__()
        layers = []
        cur = in_dim
        for i in range(nlayers - 1):
            layers.append(nn.Linear(cur, hidden))
            layers.append(nn.ReLU())
            cur = hidden
        layers.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def infer_cond_dim_from_encoder(enc) -> int:
    # try several heuristics
    model = getattr(enc, 'model', enc)
    # prefer explicit conds_mean stored on model
    if hasattr(model, 'conds_mean') and model.conds_mean is not None:
        return int(model.conds_mean.shape[-1])
    # try config
    conf = getattr(model, 'conf', None)
    if conf is not None:
        if hasattr(conf, 'feat_dim') and conf.feat_dim is not None:
            return int(conf.feat_dim)
        if hasattr(conf, 'net_beatgans_embed_channels') and conf.net_beatgans_embed_channels is not None:
            return int(conf.net_beatgans_embed_channels)
    # fallback
    return 512


def train(args):
    device = args.device if (isinstance(args.device, str) and 'cuda' in args.device and torch.cuda.is_available()) else 'cpu'

    # create output dir
    os.makedirs(args.out_dir, exist_ok=True)

    # import mopadi ImageEncoder
    try:
        from mopadi.configs.templates import tcga_brca_autoenc
        from mopadi.utils.encode import ImageEncoder
    except Exception as e:
        raise RuntimeError('Failed to import mopadi ImageEncoder - run inside the repository environment where mopadi is available') from e

    enc = ImageEncoder(tcga_brca_autoenc(), autoenc_path=args.diffusion_ckpt, feat_extractor=None, device=device)
    enc.model.ema_model.eval()

    cond_dim = infer_cond_dim_from_encoder(enc)
    print(f'Detected cond dim = {cond_dim}')

    # Adapter
    adapter = AdapterMLP(in_dim=args.genomic_dim, out_dim=cond_dim, hidden=args.adapter_hidden, nlayers=args.adapter_layers)
    adapter = adapter.to(device)

    # optionally freeze decoder / encoder parameters
    if not args.finetune_decoder:
        for p in enc.model.parameters():
            p.requires_grad = False

    # Dataset + loader
    img_size = int(getattr(enc.model.conf, 'img_size', args.img_size))
    transform = transforms.Compose([
        transforms.Resize(size=img_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    # build dataset: either from provided pairs_csv or auto-build using repo config
    if getattr(args, 'pairs_csv', None):
        dataset = ImageFeatureDataset(args.pairs_csv, image_root=None, feature_key=args.feature_key, transform=transform)
    else:
        if yaml is None:
            raise RuntimeError('PyYAML is required to auto-build pairs from config.yaml. Install with `pip install pyyaml` or provide --pairs-csv')
        cfg_path = getattr(args, 'config', 'src/config.yaml')
        if not os.path.exists(cfg_path):
            raise RuntimeError(f'Config file not found: {cfg_path}')
        with open(cfg_path, 'r') as fh:
            cfg = yaml.safe_load(fh)
        feat_dir = cfg.get('feature_selection', {}).get('feats_dir')
        if feat_dir is None:
            raise RuntimeError('Cannot discover feature dir from config; please set feature_selection.feats_dir or provide --pairs-csv')
        feat_dir_abs = os.path.normpath(os.path.join(os.getcwd(), feat_dir))
        parent = feat_dir_abs
        base = os.path.basename(feat_dir_abs)
        if base.lower() in ('train', 'test'):
            parent = os.path.dirname(feat_dir_abs)
        images_zip_dir = getattr(args, 'images_zip_dir', None) or cfg.get('feature_selection', {}).get('images_zip_dir') or cfg.get('data', {}).get('data_dir')
        # discover clinical CSV from config (common keys: clini_table, clinical_table)
        clinical_csv = cfg.get('feature_selection', {}).get('clini_table') or cfg.get('feature_selection', {}).get('clinical_table') or cfg.get('data', {}).get('clini_table')
        if clinical_csv:
            clinical_csv = os.path.normpath(os.path.join(os.getcwd(), clinical_csv))
            if not os.path.exists(clinical_csv):
                # try parent folder
                alt = os.path.join(parent, os.path.basename(clinical_csv))
                if os.path.exists(alt):
                    clinical_csv = alt
                else:
                    clinical_csv = None

        dataset = ImageFeatureDataset(pairs_csv=None, image_root=None, feature_key=args.feature_key,
                                      transform=transform, auto_build_from=parent, images_zip_dir=images_zip_dir,
                                      use_split=getattr(args, 'use_split', 'train'), clinical_csv=clinical_csv)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    opt = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)

    mse = nn.MSELoss()

    for epoch in range(args.epochs):
        adapter.train()
        total_loss = 0.0
        for i, (img_t, feat_t, img_path) in enumerate(loader):
            img_t = img_t.to(device)
            feat_t = feat_t.to(device)
            B = img_t.shape[0]

            # prepare neutral feature (zeros) for encoding
            neutral = torch.zeros(B, cond_dim, device=device, dtype=next(enc.model.parameters()).dtype)

            # encode image to noise xT
            with torch.no_grad():
                xT = enc.encode_to_noise(img_t, neutral, T=args.encode_steps)

            # adapter produces cond
            cond = adapter(feat_t.to(img_t.dtype))
            cond = cond.to(device)

            # decode
            recon = enc.decode_image(xT, cond, T=args.decode_steps)

            loss = mse(recon, img_t)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())

            if (i + 1) % args.log_every == 0:
                print(f'Epoch {epoch+1} iter {i+1}/{len(loader)} loss={total_loss / (i+1):.6f}')

        scheduler.step()

        avg_loss = total_loss / len(loader)
        print(f'End epoch {epoch+1} avg_loss={avg_loss:.6f}')

        # save adapter checkpoint
        ckpt_path = os.path.join(args.out_dir, f'adapter_epoch{epoch+1}.pth')
        torch.save({'adapter_state_dict': adapter.state_dict(), 'epoch': epoch+1, 'avg_loss': avg_loss}, ckpt_path)
        print(f'Saved {ckpt_path}')

    # final save
    final_path = os.path.join(args.out_dir, 'adapter_final.pth')
    torch.save({'adapter_state_dict': adapter.state_dict()}, final_path)
    print(f'Final adapter saved to {final_path}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs-csv', required=False, help='CSV file with columns: image_path,feature_h5 (optional)')
    parser.add_argument('--feature-key', type=str, default='feats')
    parser.add_argument('--config', type=str, default='src/config.yaml', help='Repository config YAML to discover feature dirs and clinical table')
    parser.add_argument('--images-zip-dir', type=str, default=None, help='Directory containing image zip files (optional; can be set in config)')
    parser.add_argument('--use-split', type=str, default='train', help='Which split to use when auto-building pairs from features dir (train/test)')
    parser.add_argument('--diffusion-ckpt', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--genomic-dim', type=int, default=512)
    parser.add_argument('--adapter-hidden', type=int, default=512)
    parser.add_argument('--adapter-layers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-step', type=int, default=5)
    parser.add_argument('--lr-gamma', type=float, default=0.5)
    parser.add_argument('--encode-steps', type=int, default=250)
    parser.add_argument('--decode-steps', type=int, default=100)
    parser.add_argument('--img-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--finetune-decoder', action='store_true', help='Allow decoder weights to be updated (default: adapter only)')
    parser.add_argument('--log-every', type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
