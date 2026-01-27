#!/usr/bin/env python3
"""
Reconstruct a single image tile from stochastic noise using a conditional
denoising diffusion encoder (MoPaDi ImageEncoder) guided by a patient-level
feature vector and a trained classifier.

Usage (example):
  python src/scripts/reconstruct_tile_from_genomic.py --feature-h5 /data/horse/ws/mala059b-rna2wsi/vae_output/full_train/mopadi_features/train/TCGA-BH-A0BC.h5 --feature-key feats --classifier-pth /data/horse/ws/mala059b-rna2wsi/vae_output/full_train/gen_clf_luma_basal/best_model.pth --classifier-meta /data/horse/ws/mala059b-rna2wsi/vae_output/full_train/gen_clf_luma_basal/meta.json --diffusion-ckpt /data/horse/ws/mala059b-rna2wsi/Master-Thesis-Repository/notebooks/split_ckpts/diffusion_without_encoder.ckpt --target-class LumA --out /data/horse/ws/mala059b-rna2wsi/vae_output/full_train/reconstruction/out.png

Notes:
 - Requires the `mopadi` package that provides `ImageEncoder` and templates
   (this repo contains `mopadi/` so run inside the repo's environment).
 - The classifier may be the `GenomicClassifier` used in training. The script
   will instantiate a compatible model (linear or small projection) and load
   the provided state dict.
 - Guidance: the script performs a few gradient-ascent steps on the
   conditioning vector to increase the target class logit, and then decodes
   the resulting conditioning vector to an image.
"""

import argparse
import json
import os
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class GenomicClassifier(nn.Module):
    """Minimal classifier compatible with training script.
    It supports either a single linear layer (proj_dim=0) or a small
    projection MLP (proj_dim>0) matching the training code.
    """
    def __init__(self, in_dim, num_classes, proj_dim=0):
        super().__init__()
        if proj_dim and proj_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, num_classes),
            )
        else:
            self.net = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        # x: [B, D] or [B, N, D]
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.net(x)


def load_feature_vector(h5_path: str, key: str = 'feats') -> torch.Tensor:
    with h5py.File(h5_path, 'r') as fh:
        if key in fh:
            arr = np.array(fh[key])
        elif 'features' in fh:
            arr = np.array(fh['features'])
        else:
            raise RuntimeError(f"No '{key}' or 'features' dataset in {h5_path}")
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        # average tiles -> patient-level vector
        vec = arr.mean(axis=0)
    elif arr.ndim == 1:
        vec = arr
    else:
        raise RuntimeError(f"Unsupported feature shape: {arr.shape}")
    return torch.from_numpy(vec).float()


def morph_feature_with_classifier(orig_feat: torch.Tensor, model: nn.Module, target_idx: int,
                                   steps: int = 20, lr: float = 1e-2, guidance_scale: float = 1.0,
                                   l2_reg: float = 0.0, device='cpu') -> torch.Tensor:
    """Perform gradient-ascent on the conditioning vector to increase the
    target class logit. Returns a new conditioning vector (1,D).
    """
    # Try moving model to device; if CUDA init fails, fall back to CPU and continue.
    try:
        model = model.to(device).eval()
    except Exception as e:
        print(f"Warning: moving classifier to device {device} failed: {e}\nFalling back to CPU for classifier operations.")
        device = 'cpu'
        model = model.to(device).eval()
    feat = orig_feat.clone().to(device).unsqueeze(0).detach()
    feat.requires_grad_(True)
    orig = feat.detach().clone()

    for i in range(steps):
        logits = model(feat)
        # maximize the target logit
        loss = -logits[0, target_idx]
        if l2_reg > 0:
            loss = loss + l2_reg * ((feat - orig) ** 2).sum()
        model.zero_grad()
        if feat.grad is not None:
            feat.grad.zero_()
        loss.backward()
        grad = feat.grad.data
        # normalized update step
        grad_norm = grad.norm() + 1e-12
        step = (lr * guidance_scale) * (grad / grad_norm)
        feat.data = feat.data + step
        feat.grad.data.zero_()

    return feat.detach().cpu().squeeze(0)


def find_classifier_input_dim(state_dict: dict) -> int:
    # Try to infer input dim from weights in state dict
    for k in state_dict.keys():
        if k.endswith('weight'):
            w = state_dict[k]
            # w shape [out, in]
            return int(w.shape[1])
    raise RuntimeError('Could not infer classifier input dim from state dict')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-h5', required=True)
    parser.add_argument('--feature-key', default='feats')
    parser.add_argument('--diffusion-ckpt', required=True, help='Autoencoder / diffusion checkpoint (MoPaDi)')
    parser.add_argument('--classifier-pth', required=True, help='Trained classifier state_dict (pth)')
    parser.add_argument('--classifier-meta', required=True, help='JSON with class2idx mapping saved with classifier')
    parser.add_argument('--classifier-proj-dim', type=int, default=0, help='proj_dim used during classifier training (0 if linear)')
    parser.add_argument('--target-class', required=True, help='Target class name (as in classifier meta) or integer index')
    parser.add_argument('--out', required=True, help='Output PNG path')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--guidance-steps', type=int, default=20)
    parser.add_argument('--guidance-lr', type=float, default=1e-2)
    parser.add_argument('--guidance-scale', type=float, default=1.0)
    parser.add_argument('--guidance-l2', type=float, default=0.0)
    parser.add_argument('--norm-state', type=str, default=None, help='Path to mopadi_features/norm_state.pth')
    parser.add_argument('--interp-alpha', type=float, default=1.0, help='Interpolate between original and morphed cond (0..1)')
    parser.add_argument('--clip-scale', type=float, default=5.0, help='Clip morphed cond to orig +/- clip_scale*std (0=no clip)')
    parser.add_argument('--denorm-before-decode', action='store_true', help='Denormalize conditioning before decode (if norm_state used)')
    parser.add_argument('--decode-steps', type=int, default=20, help='Number of diffusion sampling steps (T) for decoder')
    parser.add_argument('--init-image', type=str, default=None, help='Path to an image to encode to noise and then decode')
    parser.add_argument('--encode-steps', type=int, default=250, help='Number of diffusion sampling steps (T) for encoding (encode_to_noise)')
    parser.add_argument('--use-model-xT', action='store_true', help='Use model.x_T stored in checkpoint as initial noise if available')
    parser.add_argument('--n-seeds', type=int, default=1, help='Number of random seeds to sample and save')
    parser.add_argument('--avg-out', action='store_true', help='Average multiple sampled images and save an additional averaged image')
    args = parser.parse_args()

    # Robust device selection: prefer requested CUDA if available, otherwise use CPU.
    requested = args.device
    if isinstance(requested, str) and requested.lower().startswith('cuda'):
        if torch.cuda.is_available():
            device = requested
        else:
            print(f"Warning: requested device {requested} but CUDA not available. Falling back to CPU.")
            device = 'cpu'
    else:
        device = 'cpu' if 'cpu' in str(requested).lower() else requested

    # load feature vector
    feat = load_feature_vector(args.feature_h5, key=args.feature_key)
    print(f'Loaded feature vector of dim {feat.numel()} from {args.feature_h5}')

    # load classifier meta
    with open(args.classifier_meta, 'r') as fh:
        meta = json.load(fh)
    class2idx = meta.get('class2idx', None) or meta.get('class_to_idx', None) or meta.get('class2index', None)
    if class2idx is None:
        raise RuntimeError('classifier meta JSON must contain a class2idx mapping under "class2idx"')

    # determine target index
    try:
        target_idx = int(args.target_class)
    except Exception:
        if args.target_class not in class2idx:
            raise RuntimeError(f'Target class {args.target_class} not found in classifier meta')
        target_idx = int(class2idx[args.target_class])

    # load classifier state
    state = torch.load(args.classifier_pth, map_location='cpu')
    # state may be a dict containing 'class2idx' or be a raw state_dict
    if isinstance(state, dict) and any(k.startswith('net') or k == 'weight' for k in state.keys()):
        state_dict = state
    else:
        state_dict = state

    # infer input dim and number of classes
    in_dim = find_classifier_input_dim(state_dict)
    # find output dim from weight shape
    for k in state_dict.keys():
        if k.endswith('weight'):
            out_dim = int(state_dict[k].shape[0])
            break

    print(f'Instantiating classifier with in_dim={in_dim}, num_classes={out_dim}, proj_dim={args.classifier_proj_dim}')
    clf = GenomicClassifier(in_dim=in_dim, num_classes=out_dim, proj_dim=args.classifier_proj_dim)
    # load state dict flexibly
    try:
        clf.load_state_dict(state_dict)
    except Exception:
        # maybe the saved file was the model.state_dict() exactly; try matching keys
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith('net.'):
                remapped[k] = v
            else:
                remapped[f'net.{k}'] = v
        clf.load_state_dict(remapped)

    # morph feature towards target class
    # Optionally load normalization state and normalize feature before classifier
    mean = None
    std = None
    if args.norm_state is not None and os.path.exists(args.norm_state):
        norm = torch.load(args.norm_state, map_location='cpu')
        if 'conds_mean' in norm and 'conds_std' in norm:
            mean = norm['conds_mean'].float()
            std = norm['conds_std'].float()
            if mean.numel() != feat.numel() or std.numel() != feat.numel():
                print('Warning: norm_state feature dim != feature vector dim; ignoring norm_state')
                mean = None
                std = None
            else:
                feat_norm = (feat - mean) / std
                print('Applied norm_state: feature normalized for classifier (mean/std applied)')
                feat_used_for_clf = feat_norm
        else:
            print('norm_state found but missing conds_mean/conds_std; ignoring')
            feat_used_for_clf = feat
    else:
        feat_used_for_clf = feat

    feat_morphed = morph_feature_with_classifier(
        feat_used_for_clf, clf, target_idx, steps=args.guidance_steps, lr=args.guidance_lr,
        guidance_scale=args.guidance_scale, l2_reg=args.guidance_l2, device=device
    )
    print('Finished classifier-guided morphing of conditioning vector')

    # Interpolate between original and morphed (in classifier space)
    alpha = float(args.interp_alpha)
    if mean is not None and std is not None:
        orig_in_clf = feat_used_for_clf
    else:
        orig_in_clf = feat
    if alpha < 1.0:
        feat_final_clf = orig_in_clf + alpha * (feat_morphed - orig_in_clf)
    else:
        feat_final_clf = feat_morphed

    # optional clipping to keep close to original: clip per-dim to orig +/- clip_scale*std
    if args.clip_scale and args.clip_scale > 0 and mean is not None and std is not None:
        clip_val = args.clip_scale * std
        feat_final_clf = torch.max(torch.min(feat_final_clf, orig_in_clf + clip_val), orig_in_clf - clip_val)

    # Prepare conditioning for decoder: denormalize if requested
    if mean is not None and std is not None and args.denorm_before_decode:
        cond_for_decode = feat_final_clf * std + mean
    else:
        cond_for_decode = feat_final_clf

    # Debug prints
    try:
        print('orig mean/std:', float(feat.mean()), float(feat.std()))
        print('used_for_clf mean/std:', float(feat_used_for_clf.mean()), float(feat_used_for_clf.std()))
        print('morphed mean/std:', float(feat_morphed.mean()), float(feat_morphed.std()))
        print('final cond mean/std:', float(feat_final_clf.mean()), float(feat_final_clf.std()))
    except Exception:
        pass

    # attempt to import MoPaDi ImageEncoder and template
    try:
        from mopadi.configs.templates import tcga_brca_autoenc
        from mopadi.utils.encode import ImageEncoder
    except Exception as e:
        raise RuntimeError('Failed to import mopadi ImageEncoder - run inside the repository environment where mopadi is available') from e

    # instantiate encoder with provided checkpoint
    enc = ImageEncoder(tcga_brca_autoenc(), autoenc_path=args.diffusion_ckpt, feat_extractor=None, device=device)

    cond = cond_for_decode.to(device).unsqueeze(0)  # [1, D]

    # Optionally encode a real image to noise and use that as initial x_T for decoding.
    initial_noise = None
    if args.init_image:
        if not os.path.exists(args.init_image):
            raise RuntimeError(f"init-image not found: {args.init_image}")
        # determine image size expected by model
        try:
            img_size = int(getattr(enc.model.conf, 'img_size', args.patch_size))
        except Exception:
            img_size = args.patch_size
        transform = transforms.Compose([
            transforms.Resize(size=img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        img = Image.open(args.init_image).convert('RGB')
        tensor_img = transform(img).unsqueeze(0).to(device)
        # encode with neutral conditioning so encoder focuses on image
        neutral_feat = torch.zeros_like(cond).to(device)
        print(f"Encoding image {args.init_image} to noise with T={args.encode_steps}...")
        xT = enc.encode_to_noise(tensor_img, neutral_feat, T=args.encode_steps)
        initial_noise = xT
        print(f"Encoded image -> noise shape {xT.shape}")
    elif args.use_model_xT:
        # use model's stored x_T if available
        model_obj = getattr(enc, 'model', None)
        if model_obj is not None and hasattr(model_obj, 'x_T'):
            try:
                initial_noise = model_obj.x_T[:1].to(device)
                print('Using model.x_T as initial noise')
            except Exception:
                initial_noise = None

    samples = []
    for s in range(args.n_seeds):
        seed = s
        torch.manual_seed(seed)
        if initial_noise is not None:
            noise = initial_noise.clone().to(device)
        else:
            try:
                img_size = int(getattr(enc.model.conf, 'img_size', args.patch_size))
            except Exception:
                img_size = args.patch_size
            noise = torch.randn(1, 3, img_size, img_size, device=device)
        # call decode with configurable number of steps (T)
        # Some diffusion configs require particular spacings (ddim striding);
        # if the requested T is incompatible the sampler will raise a ValueError.
        # Try the requested T first, then a few sensible fallbacks before failing.
        tried = []
        recon_err = None
        # Build candidate T list: prefer the requested value, then any valid
        # DDIM counts derived from the encoder's base T (if available), then
        # a few sensible defaults.
        candidates = [int(args.decode_steps)]
        base_T = None
        try:
            base_T = int(getattr(enc.model.conf, 'T', None))
        except Exception:
            base_T = None

        if base_T is not None and base_T > 0:
            # compute valid counts by trying integer strides i
            valid_counts = set()
            for i in range(1, base_T):
                valid_counts.add(len(range(0, base_T, i)))
            valid_counts = sorted(valid_counts)
            # find closest valid counts around requested T
            # include up to 6 nearest values
            diffs = sorted(valid_counts, key=lambda x: abs(x - args.decode_steps))
            for v in diffs[:6]:
                if v not in candidates:
                    candidates.append(v)

        # sensible fallback defaults
        for c in (50, 100, 150, 250, 500, 1000):
            if c not in candidates:
                candidates.append(c)

        reconstructed = None
        for candidate_T in candidates:
            if candidate_T in tried:
                continue
            tried.append(candidate_T)
            try:
                print(f"Attempting decode with T={candidate_T}")
                reconstructed = enc.decode_image(noise, cond, T=int(candidate_T))
                recon_err = None
                break
            except ValueError as e:
                recon_err = e
                print(f"Decode with T={candidate_T} failed: {e}")
                continue

        if reconstructed is None:
            # re-raise the last error to inform the user
            raise recon_err
        # ensure in [0,1]
        recon = reconstructed.squeeze().float().cpu()
        print(f'Seed {s}: recon min/max/mean = {recon.min().item():.4f}/{recon.max().item():.4f}/{recon.mean().item():.4f}')
        # convert to uint8
        img = recon.permute(1, 2, 0).numpy()
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        out_path = args.out
        if args.n_seeds > 1:
            base, ext = os.path.splitext(args.out)
            out_path = f"{base}_seed{s}{ext}"
        Image.fromarray(img).save(out_path)
        print(f'Saved reconstructed tile (seed {s}) to {out_path}')
        samples.append(recon.numpy())

    if args.n_seeds > 1 and args.avg_out:
        avg = np.mean(np.stack(samples, axis=0), axis=0)
        avg_img = np.clip(avg * 255.0, 0, 255).astype(np.uint8)
        base, ext = os.path.splitext(args.out)
        avg_path = f"{base}_avg{ext}"
        Image.fromarray(avg_img.transpose(1, 2, 0)).save(avg_path)
        print(f'Saved averaged reconstruction to {avg_path}')


if __name__ == '__main__':
    main()
