#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reconstruct a WSI tile from a genomic feature vector using the
adapter that you just trained.
"""

# ------------------------------------------------------------
#  Imports (same as training script – keep versions consistent)
# ------------------------------------------------------------
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# MoPaDi imports – make sure the repo is on PYTHONPATH or install it
from mopadi.configs.templates import tcga_brca_autoenc
from mopadi.utils.encode import ImageEncoder

# ------------------------------------------------------------
#  Helper – same canonical‑ID function used by the dataset
# ------------------------------------------------------------
def canonical_id(fname: str) -> str:
    name = Path(fname).stem.upper()
    for sep in ("_", "."):
        name = name.replace(sep, "-")
    while "--" in name:
        name = name.replace("--", "-")
    parts = name.split("-")
    if len(parts) >= 3 and parts[0].startswith("TCGA"):
        return "-".join(parts[:3])
    return name


# ------------------------------------------------------------
#  Tiny adapter definition – must match the training architecture
# ------------------------------------------------------------
class AdapterMLP(nn.Module):
    """Same class as in the training script."""
    def __init__(self, in_dim: int, out_dim: int,
                 hidden: int = 512, nlayers: int = 2):
        super().__init__()
        layers = []
        cur = in_dim
        for _ in range(nlayers - 1):
            layers.append(nn.Linear(cur, hidden))
            layers.append(nn.ReLU(inplace=True))
            cur = hidden
        layers.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------
#  Load a *single* genomic vector from an H5 file
# ------------------------------------------------------------
def load_genomic_vector(h5_path: Path,
                        key: str = "feats") -> torch.Tensor:
    """Returns a (D,) FloatTensor."""
    with h5py.File(h5_path, "r") as f:
        if key in f:
            arr = np.array(f[key])
        elif "features" in f:
            arr = np.array(f["features"])
        else:
            raise RuntimeError(f"Key not found in {h5_path}")
        # Collapse possible (N, D) -> (D,) by averaging across N
        if arr.ndim == 2:
            vec = arr.mean(axis=0)
        elif arr.ndim == 1:
            vec = arr
        else:
            raise RuntimeError(f"Unexpected shape {arr.shape}")
    return torch.from_numpy(vec.astype(np.float32))


# ------------------------------------------------------------
#  Main reconstruction routine
# ------------------------------------------------------------
def reconstruct_one(
    enc: ImageEncoder,
    adapter: AdapterMLP,
    genomic_vec: torch.Tensor,
    device: torch.device,
    encode_steps: int = 50,
    decode_steps: int = 100,
    img_size: int = 64,
    use_amp: bool = True,
) -> torch.Tensor:
    """
    Returns a tensor (3, H, W) in the [-1, 1] range – same range that MoPaDi uses.
    """
    # 1️⃣ Put everything on the right device / dtype
    genomic_vec = genomic_vec.to(device).unsqueeze(0)   # shape (1, G)

    # 2️⃣ Create a *neutral* latent (zero conditioning) – same as training
    #    We pick a dummy image just to obtain the correct latent shape.
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    zero_cond = torch.zeros(1, enc.model.conf.feat_dim,
                            device=device, dtype=dummy.dtype)

    # 3️⃣ Encode to noisy latent x_T
    with torch.no_grad():
        x_T = enc.encode_to_noise(
            dummy,          # image (only shape matters)
            zero_cond,
            _T_=encode_steps,
        )

    # 4️⃣ Predict conditioning from the genomic vector
    with torch.no_grad():
        cond_pred = adapter(genomic_vec)            # (1, C)

        # If the latent is (B, P, C) we need to broadcast over the spatial dim.
        if x_T.dim() == 3:          # (B, P, C)
            cond_pred = cond_pred.unsqueeze(1).expand(-1, x_T.shape[1], -1)

    # 5️⃣ Decode – use AMP if requested
    if use_amp and device.type == "cuda":
        ctx = torch.amp.autocast('cuda', enabled=True)
    else:
        ctx = torch.no_grad()  # no grad, but no autocast

    with ctx:
        recon = enc.decode_image(
            x_T,               # latent
            cond_pred,         # conditioning
            decode_steps,      # DDIM steps
            deterministic=True,
        )                       # (1, 3, H, W) in [-1, 1]

    return recon.squeeze(0)    # (3, H, W)


# ------------------------------------------------------------
#  Helper – convert [-1,1] tensor → PIL image and save
# ------------------------------------------------------------
def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """
    img_tensor: (3, H, W) in [-1, 1]
    Returns a PIL RGB image (uint8).
    """
    img = (img_tensor.clamp(-1, 1) + 1.0) * 0.5   # -> [0,1]
    img = img.cpu().numpy().transpose(1, 2, 0)    # H,W,3
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


# ------------------------------------------------------------
#  Argument parser (mirrors training for reproducibility)
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a tile from a genomic vector using a trained adapter."
    )
    p.add_argument("--diffusion-ckpt", required=True,
                   help="Path to the MoPaDi diffusion checkpoint (same used for training)")
    p.add_argument("--adapter-ckpt", required=True,
                   help="Path to the adapter checkpoint (e.g. adapter_final.pth)")
    p.add_argument("--genomic-h5", required=True,
                   help="H5 file that contains the genomic vector you want to visualise")
    p.add_argument("--feature-key", default="feats",
                   help="Key inside the H5 containing the vector")
    p.add_argument("--out-img", required=True,
                   help="Where to write the reconstructed PNG/JPG")
    p.add_argument("--encode-steps", type=int, default=50,
                   help="Number of forward‑noising steps (must match training)")
    p.add_argument("--decode-steps", type=int, default=100,
                   help="DDIM steps for reverse diffusion")
    p.add_argument("--img-size", type=int, default=64,
                   help="Resolution that the MoPaDi model expects")
    p.add_argument("--device", default="cuda:0",
                   help="torch device – e.g. cuda:0 or cpu")
    p.add_argument("--use-amp", action="store_true",
                   help="Enable mixed‑precision when decoding (only on CUDA)")
    p.add_argument("--adapter-hidden", type=int, default=512,
                   help="Hidden size of the adapter (must match training)")
    p.add_argument("--adapter-layers", type=int, default=2,
                   help="Number of MLP layers (must match training)")
    p.add_argument("--genomic-dim", type=int, default=512,
                   help="Dimensionality of the genomic vector (must match training)")
    return p.parse_args()


# ------------------------------------------------------------
#  Main entry point
# ------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device(
        args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu"
    )
    print(f"[INFO] Using device: {device}")

    # -----------------------------------------------------------------
    # Load MoPaDi encoder (same as training)
    # -----------------------------------------------------------------
    try:
        from mopadi.configs.templates import tcga_brca_autoenc
        from mopadi.utils.encode import ImageEncoder
    except Exception as exc:
        raise RuntimeError(
            "MoPaDi import failed – make sure you run this from the repo root "
            "or have the package installed."
        ) from exc

    enc = ImageEncoder(
        tcga_brca_autoenc(),
        _autoenc_path_=args.diffusion_ckpt,
        _device_=device,
        _feat_extractor_=None,
    )
    enc.model.ema_model.eval()                     # keep decoder frozen
    for p in enc.model.parameters():
        p.requires_grad = False

    # -----------------------------------------------------------------
    # Infer conditioning dimension (same logic as training script)
    # -----------------------------------------------------------------
    cond_dim = getattr(enc.model.conf, "feat_dim", None)
    if cond_dim is None:
        dummy = torch.randn(
            1, 3, getattr(enc.model.conf, "img_size", 64),
            getattr(enc.model.conf, "img_size", 64),
            device=device,
        )
        with torch.no_grad():
            latent = enc.encode_to_noise(dummy,
                                         torch.zeros(1, 1, device=device),
                                         _T_=1)
        cond_dim = latent.shape[-1] if latent.dim() in (2, 3) else 512
    print(f"[INFO] Conditioning dimension = {cond_dim}")

    # -----------------------------------------------------------------
    # Build the adapter and load its weights
    # -----------------------------------------------------------------
    adapter = AdapterMLP(
        in_dim=args.genomic_dim,
        out_dim=cond_dim,
        hidden=args.adapter_hidden,
        nlayers=args.adapter_layers,
    ).to(device)

    ckpt = torch.load(args.adapter_ckpt, map_location=device)
    # The checkpoint may be either the “final” dict or a full training dict.
    if "adapter_state_dict" in ckpt:
        adapter.load_state_dict(ckpt["adapter_state_dict"])
    else:
        adapter.load_state_dict(ckpt)
    adapter.eval()
    print("[INFO] Adapter weights loaded.")

    # -----------------------------------------------------------------
    # Load the genomic vector we want to visualise
    # -----------------------------------------------------------------
    geno_vec = load_genomic_vector(Path(args.genomic_h5), key=args.feature_key)
    print(f"[INFO] Loaded genomic vector of shape {geno_vec.shape}")

    # -----------------------------------------------------------------
    # Reconstruct
    # -----------------------------------------------------------------
    recon_tensor = reconstruct_one(
        enc=enc,
        adapter=adapter,
        genomic_vec=geno_vec,
        device=device,
        encode_steps=args.encode_steps,
        decode_steps=args.decode_steps,
        img_size=args.img_size,
        use_amp=args.use_amp,
    )
    # Convert to PIL and write to disk
    recon_img = tensor_to_pil(recon_tensor)
    out_path = Path(args.out_img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recon_img.save(out_path)
    print(f"[INFO] Saved reconstructed tile -> {out_path}")


if __name__ == "__main__":
    main()