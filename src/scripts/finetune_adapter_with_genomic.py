#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adapter fine‑tuning for MoPaDi – map genomic vectors → diffusion conditioning.

Key steps
---------
1️⃣ Encode a real tile into a *noisy latent* (`x_T`) with `encode_to_noise`.
   This latent lives in the same space the diffusion UNet expects.
2️⃣ Pass the genomic vector through a tiny MLP (the *adapter*) → predicted
   conditioning vector.
3️⃣ OPTIONAL: decode `x_T` with the predicted conditioning to obtain a
   reconstructed image and compute a pixel‑L2 loss.
   (You can skip step 3 to save a lot of memory – you will only train the
   adapter on a simple conditioning‑MSE loss.)
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import argparse
import os
import random
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# --------------------------------------------------------------
#   OPTIONAL: frozen CLIP model for perceptual loss (you can ignore)
# --------------------------------------------------------------
try:
    import clip  # pip install git+https://github.com/openai/CLIP.git
    _clip_model, _ = clip.load("ViT-B/32", device="cpu")
    _clip_model.eval()
    PERCEPTUAL_MODEL = _clip_model.visual
    PERCEPTUAL_MODEL.requires_grad_(False)
except Exception:
    PERCEPTUAL_MODEL = None
    print("[INFO] Perceptual model not found – perceptual loss disabled.")


# ----------------------------------------------------------------------
#   Helper: canonical patient ID (TCGA‑AB‑1234 ...)
# ----------------------------------------------------------------------
def _canonical_id(fname: str) -> str:
    name = Path(fname).stem.upper()
    for sep in ("_", "."):
        name = name.replace(sep, "-")
    while "--" in name:
        name = name.replace("--", "-")
    parts = name.split("-")
    if len(parts) >= 3 and parts[0].startswith("TCGA"):
        return "-".join(parts[:3])
    return name


# ----------------------------------------------------------------------
#   Dataset – matches zip ↔ h5 automatically
# ----------------------------------------------------------------------
class ImageFeatureDataset(Dataset):
    def __init__(
        self,
        images_zip_dir: str,
        feature_root: str,
        split: str = "train",
        feature_key: str = "feats",
        transform=None,
        random_tile: bool = True,
    ):
        super().__init__()
        self.images_zip_dir = Path(images_zip_dir).expanduser().resolve()
        self.feature_root = Path(feature_root).expanduser().resolve()
        self.split = split
        self.feature_key = feature_key
        self.transform = transform
        self.random_tile = random_tile

        # -------- zip files ----------
        zip_paths = sorted(self.images_zip_dir.glob("*.zip"))
        self._zip_by_id = { _canonical_id(p.name): p for p in zip_paths }

        # -------- h5 files ----------
        split_dir = (self.feature_root / self.split).expanduser().resolve()
        if not split_dir.is_dir():
            raise RuntimeError(f"Split dir not found: {split_dir}")
        h5_paths = sorted(split_dir.glob("*.h5"))
        self._h5_by_id = { _canonical_id(p.name): p for p in h5_paths }

        # -------- intersect ----------
        common_ids = sorted(set(self._zip_by_id) & set(self._h5_by_id))
        if not common_ids:
            raise RuntimeError(
                f"No common patient IDs between {self.images_zip_dir} and {split_dir}"
            )
        print(f"[INFO] Found {len(common_ids)} patients (split={split})")
        self.pairs = [(self._zip_by_id[i], self._h5_by_id[i]) for i in common_ids]

    def __len__(self):
        return len(self.pairs)

    # ---- read an image from a zip file ----
    @staticmethod
    def _load_from_zip(zip_path: Path, inner_name: str) -> Image.Image:
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open(inner_name) as f:
                data = f.read()
        return Image.open(BytesIO(data)).convert("RGB")

    def __getitem__(self, idx):
        zip_path, h5_path = self.pairs[idx]

        # ---- pick a tile (random or first) ----
        with zipfile.ZipFile(zip_path, "r") as zf:
            candidates = [
                m for m in zf.namelist()
                if m.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if not candidates:
                raise RuntimeError(f"No images inside {zip_path}")
            inner_name = random.choice(candidates) if self.random_tile else candidates[0]

        img = self._load_from_zip(zip_path, inner_name)
        img_t = self.transform(img) if self.transform else transforms.ToTensor()(img)

        # ---- load the genomic vector ----
        with h5py.File(h5_path, "r") as fh:
            if self.feature_key in fh:
                arr = np.array(fh[self.feature_key])
            elif "features" in fh:
                arr = np.array(fh["features"])
            else:
                raise RuntimeError(
                    f"Neither '{self.feature_key}' nor 'features' in {h5_path}"
                )
        if arr.ndim == 2:
            vec = arr.mean(axis=0)
        elif arr.ndim == 1:
            vec = arr
        else:
            raise RuntimeError(f"Unsupported shape {arr.shape} in {h5_path}")

        feat_t = torch.from_numpy(vec.astype(np.float32))

        meta = {
            "zip_path": str(zip_path),
            "inner_name": inner_name,
            "h5_path": str(h5_path),
        }
        return img_t, feat_t, meta


# ----------------------------------------------------------------------
#   Adapter MLP
# ----------------------------------------------------------------------
class AdapterMLP(nn.Module):
    """Simple fully‑connected mapper: genomic_dim → cond_dim."""
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, nlayers: int = 2):
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


# ----------------------------------------------------------------------
#   Perceptual loss (optional)
# ----------------------------------------------------------------------
def perceptual_loss(img1, img2, model=PERCEPTUAL_MODEL):
    if model is None:
        return torch.tensor(0.0, device=img1.device)
    # CLIP expects 224×224
    img1 = F.interpolate(img1, size=(224, 224), mode="bilinear", align_corners=False)
    img2 = F.interpolate(img2, size=(224, 224), mode="bilinear", align_corners=False)
    img1 = (img1 + 1.0) / 2.0   # [-1,1] → [0,1]
    img2 = (img2 + 1.0) / 2.0
    with torch.no_grad():
        f1 = model(img1)
        f2 = model(img2)
    return F.mse_loss(f1, f2)


# ----------------------------------------------------------------------
#   Training loop
# ----------------------------------------------------------------------
def train(args):
    # --------------------------------------------------------------
    #   Device / AMP
    # --------------------------------------------------------------
    device = (
        args.device
        if ("cuda" in args.device and torch.cuda.is_available())
        else "cpu"
    )
    amp_enabled = args.use_amp and device.startswith("cuda")

    # ------------------------------------------------------------------
    #   GradScaler (only for CUDA, and only if AMP is enabled)
    # ------------------------------------------------------------------
    if amp_enabled:
        # torch.cuda.amp.GradScaler has the correct signature
        scaler = torch.amp.GradScaler(device="cuda", enabled=True)
    else:
        # On CPU (or when AMP is off) we don't need a scaler.
        # We create a tiny stub so the later code can call the same API
        class _DummyScaler:
            def scale(self, loss):
                return loss

            def step(self, optimizer):
                optimizer.step()

            def update(self):
                pass

            def zero_grad(self):
                pass

        scaler = _DummyScaler()

    # --------------------------------------------------------------
    #   Output folder
    # --------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    # --------------------------------------------------------------
    #   Load MoPaDi ImageEncoder (new API)
    # --------------------------------------------------------------
    try:
        from mopadi.configs.templates import tcga_brca_autoenc
        from mopadi.utils.encode import ImageEncoder
    except Exception as exc:
        raise RuntimeError(
            "Failed to import MoPaDi – run inside the repo root where it is installed."
        ) from exc

    enc = ImageEncoder(
        tcga_brca_autoenc(),
        autoenc_path=args.diffusion_ckpt,
        device=device,
        feat_extractor=None,
    )
    enc.model.ema_model.eval()
    if args.finetune_decoder:
        enc.model.ema_model.train()
    else:
        for p in enc.model.parameters():
            p.requires_grad = False

    # --------------------------------------------------------------
    #   Infer conditioning dimension
    # --------------------------------------------------------------
    # The decoder expects a vector of size `cond_dim`.  It is stored in
    # `enc.model.conf.feat_dim` for most MoPaDi releases.
    cond_dim = getattr(enc.model.conf, "feat_dim", None)
    if cond_dim is None:
        # fallback: run a dummy forward to discover it
        dummy = torch.randn(1, 3, getattr(enc.model.conf, "img_size", 64),
                            getattr(enc.model.conf, "img_size", 64)).to(device)
        with torch.no_grad():
            # a single forward noising step is enough to reveal the shape
            latent = enc.encode_to_noise(dummy, torch.zeros(1, 1, device=device), T=1)
        cond_dim = latent.shape[-1] if latent.dim() in (2, 3) else 512
    print(f"[INFO] Detected conditioning dimension = {cond_dim}")

    # --------------------------------------------------------------
    #   Build the adapter
    # --------------------------------------------------------------
    adapter = AdapterMLP(
        in_dim=args.genomic_dim,
        out_dim=cond_dim,
        hidden=args.adapter_hidden,
        nlayers=args.adapter_layers,
    ).to(device)

    # --------------------------------------------------------------
    #   Optimiser / scheduler
    # --------------------------------------------------------------
    trainable = list(adapter.parameters())
    if args.finetune_decoder:
        trainable += list(enc.model.parameters())
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma
    )

    # --------------------------------------------------------------
    #   Dataloader
    # --------------------------------------------------------------
    img_sz = int(getattr(enc.model.conf, "img_size", args.img_size))
    transform = transforms.Compose(
        [
            transforms.Resize(
                size=img_sz, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    dataset = ImageFeatureDataset(
        images_zip_dir=args.images_zip_dir,
        feature_root=args.feature_dir,
        split=args.use_split,
        feature_key=args.feature_key,
        transform=transform,
        random_tile=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # --------------------------------------------------------------
    #   TensorBoard (optional)
    # --------------------------------------------------------------
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(log_dir=args.out_dir)

    # --------------------------------------------------------------
    #   Training
    # --------------------------------------------------------------
    global_step = 0
    torch.backends.cudnn.benchmark = True  # speed‑up for fixed image size

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        if args.finetune_decoder:
            enc.model.ema_model.train()
        epoch_loss = 0.0

        for i, (img, feat, meta) in enumerate(loader):
            img = img.to(device)               # (B,3,H,W) in [-1,1]
            feat = feat.to(device)             # (B,G)
            B = img.shape[0]

            # ------------------- 1️⃣ Encode tile to noisy latent -------------------
            # Use a *zero* conditioning vector – we only want the image‑only latent.
            neutral = torch.zeros(B, cond_dim, device=device, dtype=img.dtype)
            with torch.no_grad():
                x_T = enc.encode_to_noise(
                    img,
                    neutral,
                    T=args.encode_steps,          # <-- you can set this smaller
                )  # shape (B, C) or (B, P, C)

            # ------------------- 2️⃣ Adapter predicts conditioning -------------------
            cond_pred = adapter(feat)           # (B, C)
            # Broadcast if the decoder expects a spatial map
            if x_T.dim() == 3:  # (B,P,C)
                cond_pred = cond_pred.unsqueeze(1).expand(-1, x_T.shape[1], -1)

            # ------------------- 3️⃣ (Optional) decode back to image ----------
            if args.decode_and_reconstruct:   # <-- set this flag if you want pixel loss
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    recon = enc.decode_image(
                        x_T=x_T,
                        cond=cond_pred,
                        T=args.decode_steps,
                        deterministic=True,
                    )  # (B,3,H,W) in [-1,1]

                    loss_pix = F.mse_loss(recon, img)

                    if args.perceptual_weight > 0.0 and PERCEPTUAL_MODEL is not None:
                        loss_perc = perceptual_loss(recon, img)
                    else:
                        loss_perc = torch.tensor(0.0, device=device)

                    loss = loss_pix + args.perceptual_weight * loss_perc
            else:
                # -----------------------------------------------------------------
                #   No decoder → just train the adapter to match the *default* conditioning.
                #   This is cheap and often enough to learn a useful mapping.
                # -----------------------------------------------------------------
                # Some MoPaDi checkpoints expose a template `conds_mean` (global conditioning).
                conds_mean = getattr(enc.model, "conds_mean", None)
                if conds_mean is None:
                    # fallback: simply push the adapter toward zero (acts as a regulariser)
                    target = torch.zeros_like(cond_pred)
                else:
                    # make sure it is a tensor on the right device/dtype
                    if not isinstance(conds_mean, torch.Tensor):
                        conds_mean = torch.as_tensor(conds_mean, device=device, dtype=cond_pred.dtype)
                    else:
                        conds_mean = conds_mean.to(device=device, dtype=cond_pred.dtype)

                    if conds_mean.ndim == 2:          # (P, C)
                        if cond_pred.dim() == 3:
                            target = conds_mean.unsqueeze(0).expand(B, -1, -1)
                        else:
                            target = conds_mean.mean(dim=0).unsqueeze(0).expand(B, -1)
                    elif conds_mean.ndim == 1:        # (C,)
                        target = conds_mean.unsqueeze(0).expand(B, -1)
                    else:
                        target = torch.zeros_like(cond_pred)

                loss = F.mse_loss(cond_pred, target)

            # ------------------- 4️⃣ Back‑prop -------------------
            optimizer.zero_grad()
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            # ------------------- logging -------------------
            if (i + 1) % args.log_every == 0:
                avg = epoch_loss / (i + 1)
                print(
                    f"[E{epoch:02d}] iter {i+1}/{len(loader)}  loss={avg:.6f}"
                )
                tb.add_scalar("train/total_loss", avg, global_step)

                if args.decode_and_reconstruct:
                    tb.add_scalar("train/pix_loss", loss_pix.item(), global_step)
                    tb.add_scalar("train/perc_loss", loss_perc.item(), global_step)

                # visualise a few reconstructions (if decoding is on)
                if args.decode_and_reconstruct and args.log_images:
                    with torch.no_grad():
                        disp_rec = recon[:4].cpu() * 0.5 + 0.5   # -> [0,1]
                        disp_gt = img[:4].cpu() * 0.5 + 0.5
                        grid = torch.cat([disp_gt, disp_rec], dim=0)
                        tb.add_images("samples/gt_vs_rec", grid, global_step)

            # --------------------------------------------------------------
            #   Small memory housekeeping – clears cached fragments that
            #   PyTorch keeps around after a large forward pass.
            # --------------------------------------------------------------
            if (i + 1) % 20 == 0:   # every ~20 batches
                torch.cuda.empty_cache()

        scheduler.step()

        epoch_avg = epoch_loss / len(loader)
        print(f"[E{epoch:02d}] epoch avg loss = {epoch_avg:.6f}")
        tb.add_scalar("epoch/avg_loss", epoch_avg, epoch)

        # ------------------- checkpoint -------------------
        ckpt_path = os.path.join(args.out_dir, f"adapter_epoch{epoch}.pth")
        torch.save(
            {
                "epoch": epoch,
                "adapter_state_dict": adapter.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "avg_loss": epoch_avg,
            },
            ckpt_path,
        )
        print(f"[INFO] Saved checkpoint → {ckpt_path}")

    # ------------------- final adapter only -------------------
    final_path = os.path.join(args.out_dir, "adapter_final.pth")
    torch.save({"adapter_state_dict": adapter.state_dict()}, final_path)
    print(f"[INFO] Final adapter saved to {final_path}")
    tb.close()


# ----------------------------------------------------------------------
#   Argument parsing
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Adapter fine‑tuning for MoPaDi – map genomic ↔ diffusion conditioning"
    )
    # ---------- data ----------
    p.add_argument("--images-zip-dir", required=True,
                   help="Folder containing one *.zip per patient.")
    p.add_argument("--feature-dir", required=True,
                   help="Root folder containing train/ and test/ sub‑folders with *.h5 files.")
    p.add_argument("--use-split", default="train", choices=["train", "test"])
    p.add_argument("--feature-key", default="feats")
    # ---------- model ----------
    p.add_argument("--diffusion-ckpt", required=True,
                   help="Path to the pretrained MoPaDi diffusion checkpoint.")
    p.add_argument("--genomic-dim", type=int, default=512,
                   help="Dimensionality of the genomic vectors.")
    p.add_argument("--adapter-hidden", type=int, default=512)
    p.add_argument("--adapter-layers", type=int, default=2)
    # ---------- training ----------
    p.add_argument("--batch-size", type=int, default=2,
                   help="Batch size – lower this if you hit OOM.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-step", type=int, default=5)
    p.add_argument("--lr-gamma", type=float, default=0.5)
    p.add_argument("--encode-steps", type=int, default=250,
                   help="How many forward‑noising steps `encode_to_noise` uses. "
                        "Lower → much less memory.")
    p.add_argument("--decode-steps", type=int, default=100,
                   help="DDIM steps for the reverse diffusion (if decoding).")
    p.add_argument("--img-size", type=int, default=64,
                   help="Resize tiles to this resolution (lower → less memory).")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers – keep 0 on a single‑GPU node.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--finetune-decoder", action="store_true",
                   help="If set, the diffusion UNet will also be updated.")
    p.add_argument("--decode-and-reconstruct", action="store_true",
                   help="If set, we decode `x_T` and compute a pixel‑L2 loss "
                        "(adds a lot of memory).")
    p.add_argument("--use-amp", action="store_true",
                   help="Enable mixed‑precision (float16) training.")
    p.add_argument("--log-images", action="store_true",
                   help="Write GT vs recon images to TensorBoard (requires decoding).")
    p.add_argument("--perceptual-weight", type=float, default=0.0,
                   help="Weight for CLIP perceptual loss (0 = disabled).")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out-dir", required=True,
                   help="Where to save checkpoints and TensorBoard logs.")
    return p.parse_args()


# ----------------------------------------------------------------------
#   Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    train(args)
