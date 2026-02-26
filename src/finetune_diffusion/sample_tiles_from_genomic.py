#!/usr/bin/env python
"""Sample tiles from a finetuned genomic-conditioned diffusion model.

Two modes:
 - `random`: generate tiles from random noise conditioned on genomic vectors
 - `reconstruct`: add noise to a real tile and decode guided by genomic vector

Usage examples are printed with `-h`.

python src/finetune_diffusion/sample_tiles_from_genomic.py \
  --ckpt path/to/last.ckpt \
  --genomic-h5-dir /path/to/genomic_h5_dir \
  --mode random \
  --n-samples 4 \
  --out-dir ./samples_random

python src/finetune_diffusion/sample_tiles_from_genomic.py \
  --ckpt path/to/last.ckpt \
  --genomic-h5-dir /path/to/genomic_h5_dir \
  --mode reconstruct \
  --tile-file /path/to/tile.png \
  --out-dir ./samples_recon \
  --n-samples 4
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import glob
import h5py
import numpy as np
import copy
import torch
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image

# Import helpers from finetune script module
from finetune_diffusion.finetune_diffusion_with_genomic import (
	ProjectionHead,
	make_tile_transform,
	tcga_brca_autoenc,
	LitDiffusionGenomicV4,
	canonical_patient_id,
)


def load_genomic_vector(h5_path: Path, key: str = "feats") -> torch.Tensor:
	with h5py.File(h5_path, "r") as f:
		dset = f[key]
		arr: np.ndarray = np.asarray(dset)
	if arr.ndim == 2:
		arr = arr.mean(axis=0)
	return torch.from_numpy(arr.astype(np.float32))


def load_tile_image(path: Path, img_size: int) -> torch.Tensor:
	img = Image.open(path).convert("RGB")
	xform = make_tile_transform(img_size)
	out = xform(img)
	# help static analyzers: ensure we return a Tensor supporting .unsqueeze
	assert isinstance(out, torch.Tensor)
	return out


def main():
	parser = argparse.ArgumentParser(description="Sample tiles from genomic-conditioned diffusion")
	parser.add_argument("--ckpt", required=True, help="Diffusion checkpoint (last.ckpt or .ckpt file)")
	parser.add_argument("--proj-ckpt", default=None, help="Optional projection head checkpoint")
	parser.add_argument("--genomic-h5-dir", required=True, help="Directory with per-patient .h5 genomic files")
	parser.add_argument("--tiles-zip-dir", default=None, help="Optional: directory of tile .zip archives (for reconstruct mode) or individual tile files")
	parser.add_argument("--mode", choices=["random", "reconstruct"], default="random")
	parser.add_argument("--patient", default=None, help="Optional patient id to sample (canonical TCGA-XX-YYYY)")
	parser.add_argument("--tile-file", default=None, help="Path to a tile image file to reconstruct (reconstruct mode)")
	parser.add_argument("--out-dir", required=True)
	parser.add_argument("--n-samples", type=int, default=4)
	parser.add_argument("--img-size", type=int, default=512)
	parser.add_argument("--proj-hidden-dim", type=int, default=512)
	parser.add_argument("--proj-layers", type=int, default=4)
	parser.add_argument("--proj-dropout", type=float, default=0.1)
	parser.add_argument("--device", default=None, help="torch device (e.g., cuda:0 or cpu)")
	parser.add_argument("--seed", type=int, default=42)

	args = parser.parse_args()

	torch.manual_seed(args.seed)

	device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	# Build minimal MoPaDi config and model wrapper (matching training)
	conf = tcga_brca_autoenc()
	conf.img_size = args.img_size
	conf.sample_size = args.n_samples

	proj_head = ProjectionHead(in_dim=512, out_dim=512, hidden_dim=args.proj_hidden_dim,
							   num_layers=args.proj_layers, dropout=args.proj_dropout)

	# Instantiate LitDiffusionGenomicV4. We only need its `model`, `eval_sampler`, and `projection_head`.
	lit = LitDiffusionGenomicV4(
		conf=conf,
		projection_head=proj_head,
		genomic_h5_dir=args.genomic_h5_dir,
		tiles_zip_dir=(args.tiles_zip_dir or ""),
		tiles_per_patient=1,
		split="all",
		img_size=args.img_size,
		proj_lr=1e-4,
		proj_warmup=0,
		n_log_samples=args.n_samples,
	)

	# Load checkpoint (same logic as finetune script)
	ckpt = torch.load(args.ckpt, map_location="cpu")
	if "state_dict" in ckpt:
		state = dict(ckpt["state_dict"])
		# Drop x_T mismatch if present
		if "x_T" in state and hasattr(lit, "x_T") and state["x_T"].shape != getattr(lit, "x_T").shape:
			del state["x_T"]
		lit.load_state_dict(state, strict=False)
		print(f"Loaded state_dict from {args.ckpt}")
	else:
		# Bare model weights
		try:
			lit.model.load_state_dict(ckpt, strict=False)
			lit.ema_model = copy.deepcopy(lit.model)
			# Freeze EMA model parameters and set eval mode
			lit.ema_model.requires_grad_(False)
			lit.ema_model.eval()
			print(f"Loaded bare model weights from {args.ckpt}")
		except Exception:
			# Fallback: try to load into full lit
			lit.load_state_dict(ckpt, strict=False)
			print(f"Loaded checkpoint into wrapper from {args.ckpt}")

	# Load projection head if separate ckpt provided
	if args.proj_ckpt:
		ppath = Path(args.proj_ckpt)
		if ppath.exists():
			pck = torch.load(ppath, map_location="cpu")
			# Accept either {'state_dict': ...} or raw
			state = pck.get("state_dict", pck) if isinstance(pck, dict) else pck
			try:
				lit.projection_head.load_state_dict(state, strict=False)
				print(f"Loaded projection head from {args.proj_ckpt}")
			except Exception as e:
				print(f"Could not load proj ckpt: {e}")

	lit.to(device)
	lit.eval()
	lit.projection_head.to(device).eval()

	# Gather genomic files
	gdir = Path(args.genomic_h5_dir)
	h5_files = sorted(gdir.glob("**/*.h5"))
	if not h5_files:
		raise RuntimeError(f"No .h5 genomic files found in {gdir}")

	# Filter by patient if requested
	if args.patient:
		patient_key = canonical_patient_id(args.patient)
		h5_files = [p for p in h5_files if canonical_patient_id(p.name) == patient_key]
		if not h5_files:
			raise RuntimeError(f"No genomic file found for patient {args.patient}")

	# Sampling
	if args.mode == "random":
		for h5p in tqdm(h5_files, desc="Genomic files"):
			pid = canonical_patient_id(h5p.name)
			vec = load_genomic_vector(h5p).to(device)
			cond = lit.projection_head(vec.unsqueeze(0))
			# repeat cond for batch
			cond = cond.repeat(args.n_samples, 1)

			noise = torch.randn(args.n_samples, 3, args.img_size, args.img_size, device=device)

			with torch.no_grad():
				try:
					gen = lit.eval_sampler.sample(model=lit.model, noise=noise, cond=cond)
				except Exception:
					# Fallback to EMA model if available, otherwise use lit.model
					fallback_model = getattr(lit, "ema_model", None) or lit.model
					gen = lit.eval_sampler.sample(model=fallback_model, noise=noise, cond=cond)

			# gen expected in [-1,1]
			save_path = out_dir / f"{pid}_random_samples.png"
			save_image((gen + 1) / 2, save_path, nrow=min(4, args.n_samples))

	else:  # reconstruct
		if args.tile_file is None:
			raise RuntimeError("--tile-file required in reconstruct mode")
		tile_path = Path(args.tile_file)
		if not tile_path.exists():
			raise RuntimeError(f"Tile file not found: {tile_path}")

		img = load_tile_image(tile_path, args.img_size).unsqueeze(0).to(device)

		# If patient specified, find matching genomic h5, otherwise try to infer from tile filename
		if args.patient:
			h5p = next((p for p in h5_files if canonical_patient_id(p.name) == canonical_patient_id(args.patient)), None)
		else:
			# attempt to extract patient id from tile filename
			pid = canonical_patient_id(tile_path.name)
			h5p = next((p for p in h5_files if canonical_patient_id(p.name) == pid), None)

		if h5p is None:
			raise RuntimeError("No matching genomic file found for the provided tile (specify --patient or ensure filenames include TCGA- ids)")

		vec = load_genomic_vector(h5p).to(device)
		cond = lit.projection_head(vec.unsqueeze(0))
		cond = cond.repeat(args.n_samples, 1)

		# Prepare noisy starts: repeat the single input tile
		img_batch = img.repeat(args.n_samples, 1, 1, 1)
		noise = torch.randn_like(img_batch)

		with torch.no_grad():
			try:
				gen = lit.eval_sampler.sample(model=lit.model, noise=noise, cond=cond, x_start=img_batch)
			except Exception:
				fallback_model = getattr(lit, "ema_model", None) or lit.model
				gen = lit.eval_sampler.sample(model=fallback_model, noise=noise, cond=cond, x_start=img_batch)

		# Save input and generated grid
		pid = canonical_patient_id(h5p.name)
		save_image((img_batch + 1) / 2, out_dir / f"{pid}_input_tile.png", nrow=1)
		save_image((gen + 1) / 2, out_dir / f"{pid}_reconstruct_samples.png", nrow=min(4, args.n_samples))

	print("Done. Images saved to:", out_dir)


if __name__ == "__main__":
	main()


