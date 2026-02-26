#!/usr/bin/env python
"""Sample tiles from a finetuned genomic-conditioned diffusion model.

Two modes:
 - `random-noise`: generate tiles from random noise conditioned on genomic vectors
 - `encode-decode`: add noise to a real tile and decode guided by genomic vector

Usage examples are printed with `-h`.

python src/finetune_diffusion/sample_tiles_from_genomic.py \
	--ckpt path/to/last.ckpt \
	--genomic-h5-dir /path/to/genomic_h5_dir \
	--mode random-noise \
	--n-samples 4 \
	--out-dir ./samples_random

python src/finetune_diffusion/sample_tiles_from_genomic.py \
	--ckpt path/to/last.ckpt \
	--genomic-h5-dir /path/to/genomic_h5_dir \
	--mode encode-decode \
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
import zipfile
from io import BytesIO

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


def find_genomic_map(genomic_h5_dir: str, split: str = "all") -> dict:
	p = Path(genomic_h5_dir)
	train_dir = p / "train"
	test_dir = p / "test"
	files = {}
	if train_dir.is_dir() or test_dir.is_dir():
		if split in ("train", "all") and train_dir.is_dir():
			files.update({canonical_patient_id(f.name): f for f in train_dir.glob("*.h5")})
		if split in ("test", "all") and test_dir.is_dir():
			files.update({canonical_patient_id(f.name): f for f in test_dir.glob("*.h5")})
		return files
	return {canonical_patient_id(f.name): f for f in p.glob("*.h5")}


def list_tiles_in_zip(zpath: Path) -> list[str]:
	try:
		with zipfile.ZipFile(zpath, "r") as zf:
			names = [n for n in zf.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg"))]
			return names
	except zipfile.BadZipFile:
		return []


def load_tiles_from_zip(zpath: Path, names: list[str], img_size: int):
	imgs = []
	with zipfile.ZipFile(zpath, "r") as zf:
		for n in names:
			with zf.open(n) as fh:
				im = Image.open(BytesIO(fh.read())).convert("RGB")
				imgs.append(make_tile_transform(img_size)(im))
	if imgs:
		return torch.stack(imgs, dim=0)
	return torch.empty(0)


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
	parser.add_argument("--mode", choices=["random-noise", "encode-decode"], default="random-noise")
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

	# Prefer EMA model by default for sampling; fall back to `lit.model` if unavailable or on error
	ema_model = getattr(lit, "ema_model", None)
	primary_model = ema_model or lit.model
	primary_model.to(device).eval()

	# Map genomic .h5 files by canonical patient id
	genomic_map = find_genomic_map(args.genomic_h5_dir, split="all")
	if not genomic_map:
		raise RuntimeError(f"No .h5 genomic files found in {args.genomic_h5_dir}")

	# Collect zip files if provided
	zdir = Path(args.tiles_zip_dir) if args.tiles_zip_dir else None
	zip_files = []
	if zdir is not None and zdir.exists():
		zip_files = sorted(zdir.glob("*.zip"))
	# If user supplied a single tile file instead of zips, we'll handle that later

	# Sampling
	if args.mode == "random-noise":
		# If zip dir provided, iterate zip files (one per patient)
		if zip_files:
			for zpath in tqdm(zip_files, desc="Tile zips"):
				pid = canonical_patient_id(zpath.name)
				if pid not in genomic_map:
					print(f"[WARN] No genomic .h5 for {pid}, skipping {zpath}")
					continue
				vec = load_genomic_vector(genomic_map[pid]).to(device)
				cond = lit.projection_head(vec.unsqueeze(0)).repeat(args.n_samples, 1)

				noise = torch.randn(args.n_samples, 3, args.img_size, args.img_size, device=device)
				with torch.no_grad():
					try:
						gen = lit.eval_sampler.sample(model=primary_model, noise=noise, cond=cond)
					except Exception:
						alt = lit.model if primary_model is not lit.model else (ema_model or lit.model)
						gen = lit.eval_sampler.sample(model=alt, noise=noise, cond=cond)

				if not isinstance(gen, torch.Tensor):
					raise RuntimeError("Sampler returned non-tensor for random-noise mode")

				# Prepare names for generated files. If the source zip contains tiles, keep their basenames
				tile_names = list_tiles_in_zip(zpath)
				sel = tile_names[:args.n_samples] if tile_names else []
				gen_zip_path = out_dir / f"{pid}_generated_tiles.zip"
				in_zip_path = out_dir / f"{pid}_input_tiles.zip"
				# Write input tiles (copy originals) and generated tiles into zips
				with zipfile.ZipFile(gen_zip_path, "w") as gz, zipfile.ZipFile(in_zip_path, "w") as iz:
					# copy selected input tiles if available
					with zipfile.ZipFile(zpath, "r") as srcz:
						for n in sel:
							data = srcz.read(n)
							basename = Path(n).name
							iz.writestr(basename, data)
					# write generated images; name by original basename if available, else sample index
					for i in range(gen.shape[0]):
						out_name = None
						if i < len(sel):
							out_name = Path(sel[i]).name
						else:
							out_name = f"sample_{i}.png"
						img = gen[i].detach().cpu()
						img = ((img + 1) / 2).clamp(0, 1)
						arr = (img * 255).to(torch.uint8).permute(1, 2, 0).numpy()
						pil = Image.fromarray(arr)
						buf = BytesIO()
						pil.save(buf, format="PNG")
						gz.writestr(out_name, buf.getvalue())
		else:
			# No zip dir: use genomic_map entries directly
			for pid, h5p in tqdm(genomic_map.items(), desc="Genomic patients"):
				vec = load_genomic_vector(h5p).to(device)
				cond = lit.projection_head(vec.unsqueeze(0)).repeat(args.n_samples, 1)

				noise = torch.randn(args.n_samples, 3, args.img_size, args.img_size, device=device)
				with torch.no_grad():
					try:
						gen = lit.eval_sampler.sample(model=primary_model, noise=noise, cond=cond)
					except Exception:
						alt = lit.model if primary_model is not lit.model else (ema_model or lit.model)
						gen = lit.eval_sampler.sample(model=alt, noise=noise, cond=cond)

				if not isinstance(gen, torch.Tensor):
					raise RuntimeError("Sampler returned non-tensor for random-noise mode")

				gen_zip_path = out_dir / f"{pid}_generated_tiles.zip"
				with zipfile.ZipFile(gen_zip_path, "w") as gz:
					for i in range(gen.shape[0]):
						out_name = f"{pid}_sample_{i}.png"
						img = gen[i].detach().cpu()
						img = ((img + 1) / 2).clamp(0, 1)
						arr = (img * 255).to(torch.uint8).permute(1, 2, 0).numpy()
						pil = Image.fromarray(arr)
						buf = BytesIO()
						pil.save(buf, format="PNG")
						gz.writestr(out_name, buf.getvalue())

	elif args.mode == "encode-decode":
		# Prefer zip dir if provided
		if zip_files:
			for zpath in tqdm(zip_files, desc="Tile zips"):
				pid = canonical_patient_id(zpath.name)
				if pid not in genomic_map:
					print(f"[WARN] No genomic .h5 for {pid}, skipping {zpath}")
					continue
				tile_names = list_tiles_in_zip(zpath)
				if not tile_names:
					print(f"[WARN] No image tiles in {zpath}")
					continue
				# choose up to n_samples tiles
				sel = tile_names[:args.n_samples]
				img_batch = load_tiles_from_zip(zpath, sel, args.img_size).to(device)
				if img_batch.numel() == 0:
					continue

				vec = load_genomic_vector(genomic_map[pid]).to(device)
				cond = lit.projection_head(vec.unsqueeze(0)).repeat(img_batch.shape[0], 1)

				# Forward-diffuse the input images to x_T using the sampler's q_sample
				eps = torch.randn_like(img_batch)
				T = getattr(lit.eval_sampler, "num_timesteps", None)
				if T is None:
					# fallback to attribute name used elsewhere
					T = getattr(lit.eval_sampler, "_num_timesteps", None) or getattr(lit.eval_sampler, "num_steps", 1000)
				t_batch = torch.tensor([T - 1] * img_batch.shape[0], device=device)
				x_t = lit.eval_sampler.q_sample(img_batch, t_batch, noise=eps)
				with torch.no_grad():
					try:
						gen = lit.eval_sampler.sample(model=primary_model, noise=x_t, cond=cond, x_start=img_batch)
					except Exception:
						alt = lit.model if primary_model is not lit.model else (ema_model or lit.model)
						gen = lit.eval_sampler.sample(model=alt, noise=x_t, cond=cond, x_start=img_batch)

				if not isinstance(gen, torch.Tensor):
					raise RuntimeError("Sampler returned non-tensor for encode-decode mode")

				# Write input tiles and generated tiles to zip files, preserving original basenames
				in_zip_path = out_dir / f"{pid}_input_tiles.zip"
				gen_zip_path = out_dir / f"{pid}_generated_tiles.zip"
				with zipfile.ZipFile(in_zip_path, "w") as iz, zipfile.ZipFile(gen_zip_path, "w") as gz, zipfile.ZipFile(zpath, "r") as srcz:
					for n in sel:
						data = srcz.read(n)
						basename = Path(n).name
						iz.writestr(basename, data)
					# generated images: map one-to-one to sel entries
					for i in range(gen.shape[0]):
						if i < len(sel):
							out_name = Path(sel[i]).name
						else:
							out_name = f"gen_{i}.png"
						img = gen[i].detach().cpu()
						img = ((img + 1) / 2).clamp(0, 1)
						arr = (img * 255).to(torch.uint8).permute(1, 2, 0).numpy()
						pil = Image.fromarray(arr)
						buf = BytesIO()
						pil.save(buf, format="PNG")
						gz.writestr(out_name, buf.getvalue())
		else:
			# Fall back to single tile-file behavior
			if args.tile_file is None:
				raise RuntimeError("--tile-file required when no --tiles-zip-dir is provided")
			tile_path = Path(args.tile_file)
			if not tile_path.exists():
				raise RuntimeError(f"Tile file not found: {tile_path}")

			img = load_tile_image(tile_path, args.img_size).unsqueeze(0).to(device)

			# If patient specified, find matching genomic h5, otherwise try to infer from tile filename
			if args.patient:
				pid_key = canonical_patient_id(args.patient)
				h5p = genomic_map.get(pid_key)
			else:
				pid_key = canonical_patient_id(tile_path.name)
				h5p = genomic_map.get(pid_key)

			if h5p is None:
				raise RuntimeError("No matching genomic file found for the provided tile (specify --patient or ensure filenames include TCGA- ids)")

			vec = load_genomic_vector(h5p).to(device)
			cond = lit.projection_head(vec.unsqueeze(0)).repeat(args.n_samples, 1)

			img_batch = img.repeat(args.n_samples, 1, 1, 1)
			eps = torch.randn_like(img_batch)
			T = getattr(lit.eval_sampler, "num_timesteps", None)
			if T is None:
				T = getattr(lit.eval_sampler, "_num_timesteps", None) or getattr(lit.eval_sampler, "num_steps", 1000)
			t_batch = torch.tensor([T - 1] * img_batch.shape[0], device=device)
			x_t = lit.eval_sampler.q_sample(img_batch, t_batch, noise=eps)

			with torch.no_grad():
				try:
					gen = lit.eval_sampler.sample(model=primary_model, noise=x_t, cond=cond, x_start=img_batch)
				except Exception:
					alt = lit.model if primary_model is not lit.model else (ema_model or lit.model)
					gen = lit.eval_sampler.sample(model=alt, noise=x_t, cond=cond, x_start=img_batch)

				if not isinstance(gen, torch.Tensor):
					raise RuntimeError("Sampler returned non-tensor for encode-decode mode")

				pid = canonical_patient_id(h5p.name)
				in_zip_path = out_dir / f"{pid}_input_tiles.zip"
				gen_zip_path = out_dir / f"{pid}_generated_tiles.zip"
				# write the single input tile and all generated variants
				with zipfile.ZipFile(in_zip_path, "w") as iz, zipfile.ZipFile(gen_zip_path, "w") as gz:
					# original tile
					with open(tile_path, "rb") as f:
						iz.writestr(Path(tile_path).name, f.read())
					# generated tiles
					for i in range(gen.shape[0]):
						out_name = f"{Path(tile_path).stem}_gen_{i}.png"
						img = gen[i].detach().cpu()
						img = ((img + 1) / 2).clamp(0, 1)
						arr = (img * 255).to(torch.uint8).permute(1, 2, 0).numpy()
						pil = Image.fromarray(arr)
						buf = BytesIO()
						pil.save(buf, format="PNG")
						gz.writestr(out_name, buf.getvalue())

	else:
		raise RuntimeError(f"Unknown mode: {args.mode}")

	print("Done. Images saved to:", out_dir)


if __name__ == "__main__":
	main()


