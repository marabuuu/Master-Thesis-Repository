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
	parser.add_argument("--debug-save", action="store_true", help="Save debug images (original, x_t, generated) to out-dir")
	parser.add_argument("--debug-n", type=int, default=4, help="Number of debug samples to save per patient")
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
	# track whether projection head weights were loaded from the main ckpt or external proj_ckpt
	proj_loaded_from_ckpt = False
	proj_loaded_from_projckpt = False
	if "state_dict" in ckpt:
		state = dict(ckpt["state_dict"])
		# Drop x_T mismatch if present
		if "x_T" in state and hasattr(lit, "x_T") and state["x_T"].shape != getattr(lit, "x_T").shape:
			del state["x_T"]
		# detect if projection head is present in the saved state
		if any("projection_head" in k for k in state.keys()):
			proj_loaded_from_ckpt = True
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
			# if the bare checkpoint is actually a state-dict-like mapping, check for projection_head keys
			if isinstance(ckpt, dict) and any("projection_head" in k for k in ckpt.keys()):
				proj_loaded_from_ckpt = True
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
				proj_loaded_from_projckpt = True
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

	# Log which model is used for sampling and write metadata to out_dir
	try:
		model_used = "EMA" if (ema_model is not None and primary_model is ema_model) else "BASE"
		if model_used == "EMA":
			print("[INFO] Sampling with EMA model (lit.ema_model)")
		else:
			print("[INFO] Sampling with base model (lit.model)")
		# Decide where projection weights came from
		if proj_loaded_from_projckpt:
			proj_source = "proj_ckpt"
		elif proj_loaded_from_ckpt:
			proj_source = "main_ckpt"
		else:
			proj_source = "none"
		# compute projection head parameter L2 norm (helps detect uninitialized random weights)
		try:
			ph_norm_sq = torch.tensor(0.0, device=device)
			for p in lit.projection_head.parameters():
				val = p.detach().float().norm()
				ph_norm_sq = ph_norm_sq + (val * val)
			ph_norm = float(torch.sqrt(ph_norm_sq))
		except Exception:
			ph_norm = None
		meta = {
			"model_used": model_used,
			"ckpt": str(args.ckpt),
			"proj_ckpt": str(args.proj_ckpt) if args.proj_ckpt else "",
			"proj_loaded_from": proj_source,
			"proj_param_l2_norm": str(ph_norm) if ph_norm is not None else "",
			"mode": args.mode,
			"tiles_zip_dir": args.tiles_zip_dir or "",
			"seed": str(args.seed),
		}
		meta_path = out_dir / "sampling_metadata.txt"
		meta_lines = [f"{k}: {v}" for k, v in meta.items()]
		meta_path.write_text("\n".join(meta_lines))
	except Exception as e:
		print(f"[WARN] Could not write sampling metadata: {e}")

	# Map genomic .h5 files by canonical patient id
	genomic_map = find_genomic_map(args.genomic_h5_dir, split="all")
	if not genomic_map:
		raise RuntimeError(f"No .h5 genomic files found in {args.genomic_h5_dir}")

	# Collect zip files if provided
	zdir = Path(args.tiles_zip_dir) if args.tiles_zip_dir else None
	zip_files = []
	if zdir is not None and zdir.exists():
		zip_files = sorted(zdir.glob("*.zip"))
	# If user requested a specific patient, filter available zip files / genomic_map
	if args.patient:
		pid_key = canonical_patient_id(args.patient)
		if zip_files:
			# find matching zip(s) by canonical name
			matched = [z for z in zip_files if canonical_patient_id(z.name) == pid_key]
			if not matched:
				print(f"[WARN] No tile zip matching patient {pid_key} in {zdir}; continuing but nothing will be processed for that patient")
			zip_files = matched
		else:
			# restrict genomic_map lookup later by patient key (genomic_map is built below)
			# no action here; genomic_map filtering happens after it's created
			pass

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
				# Write generated tiles only; input tiles already exist in the source zip directory
				with zipfile.ZipFile(gen_zip_path, "w") as gz:
					for i in range(gen.shape[0]):
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

				# Write generated tiles only; input tiles remain in the source zip directory
				gen_zip_path = out_dir / f"{pid}_generated_tiles.zip"
				with zipfile.ZipFile(gen_zip_path, "w") as gz:
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

				# Debug: save original, x_t (noisy), and generated images plus basic MSE
				if args.debug_save:
					from pathlib import Path as _P
					_debug_dir = out_dir / f"debug_{pid}"
					_debug_dir.mkdir(parents=True, exist_ok=True)
					metrics_lines = ["pid,basename,mse"]
					n_save = min(args.debug_n, gen.shape[0], len(sel) if sel else gen.shape[0])
					for i in range(n_save):
						# ensure variables exist for static analysis
						member = None
						orig_pil = None
						# original image from zip
						if i < len(sel):
							member = sel[i]
							with zipfile.ZipFile(zpath, "r") as srcz:
								orig_bytes = srcz.read(member)
								orig_pil = Image.open(BytesIO(orig_bytes)).convert("RGB")
								orig_pil.save(_debug_dir / f"orig_{Path(member).name}")
						# save x_t (noisy) - convert from tensor x_t range to image
						x_img = x_t[i].detach().cpu()
						x_img = ((x_img + 1) / 2).clamp(0, 1)
						x_arr = (x_img * 255).to(torch.uint8).permute(1, 2, 0).numpy()
						Image.fromarray(x_arr).save(_debug_dir / f"x_t_{i}.png")
						# build and save generated image from tensor (avoid relying on outer 'pil' var)
						gen_img = gen[i].detach().cpu()
						gen_img = ((gen_img + 1) / 2).clamp(0, 1)
						gen_arr = (gen_img * 255).to(torch.uint8).permute(1, 2, 0).numpy()
						gen_pil = Image.fromarray(gen_arr)
						gen_pil.save(_debug_dir / f"gen_{i}.png")
						# compute simple MSE if original available
						if orig_pil is not None:
							orig_arr = np.asarray(orig_pil.resize(gen_pil.size)).astype(np.float32)
							gen_arr_f = np.asarray(gen_pil).astype(np.float32)
							mse = float(np.mean((orig_arr - gen_arr_f) ** 2))
							member_name = Path(member).name if member is not None else ""
							metrics_lines.append(f"{pid},{member_name},{mse:.4f}")
					# end per-sample
					# write metrics file
					with open(_debug_dir / "metrics.csv", "w") as mf:
						mf.write("\n".join(metrics_lines))
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
				gen_zip_path = out_dir / f"{pid}_generated_tiles.zip"
				# write generated variants only; the original tile exists at the provided path or in the tiles zip dir
				with zipfile.ZipFile(gen_zip_path, "w") as gz:
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


