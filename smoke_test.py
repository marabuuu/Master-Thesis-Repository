#!/usr/bin/env python
"""
Smoke test for mopadi genomic scratch training.

Runs 5 training steps + 1 validation pass with a single GPU, then exits.
Catches: import errors, config parse failures, missing data paths,
         model init problems, dataset errors, bf16 compatibility.

Usage (from repo root):
    srun --partition=capella-interactive --gres=gpu:h100_1g.12gb:1 \
         --time=10 --cpus-per-task=4 --mem=16G \
         python smoke_test.py
"""
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, "/data/cat/ws/mala059b-rna2wsi/mopadi/src")

print("[smoke] Loading config...")
from run_pipeline import load_config
full_cfg = load_config(str(repo_root / "src/config.yaml"), repo_root=repo_root)
cfg = full_cfg["mopadi_genomic_training"].copy()

# ── Smoke overrides ────────────────────────────────────────────────────────
# MIG slice is 12 GB — keep batch_size=1 and fp32 for headroom.
# We keep bf16=True to explicitly test H100 bf16 support.
cfg["gpus"] = [0]
cfg["batch_size"] = 1
cfg["accumulate_grad_batches"] = 1
cfg["num_workers"] = 1
cfg["img_size"] = 256
cfg["do_resize"] = True
cfg["total_samples"] = 2         # 2 steps, then done (fits 5-min interactive window)
cfg["val_every_steps"] = 1       # trigger validation quickly
cfg["limit_val_batches"] = 1     # one val batch is enough for smoke
cfg["save_every_samples"] = 100000  # no checkpoints during smoke test
cfg["reconstruct_every_samples"] = 100000
cfg["sample_size"] = 1

# Write to a temp dir so nothing pollutes the real experiment folder
tmpdir = tempfile.mkdtemp(prefix="mopadi_smoke_")
cfg["base_dir"] = tmpdir
print(f"[smoke] Output dir: {tmpdir}")

# ── Run ────────────────────────────────────────────────────────────────────
print("[smoke] Importing training module...")
from src.mopadi_genomic.run_genomic_training import run_genomic_training

print("[smoke] Starting 2-step run...")
run_genomic_training(cfg, verbose=True)

print()
print("=" * 60)
print("[smoke] PASSED — safe to submit the full job.")
print("=" * 60)
