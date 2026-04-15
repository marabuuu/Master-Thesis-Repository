#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared utilities for the reconstruction module.

Functions here are used by both reconstruct_tiles and investigate_noising.
Keep this module free of heavy imports so it can be imported cheaply.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def extract_patient_id(name: str) -> str:
    """Extract TCGA-XX-XXXX patient identifier from a filename or string."""
    stem = Path(name).stem.upper()
    for sep in ("_", "."):
        stem = stem.replace(sep, "-")
    while "--" in stem:
        stem = stem.replace("--", "-")
    parts = stem.split("-")
    if len(parts) >= 3 and parts[0].startswith("TCGA"):
        return "-".join(parts[:3])
    return stem


def tensor_to_image(x: torch.Tensor) -> np.ndarray:
    """Convert a (C, H, W) tensor in [-1, 1] to a uint8 HWC numpy array."""
    if x.ndim == 4:
        x = x[0]
    x = x.cpu().detach()
    x = ((x + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
    return x.permute(1, 2, 0).numpy()


def _ensure_mopadi_import_path() -> None:
    """Ensure local mopadi source is importable for checkpoint unpickling.

    Some checkpoints store objects under ``mopadi.configs`` in hyperparameters.
    When loading outside SLURM, ``PYTHONPATH`` may miss the local mopadi source.
    Checks the ``MOPADI_SRC`` environment variable first, then falls back to
    common relative locations within the repository tree.
    """
    env_path = os.getenv("MOPADI_SRC")
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path(env_path) if env_path else None,
        repo_root / "mopadi" / "src",
        repo_root.parent / "mopadi" / "src",
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        candidate = candidate.resolve()
        if (candidate / "mopadi").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
                logger.info(f"Added mopadi import path: {candidate_str}")
            break


def _sanitize_joint_cfg_for_inference(joint_cfg: dict) -> dict:
    """Return a copy of joint_cfg with constructor-only preload checkpoints disabled.

    During reconstruction the full state dict is loaded from the target checkpoint,
    so constructor-side optional preload checkpoints are unnecessary and can stall
    if paths point to slow or unavailable network mounts.
    """
    cfg = dict(joint_cfg)
    cfg["diffusion_ckpt"] = None
    cfg["encoder_ckpt"] = None
    return cfg
