#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared config utilities used across training and extraction entrypoints.

Centralises two helper functions that were previously duplicated verbatim
in joint_training, cross_attention_joint_training,
gene_token_cross_attention_joint_training, and encoding/extract_joint_latents.
"""

from __future__ import annotations

import copy
from pathlib import Path


def resolve_config_paths(config_dict: dict, repo_root: Path) -> dict:
    """Recursively resolve relative paths in a config dict against *repo_root*.

    Workspace layout assumption
    ---------------------------
    ``data/``, ``dataframes/``, and ``experiments/`` live **next to** the repo
    root (i.e. at ``repo_root.parent/``).  Any relative value that starts with
    one of those prefixes is resolved against ``repo_root.parent`` rather than
    ``repo_root`` itself; the parent path is preferred whenever it already
    exists or when the repo-local candidate does not.

    All other relative paths (``./``, ``../``, or containing ``slurm/`` /
    ``src/``) are resolved against ``repo_root``.
    """
    def _resolve_path(value: str) -> str:
        repo_candidate = (repo_root / value).resolve()
        normalized = value[2:] if value.startswith("./") else value

        if normalized.startswith(("data/", "dataframes/", "experiments/")):
            parent_candidate = (repo_root.parent / normalized).resolve()
            if parent_candidate.exists() or not repo_candidate.exists():
                return str(parent_candidate)

        return str(repo_candidate)

    if isinstance(config_dict, dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                resolve_config_paths(value, repo_root)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        resolve_config_paths(item, repo_root)
            elif isinstance(value, str):
                if not value.startswith("/") and (
                    value.startswith("./")
                    or value.startswith("../")
                    or any(
                        part in value
                        for part in ["data/", "experiments/", "dataframes/", "slurm/", "src/"]
                    )
                ):
                    config_dict[key] = _resolve_path(value)

    return config_dict


def deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* on top of *base*, returning a new dict.

    Nested dicts are merged recursively; all other types are replaced.
    *base* is deep-copied so the original is never mutated.
    """
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged
