#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A small utility to inject missing hyperparameters into a joint training checkpoint.

Because we forgot to call `self.save_hyperparameters()` in the `JointLitModel.__init__`,
PyTorch Lightning only saved the `conf` attributes from mopadi. 
This script reads your config and the gene expression CSV to restore the missing
'conf', 'joint_cfg', and 'n_genes' straight into the checkpoint file's 'hyper_parameters'
dictionary, so that `reconstruct_tiles.py` will pass its strict checks!
"""

import argparse
import sys
import torch
import yaml

from src.drafts.joint_training.model import build_conf
from src.drafts.joint_training.train import _count_genes

def main():
    parser = argparse.ArgumentParser(description="Patch joint checkpoint with missing hyper_parameters")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint to fix (e.g., experiments/joint_training/joint/last.ckpt)")
    parser.add_argument("--config", type=str, required=True, help="Path to the config.yaml used for training")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        sys.exit(1)

    print(f"Loading config: {args.config}")
    with open(args.config, "r") as f:
        full_cfg = yaml.safe_load(f)
    
    joint_cfg = full_cfg.get("joint_training")
    if not joint_cfg:
        print("Could not find 'joint_training' in the config.yaml!")
        sys.exit(1)

    # 1. Verify this actually IS a joint training checkpoint
    state_keys = ckpt.get("state_dict", {}).keys()
    if not any(k.startswith("projection.") for k in state_keys) and not any(k.startswith("vae.") for k in state_keys):
        print("Wait! The state_dict doesn't contain 'projection.' or 'vae.' keys.")
        print("This really looks like a base mopadi checkpoint. I cannot patch this safely.")
        sys.exit(1)
    print("✅ Verified `state_dict` has joint network layers (e.g., projection head).")

    # 2. Re-create the missing objects
    n_genes = _count_genes(joint_cfg)
    conf = build_conf(joint_cfg)

    print(f"Computed n_genes: {n_genes}")
    print(f"Reconstructed conf: {type(conf)}")

    # 3. Patch the checkpoint
    if "hyper_parameters" not in ckpt:
        ckpt["hyper_parameters"] = {}

    ckpt["hyper_parameters"]["conf"] = conf
    ckpt["hyper_parameters"]["joint_cfg"] = joint_cfg
    ckpt["hyper_parameters"]["n_genes"] = n_genes

    print("Patched checkpoint 'hyper_parameters' dictionary!")

    # 4. Save it out (we'll save to a new file and then overwrite if they want, or just overwrite directly)
    backup_path = args.ckpt + ".backup"
    import shutil
    import os
    if not os.path.exists(backup_path):
        shutil.copy(args.ckpt, backup_path)
        print(f"Created a backup at: {backup_path}")

    torch.save(ckpt, args.ckpt)
    print(f"✅ Successfully overwrote {args.ckpt} with patched hyperparameters!")
    print("\nYou can now safely run your reconstruction pipeline.")

if __name__ == "__main__":
    main()
