#!/bin/bash
#SBATCH --job-name=pam50_1hot_256
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --export=ALL
#SBATCH --cpus-per-task=30
#SBATCH --mem=160G
#SBATCH --time=6-00:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

################################################################################
# BRCA PAM50 CFG v2 — orthogonal conditioning, 256×256px (DGX, 4×A100 80GB)
#
# Identical to brca_pam50_cfg_v2_1hot but trained on 256×256 tiles
# (BRCA-tumor-tiles-256) so results are directly comparable to
# brca_pam50_cfg_v2_256 (real RNA-seq, 256px).
#
# conditioning_type: one_hot   normalize_feats: false
# 5-class orthogonal binary codes (Basal/Her2/LumA/LumB/Normal),
# assigned round-robin across feat_dim=512 positions.
# genomic_feature_dir: null  (no H5 files needed)
#
# Per-patient tile caps (same as v2):
#   LumA   506 pts ×  45 cap → ~22.8K tiles
#   LumB   195 pts × 120 cap → ~23.4K tiles
#   Basal  168 pts × 135 cap → ~22.7K tiles
#   Her2    74 pts × 300 cap → ~22.2K tiles
#   Normal  38 pts  uncapped → ~41K tiles
# WeightedRandomSampler equalises batch representation.
#
# Submit:
#   cd Master-Thesis-Repository
#   sbatch slurm/brca_pam50_cfg_v2_1hot_256_dgx.sh
################################################################################

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
    source "$REPO_DIR/.venv/bin/activate"
else
    echo "[ERROR] venv not found at $REPO_DIR/.venv/bin/activate"
    exit 1
fi

cd "$REPO_DIR" || exit 1

export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_ASYNC_ERROR_HANDLING || true

echo "============================================================"
echo "BRCA PAM50 CFG v2 — orthogonal conditioning, 256px — DGX 4×A100"
echo "============================================================"
echo "Node:    $(hostname)"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "GPUs:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Python:  $(python --version)"
echo "Data:    ../data/BRCA-tumor-tiles-256  (256×256px)"
echo "Cond:    one_hot (5-class orthogonal 512-dim binary codes, no RNA-seq)"
echo "Init:    from scratch (no backbone_ckpt_path)"
echo "Output:  experiments/20260621_brca_pam50_cfg_v2_1hot_256"
echo "============================================================"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), "| gpus:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  gpu{i}:", torch.cuda.get_device_name(i))
PY

echo ""
echo "[RUN] Starting BRCA PAM50 CFG v2 orthogonal 256px training..."
srun --label --kill-on-bad-exit=1 python run_pipeline.py \
    --config src/config.yaml \
    --stage brca_pam50_cfg_v2_1hot_256

echo ""
echo "============================================================"
echo "[OK] Training complete."
echo "  Checkpoints: experiments/20260621_brca_pam50_cfg_v2_1hot_256/gda/autoenc/"
echo "  TensorBoard: experiments/20260621_brca_pam50_cfg_v2_1hot_256/gda"
echo "  Watch:  cond/signal         — must grow"
echo "          cond/gap            — should turn positive"
echo "          cond/basal_luma_sep — must grow (Basal vs LumA separation)"
echo "          loss/val_ckpt       — should remain stable/decreasing"
echo "============================================================"
