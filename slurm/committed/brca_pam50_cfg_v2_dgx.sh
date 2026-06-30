#!/bin/bash
#SBATCH --job-name=brca_pam50_cfg_v2
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --export=ALL
#SBATCH --cpus-per-task=30
#SBATCH --mem=160G
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

################################################################################
# BRCA PAM50 CFG v2 backbone (DGX, 4×A100 80GB)
#
# CfgBackboneLitModel (model_training): backbone IS the conditioned denoiser.
# RNA-seq features feed directly into the backbone style MLP with CFG dropout.
# All 5 PAM50 subtypes: Basal/LumA/LumB/Her2/Normal.
# Per-patient tile caps calibrated so each subtype contributes ~22K total tiles:
#   LumA   506 pts ×  45 cap → ~22.8K tiles
#   LumB   195 pts × 120 cap → ~23.4K tiles
#   Basal  168 pts × 135 cap → ~22.7K tiles
#   Her2    74 pts × 300 cap → ~22.2K tiles
#   Normal  38 pts  uncapped → ~41K tiles  (too few patients to cap)
# WeightedRandomSampler equalises batch representation across all subtypes.
# From-scratch training — no warm start.
#
# batch_size=16 (4/GPU), trainer accum=1 → effective global batch = 16
# max_steps = 200M / 16 = 12.5M; checkpoints every 50K samples (~3K steps)
#
# Key diagnostics in TensorBoard:
#   cond/signal  — must grow (backbone using PAM50 RNA-seq features)
#   cond/gap     — should turn positive (shuffled > matched patient genomics)
#   loss/val_ckpt — baseline denoising quality
#
# Submit:
#   cd Master-Thesis-Repository
#   sbatch slurm/brca_pam50_cfg_v2_dgx.sh
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
echo "BRCA PAM50 CFG v2 backbone — DGX 4×A100"
echo "============================================================"
echo "Node:    $(hostname)"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "GPUs:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Python:  $(python --version)"
echo "Data:    ../data/BRCA-tumor-tiles-final"
echo "Genomics: experiments/20260528_genomic_features/genomic_h5"
echo "Init:    from scratch (no backbone_ckpt_path)"
echo "Output:  experiments/20260605_brca_pam50_cfg_v2"
echo "============================================================"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), "| gpus:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  gpu{i}:", torch.cuda.get_device_name(i))
PY

echo ""
echo "[RUN] Starting BRCA PAM50 CFG v2 backbone training..."
srun --label --kill-on-bad-exit=1 python run_pipeline.py \
    --config src/config.yaml \
    --stage brca_pam50_cfg_v2

echo ""
echo "============================================================"
echo "[OK] Training complete."
echo "  Checkpoints: experiments/20260605_brca_pam50_cfg_v2/gda/autoenc/"
echo "  TensorBoard: experiments/20260605_brca_pam50_cfg_v2/gda"
echo "  Watch:  cond/signal  — must grow"
echo "          cond/gap     — should turn positive"
echo "          loss/val_ckpt — should remain stable/decreasing"
echo "============================================================"
