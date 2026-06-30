#!/bin/bash
#SBATCH --job-name=poc_128_rna_30M
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --export=ALL
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

################################################################################
# PoC 128×128 — real RNA-seq conditioning, 30M samples
#
# Main experiment: conditions on real CONCH genomic features from H5 files.
# normalize_feats=false: raw features enter style MLP without L2 normalization.
# Identical hyperparams to poc_128_1hot_nonorm_30M for fair comparison.
#
# Submit:
#   cd Master-Thesis-Repository
#   sbatch slurm/poc_128_rna_30M_dgx.sh
################################################################################

set -euo pipefail

GENHIST_DIR=/mnt/bulk-saturn/maralampert/genhist
REPO_DIR="$GENHIST_DIR/Master-Thesis-Repository"

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
echo "PoC 128×128 RNA conditioning — 30M samples (DGX 4×A100)"
echo "============================================================"
echo "Node:    $(hostname)"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "GPUs:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Python:  $(python --version)"
echo "Output:  experiments/20260605_poc_128_rna_30M"
echo "============================================================"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), "| gpus:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  gpu{i}:", torch.cuda.get_device_name(i))
PY

echo ""
echo "[RUN] Starting RNA-conditioning 30M training run..."
srun --label --kill-on-bad-exit=1 python run_pipeline.py \
    --config src/config.yaml \
    --stage poc_128_rna_30M

echo ""
echo "============================================================"
echo "[OK] Training complete."
echo "  Checkpoints: experiments/20260605_poc_128_rna_30M/gda/autoenc/"
echo "  TensorBoard: experiments/20260605_poc_128_rna_30M/gda"
echo "============================================================"
