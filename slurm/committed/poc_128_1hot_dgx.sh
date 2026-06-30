#!/bin/bash
#SBATCH --job-name=poc_128_1hot
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
# PoC 128×128 — 1-hot CFG backbone (DGX, 4×A100 80GB)
#
# Binary conditioning diagnostic: BRCA→e₁=[1,0,...,0], LIHC→e₂=[0,1,...,0]
# Trains from scratch — no warm-start checkpoint.
# 4-level UNet (net_ch=64, mult=[1,2,4,8]), 16×16 bottleneck, ~80M params.
#
# batch_size=32 (8/GPU), accum=1 → effective global batch = 32
# max_steps = 1M / 32 = 31,250 → ~1-2h at ~4-5 it/s (128px is fast)
#
# Key diagnostics in TensorBoard:
#   cond/signal  — must grow (expect to see movement within first 100K samples)
#   cond/gap     — should turn positive (BRCA tiles score better with e₁ than e₂)
#   loss/val_ckpt — should remain stable/decreasing
#
# Submit:
#   cd Master-Thesis-Repository
#   sbatch slurm/poc_128_1hot_dgx.sh
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
echo "PoC 128×128 1-hot CFG — BRCA vs LIHC (DGX 4×A100)"
echo "============================================================"
echo "Node:    $(hostname)"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "GPUs:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Python:  $(python --version)"
echo "Data:    ./data/PoC-BRCA-LIHC-tumor-tiles-128"
echo "Init:    from scratch (no backbone_ckpt_path)"
echo "Output:  ./experiments/20260603_poc_128_1hot"
echo "ETA:     ~1-2h (31K steps)"
echo "============================================================"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), "| gpus:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  gpu{i}:", torch.cuda.get_device_name(i))
PY

echo ""
echo "[RUN] Starting PoC 128×128 1-hot CFG backbone training..."
srun --label --kill-on-bad-exit=1 python run_pipeline.py \
    --config src/config.yaml \
    --stage poc_128_1hot

echo ""
echo "============================================================"
echo "[OK] Training complete."
echo "  Checkpoints: experiments/20260603_poc_128_1hot/gda/autoenc/"
echo "  TensorBoard: experiments/20260603_poc_128_1hot/gda"
echo "  Watch:  cond/signal  — must grow (first signal expected ~100K samples)"
echo "          cond/gap     — should turn positive"
echo "          loss/val_ckpt — should remain stable"
echo "============================================================"
