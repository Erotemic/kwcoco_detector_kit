#!/usr/bin/env bash
# Generic single-node Slurm submit template for OpenGroundingDINO on
# 4x NVIDIA RTX A6000 (48 GB each). This is intentionally dataset-agnostic:
# point TRAIN_KWCOCO / VALI_KWCOCO / TEST_KWCOCO at already-prepared kwcoco
# files and set CATEGORY_NAME for the class to train.
#
# Submit:
#   sbatch --export=ALL,TRAIN_KWCOCO=/path/train.kwcoco.zip,VALI_KWCOCO=/path/vali.kwcoco.zip,TEST_KWCOCO=/path/test.kwcoco.zip,CATEGORY_NAME=sealion examples/slurm/a6000_4x_opengroundingdino.sbatch
#
# Notes:
# - `nvidia-smi CUDA Version: 13.2` is the driver's runtime capability. The
#   OpenGroundingDINO CUDA extension must be compiled with an nvcc toolkit
#   matching `torch.version.cuda` (for example, a cu130 PyTorch wheel wants
#   nvcc 13.0 even when the driver advertises CUDA 13.2).
# - Load the CUDA module that matches the PyTorch wheel used in PYTHON_BIN.
#   Override CUDA_MODULE or KCD_TORCH_INDEX_URL for your cluster image.

#SBATCH --job-name=kcd-ogdino-a6000x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module purge || true
if [ -n "${CUDA_MODULE:-}" ]; then
    module load "$CUDA_MODULE"
fi

KIT_DPATH="${KIT_DPATH:-$HOME/code/kwcoco_detector_kit}"
PYTHON_BIN="${PYTHON_BIN:-python}"
KCD_ROOT="${KCD_ROOT:-${SCRATCH:-$HOME/scratch}/kcd_runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
KCD_CACHE_ROOT="${KCD_CACHE_ROOT:-${SCRATCH:-$HOME/scratch}/kcd_cache/opengroundingdino}"
KCD_OPENGROUNDINGDINO_REPO_DPATH="${KCD_OPENGROUNDINGDINO_REPO_DPATH:-$KIT_DPATH/tpl/Open-GroundingDino}"

TRAIN_KWCOCO="${TRAIN_KWCOCO:?set TRAIN_KWCOCO=/path/to/train.kwcoco.zip}"
VALI_KWCOCO="${VALI_KWCOCO:?set VALI_KWCOCO=/path/to/vali.kwcoco.zip}"
TEST_KWCOCO="${TEST_KWCOCO:?set TEST_KWCOCO=/path/to/test.kwcoco.zip}"
CATEGORY_NAME="${CATEGORY_NAME:-widget}"

VARIANT="${VARIANT:-opengroundingdino_swint}"
INPUT_HW="${INPUT_HW:-800,800}"
NUM_EPOCHS="${NUM_EPOCHS:-24}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
BACKBONE_LR="${BACKBONE_LR:-1e-5}"
KCD_DO_EVAL="${KCD_DO_EVAL:-0}"

export KCD_ROOT
export KCD_CACHE_ROOT
export KCD_OPENGROUNDINGDINO_REPO_DPATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NNODES="${SLURM_NNODES:-1}"
export NODE_RANK="${SLURM_NODEID:-0}"
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
export PORT="${PORT:-29500}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TEXT_ENCODER_TYPE="${TEXT_ENCODER_TYPE:-bert-base-uncased}"

mkdir -p "$KCD_ROOT" "$KCD_CACHE_ROOT/pretrained"
PRETRAIN_MODEL_PATH="${PRETRAIN_MODEL_PATH:-$KCD_CACHE_ROOT/pretrained/groundingdino_swint_ogc.pth}"
export PRETRAIN_MODEL_PATH

echo "=== Slurm / GPU context ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

echo
echo "=== Python / CUDA ABI context ==="
"$PYTHON_BIN" - <<'PY'
import shutil
import subprocess
import torch

print(f"python torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
nvcc = shutil.which("nvcc")
print(f"nvcc={nvcc}")
if nvcc:
    print(subprocess.check_output([nvcc, "--version"], text=True).strip())
PY

if [ ! -f "$PRETRAIN_MODEL_PATH" ]; then
    echo "Missing PRETRAIN_MODEL_PATH=$PRETRAIN_MODEL_PATH" >&2
    echo "Run setup/download once, or set PRETRAIN_MODEL_PATH to the Swin-T checkpoint." >&2
    exit 1
fi

echo
echo "=== OpenGroundingDINO 4x A6000 train ==="
"$PYTHON_BIN" -m kwcoco_detector_kit sweep \
    --train_kwcoco "$TRAIN_KWCOCO" \
    --vali_kwcoco "$VALI_KWCOCO" \
    --test_kwcoco "$TEST_KWCOCO" \
    --kcd_root "$KCD_ROOT" \
    --trainer opengroundingdino \
    --variant "$VARIANT" \
    --input_hw "$INPUT_HW" \
    --train_policy fixed \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --val_batch_size "$VAL_BATCH_SIZE" \
    --num_classes 1 \
    --category_name "$CATEGORY_NAME" \
    --lr "$LR" \
    --backbone_lr "$BACKBONE_LR" \
    --use_amp true \
    --scale_tier 2-4xL \
    --num_gpus 4 \
    --distributed \
    --no-do_export \
    --no-do_bench \
    $(if [ "$KCD_DO_EVAL" = "1" ]; then echo --do_eval true; else echo --no-do_eval; fi)

echo
echo "Done. KCD_ROOT=$KCD_ROOT"
