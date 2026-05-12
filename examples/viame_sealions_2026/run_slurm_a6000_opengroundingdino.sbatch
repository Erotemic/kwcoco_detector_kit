#!/usr/bin/env bash
# VIAME Sea Lions wrapper for the generic 4x RTX A6000 OpenGroundingDINO
# Slurm template. This script tiles/reuses the VIAME training-ready splits
# before launching a 4-GPU distributed fine-tune.

#SBATCH --job-name=viame-ogdino-a6000x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module purge || true
if [ -n "${CUDA_MODULE:-}" ]; then
    module load "$CUDA_MODULE"
fi

KIT_DPATH="${KIT_DPATH:-/home/joncrall/code/kwcoco_detector_kit}"
DATA_DPATH="${DATA_DPATH:-/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026}"
PYTHON_BIN="${PYTHON_BIN:-python}"

KCD_ROOT="${KCD_ROOT:-$DATA_DPATH/training_runs/ogdino_swint_a6000x4/${SLURM_JOB_ID:-manual}}"
KCD_CACHE_ROOT="${KCD_CACHE_ROOT:-$DATA_DPATH/training_runs/cache/ogdino_swint}"
RAW_DPATH="${RAW_DPATH:-$DATA_DPATH/training_ready_v1}"
TRAIN_RAW="${TRAIN_RAW:-$RAW_DPATH/train.kwcoco.zip}"
VALI_RAW="${VALI_RAW:-$RAW_DPATH/vali.kwcoco.zip}"
TEST_RAW="${TEST_RAW:-$RAW_DPATH/test.kwcoco.zip}"

TILE_SIZE="${TILE_SIZE:-800}"
SOURCE_SCALES="${SOURCE_SCALES:-1.0,0.5,0.25}"
STRIDE_FRAC="${STRIDE_FRAC:-0.75}"
MIN_GT_AREA_FRAC="${MIN_GT_AREA_FRAC:-0.0001}"
MIN_KEEP_FRACTION="${MIN_KEEP_FRACTION:-0.20}"
TILE_CACHE_NAME="tiles_t${TILE_SIZE}_scales_${SOURCE_SCALES}_stride_${STRIDE_FRAC}_pos"
TILE_CACHE_NAME="${TILE_CACHE_NAME//[^A-Za-z0-9._-]/_}"
TILE_DPATH="${TILE_DPATH:-$KCD_CACHE_ROOT/$TILE_CACHE_NAME}"
TRAIN_TILES="${TRAIN_TILES:-$TILE_DPATH/train_tiles_pos.kwcoco.zip}"
VALI_TILES="${VALI_TILES:-$TILE_DPATH/vali_tiles_pos.kwcoco.zip}"
TEST_TILES="${TEST_TILES:-$TILE_DPATH/test_tiles_pos.kwcoco.zip}"

NUM_EPOCHS="${NUM_EPOCHS:-24}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
BACKBONE_LR="${BACKBONE_LR:-1e-5}"
KCD_DO_EVAL="${KCD_DO_EVAL:-0}"

export KCD_ROOT
export KCD_CACHE_ROOT
export KCD_OPENGROUNDINGDINO_REPO_DPATH="${KCD_OPENGROUNDINGDINO_REPO_DPATH:-$KIT_DPATH/tpl/Open-GroundingDino}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NNODES="${SLURM_NNODES:-1}"
export NODE_RANK="${SLURM_NODEID:-0}"
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
export PORT="${PORT:-29500}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TEXT_ENCODER_TYPE="${TEXT_ENCODER_TYPE:-bert-base-uncased}"

mkdir -p "$KCD_ROOT" "$KCD_CACHE_ROOT/pretrained" "$TILE_DPATH"
PRETRAIN_MODEL_PATH="${PRETRAIN_MODEL_PATH:-$KCD_CACHE_ROOT/pretrained/groundingdino_swint_ogc.pth}"
export PRETRAIN_MODEL_PATH

echo "=== Slurm / GPU context ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "HOSTNAME=$(hostname)"
echo "KCD_ROOT=$KCD_ROOT"
echo "KCD_CACHE_ROOT=$KCD_CACHE_ROOT"
echo "TILE_DPATH=$TILE_DPATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

if [ ! -f "$PRETRAIN_MODEL_PATH" ]; then
    echo "Missing PRETRAIN_MODEL_PATH=$PRETRAIN_MODEL_PATH" >&2
    echo "Run examples/viame_sealions_2026/setup_host_env.sh once on the cluster login/build node, or set PRETRAIN_MODEL_PATH." >&2
    exit 1
fi

tile_one() {
    local src="$1"
    local dst="$2"
    if [ -f "$dst" ]; then
        echo "Reusing tiles: $dst"
        return
    fi
    "$PYTHON_BIN" -m kwcoco_detector_kit tile \
        "$src" "$dst" \
        --mode multiscale \
        --category_name sealion \
        --tile_size "$TILE_SIZE" \
        --source_scales "$SOURCE_SCALES" \
        --stride_frac "$STRIDE_FRAC" \
        --min_gt_area_frac "$MIN_GT_AREA_FRAC" \
        --min_keep_fraction "$MIN_KEEP_FRACTION" \
        --keep_negative false \
        --oversize_factor 1.0 \
        --jpeg_quality 90
}

echo
echo "=== 1. Positive-only tiling ==="
tile_one "$TRAIN_RAW" "$TRAIN_TILES"
tile_one "$VALI_RAW" "$VALI_TILES"
tile_one "$TEST_RAW" "$TEST_TILES"

echo
echo "=== 2. OpenGroundingDINO Swin-T 4x A6000 tune ==="
"$PYTHON_BIN" -m kwcoco_detector_kit sweep \
    --train_kwcoco "$TRAIN_TILES" \
    --vali_kwcoco "$VALI_TILES" \
    --test_kwcoco "$TEST_TILES" \
    --kcd_root "$KCD_ROOT" \
    --trainer opengroundingdino \
    --variant opengroundingdino_swint \
    --input_hw "$TILE_SIZE,$TILE_SIZE" \
    --train_policy fixed \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --val_batch_size "$VAL_BATCH_SIZE" \
    --num_classes 1 \
    --category_name sealion \
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
