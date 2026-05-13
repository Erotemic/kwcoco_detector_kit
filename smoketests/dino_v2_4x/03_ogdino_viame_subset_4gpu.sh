#!/usr/bin/env bash
# Real-data subset smoke: OpenGroundingDINO on a tiny absolute-path subset of
# the prepared VIAME sea-lion kwcoco splits, 4 GPUs.
set -euo pipefail

export CATEGORY_NAME="${CATEGORY_NAME:-sealion}"
source "$(dirname "$0")/common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export PORT="${PORT:-29500}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"

INPUT_SIZE="${INPUT_SIZE:-800}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DATA_DPATH="${DATA_DPATH:-/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026}"
RAW_DPATH="${RAW_DPATH:-$DATA_DPATH/training_ready_v1}"

TRAIN_RAW="${TRAIN_RAW:-$RAW_DPATH/train.kwcoco.zip}"
VALI_RAW="${VALI_RAW:-$RAW_DPATH/vali.kwcoco.zip}"
TEST_RAW="${TEST_RAW:-$RAW_DPATH/test.kwcoco.zip}"

ROOT="$KCD_SMOKE_ROOT/03_ogdino_viame_subset_4gpu"
DATA="$ROOT/data"
RUN_ROOT="$ROOT/run"

print_env_summary
ensure_pretrain
require_file "$TRAIN_RAW" "missing VIAME train split"
require_file "$VALI_RAW" "missing VIAME validation split"
require_file "$TEST_RAW" "missing VIAME test split"

make_subset_abs "$TRAIN_RAW" "$DATA/train_subset.kwcoco.zip" "${VIAME_SUBSET_TRAIN_IMAGES:-16}"
make_subset_abs "$VALI_RAW" "$DATA/vali_subset.kwcoco.zip" "${VIAME_SUBSET_VALI_IMAGES:-8}"
make_subset_abs "$TEST_RAW" "$DATA/test_subset.kwcoco.zip" "${VIAME_SUBSET_TEST_IMAGES:-8}"

run_ogdino_sweep "$RUN_ROOT" \
    "$DATA/train_subset.kwcoco.zip" \
    "$DATA/vali_subset.kwcoco.zip" \
    "$DATA/test_subset.kwcoco.zip" \
    4 1 "$BATCH_SIZE" 2-4xL

echo
echo "PASS: OpenGroundingDINO VIAME subset 4-GPU smoke"
echo "RUN_ROOT=$RUN_ROOT"
