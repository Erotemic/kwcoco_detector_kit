#!/usr/bin/env bash
# Distributed target-model smoke: OpenGroundingDINO on tiny kwcoco demo data,
# 4 GPUs. This mainly proves torch.distributed, NCCL, launcher wiring, and
# per-rank data loading.
set -euo pipefail
source "$(dirname "$0")/common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export PORT="${PORT:-29500}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
INPUT_SIZE="${INPUT_SIZE:-320}"
BATCH_SIZE="${BATCH_SIZE:-1}"

ROOT="$KCD_SMOKE_ROOT/02_ogdino_demo_4gpu"
DATA="$ROOT/data"
RUN_ROOT="$ROOT/run"

print_env_summary
ensure_pretrain
make_demo_splits "$DATA"
run_ogdino_sweep "$RUN_ROOT" \
    "$DATA/train.kwcoco.zip" \
    "$DATA/vali.kwcoco.zip" \
    "$DATA/test.kwcoco.zip" \
    4 1 "$BATCH_SIZE" 2-4xL

echo
echo "PASS: OpenGroundingDINO demo 4-GPU smoke"
echo "RUN_ROOT=$RUN_ROOT"
