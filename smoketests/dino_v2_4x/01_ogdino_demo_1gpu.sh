#!/usr/bin/env bash
# Target-model smoke: OpenGroundingDINO on tiny kwcoco demo data, 1 GPU.
# Uses one tiny epoch, so "few optimization steps" is controlled by the
# number of demo images and batch size.
set -euo pipefail
source "$(dirname "$0")/common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
INPUT_SIZE="${INPUT_SIZE:-320}"
BATCH_SIZE="${BATCH_SIZE:-1}"

ROOT="$KCD_SMOKE_ROOT/01_ogdino_demo_1gpu"
DATA="$ROOT/data"
RUN_ROOT="$ROOT/run"

print_env_summary
ensure_pretrain
make_demo_splits "$DATA"
run_ogdino_sweep "$RUN_ROOT" \
    "$DATA/train.kwcoco.zip" \
    "$DATA/vali.kwcoco.zip" \
    "$DATA/test.kwcoco.zip" \
    1 0 "$BATCH_SIZE" M

echo
echo "PASS: OpenGroundingDINO demo 1-GPU smoke"
echo "RUN_ROOT=$RUN_ROOT"
