#!/usr/bin/env bash
# Full real-data launch: delegates to the VIAME example runner with 4-GPU
# distributed settings. Run this only after 00-03 pass.
set -euo pipefail
source "$(dirname "$0")/common.sh"

DATA_DPATH="${DATA_DPATH:-/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026}"
export KIT_DPATH
export DATA_DPATH
export KCD_ROOT="${KCD_ROOT:-$KCD_SMOKE_ROOT/04_ogdino_viame_full_4gpu/run}"
export KCD_CACHE_ROOT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export PORT="${PORT:-29500}"
export NUM_GPUS="${NUM_GPUS:-4}"
export KCD_DISTRIBUTED="${KCD_DISTRIBUTED:-1}"
export SCALE_TIER="${SCALE_TIER:-2-4xL}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
export NUM_EPOCHS="${NUM_EPOCHS:-24}"
export KCD_DO_EVAL="${KCD_DO_EVAL:-0}"

print_env_summary
ensure_pretrain

run_cmd bash "$KIT_DPATH/examples/viame_sealions_2026/run_3090_opengroundingdino.sh"

echo
echo "PASS: full VIAME 4-GPU launch completed"
echo "KCD_ROOT=$KCD_ROOT"
