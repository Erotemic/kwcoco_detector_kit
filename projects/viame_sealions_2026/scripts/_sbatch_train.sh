#!/usr/bin/env bash
# Internal boilerplate. Sbatch script — runs on the compute node, sets
# up NCCL diagnostics + dev mounts, then invokes _launch_train.sh
# inside the kit's docker image.
#
# Called by `sbatch` from _submit_train.sh. Don't invoke directly.
# All KCD_* hyperparams must be forwarded via the submit script's
# --export=ALL,KCD_*=...
#
# Resource directives (--gres, --time, --cpus-per-task, --mem) are
# supplied on the sbatch CLI, so this script intentionally omits
# #SBATCH resource lines — the CLI is the single source of truth.
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail

if [ -z "${KCD_REPO_ROOT:-}" ]; then
    KCD_REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
fi
source "$KCD_REPO_ROOT/scripts/paths.sh"

: "${KCD_RUN_NAME:?_sbatch_train.sh: missing KCD_RUN_NAME (sbatch --export forwarding failed?)}"
: "${KCD_NUM_GPUS:?_sbatch_train.sh: missing KCD_NUM_GPUS}"

KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-arisia}"
KCD_ROOT="$KCD_RUNS_DPATH/$KCD_RUN_NAME"
export KCD_ROOT

echo "=== Slurm context ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<manual>}"
echo "HOSTNAME=$(hostname)"
echo "RUN_NAME=$KCD_RUN_NAME"
echo "KCD_ROOT=$KCD_ROOT"
echo "IMAGE=$KCD_IMAGE"
echo "SCHEME=$KCD_SCHEME  VARIANT=$KCD_VARIANT  GPUS=$KCD_NUM_GPUS"
nvidia-smi -L || true
echo

# NCCL diagnostics. Flight recorder only by default (cheap; only fires
# on watchdog timeout). Set KCD_NCCL_DEBUG=verbose for live collective
# logging during active debugging; KCD_NCCL_DEBUG=0 to disable.
KCD_NCCL_DEBUG="${KCD_NCCL_DEBUG:-1}"
NCCL_DEBUG_FLAGS=()
if [ "$KCD_NCCL_DEBUG" != "0" ]; then
    NCCL_DEBUG_FLAGS=(
        -e TORCH_FR_BUFFER_SIZE=20000
        -e TORCH_NCCL_TRACE_BUFFER_SIZE=20000
        -e TORCH_NCCL_DUMP_ON_TIMEOUT=1
        -e TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$KCD_ROOT/nccl_traces/rank_"
        -e TORCH_NCCL_DESYNC_DEBUG=1
    )
fi
if [ "$KCD_NCCL_DEBUG" = "verbose" ]; then
    NCCL_DEBUG_FLAGS+=(-e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=COLL)
fi

# Dev override: mount host's tpl/DEIMv2 over the image's baked copy.
DEV_MOUNT_FLAGS=()
if [ "${KCD_DEV_MOUNT_DEIMV2:-0}" = "1" ]; then
    if [ -d "$KCD_KIT_DPATH/tpl/DEIMv2" ]; then
        DEV_MOUNT_FLAGS+=(-v "$KCD_KIT_DPATH/tpl/DEIMv2:/opt/kwcoco_detector_kit/tpl/DEIMv2")
        echo "DEV: mounting host tpl/DEIMv2 over image's copy" >&2
    else
        echo "WARNING: KCD_DEV_MOUNT_DEIMV2=1 but $KCD_KIT_DPATH/tpl/DEIMv2 not found" >&2
    fi
fi

# Shared shm size scales with gpu count: DataLoader workers' ipc.
shm_gb=$(( 16 + 8 * KCD_NUM_GPUS ))

# Pass through every KCD_* env the launcher needs. Listed explicitly
# so missing forwards fail loud at submit-time, not silently.
KCD_ENV_FLAGS=()
for v in KCD_RUN_NAME KCD_SCHEME KCD_VARIANT KCD_NUM_GPUS KCD_PER_GPU_BATCH \
         KCD_NUM_EPOCHS KCD_INPUT_HW KCD_TRAIN_POLICY KCD_LR KCD_BACKBONE_LR \
         KCD_USE_AMP KCD_INIT_CHECKPOINT KCD_TRAIN_FROM_SCRATCH \
         KCD_CATEGORY_NAMES KCD_TILE_SIZE KCD_TILE_SOURCE_SCALES \
         KCD_TILE_STRIDE_FRAC KCD_TILE_MIN_GT_AREA_FRAC \
         KCD_TILE_MIN_KEEP_FRACTION KCD_TILE_OVERSIZE_FACTOR \
         KCD_TILE_KEEP_NEGATIVE KCD_SCALE_TIER; do
    val="${!v:-}"
    KCD_ENV_FLAGS+=(-e "$v=$val")
done

docker run --rm \
    --gpus all \
    --ipc=host \
    --shm-size="${shm_gb}g" \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
    -e SLURM_JOB_ID="${SLURM_JOB_ID:-manual}" \
    "${KCD_ENV_FLAGS[@]}" \
    "${NCCL_DEBUG_FLAGS[@]}" \
    -v "$KCD_DATA_ROOT:$KCD_DATA_ROOT" \
    -v "$KCD_DATA_DPATH:$KCD_DATA_DPATH" \
    -v "$KCD_REPO_ROOT:$KCD_REPO_ROOT" \
    "${DEV_MOUNT_FLAGS[@]}" \
    -w "$KCD_REPO_ROOT" \
    "$KCD_IMAGE" \
    bash "$KCD_REPO_ROOT/scripts/_launch_train.sh"

echo
echo "Done. Output under: $KCD_ROOT"
