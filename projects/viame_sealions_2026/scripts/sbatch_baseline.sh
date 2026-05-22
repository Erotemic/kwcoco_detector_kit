#!/usr/bin/env bash
# Slurm wrapper around launch_baseline.sh — single-GPU baseline.
#
# Submits a 1xGPU job that runs the generic launcher inside the kit's
# docker image. Forward variant/scheme overrides through --export.
#
# Submit:
#     # default = deimv2_hgnetv2_n on pup_vs_nonpup
#     sbatch projects/viame_sealions_2026/scripts/sbatch_baseline.sh
#
# Configurable via env:
#   KCD_SCHEME / KCD_VARIANT / KCD_INPUT_HW / KCD_TRAIN_POLICY /
#   KCD_CATEGORY_NAMES / KCD_NUM_EPOCHS / KCD_PER_GPU_BATCH
#   KCD_IMAGE        docker image tag (default: kwcoco-detector-kit:ogdino-cu132-arisia)
#   KCD_GRES         GPU resource string (default: gpu:1)
#   KCD_TIME_LIMIT   walltime (default: 48:00:00 — baselines are smaller than the big run)

#SBATCH --job-name=sealion-baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

if [ -z "${KCD_REPO_ROOT:-}" ]; then
    KCD_REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
fi
source "$KCD_REPO_ROOT/scripts/paths.sh"

KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-arisia}"

echo "=== Slurm context ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<manual>}"
echo "HOSTNAME=$(hostname)"
echo "REPO=$KCD_REPO_ROOT"
echo "IMAGE=$KCD_IMAGE"
echo "SCHEME=${KCD_SCHEME:-pup_vs_nonpup}"
echo "VARIANT=${KCD_VARIANT:-deimv2_hgnetv2_n}"
nvidia-smi -L || true
echo

# NCCL flight recorder (same defaults as the multi-GPU sbatch — single
# GPU still does init_process_group, and the trace overhead is
# negligible).
KCD_NCCL_DEBUG="${KCD_NCCL_DEBUG:-1}"
NCCL_DEBUG_FLAGS=()
if [ "$KCD_NCCL_DEBUG" != "0" ]; then
    NCCL_DEBUG_FLAGS=(
        -e TORCH_FR_BUFFER_SIZE=20000
        -e TORCH_NCCL_TRACE_BUFFER_SIZE=20000
        -e TORCH_NCCL_DUMP_ON_TIMEOUT=1
        -e TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$KCD_TRAINING_ROOT/baseline_${KCD_VARIANT:-deimv2_hgnetv2_n}_${KCD_SCHEME:-pup_vs_nonpup}/nccl_traces/rank_"
        -e TORCH_NCCL_DESYNC_DEBUG=1
    )
fi
if [ "$KCD_NCCL_DEBUG" = "verbose" ]; then
    NCCL_DEBUG_FLAGS+=(-e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=COLL)
fi

# Dev override: mount host's tpl/DEIMv2 (for testing patches without
# rebuilding the image).
DEV_MOUNT_FLAGS=()
if [ "${KCD_DEV_MOUNT_DEIMV2:-0}" = "1" ]; then
    if [ -d "$KCD_KIT_DPATH/tpl/DEIMv2" ]; then
        DEV_MOUNT_FLAGS+=(-v "$KCD_KIT_DPATH/tpl/DEIMv2:/opt/kwcoco_detector_kit/tpl/DEIMv2")
        echo "DEV: mounting host tpl/DEIMv2 over image's copy" >&2
    else
        echo "WARNING: KCD_DEV_MOUNT_DEIMV2=1 but $KCD_KIT_DPATH/tpl/DEIMv2 not found" >&2
    fi
fi

docker run --rm \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    -e KCD_SCHEME="${KCD_SCHEME:-pup_vs_nonpup}" \
    -e KCD_VARIANT="${KCD_VARIANT:-deimv2_hgnetv2_n}" \
    -e KCD_INPUT_HW="${KCD_INPUT_HW:-}" \
    -e KCD_TRAIN_POLICY="${KCD_TRAIN_POLICY:-}" \
    -e KCD_CATEGORY_NAMES="${KCD_CATEGORY_NAMES:-}" \
    -e KCD_NUM_EPOCHS="${KCD_NUM_EPOCHS:-30}" \
    -e KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-}" \
    -e KCD_INIT_CHECKPOINT="${KCD_INIT_CHECKPOINT:-}" \
    -e KCD_TRAIN_FROM_SCRATCH="${KCD_TRAIN_FROM_SCRATCH:-0}" \
    -e SLURM_JOB_ID="${SLURM_JOB_ID:-manual}" \
    "${NCCL_DEBUG_FLAGS[@]}" \
    -v "$KCD_DATA_ROOT:$KCD_DATA_ROOT" \
    -v "$KCD_REPO_ROOT:$KCD_REPO_ROOT" \
    "${DEV_MOUNT_FLAGS[@]}" \
    -w "$KCD_REPO_ROOT" \
    "$KCD_IMAGE" \
    bash "$KCD_REPO_ROOT/scripts/launch_baseline.sh"

echo
echo "Done. Output under: $KCD_TRAINING_ROOT/baseline_${KCD_VARIANT:-deimv2_hgnetv2_n}_${KCD_SCHEME:-pup_vs_nonpup}"
