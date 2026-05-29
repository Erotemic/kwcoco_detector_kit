#!/usr/bin/env bash
# Slurm wrapper for projects/.../scripts/rescore_per_checkpoint.py.
#
# Submits a 1-GPU job that scores every saved checkpoint in <run_name>'s
# run dir against vali (default) or test, with distractor pruning. Uses
# the kit's docker image so the predictor + DEIMv2 deps are the same
# ones the training run used.
#
# Usage (from kit root):
#   bash projects/viame_sealions_2026/scripts/submit_rescore_per_checkpoint.sh <run_name> [vali|test|both]
#
# Examples:
#   bash projects/.../submit_rescore_per_checkpoint.sh \
#       lifestage_6cls_deimv2_hgnetv2_n_1gpu_arisia_v4
#   bash projects/.../submit_rescore_per_checkpoint.sh \
#       pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_v6 test
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <run_name> [vali|test|both]" >&2
    exit 1
fi
RUN_NAME="$1"
EVAL_TARGET="${2:-vali}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

RUN_DIR="$KCD_TRAINING_ROOT/runs/$RUN_NAME"
if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: run dir not found: $RUN_DIR" >&2
    exit 1
fi

LOG_DPATH="${LOG_DPATH:-$KCD_SLURM_LOG_DPATH}"
mkdir -p "$LOG_DPATH"
KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-arisia}"

JOB_NAME="rescore_${RUN_NAME}_${EVAL_TARGET}"

# Hand-rolled sbatch — no need for the full _sbatch_train.sh boilerplate
# since this is just predict + score, no training.
sbatch_args=(
    --parsable
    --job-name="$JOB_NAME"
    --gres=gpu:1
    --cpus-per-task=4
    --mem=24G
    --time=02:00:00
    --nodes=1
    --ntasks=1
    --output="$LOG_DPATH/%x-%j.out"
    --error="$LOG_DPATH/%x-%j.out"
    --export=ALL,RUN_DIR="$RUN_DIR",EVAL_TARGET="$EVAL_TARGET",KCD_IMAGE="$KCD_IMAGE",KCD_REPO_ROOT="$KCD_REPO_ROOT",KCD_DATA_ROOT="$KCD_DATA_ROOT",KCD_DATA_DPATH="$KCD_DATA_DPATH"
    --chdir="$KCD_REPO_ROOT"
)

job_id=$(sbatch "${sbatch_args[@]}" --wrap='
set -euo pipefail
echo "=== rescore_per_checkpoint context ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  HOSTNAME=$(hostname)"
echo "RUN_DIR=$RUN_DIR  EVAL_TARGET=$EVAL_TARGET  IMAGE=$KCD_IMAGE"
nvidia-smi -L || true

# Host-mount the kit package so this works against an image that
# predates per_checkpoint_eval.py (added today). Once the image is
# rebuilt this overlay becomes a no-op.
KIT_DPATH="${KCD_KIT_DPATH:-$KCD_REPO_ROOT/../..}"
KIT_PACKAGE="$KIT_DPATH/kwcoco_detector_kit"
DEV_MOUNT=()
if [ -d "$KIT_PACKAGE" ]; then
    DEV_MOUNT+=(-v "$KIT_PACKAGE:/opt/kwcoco_detector_kit/kwcoco_detector_kit:ro")
fi

docker run --rm \
    --gpus all \
    --ipc=host \
    --shm-size=24g \
    -v "$KCD_DATA_ROOT:$KCD_DATA_ROOT" \
    -v "$KCD_DATA_DPATH:$KCD_DATA_DPATH" \
    -v "$KCD_REPO_ROOT:$KCD_REPO_ROOT" \
    "${DEV_MOUNT[@]}" \
    -w "$KCD_REPO_ROOT" \
    "$KCD_IMAGE" \
    python3 "$KCD_REPO_ROOT/scripts/rescore_per_checkpoint.py" \
        --run-dir "$RUN_DIR" \
        --eval-target "$EVAL_TARGET" \
        --device cuda
')
echo "  job:  $job_id"
echo "  log:  $LOG_DPATH/${JOB_NAME}-${job_id}.out"

# Optional: tail follow. follow_job.py takes (positional jobid,
# --stdout <path>, --poll <interval>).
FOLLOW="$KCD_KIT_DPATH/smoketests/dino_v2_4x/slurm/follow_job.py"
if [ -f "$FOLLOW" ]; then
    python3 "$FOLLOW" "$job_id" --stdout "$LOG_DPATH/${JOB_NAME}-${job_id}.out" --poll 1.0 || true
fi
