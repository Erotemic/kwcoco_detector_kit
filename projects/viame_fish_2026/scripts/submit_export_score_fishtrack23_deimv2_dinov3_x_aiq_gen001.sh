#!/usr/bin/env bash
# Export + score gen001's existing checkpoint, WITHOUT claiming training finished.
#
# Produces a candidate model and an honest held-out number from the epoch-12
# checkpoint that gen001 left behind when it deadlocked at epoch 13. Does not
# train, does not resume, and does not write the `.train_complete` marker --
# `sweep` continues to see an unfinished run, which it is.
#
# To actually finish the schedule, use
#   submit_resume_fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen001.sh
# The two are independent: exporting now does not compromise resuming later.
#
# One GPU and a short walltime -- this is inference over 33,434 test images plus
# an ONNX trace, not training.
#
# Submit (from the kit root, on aiq-gpu):
#   bash projects/viame_fish_2026/scripts/submit_export_score_fishtrack23_deimv2_dinov3_x_aiq_gen001.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The run whose checkpoint we are exporting. KCD_ROOT is derived from this, so
# it must match the training run's name exactly.
export KCD_RUN_NAME="${KCD_RUN_NAME:-fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen001}"

# Drive the export/score launcher instead of the training one, reusing the
# same sbatch + docker + GPU-pinning machinery.
export KCD_LAUNCH_SCRIPT=_launch_export_score.sh

# _submit_train.sh's contract still applies (it pre-flights the bundles and the
# init checkpoint), so these have to be set even though nothing trains.
export KCD_VARIANT=deimv2_dinov3_x
export KCD_CATEGORY_NAMES=fish
export KCD_NUM_GPUS="${KCD_NUM_GPUS:-1}"
export KCD_NUM_EPOCHS=20
export KCD_INPUT_HW="${KCD_INPUT_HW:-[1024, 1024]}"
export KCD_LR=1e-3
export KCD_BACKBONE_LR=5e-5

# Whole-image eval, matching how the model was trained. Tiled eval would
# measure something the model never saw.
export KCD_TILED_EVAL="${KCD_TILED_EVAL:-False}"
export KCD_EVAL_DEVICE="${KCD_EVAL_DEVICE:-cuda}"

export KCD_TIME_LIMIT="${KCD_TIME_LIMIT:-08:00:00}"
export KCD_CPUS_PER_TASK="${KCD_CPUS_PER_TASK:-16}"
export KCD_MEM="${KCD_MEM:-64G}"

export KCD_NO_SLURM="${KCD_NO_SLURM:-0}"
export KCD_DOCKER_GPU_MODE="${KCD_DOCKER_GPU_MODE:-gpus}"
export KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-aiq}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
