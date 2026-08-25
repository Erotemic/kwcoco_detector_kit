#!/usr/bin/env bash
# Establish the baseline B that gen006's success criterion is defined against.
#
# Scores gen001 and gen003 on the FULL vali split under the frozen final
# inference protocol -- true-tiled, 1229px source window, 0.25 overlap,
# keep_full -- each under its own preprocessing contract.
#
# gen005 is excluded: it aborted at epoch 3 and its surviving checkpoint is
# epoch 1. Add it to KCD_BASELINE_RUNS if you want it as a diagnostic.
#
# RUN THIS BEFORE gen006, and record B plus this protocol in the journal. A
# success threshold fixed after seeing the result is not a threshold.
#
# One GPU; inference only, no training.
#
#   bash projects/viame_fish_2026/scripts/submit_baseline_vali.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

export KCD_LAUNCH_SCRIPT=_launch_baseline_vali.sh
export KCD_RUN_NAME="${KCD_RUN_NAME:-fishtrack23_baseline_vali_1229}"

# The frozen protocol. 1229 is the tile cache's ACTUAL on-disk tile size, taken
# from its metadata rather than recomputed as 1024*1.2 -- the tiler rounds, and
# the stored integer is the authority.
export KCD_TILED_EVAL=True
export KCD_TILED_EVAL_WINDOW="${KCD_TILED_EVAL_WINDOW:-$KCD_TILE_SIZE_ONDISK}"
export KCD_TILED_EVAL_OVERLAP="${KCD_TILED_EVAL_OVERLAP:-0.25}"
export KCD_TILED_EVAL_BATCH="${KCD_TILED_EVAL_BATCH:-64}"
export KCD_EVAL_DEVICE="${KCD_EVAL_DEVICE:-cuda}"

# _submit_train.sh's contract still applies even though nothing trains.
export KCD_VARIANT=deimv2_dinov3_x
export KCD_CATEGORY_NAMES=fish
export KCD_NUM_GPUS="${KCD_NUM_GPUS:-1}"
export KCD_NUM_EPOCHS=1
export KCD_INPUT_HW="${KCD_INPUT_HW:-[1024, 1024]}"
export KCD_LR=5e-4
export KCD_BACKBONE_LR=1e-5

export KCD_TIME_LIMIT="${KCD_TIME_LIMIT:-12:00:00}"
export KCD_CPUS_PER_TASK="${KCD_CPUS_PER_TASK:-16}"
export KCD_MEM="${KCD_MEM:-96G}"
export KCD_NO_SLURM="${KCD_NO_SLURM:-0}"
export KCD_DOCKER_GPU_MODE="${KCD_DOCKER_GPU_MODE:-gpus}"
export KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-aiq}"

echo "baseline B on full vali"
echo "  window:  ${KCD_TILED_EVAL_WINDOW}px source (tile cache metadata)"
echo "  overlap: $KCD_TILED_EVAL_OVERLAP  keep_full: yes"
echo "  vali:    $KCD_VALI_KWCOCO"
echo

exec bash "$SCRIPT_DIR/_submit_train.sh"
