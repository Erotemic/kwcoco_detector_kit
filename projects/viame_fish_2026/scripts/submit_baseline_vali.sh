#!/usr/bin/env bash
# Establish B, and pick gen006's best epoch, under ONE frozen protocol.
#
# Scores gen001, gen003 and EVERY staged gen006 epoch on vali under the frozen
# final-inference protocol -- true-tiled, 1229px source window, 0.25 overlap,
# keep_full, bf16 -- each checkpoint under its own preprocessing contract.
#
# ## Two stages
#
# vali is 35,111 images and gen006 staged 14 epochs, so 16 checkpoints x full
# vali is days of GPU. Run it in two passes:
#
#   stage 1   KCD_EVAL_STRIDE=8   (the default)
#             16 checkpoints on every 8th image = 4,389 images each. That is
#             ~70k image-inferences, the same total this job was already
#             budgeted 12h for when it scored 2 checkpoints on all 35,111.
#             Ranks the epochs. Produces NO reportable number.
#
#   stage 2   KCD_EVAL_STRIDE=1 KCD_BASELINE_RUNS="<baselines> <winner-run>"
#             the finalists on the full split. This is where B and the chosen
#             checkpoint come from, and it is the only output that may be
#             journalled or compared to gen003's 0.5406.
#
# Stage 1 is a paired within-run ranking: every checkpoint sees the identical
# subsample, so the subsample's bias cancels. It is NOT an estimate of B, and
# score_epochs.py stamps the stride into both the output paths and the summary
# json so the two stages cannot be quietly mixed.
#
# gen005 is excluded: it aborted at epoch 3 and its surviving checkpoint is
# epoch 1. Add it to KCD_BASELINE_RUNS if you want it as a diagnostic.
#
# The test split stays untouched until exactly one checkpoint is chosen here.
#
# One GPU; inference only, no training.
#
#   bash projects/viame_fish_2026/scripts/submit_baseline_vali.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

export KCD_LAUNCH_SCRIPT=_launch_baseline_vali.sh
export KCD_EVAL_STRIDE="${KCD_EVAL_STRIDE:-8}"
export KCD_RUN_NAME="${KCD_RUN_NAME:-fishtrack23_vali_1229_s${KCD_EVAL_STRIDE}}"

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

export KCD_TIME_LIMIT="${KCD_TIME_LIMIT:-24:00:00}"
export KCD_CPUS_PER_TASK="${KCD_CPUS_PER_TASK:-16}"
export KCD_MEM="${KCD_MEM:-96G}"
export KCD_NO_SLURM="${KCD_NO_SLURM:-0}"
export KCD_DOCKER_GPU_MODE="${KCD_DOCKER_GPU_MODE:-gpus}"
export KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-aiq}"

echo "vali scoring -- B + gen006 epoch curve"
echo "  window:  ${KCD_TILED_EVAL_WINDOW}px source (tile cache metadata)"
echo "  overlap: $KCD_TILED_EVAL_OVERLAP  keep_full: yes"
echo "  vali:    $KCD_VALI_KWCOCO"
echo "  stride:  $KCD_EVAL_STRIDE"
if [ "$KCD_EVAL_STRIDE" != "1" ]; then
  echo "  STAGE 1: ranks checkpoints only. Nothing here is B."
else
  echo "  STAGE 2: full split. This is the reportable number."
fi
echo

exec bash "$SCRIPT_DIR/_submit_train.sh"
