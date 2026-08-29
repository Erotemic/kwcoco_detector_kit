#!/usr/bin/env bash
# Final test numbers for the record: all four generations, one protocol.
#
#   bash projects/viame_fish_2026/scripts/submit_test_score.sh
#
# ## What this is for
#
# The tiling hypothesis is already falsified on vali -- under true-tiled
# deployment geometry the WHOLE-FRAME runs beat the TILE-trained ones:
#
#   gen003 (whole frame)  0.7689   <- B
#   gen001 (whole frame)  0.7658
#   gen007 (tiles)        0.7311   epoch 6
#
# This adds the held-out split to that picture so a future brainstorm can see
# how far the confident predictions missed, on the split that was supposed to
# be the arbiter. gen006 is included because it is the run gen007 was designed
# to improve on and it was never scored under deployment geometry at all.
#
# ## Comparability, which is the whole point
#
# gen001 and gen003 already have test numbers (~0.7285 for gen003), but those
# are WHOLE-IMAGE, taken before true-tiled inference existed. Comparing a
# true-tiled gen006 number against them would repeat exactly the protocol
# mismatch that made the earlier RF-DETR comparison useless. Everything is
# rescored here, true-tiled, in bf16, on the full split.
#
# ## Selection happened on vali
#
# Each run's checkpoint comes from the vali ranking summary, not from anything
# measured here. gen006 has 14 staged epochs and NO vali entry yet, so rank it
# first or this exits with an explanation:
#
#   KCD_NO_SLURM=1 KCD_EVAL_STRIDE=8 \
#     KCD_BASELINE_RUNS="...gen006_final" \
#     bash projects/viame_fish_2026/scripts/submit_baseline_vali.sh
#
# One GPU; inference only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

export KCD_LAUNCH_SCRIPT=_launch_test_score.sh
export KCD_RUN_NAME="${KCD_RUN_NAME:-fishtrack23_test_score_1229}"

export KCD_TEST_SCORE_RUNS="${KCD_TEST_SCORE_RUNS:-\
fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen001 \
fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen003_bf16_fresh \
fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen006_final \
fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen007_seqbalance}"

# Same frozen protocol as the vali scoring, from the tile cache's own metadata.
export KCD_TILED_EVAL=True
export KCD_TILED_EVAL_WINDOW="${KCD_TILED_EVAL_WINDOW:-$KCD_TILE_SIZE_ONDISK}"
if ! [[ "$KCD_TILED_EVAL_WINDOW" =~ ^[0-9]+$ ]] || [ "$KCD_TILED_EVAL_WINDOW" -lt 64 ]; then
    echo "ERROR: tiled-eval window did not resolve: '$KCD_TILED_EVAL_WINDOW'" >&2
    exit 1
fi
export KCD_TILED_EVAL_OVERLAP="${KCD_TILED_EVAL_OVERLAP:-0.25}"
export KCD_TILED_EVAL_BATCH="${KCD_TILED_EVAL_BATCH:-64}"
export KCD_EVAL_DEVICE="${KCD_EVAL_DEVICE:-cuda}"
export KCD_AMP_DTYPE="${KCD_AMP_DTYPE:-bfloat16}"

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
export KCD_NO_SLURM="${KCD_NO_SLURM:-1}"     # no slurm on this host
export KCD_DOCKER_GPU_MODE="${KCD_DOCKER_GPU_MODE:-gpus}"
export KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-aiq}"

echo "test scoring -- 4 generations, one protocol"
echo "  window:  ${KCD_TILED_EVAL_WINDOW}px source (tile cache metadata)"
echo "  overlap: $KCD_TILED_EVAL_OVERLAP  keep_full: yes  amp: $KCD_AMP_DTYPE"
echo "  test:    $KCD_TEST_KWCOCO"
echo "  runs:    $KCD_TEST_SCORE_RUNS"
echo "  NOTE: checkpoints come from the VALI ranking; nothing is chosen here."
echo

exec bash "$SCRIPT_DIR/_submit_train.sh"
