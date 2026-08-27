#!/usr/bin/env bash
# Generation 7 -- treat the corpus as 439 video sequences, not 495,514 samples.
#
# gen006 ran the upstream recipe correctly and cleanly: 14/14 epochs, zero
# NaN, zero destructive reloads, every mechanism repaired. It still peaked at
# EPOCH 4 OF 14 and declined monotonically to epoch 11 while training loss fell
# from 33.20 to 26.80. Loss down and validation down together is not a run that
# ended too early; it is a model fitting something that does not transfer.
#
# So gen007 changes what an epoch IS, not how long the schedule runs.
#
# ## The measurement this is built on
#
# From `kwcoco_detector_kit.data.sequence_balance measure` over the actual
# train corpus (recorded in docs/journals/, re-runnable, not estimated):
#
#   SEQUENCE      439 groups / 495,514 tiles
#                 min 30, median 260, mean 1129, max 13,450  -> 448x imbalance
#                 gini 0.747
#                 EFFECTIVE COUNT 81 of 439  (18.3%)
#                 the top 10% of sequences hold 66.4% of all tiles
#
#   TRACK         14,103 tracks / 780,566 tile-annotations
#                 min 1, median 31, mean 55, max 4,794       -> 4794x imbalance
#                 gini 0.573
#                 EFFECTIVE COUNT 2,963 of 14,103  (21.0%)
#
#   SOURCE FRAME  251,143 groups, gini 0.013, effective 99.3%
#
# Read the effective counts first. Uniform sampling over 495,514 tiles draws
# from something that behaves like 81 sequences and 2,963 tracks. Four fifths
# of the nominal diversity is redundancy -- one fish, tracked for 4,794 tiles,
# outvotes 150 median-length tracks by itself. That is a mechanism for the
# exact curve gen006 produced.
#
# The source-frame line is why this script balances sequences and tracks and
# NOT frames: tiles are already almost perfectly uniform per frame (2 per
# frame, gini 0.013), so frame-level balancing would cost complexity and do
# nothing. frame_alpha stays 0.
#
# ## Why alpha = 0.5, not 1.0
#
# Also measured, by sweeping alpha and recomputing the effective counts:
#
#   seq_a track_a cap | eff_seq  eff_track | max draws/tile/epoch
#   0.00   0.00  none |      81       1454 |   0.19    (gen006, today)
#   0.50   0.50     8 |     238       4473 |   1.55    <- chosen
#   0.75   0.50     8 |     334       3059 |   1.55
#   1.00   0.50    32 |     394       1729 |   6.20
#
# Full flattening (seq_alpha=1.0) buys sequence diversity by DESTROYING track
# diversity -- 4,473 effective tracks down to 1,729 -- because it pours mass
# into short sequences, and short sequences are short precisely because they
# contain few tracks. Half-flattening triples effective sequences (81 -> 238)
# and triples effective tracks (1,454 -> 4,473) at the same time. It is the
# only row that improves both.
#
# max_oversample=8 is what keeps the cure from becoming the disease. Uncapped,
# seq_alpha=0.5 draws some single tile 7.7 times per epoch -- memorisation of
# a different tile. Capped, 1.55, and effective track count goes UP not down.
#
# ## The update budget
#
#   epoch_length 96,000 tiles x 28 epochs / batch 32 = 84,000 updates
#
# gen006 peaked at epoch 4 of 14 = ~62k updates and never beat it over the
# remaining ~155k. 84k gives that observed peak room plus margin, at a third
# of gen006's 217k. Each tile is now seen ~5.4 times across the whole run
# rather than 14 times, and WHICH tiles are seen is redrawn every epoch, so
# the schedule's 28 stages also give the LR cosine and the augmentation state
# machine real resolution instead of gen006's coarse 14.
#
# REFINE THIS IF THE STRIDE-8 CURVE ARRIVES. If gen006's tiled epoch ranking
# peaks materially later than epoch 4, raise KCD_NUM_EPOCHS accordingly; the
# 84k here encodes "peaks at 4" as the current best estimate, not a law.
#
# ## Lighter augmentation
#
# aug_profile=tiled_light drops Mosaic, RandomZoomOut and RandomIoUCrop and
# disables mixup/copyblend. All five assume each sample is an independent
# scene. These samples are 1229px crops of video frames: the crop diversity is
# already supplied by the tiler, and compositing two reefs into one frame asks
# the model to detect fish in scenes the sensor cannot produce. Photometric
# distortion and horizontal flip stay -- neither assumes scene independence.
#
# ## Lower fine-tuning LR
#
# 2.5e-4 / 5e-6, both halved from gen006, keeping upstream's 50:1 head/backbone
# ratio intact. Halving the pair rather than retuning it: the ratio is what
# DINOv3-X was tuned with, the magnitude is what a narrow-domain fine-tune of a
# COCO checkpoint does not need at full strength.
#
# ## NOT LAUNCHED AUTOMATICALLY
#
# This script is reviewed before it is run. Nothing in it is irreversible, but
# it occupies all four GPUs for ~11 h.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

# Inherited experiment-defining variables are cleared, exactly as gen006 does.
# KCD_TILE_SIZE_ONDISK is deliberately excluded: paths.sh has already derived
# it from the tile cache metadata, and unsetting it leaves the eval window
# EMPTY rather than forcing a recomputation.
for _stale in KCD_FLAT_EPOCH KCD_INIT_CHECKPOINT KCD_TRAIN_FROM_SCRATCH \
              KCD_FORCE_EVAL KCD_BALANCE_EPOCH_LENGTH KCD_AUG_PROFILE; do
    if [ -n "${!_stale:-}" ]; then
        echo "  note: clearing inherited $_stale=${!_stale}" >&2
        unset "$_stale"
    fi
done
unset _stale

kcd_require_init_checkpoint "deimv2_dinov3_x" || exit 1
export KCD_RESUME_CKPT=fresh

# ============================================================
# Hyperparameters
# ============================================================
export KCD_VARIANT=deimv2_dinov3_x
export KCD_CATEGORY_NAMES=fish
export KCD_NUM_GPUS=4
export KCD_PER_GPU_BATCH=8              # global 32, upstream's LR pairing
export KCD_VAL_BATCH_MULT=1
export KCD_NUM_EPOCHS=28
export KCD_INPUT_HW="[1024, 1024]"
export KCD_TRAIN_POLICY=fixed
export KCD_LR=2.5e-4                    # halved from gen006
export KCD_BACKBONE_LR=5e-6             # halved from gen006; 50:1 preserved
export KCD_USE_AMP=true
export KCD_AMP_DTYPE=bfloat16           # the only precision to finish a run here

# KCD_FLAT_EPOCH stays unset so the recipe supplies it. At 28 epochs expect
# policy [2, 14, 24], flat 14, no_aug 4, stop 24, matcher 22, wd 1.25e-4 --
# and mixup/copyblend reported as the disabled sentinel (40000, 15000).

# ============================================================
# Sequence/track-aware sampling
# ============================================================
export KCD_AUG_PROFILE=tiled_light
export KCD_BALANCE_SEQUENCE=True
export KCD_BALANCE_SEQ_ALPHA=0.5
export KCD_BALANCE_TRACK_ALPHA=0.5
export KCD_BALANCE_EMPTY_WEIGHT=1.0     # a negative tile ~ an average positive
export KCD_BALANCE_MAX_OVERSAMPLE=8
export KCD_BALANCE_EPOCH_LENGTH=96000   # x28 / batch 32 = 84k updates
export KCD_BALANCE_SEED=0

# Sequence identity lives ONLY in the untiled bundle: the tiler stamps
# tile_source_gid on each tile but does not copy video_id. paths.sh exports
# this; assert it rather than silently degrading to frame-level grouping,
# which the measurement shows would do nothing (gini 0.013).
: "${KCD_TILE_SOURCE_KWCOCO:?paths.sh did not export KCD_TILE_SOURCE_KWCOCO}"
kcd_require_path "untiled source bundle" "$KCD_TILE_SOURCE_KWCOCO"
if [ "$KCD_TILE_SOURCE_KWCOCO" = "$KCD_TRAIN_KWCOCO" ]; then
    echo "ERROR: KCD_TILE_SOURCE_KWCOCO == KCD_TRAIN_KWCOCO." >&2
    echo "  Training must run on the TILED bundle and balancing must read the" >&2
    echo "  UNTILED one. Equal paths means KCD_TRAIN_KWCOCO was never pointed" >&2
    echo "  at the tile cache, and every tile would be its own sequence." >&2
    exit 1
fi

# ============================================================
# Eval: windowed at the TILE size, not the model input
# ============================================================
export KCD_TILED_EVAL=True
export KCD_TILED_EVAL_WINDOW="${KCD_TILED_EVAL_WINDOW:-$KCD_TILE_SIZE_ONDISK}"
if ! [[ "$KCD_TILED_EVAL_WINDOW" =~ ^[0-9]+$ ]] || [ "$KCD_TILED_EVAL_WINDOW" -lt 64 ]; then
    echo "ERROR: tiled-eval window did not resolve: '$KCD_TILED_EVAL_WINDOW'" >&2
    echo "  It comes from the tile cache metadata via KCD_TILE_SIZE_ONDISK." >&2
    exit 1
fi
export KCD_TILED_EVAL_OVERLAP=0.25
export KCD_TILED_EVAL_BATCH="${KCD_TILED_EVAL_BATCH:-64}"
export KCD_EVAL_DEVICE="${KCD_EVAL_DEVICE:-cuda}"

# Selection happens afterwards, under deployment geometry, from staged epochs.
export KCD_SELECTION_JOURNAL=True
export KCD_DO_EVAL=False
export KCD_DO_EXPORT=False
export KCD_DO_BENCH=False

# ============================================================
# Execution
# ============================================================
export KCD_TIME_LIMIT="${KCD_TIME_LIMIT:-24:00:00}"
export KCD_NO_SLURM="${KCD_NO_SLURM:-1}"   # no slurm on aiq-gpu; run directly
export KCD_DOCKER_GPU_MODE="${KCD_DOCKER_GPU_MODE:-gpus}"
export KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-aiq}"
export KCD_TRAIN_NUM_WORKERS="${KCD_TRAIN_NUM_WORKERS:-8}"
export KCD_VAL_NUM_WORKERS="${KCD_VAL_NUM_WORKERS:-4}"
export KCD_NCCL_BLOCKING_WAIT="${KCD_NCCL_BLOCKING_WAIT:-0}"

RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

echo "gen007 -- sequence/track-balanced sampling"
echo "  init:       COCO pretrained, fresh"
echo "  train:      $KCD_TRAIN_KWCOCO   (tiled)"
echo "  sequences:  $KCD_TILE_SOURCE_KWCOCO   (untiled, for sequence identity)"
echo "  batch:      $KCD_PER_GPU_BATCH/gpu x $KCD_NUM_GPUS = $(( KCD_PER_GPU_BATCH * KCD_NUM_GPUS ))"
echo "  epochs:     $KCD_NUM_EPOCHS x $KCD_BALANCE_EPOCH_LENGTH tiles = $(( KCD_NUM_EPOCHS * KCD_BALANCE_EPOCH_LENGTH / (KCD_PER_GPU_BATCH * KCD_NUM_GPUS) )) updates"
echo "  lr:         $KCD_LR / $KCD_BACKBONE_LR   amp $KCD_AMP_DTYPE"
echo "  sampling:   seq_alpha $KCD_BALANCE_SEQ_ALPHA, track_alpha $KCD_BALANCE_TRACK_ALPHA, cap $KCD_BALANCE_MAX_OVERSAMPLE"
echo "              expect effective sequences 81 -> ~238, tracks ~1454 -> ~4473"
echo "  aug:        $KCD_AUG_PROFILE (no Mosaic/ZoomOut/IoUCrop, no mixup/copyblend)"
echo "  schedule:   from the recipe -- expect policy [2, 14, 24], flat 14,"
echo "              no_aug 4, stop 24, matcher 22, mixup (40000, 15000)"
echo "  eval:       tiled, ${KCD_TILED_EVAL_WINDOW}px window, overlap $KCD_TILED_EVAL_OVERLAP"
echo "  expect:     ~11 h"
echo

exec bash "$SCRIPT_DIR/_submit_train.sh"
