#!/usr/bin/env bash
# Generation 6 -- the run the remaining allocation is spent on.
#
# Almost nothing here is a hyperparameter choice. The schedule, the weight
# decay, the augmentation windows and the loss-scaler setting are all now READ
# from DEIMv2's own DINOv3-X config and rescaled, because the kit's copies of
# them had drifted (see trainers/_deimv2_recipe.py). What this script actually
# chooses is: how long, at what batch, on which data, in which precision.
#
# ## Why 14 epochs at global batch 32
#
# It reproduces the upstream DINOv3-X training budget in BOTH senses at once,
# which the earlier proposals did not:
#
#   upstream COCO   118,287 imgs / batch 32 x 58 epochs = 214k updates, 6.9M views
#   this run        495,514 tiles / batch 32 x 14 epochs = 217k updates, 6.9M views
#
# and the recipe's landmarks land within a couple of thousand updates of
# upstream's at every stage:
#
#   upstream e4  ~14.8k updates  ->  e1   ~15.5k     augmentation begins
#   upstream e29 ~107k           ->  e7   ~108k      flat LR ends, Mosaic+MixUp end
#   upstream e45 ~166k           ->  e11  ~170k      matcher changes
#   upstream e50 ~185k           ->  e12  ~186k      augmentation ends, EMA stage
#   upstream e58 ~214k           ->  e14  ~217k      finish
#
# Batch 32 rather than 64 also restores the LR/batch pairing DINOv3-X was tuned
# with. The earlier fish runs doubled the batch and left lr at 5e-4; upstream's
# README says a batch change should be accompanied by LR/EMA/warmup scaling.
# Rather than introduce an untested LR on the final run, go back to 32.
#
# Cost: batch 32 is ~30% slower per sample than 64 on these cards (measured
# 63 vs 88.9 img/s), so ~33 h rather than ~24 h. Inside the 56 h allocation
# with ~23 h of margin, which is the right trade for a one-shot.
#
# ## What is inherited and why it is not re-litigated here
#
#   fresh COCO init     the checkpoint is optimised for normalized DINO input;
#                       warm-starting a fish checkpoint that adapted to the OLD
#                       unnormalized contract would carry that adaptation in.
#   normalized input    DINOv3 only -- upstream normalizes in its config and in
#                       every inference tool; the kit emitted it nowhere. Now
#                       applied in train, val and both predictor paths, and
#                       recorded in the exported modelspec.
#   1229px tile cache   kept as-is. Retiling at oversize 1.0 would give ~20%
#                       more linear object resolution but triple the tile count
#                       and force a different schedule -- too much change for
#                       the last run. 1229->1024 is 0.833, the same
#                       source-to-model scale the sea-lion pipeline uses.
#   bf16                three fp16 non-finite aborts, all inside augmented
#                       epochs; zero for bf16, which is the only configuration
#                       to have completed a schedule here.
#   negatives kept      77.4% of tiles contain fish already; tiled deployment
#                       creates many chances for false positives, so genuine
#                       background is worth keeping.
#
# ## PREREQUISITE -- do not skip
#
#   bash projects/viame_fish_2026/scripts/submit_baseline_vali.sh
#
# That fixes B (gen001 vs gen003 on full vali, true-tiled 1229) and it must be
# recorded in the journal BEFORE this launches. Success is pre-registered as
# B + 0.01 AP50 with a sequence-bootstrap 90% CI whose lower bound clears zero;
# B + 0.02 is a strong result; AP50:95 must not regress by more than 0.01. The
# held-out test split stays untouched until a checkpoint is selected.
#
# ## Submit (from the kit root, on aiq-gpu, AFTER rebuilding the image)
#
#   nvidia-smi   # a stray vLLM server explains job 296's slow steps and 489's stall
#   bash projects/viame_fish_2026/scripts/submit_train_fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen006_final.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

kcd_require_path "tiled train bundle" "$KCD_TILE_TRAIN_KWCOCO" || {
    echo "  Build it first: bash $SCRIPT_DIR/submit_build_tiles.sh" >&2; exit 1; }
kcd_require_path "tiled vali bundle" "$KCD_TILE_VALI_KWCOCO" || exit 1
export KCD_TRAIN_KWCOCO="$KCD_TILE_TRAIN_KWCOCO"
export KCD_VALI_KWCOCO="$KCD_TILE_VALI_KWCOCO"
export KCD_TEST_KWCOCO="$VF_TEST_KWCOCO"        # untouched until selection

kcd_require_init_checkpoint "deimv2_dinov3_x" || exit 1
export KCD_RESUME_CKPT="${KCD_RESUME_CKPT:-fresh}"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_VARIANT=deimv2_dinov3_x
export KCD_CATEGORY_NAMES=fish
export KCD_NUM_GPUS="${KCD_NUM_GPUS:-4}"
export KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-8}"      # global 32, upstream's
export KCD_VAL_BATCH_MULT="${KCD_VAL_BATCH_MULT:-1}"
export KCD_NUM_EPOCHS="${KCD_NUM_EPOCHS:-14}"
export KCD_INPUT_HW="${KCD_INPUT_HW:-[1024, 1024]}"
export KCD_TRAIN_POLICY="${KCD_TRAIN_POLICY:-fixed}"
export KCD_LR="${KCD_LR:-5e-4}"
export KCD_BACKBONE_LR="${KCD_BACKBONE_LR:-1e-5}"
export KCD_USE_AMP=true
export KCD_AMP_DTYPE="${KCD_AMP_DTYPE:-bfloat16}"

# KCD_FLAT_EPOCH is deliberately NOT set. The recipe supplies 7 (upstream's
# 29/58 rescaled), which is the value we want, and leaving it unset means the
# run demonstrates the recipe working rather than overriding it. weight_decay
# (1.25e-4), the policy [1, 7, 12], mixup [1, 7], copyblend [1, 12],
# stop_epoch 12 and matcher_change 11 all come from the same place.

# ============================================================
# Eval: windowed at the TILE size, not the model input.
# ============================================================
# 1229 is the tile cache's actual on-disk size, resolved from its metadata in
# paths.sh rather than recomputed as 1024*1.2. A 1024 window would slide at a
# different object scale than training used.
export KCD_TILED_EVAL="${KCD_TILED_EVAL:-True}"
export KCD_TILED_EVAL_WINDOW="${KCD_TILED_EVAL_WINDOW:-$KCD_TILE_SIZE_ONDISK}"
export KCD_TILED_EVAL_OVERLAP="${KCD_TILED_EVAL_OVERLAP:-0.25}"
export KCD_TILED_EVAL_BATCH="${KCD_TILED_EVAL_BATCH:-64}"
export KCD_EVAL_DEVICE="${KCD_EVAL_DEVICE:-cuda}"

# ============================================================
# Slurm on aiq
# ============================================================
export KCD_TIME_LIMIT="${KCD_TIME_LIMIT:-56:00:00}"     # ~33 h expected
export KCD_NO_SLURM="${KCD_NO_SLURM:-0}"
export KCD_DOCKER_GPU_MODE="${KCD_DOCKER_GPU_MODE:-gpus}"
export KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-aiq}"
export KCD_TRAIN_NUM_WORKERS="${KCD_TRAIN_NUM_WORKERS:-8}"
export KCD_VAL_NUM_WORKERS="${KCD_VAL_NUM_WORKERS:-4}"
export KCD_NCCL_BLOCKING_WAIT="${KCD_NCCL_BLOCKING_WAIT:-0}"

RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

echo "gen006 -- final run"
echo "  init:      COCO pretrained, fresh"
echo "  data:      $KCD_TRAIN_KWCOCO"
echo "  batch:     $KCD_PER_GPU_BATCH/gpu x $KCD_NUM_GPUS = $(( KCD_PER_GPU_BATCH * KCD_NUM_GPUS ))  (upstream's pairing)"
echo "  epochs:    $KCD_NUM_EPOCHS   lr $KCD_LR / $KCD_BACKBONE_LR   amp $KCD_AMP_DTYPE"
echo "  schedule:  from the upstream recipe -- expect policy [1, 7, 12],"
echo "             stop 12, matcher 11, flat 7, wd 1.25e-4 in the banner"
echo "  eval:      tiled, ${KCD_TILED_EVAL_WINDOW}px source window, overlap $KCD_TILED_EVAL_OVERLAP"
echo "  expect:    ~33 h"
echo

exec bash "$SCRIPT_DIR/_submit_train.sh"
