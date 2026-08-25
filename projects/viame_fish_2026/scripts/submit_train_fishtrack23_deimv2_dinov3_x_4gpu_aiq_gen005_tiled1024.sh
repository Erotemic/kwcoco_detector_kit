#!/usr/bin/env bash
# Generation 5 -- native-resolution 1024px windows. No resize anywhere.
#
# ## The problem this attacks
#
# Every run so far trained on whole frames resized to the model input. For the
# 97% of this corpus that is 1920 wide, that is an x-scale of 0.533 and a
# y-scale of 0.853: nearly half the horizontal resolution discarded, plus a
# 1.6-1.78x aspect distortion. A p1 fish (42x44 native) reached the model as
# 22x38 px.
#
# The RF-DETR baseline never paid that. Its VIAME config uses adaptive
# chipping -- chip_width 720, chip_step 480, chip_adaptive_thresh 1.6 Mpx --
# so every image above 1.6 Mpx is cut into 720px windows at NATIVE scale.
#
# gen003's metrics show what it cost us: AP_small 0.192 against AP_large
# 0.653. And two schedule experiments (gen003 at 24 epochs, gen004 at 48) have
# now landed within noise of each other on held-out test, which is what being
# resolution-limited rather than schedule-limited looks like.
#
# The sea-lion project measured the fix on the same failure mode: windowed
# native-resolution eval lifted pup AP 0.123 -> 0.838.
#
# ## A BIGGER window than RF-DETR, which is also cheaper
#
# 1024 covers 2.0x the area of 720, so fewer windows cover a frame:
#
#            window  stride  overlap  tiles/frame
#   RF-DETR    720     480     33%       7.90
#   THIS       1024    768     25%       5.87
#
# More context per window, full native resolution, and 26% fewer crops than
# the baseline. And because the window equals the model input, every tile is
# fed 1:1 -- no resize, no distortion, anywhere in the pipeline.
#
# ## Schedule -- sized from the MEASURED tiling, not the estimate
#
# I predicted 5.87 tiles/frame from a dense-grid calculation. The tiler emits a
# tile only where one is needed, so job 494 actually produced:
#
#   495,514 train tiles = 1.97/frame   780,566 annotations (1.58/tile)
#   111,835 empty (22.6%)              59 GB on disk
#
# Every number moved in our favour: a third of the predicted tiles, a quarter
# of the predicted empties (so keep_negative is a non-issue), and 59 GB rather
# than 95. But it also means an epoch is far cheaper than planned, and the
# original 10 epochs would have been only ~77k optimizer steps -- FEWER than
# gen003's 94k, the opposite of what this run needs.
#
#   7,742 steps/epoch at batch 64
#   93 min train + ~9 min vali eval + save  =>  ~103 min/epoch
#   24 epochs => ~41 h and ~186k steps (2.0x gen003, 1.4x gen001)
#
# 24 rather than 28: 28 lands at 48.2 h, which leaves no margin once per-epoch
# eval is counted. 24 keeps ~7 h of slack inside the walltime.
#
# flat_epoch 8 of 24 holds the 1:2 flat:cosine ratio that gen003's gains came
# from, rather than gen004/492's 1:1, which spent 21 epochs oscillating at
# constant LR and never beat its own epoch-1 score.
#
# ## Eval
#
# KCD_TILED_EVAL=True slides the same 1024px native window over each full test
# image and NMS-merges (eval/tiled_predictor.py), with keep_full merging a
# whole-image pass so fish larger than a window still get recalled. The test
# bundle is deliberately NOT tiled, so the headline number stays directly
# comparable to gen001's 0.7272 and gen003's 0.7285.
#
# Note this is the first fish run where KCD_TILED_EVAL does anything at all:
# _launch_train.sh never forwarded it to the sweep before, so every previous
# run's setting was inert.
#
# ## PREREQUISITE
#
#   bash projects/viame_fish_2026/scripts/submit_build_tiles.sh
#
# Measured: 495,514 train tiles / 59 GB (job 494), plus vali. The train script
# refuses to start without both.
#
# Submit (from the kit root, on aiq-gpu, AFTER tiling and an image rebuild):
#   bash projects/viame_fish_2026/scripts/submit_train_fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen005_tiled1024.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

# Train and vali come from the tile bundles; test stays whole-frame.
kcd_require_path "tiled train bundle" "$KCD_TILE_TRAIN_KWCOCO" || {
    echo "  Build it first:" >&2
    echo "    bash projects/viame_fish_2026/scripts/submit_build_tiles.sh" >&2
    exit 1
}
kcd_require_path "tiled vali bundle" "$KCD_TILE_VALI_KWCOCO" || exit 1
export KCD_TRAIN_KWCOCO="$KCD_TILE_TRAIN_KWCOCO"
export KCD_VALI_KWCOCO="$KCD_TILE_VALI_KWCOCO"
export KCD_TEST_KWCOCO="$VF_TEST_KWCOCO"        # whole frames, on purpose

kcd_require_init_checkpoint "deimv2_dinov3_x" || exit 1
export KCD_RESUME_CKPT="${KCD_RESUME_CKPT:-fresh}"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_VARIANT=deimv2_dinov3_x
export KCD_CATEGORY_NAMES=fish
export KCD_NUM_GPUS="${KCD_NUM_GPUS:-4}"
# A 1024px tile is the same tensor as a 1024px-resized frame, so gen003's
# measured 37.8 GB of 96 per GPU still holds at batch 16.
export KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-16}"
export KCD_VAL_BATCH_MULT="${KCD_VAL_BATCH_MULT:-1}"
export KCD_NUM_EPOCHS="${KCD_NUM_EPOCHS:-24}"           # policy -> [2, 12, 23]
export KCD_FLAT_EPOCH="${KCD_FLAT_EPOCH:-8}"            # 16 cosine epochs
export KCD_INPUT_HW="${KCD_INPUT_HW:-[1024, 1024]}"
export KCD_TRAIN_POLICY="${KCD_TRAIN_POLICY:-fixed}"
export KCD_LR="${KCD_LR:-5e-4}"
export KCD_BACKBONE_LR="${KCD_BACKBONE_LR:-1e-5}"
export KCD_USE_AMP=true
export KCD_AMP_DTYPE="${KCD_AMP_DTYPE:-float16}"

# ============================================================
# Eval: windowed, matching how the model now trains.
# ============================================================
export KCD_TILED_EVAL="${KCD_TILED_EVAL:-True}"
export KCD_TILED_EVAL_WINDOW="${KCD_TILED_EVAL_WINDOW:-$KCD_TILE_SIZE}"
export KCD_TILED_EVAL_OVERLAP="${KCD_TILED_EVAL_OVERLAP:-0.25}"
export KCD_TILED_EVAL_BATCH="${KCD_TILED_EVAL_BATCH:-64}"
export KCD_EVAL_DEVICE="${KCD_EVAL_DEVICE:-cuda}"

# ============================================================
# Slurm on aiq
# ============================================================
export KCD_TIME_LIMIT="${KCD_TIME_LIMIT:-56:00:00}"     # ~41 h expected
export KCD_NO_SLURM="${KCD_NO_SLURM:-0}"
export KCD_DOCKER_GPU_MODE="${KCD_DOCKER_GPU_MODE:-gpus}"
export KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-aiq}"
export KCD_TRAIN_NUM_WORKERS="${KCD_TRAIN_NUM_WORKERS:-8}"
export KCD_VAL_NUM_WORKERS="${KCD_VAL_NUM_WORKERS:-4}"

export KCD_NCCL_BLOCKING_WAIT="${KCD_NCCL_BLOCKING_WAIT:-0}"

# ============================================================
# Run identity
# ============================================================
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

echo "gen005 -- native-resolution ${KCD_TILE_SIZE}px windows"
echo "  train:     $KCD_TRAIN_KWCOCO"
echo "  vali:      $KCD_VALI_KWCOCO"
echo "  test:      $KCD_TEST_KWCOCO  (whole frames + tiled eval)"
echo "  batch:     $KCD_PER_GPU_BATCH/gpu x $KCD_NUM_GPUS = $(( KCD_PER_GPU_BATCH * KCD_NUM_GPUS ))"
echo "  epochs:    $KCD_NUM_EPOCHS  (flat $KCD_FLAT_EPOCH, cosine $(( KCD_NUM_EPOCHS - KCD_FLAT_EPOCH )))"
echo "  eval:      tiled, window $KCD_TILED_EVAL_WINDOW overlap $KCD_TILED_EVAL_OVERLAP"
echo "  expect:    ~1.7 h/epoch, ~41 h total (~186k steps)"
echo

exec bash "$SCRIPT_DIR/_submit_train.sh"
