#!/usr/bin/env bash
# Generation 1 -- FishTrack23 single-class fish detector, DEIMv2-DINOv3-X,
# 1024px whole-frame, aiq-gpu Blackwell, 4-GPU.
#
#   variant:  deimv2_dinov3_x   (50.3M params, 57.8 COCO AP)
#   gpus:     4 (aiq-gpu: 4x RTX PRO 6000 Blackwell, 96 GB each)
#   res:      1024 whole frames -- NO tiling
#   classes:  fish (single, folded from 150 species via the corpus labels.txt)
#   launcher: slurm
#
# ## Why this is complementary rather than redundant
#
# The existing RF-DETR model (fishtrack23_rfdetr_l_seg720_4gpu_viame0227_gen001,
# completed 2026-08-07) is also single-class. This run is architecturally
# distinct -- DINOv3 backbone + DEIM head vs RF-DETR's ViT -- trained on whole
# frames rather than 720px chips, and, unlike that run, against a genuinely
# held-out test set. Two models that fail on the same images are not worth
# ensembling; these plausibly do not.
#
# ## Class scheme: deliberately identical to the baseline
#
# The corpus ships Train/labels.txt: one line, output class `fish` followed by
# 321 aliases. RF-DETR trained through that exact file (its
# rf_detr_mgpu_params.json records class_names: ["fish"]), so we use it too.
# The four non_fish_* categories absent from that file (25,392 boxes, 3.8%) are
# dropped here exactly as VIAME dropped them.
#
# ## Resolution: 1024 whole-frame, no tiling
#
# Box size percentiles over all 665,228 boxes, on 1920x1200 imagery:
#
#     p1  42 x 44      p25 103 x 77     p75 220 x 154
#     p5  59 x 54      p50 150 x 109    p95 359 x 230
#
# A whole-frame resize to 1024 is a 0.53x scale, so the 1st-percentile box is
# still ~22 px wide and the median ~80 px. There is no small-object regime here
# to tile for -- which also means RF-DETR's small_box_area=75 /
# small_action=remove (objects under ~8.7x8.7 px) deleted essentially nothing,
# contrary to what the 2026-08-14 orientation journal assumed.
#
# ## Batch and LR
#
# Memory: the sea-lion project measured dinov3_x at 1280px as
# peak ~= -3160 + 13492 * batch_per_gpu MB. Area-scaling to 1024px
# ((1024/1280)^2 = 0.64) gives ~8.6 GB per unit of batch, so batch=6/GPU lands
# near 49 GB of 96 GB -- roughly half, deliberately conservative for a first
# run on a corpus whose per-image annotation density (up to hundreds of fish in
# a school) is much higher than sea lions.
#
# LR by sqrt scaling (Goyal et al. 2017) from the sea-lion reference point
# (total_batch=4, LR=4e-4):
#     total_batch = 4 * 6 = 24
#     LR = 4e-4 * sqrt(24/4) = 4e-4 * 2.449 ~= 1e-3
#     backbone = same 1/20 ratio -> 5e-5
# These are the identical (batch, LR) pair validated by sea-lion gen007, just
# at a lower resolution, which is the lowest-risk starting point.
#
# ## Epochs
#
# 20, matching the RF-DETR baseline's 20 so a like-for-like comparison is not
# confounded by training length. At ~250k train images and total batch 24 that
# is ~10.4k steps/epoch; expect roughly 20-24 h. The baseline's own best EMA
# was still improving slowly at epoch 15, so 20 is not obviously enough for
# either model -- worth revisiting from the training curve.
#
# Submit (from the kit root, on aiq-gpu):
#   bash projects/viame_fish_2026/scripts/submit_train_fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen001.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_VARIANT=deimv2_dinov3_x
export KCD_CATEGORY_NAMES=fish
export KCD_NUM_GPUS="${KCD_NUM_GPUS:-4}"
export KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-6}"     # total = 4 * 6 = 24
export KCD_VAL_BATCH_MULT="${KCD_VAL_BATCH_MULT:-1}"
export KCD_NUM_EPOCHS="${KCD_NUM_EPOCHS:-20}"
export KCD_INPUT_HW="${KCD_INPUT_HW:-[1024, 1024]}"
export KCD_TRAIN_POLICY="${KCD_TRAIN_POLICY:-fixed}"
export KCD_LR="${KCD_LR:-1e-3}"
export KCD_BACKBONE_LR="${KCD_BACKBONE_LR:-5e-5}"
export KCD_USE_AMP=true

# ============================================================
# Eval -- whole-image, because that is how the model trains.
# The sea-lion project defaults to tiled eval; that exists to stop small
# objects vanishing when a huge aerial is resized to the model input. Not
# applicable here (see the box percentiles above), and using it would measure
# something different from what we train.
# ============================================================
export KCD_TILED_EVAL="${KCD_TILED_EVAL:-False}"
export KCD_EVAL_DEVICE="${KCD_EVAL_DEVICE:-cuda}"

# ============================================================
# Slurm on aiq
# ============================================================
export KCD_NO_SLURM="${KCD_NO_SLURM:-0}"
export KCD_DOCKER_GPU_MODE="${KCD_DOCKER_GPU_MODE:-gpus}"

# Let the NCCL watchdog actually kill a stalled collective.
#
# Job 293 deadlocked at epoch 13 and sat there for two days: all four ranks
# stopped in the same frame (`_engine_run_backward`, det_engine.py:68), every
# GPU pegged at 100% utilization but drawing 78-81W of 300W -- an all-reduce
# spinning forever, confirmed by py-spy on all four ranks.
#
# It should have died in ten minutes. The shared launcher sets
# TORCH_NCCL_ASYNC_ERROR_HANDLING=1 and TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600,
# which together are supposed to abort exactly this -- but it also defaults
# TORCH_NCCL_BLOCKING_WAIT=1, and a blocking wait runs in the main thread where
# the async watchdog cannot preempt it. The two settings cancel out.
#
# Turning the blocking wait off restores the watchdog, so a stall becomes a
# 10-minute failure with a resumable checkpoint instead of a lost weekend. It
# does not prevent the stall; it bounds the damage.
#
# Scoped to this project deliberately: the same default is in the sea-lion
# launcher and deserves the same treatment, but changing shared infrastructure
# is not a side effect this run should have.
export KCD_NCCL_BLOCKING_WAIT="${KCD_NCCL_BLOCKING_WAIT:-0}"
export KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-aiq}"
export KCD_TRAIN_NUM_WORKERS="${KCD_TRAIN_NUM_WORKERS:-8}"
export KCD_VAL_NUM_WORKERS="${KCD_VAL_NUM_WORKERS:-4}"

# ============================================================
# Run identity
# ============================================================
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
