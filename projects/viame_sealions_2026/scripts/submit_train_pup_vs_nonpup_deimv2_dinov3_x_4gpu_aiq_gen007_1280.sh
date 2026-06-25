#!/usr/bin/env bash
# Generation 7 — pup_vs_nonpup, 1280px, dinov3_X, aiq-gpu Blackwell, 4-GPU.
#
#   scheme:   pup_vs_nonpup
#   variant:  deimv2_dinov3_x   (50.3M params)
#   gpus:     4 (aiq-gpu: 4x RTX PRO 6000 Blackwell, 96 GB each)
#   res:      1280 (uses the prebuilt 0441d89e tile cache)
#   launcher: slurm
#
# Motivation: gen006 4-GPU (job 25) came in slightly below gen006 2-GPU (job 23)
# on standard eval (0.882 vs 0.899 nocls AP).  Root cause: total batch went
# from 4 (2-GPU × 2/GPU) to 16 (4-GPU × 4/GPU) with LR unchanged at 4e-4 —
# effectively 4× less gradient signal per step relative to batch noise.
#
# Fix: increase per-GPU batch to 6 (total=24) AND scale LR by sqrt(24/4)=√6≈2.45
# from the 2-GPU baseline (LR=4e-4), giving LR=1e-3.  Backbone LR scaled in the
# same ratio (2e-5 → 5e-5).  This follows the square-root scaling rule (Goyal
# et al. 2017) which is standard practice when changing batch size.
#
# Memory analysis (two measured data points):
#   batch=2/GPU (job 23, 2-GPU): 23,824 MB peak per GPU
#   batch=4/GPU (job 25, 4-GPU): 50,808 MB peak per GPU
#   Linear model: peak ≈ -3,160 + 13,492 × batch_per_gpu
#   batch=6/GPU estimate: ~77,792 MB (79% of 96 GB; ~20 GB headroom)  → SAFE
#   batch=7/GPU estimate: ~91,284 MB (93%)                             → too close
#   batch=6 is the right choice.
#
# LR derivation (sqrt scaling from 2-GPU reference):
#   base: total_batch=4, LR=4e-4
#   new:  total_batch=24, LR = 4e-4 × sqrt(24/4) = 4e-4 × 2.449 ≈ 1e-3
#   backbone: same ratio 1/20 → 5e-5
#
# Epochs: 45 (same as gen006 4-GPU).  With larger batch, each epoch has fewer
# steps (N/24 vs N/16), but with sqrt-scaled LR the learning dynamics per data-
# pass are preserved.  Wall time estimate: ~6.5 h (steps/epoch × 0.44 s/step).
#
# BALANCE: file mode (sampler diverged NaN in gen006 X@640; not retried at 1280).
#
# NOTE: the ONNX export batch=32 bug is fixed in tpl/DEIMv2 (batch→1) but the
# aiq image has not been rebuilt yet.  Training is unaffected; rebuild before
# the post-training export step.
#
# Submit (slurm writes to slurm_logs; follow with follow_job.sh <jobid>):
#   aiq-gpu$ KCD_IMAGE=kwcoco-detector-kit:ogdino-cu132-aiq \
#   KCD_TILE_CACHE_DPATH=/data/users/jon.crall/kcd_sealion/ssd-data/tile_cache \
#     bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_dinov3_x_4gpu_aiq_gen007_1280.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion
export KCD_VARIANT=deimv2_dinov3_x
export KCD_NUM_GPUS="${KCD_NUM_GPUS:-4}"
# batch=6/GPU: linear model predicts 77,792 MB (79% of 96 GB; 20 GB headroom).
# total_batch = 4 × 6 = 24.
export KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-6}"    # total = 4 * 6 = 24
export KCD_VAL_BATCH_MULT="${KCD_VAL_BATCH_MULT:-1}"
# 45 epochs: flat_epoch=22 (kit sets num_epochs//2).
export KCD_NUM_EPOCHS="${KCD_NUM_EPOCHS:-45}"
export KCD_INPUT_HW="${KCD_INPUT_HW:-[1280, 1280]}"
export KCD_TRAIN_POLICY=fixed
# LR: sqrt scaling from 2-GPU reference (total_batch=4, LR=4e-4).
#   new LR = 4e-4 × sqrt(24/4) = 4e-4 × 2.449 = 9.8e-4 → 1e-3.
export KCD_LR="${KCD_LR:-1e-3}"
export KCD_BACKBONE_LR="${KCD_BACKBONE_LR:-5e-5}"
export KCD_USE_AMP=true

# ============================================================
# Backend + balance — file mode (same as gen006 for comparability)
# ============================================================
export KCD_USE_WEBDATASET=0
export KCD_BALANCE_TARGET_JSON='{"<empty>": 0.4, "pup": 0.2, "nonpup_sealion": 0.4}'
export KCD_BALANCE_MAX_OVERSAMPLE=1

# ============================================================
# Tile params — 1280 (prebuilt 0441d89e cache)
# ============================================================
export KCD_TILE_SIZE="${KCD_TILE_SIZE:-1280}"

# ============================================================
# Eval: tiled at 1280
# ============================================================
export KCD_TILED_EVAL="${KCD_TILED_EVAL:-True}"
export KCD_EVAL_DEVICE="${KCD_EVAL_DEVICE:-cuda}"

# ============================================================
# Slurm on aiq
# ============================================================
export KCD_NO_SLURM="${KCD_NO_SLURM:-0}"
export KCD_DOCKER_GPU_MODE="${KCD_DOCKER_GPU_MODE:-gpus}"
export KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-aiq}"
export KCD_TRAIN_NUM_WORKERS="${KCD_TRAIN_NUM_WORKERS:-8}"
export KCD_VAL_NUM_WORKERS="${KCD_VAL_NUM_WORKERS:-4}"

# ============================================================
# Run identity
# ============================================================
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
