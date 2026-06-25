#!/usr/bin/env bash
# Generation 6 — pup_vs_nonpup, 1280px, dinov3_X, aiq-gpu 2-GPU, FILE balance.
#
#   scheme:   pup_vs_nonpup
#   variant:  deimv2_dinov3_x   (50.3M params)
#   gpus:     2 (aiq-gpu: 2x RTX PRO 6000 Blackwell, 96GB each)
#   res:      1280 (prebuilt 0441d89e tile cache — no rebuild needed)
#   balance:  file (static duplication, same as all gen005 1280 runs)
#   launcher: slurm
#
# ABLATIVE QUESTION: Does 1280px input resolution improve pup AP over 640px
# for the X backbone?  Companion to the X@640 sampler run:
#
#   aiq (this run):   X@1280 file   2-GPU   — tests resolution lever on X
#   aiq (parallel):   X@640  sampler 2-GPU  — tests balance mode on X
#   aiq (future):     X@1280 (sampler?) 4-GPU — best settings, full scale
#
# TOTAL BATCH: 2 GPU × 2 batch = 4 (same as the 4-GPU design 4×1=4).
# LR stays at 4e-4 (no rescaling needed).
#
# MEMORY NOTE: at epoch 0 (before Mosaic) expect ~25-35GB/GPU for X@1280.
# At epoch 2 Mosaic activates; expect ~50-60GB/GPU (still within 96GB).
# If headroom after epoch 0-1 shows <50GB peak, consider raising
# KCD_PER_GPU_BATCH to 4 (total batch 8) and scaling LR to 8e-4.
#
# Pre-flight: 0441d89e (1280px) tile cache must exist on aiq at
# /data/users/jon.crall/kcd_sealion/ssd-data/tile_cache.
#
# Submit (slurm writes to slurm_logs; follow with follow_job.sh <jobid>):
#   KCD_IMAGE=kwcoco-detector-kit:ogdino-cu132-aiq \
#   KCD_TILE_CACHE_DPATH=/data/users/jon.crall/kcd_sealion/ssd-data/tile_cache \
#     bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_dinov3_x_2gpu_aiq_gen006_1280.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion
export KCD_VARIANT=deimv2_dinov3_x
export KCD_NUM_GPUS="${KCD_NUM_GPUS:-2}"
# total batch = 2 * 2 = 4 — same as the planned 4x1 design so LR unchanged.
# Monitor peak GPU mem after epoch 0; raise to 4 if <50GB and scale LR to 8e-4.
export KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-2}"
export KCD_VAL_BATCH_MULT="${KCD_VAL_BATCH_MULT:-1}"
export KCD_NUM_EPOCHS="${KCD_NUM_EPOCHS:-30}"
export KCD_INPUT_HW="${KCD_INPUT_HW:-[1280, 1280]}"
export KCD_TRAIN_POLICY=fixed
# LR matched to total batch=4 (same formula as gen005 S@1280 / 4-GPU X@1280 plan).
export KCD_LR="${KCD_LR:-4e-4}"
export KCD_BACKBONE_LR="${KCD_BACKBONE_LR:-2e-5}"
export KCD_USE_AMP=true

# ============================================================
# Backend + balance — FILE mode (same as gen005 for comparability)
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
