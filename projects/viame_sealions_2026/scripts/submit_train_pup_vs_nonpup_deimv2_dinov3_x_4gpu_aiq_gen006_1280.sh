#!/usr/bin/env bash
# Generation 6 — pup_vs_nonpup, 1280px, dinov3_X, aiq-gpu Blackwell, 4-GPU.
#
#   scheme:   pup_vs_nonpup
#   variant:  deimv2_dinov3_x   (50.3M params)
#   gpus:     4 (aiq-gpu: 4x RTX PRO 6000 Blackwell, 96 GB each)
#   res:      1280 (uses the prebuilt 0441d89e tile cache)
#   launcher: slurm
#
# This is the scaled-up successor to gen006 X@1280 2-GPU (job 23, pup AP 0.875).
# Memory analysis (2026-06-21, dev/analyze_gpu_memory.py) found:
#   peak max_mem at batch=2/GPU: 23,824 MB per GPU (linear estimate at batch=4:
#   ~47,648 MB conservative; ~35,600 MB const+linear — both well within 96 GB).
# → batch=4/GPU is safe; 74 GB headroom at batch=2 leaves room for 2× growth.
#
# LR: held at 4e-4 (same as 2-GPU reference). Historical runs (gen005 X@640
# 4-GPU) show no linear LR scaling is applied across batch changes; 4e-4 is
# the proven value for this config.
#
# Epochs: 45 instead of 30.  Kit auto-sets flat_epoch = epochs//2 = 22 (vs 15
# in the 30-epoch run), giving 7 more high-LR epochs.  The 30-epoch run's in-
# loop mAP was still rising at ep29 (+0.001/epoch) but the LR was already at
# its minimum — training from scratch with more epochs is the correct lever.
#
# BALANCE: file mode (sampler diverged to NaN in gen006 X@640 sampler run).
#
# Pre-flight: 0441d89e (1280px) tile cache must exist on aiq.
#
# Submit (slurm writes to slurm_logs; follow with follow_job.sh <jobid>):
#   aiq-gpu$ KCD_IMAGE=kwcoco-detector-kit:ogdino-cu132-aiq \
#   KCD_TILE_CACHE_DPATH=/data/users/jon.crall/kcd_sealion/ssd-data/tile_cache \
#     bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_dinov3_x_4gpu_aiq_gen006_1280.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion
export KCD_VARIANT=deimv2_dinov3_x
export KCD_NUM_GPUS="${KCD_NUM_GPUS:-4}"
# batch=4/GPU confirmed safe by memory analysis: peak 23,824 MB at batch=2/GPU;
# linear estimate at 4/GPU ≈ 47,648 MB (conservative); well within 96 GB.
export KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-4}"    # total = 4 * 4 = 16
export KCD_VAL_BATCH_MULT="${KCD_VAL_BATCH_MULT:-1}"
# 45 epochs: flat_epoch=22 (7 more high-LR epochs than 30-epoch ref).
export KCD_NUM_EPOCHS="${KCD_NUM_EPOCHS:-45}"
export KCD_INPUT_HW="${KCD_INPUT_HW:-[1280, 1280]}"
export KCD_TRAIN_POLICY=fixed
# LR unchanged from 2-GPU reference (job 23, pup AP 0.875). No linear
# scaling applied — consistent with gen005 X@640 4-GPU practice.
export KCD_LR="${KCD_LR:-4e-4}"
export KCD_BACKBONE_LR="${KCD_BACKBONE_LR:-2e-5}"
export KCD_USE_AMP=true

# ============================================================
# Backend + balance — file mode (same as gen005 1280 for comparability)
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
