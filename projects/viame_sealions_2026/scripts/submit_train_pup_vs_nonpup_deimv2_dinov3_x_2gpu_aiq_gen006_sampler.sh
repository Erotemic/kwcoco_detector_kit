#!/usr/bin/env bash
# Generation 6 — pup_vs_nonpup, 640px, dinov3_X, aiq-gpu 2-GPU, SAMPLER balance.
#
#   scheme:   pup_vs_nonpup
#   variant:  deimv2_dinov3_x   (50.3M params)
#   gpus:     2 (aiq-gpu: 2x RTX PRO 6000 Blackwell, 96GB each)
#   res:      640 (reuses the existing b9540ace tile cache — no rebuild)
#   launcher: slurm
#
# ABLATIVE QUESTION: Does sampler balance improve training for the X backbone
# vs gen005 X@640 file balance?
#
# gen005 X@640 (4-GPU aiq, file balance): overall tiled AP 0.892, pup 0.864
# This run (2-GPU aiq, sampler balance): same backbone + resolution.
# Confound: 2 vs 4 GPUs (total batch 16 vs 32); note when interpreting.
#
# Fills the sampler-vs-file ablation for the X backbone:
#   namek:   S@640 sampler 1-GPU    (running)
#   arisia:  S@640 sampler 2-GPU    (full_8cls gen006)
#   aiq:     X@640 sampler 2-GPU    (THIS RUN — 2 GPUs available now)
#   aiq:     X@1280 file   4-GPU    (gen006_1280 — run when 4 GPUs free)
#
# BALANCE: same natural epoch (315k, NFS-bound for 640px pup_vs_nonpup)
# and same target JSON as namek gen006.
#
# Submit (slurm writes to slurm_logs; follow with follow_job.sh <jobid>):
#   KCD_IMAGE=kwcoco-detector-kit:ogdino-cu132-aiq \
#   KCD_TILE_CACHE_DPATH=/data/users/jon.crall/kcd_sealion/ssd-data/tile_cache \
#     bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_dinov3_x_2gpu_aiq_gen006_sampler.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion
export KCD_VARIANT=deimv2_dinov3_x
export KCD_NUM_GPUS="${KCD_NUM_GPUS:-2}"
export KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-8}"   # total = 2 * 8 = 16
export KCD_VAL_BATCH_MULT="${KCD_VAL_BATCH_MULT:-1}"
export KCD_NUM_EPOCHS="${KCD_NUM_EPOCHS:-30}"
export KCD_INPUT_HW="${KCD_INPUT_HW:-[640, 640]}"
export KCD_TRAIN_POLICY=fixed
export KCD_LR="${KCD_LR:-5e-4}"
export KCD_BACKBONE_LR="${KCD_BACKBONE_LR:-2.5e-5}"
export KCD_USE_AMP=true

# ============================================================
# Backend: JPEG CocoDetection
# ============================================================
export KCD_USE_WEBDATASET=0

# ============================================================
# Class balance — SAMPLER MODE
# ============================================================
export KCD_BALANCE_MODE=sampler
export KCD_BALANCE_TARGET_JSON='{"<empty>": 0.4, "pup": 0.2, "nonpup_sealion": 0.4}'
export KCD_BALANCE_EPOCH_LENGTH=315000
export KCD_BALANCE_MAX_OVERSAMPLE=1

# ============================================================
# Tile params — canonical 640 defaults (reuse b9540ace cache)
# ============================================================

# ============================================================
# Eval: tiled
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
