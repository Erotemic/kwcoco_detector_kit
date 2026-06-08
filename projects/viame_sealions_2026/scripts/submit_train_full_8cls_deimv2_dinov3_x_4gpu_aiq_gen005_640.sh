#!/usr/bin/env bash
# Generation 5 — full_8cls (full taxonomy), 640px, dinov3_X, aiq-gpu Blackwell.
#
#   scheme:   full_8cls  (juvenile,bull,female,subadult_male,pup,dead_pup,
#                         northern_fur_seal,dead_nonpup — 8 classes)
#   variant:  deimv2_dinov3_x   (50.3M params)
#   gpus:     4 (aiq-gpu: 4x RTX PRO 6000 Blackwell, 96GB each)
#   res:      640 (REUSES the existing b9540ace tile cache — no rebuild;
#                  that cache already carries all source categories)
#   launcher: standalone docker (no slurm) -> KCD_NO_SLURM=1
#
# PURPOSE: a comparison run against a collaborator training these exact 8
# classes in this exact index order. The class order is the COMPARISON
# CONTRACT (see docs/class_schemes.yaml: full_8cls.target_order) — keep
# KCD_CATEGORY_NAMES identical to target_order.
#
# Same recipe as the validated pup_vs_nonpup dinov3_X aiq run
# (submit_train_pup_vs_nonpup_deimv2_dinov3_x_4gpu_aiq_gen005_640.sh); only
# the scheme + category set + balance change. The images are identical, so
# GT density (hence memory) is ~unchanged — batch 16 carried that run, but
# watch `max mem`: X@640 sat near 70GB and dense Mosaic batches can spike a
# rank (the earlier OOM was a GPU zombie, not batch 16 itself; drop to 12 if
# a rank approaches the ceiling).
#
# CAVEAT for the comparison: dead_pup and especially dead_nonpup (~49
# instances corpus-wide) are data-starved — their per-class AP will be very
# noisy regardless of balance. NFS is a real output class but stays excluded
# from our class-agnostic selection AP (distractor; NFS is always a negative).
#
# Run it in tmux, capture with tee (NOT nohup):
#   KCD_IMAGE=kwcoco-detector-kit:ogdino-cu132-aiq \
#   KCD_NO_SLURM=1 \
#   KCD_TILE_CACHE_DPATH=/data/users/jon.crall/kcd_sealion/ssd-data/tile_cache \
#     bash projects/viame_sealions_2026/scripts/submit_train_full_8cls_deimv2_dinov3_x_4gpu_aiq_gen005_640.sh \
#     2>&1 | tee /data/users/jon.crall/kcd_sealion/aiq_full8_x_640.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Hyperparameters (== aiq dinov3_x 640 run; scheme + balance change) -
export KCD_SCHEME=full_8cls
# Order MUST equal full_8cls.target_order (the comparison contract).
export KCD_CATEGORY_NAMES=juvenile,bull,female,subadult_male,pup,dead_pup,northern_fur_seal,dead_nonpup
export KCD_DISTRACTOR_CLASSES="${KCD_DISTRACTOR_CLASSES:-northern_fur_seal}"
export KCD_VARIANT=deimv2_dinov3_x
export KCD_NUM_GPUS="${KCD_NUM_GPUS:-4}"
export KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-16}"   # total = 4 * 16 = 64
export KCD_VAL_BATCH_MULT="${KCD_VAL_BATCH_MULT:-1}"
export KCD_NUM_EPOCHS="${KCD_NUM_EPOCHS:-30}"
export KCD_INPUT_HW="${KCD_INPUT_HW:-[640, 640]}"
export KCD_TRAIN_POLICY=fixed
export KCD_LR="${KCD_LR:-5e-4}"
export KCD_BACKBONE_LR="${KCD_BACKBONE_LR:-2.5e-5}"
export KCD_USE_AMP=true

# ---- Backend + balance (8-class rebalance) -----------------------------
# Pup (binding constraint) gets a slight boost; the data-starved dead classes
# get small targets (oversample-capped anyway at MAX_OVERSAMPLE=1, so they
# appear at ~their natural low rate). Sums to 1.0.
export KCD_USE_WEBDATASET=0
export KCD_BALANCE_TARGET_JSON='{"<empty>": 0.25, "juvenile": 0.1, "bull": 0.1, "female": 0.1, "subadult_male": 0.1, "pup": 0.15, "dead_pup": 0.05, "northern_fur_seal": 0.1, "dead_nonpup": 0.05}'
export KCD_BALANCE_MAX_OVERSAMPLE=1

# ---- Tile params: 640 defaults (existing b9540ace cache) ---------------
# Left at paths.sh canonical defaults (KCD_TILE_SIZE=640, scales 1.0,0.5).
# The b9540ace cache was tiled with the full source category set, so all 8
# target classes survive the scheme application — no retiling needed.

# ---- Eval: tiled (windowed) on GPU -------------------------------------
export KCD_TILED_EVAL="${KCD_TILED_EVAL:-True}"
export KCD_EVAL_DEVICE="${KCD_EVAL_DEVICE:-cuda}"

# ---- Standalone docker on a dedicated box ------------------------------
export KCD_NO_SLURM="${KCD_NO_SLURM:-1}"
export KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-aiq}"
export KCD_TRAIN_NUM_WORKERS="${KCD_TRAIN_NUM_WORKERS:-8}"
export KCD_VAL_NUM_WORKERS="${KCD_VAL_NUM_WORKERS:-4}"

# ---- Run identity ------------------------------------------------------
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
