#!/usr/bin/env bash
# Generation 4 — class-balanced JPEG backend, 2-GPU + bigger backbone.
#
#   scheme:   pup_vs_nonpup (P1, binding constraint)
#   variant:  deimv2_hgnetv2_s  (~3-4x more params than _n)
#   gpus:     2 (arisia)
#   gen:      004
#
# Bigger-leap companion to the 1-GPU ablation
# (submit_train_pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_gen004_balanced.sh).
# Same class-balance target so the comparison isolates
# "bigger backbone + 2-GPU" vs the 1-GPU baseline.
#
# Compared to the 1-GPU ablation:
#   1. KCD_VARIANT     n -> s     (~3-4x params; better feature
#                                  capacity for the rare pup class)
#   2. KCD_NUM_GPUS    1 -> 2     (effective batch 32 vs 16; better
#                                  grad stability)
#   3. KCD_LR          5.66e-4 -> 8.00e-4  (sqrt(2) scaling for 2x
#                                  effective batch; safer than
#                                  linear scaling for the larger
#                                  backbone)
#   4. KCD_BACKBONE_LR 2.83e-4 -> 4.00e-4  (sqrt(2) on the
#                                  pretrained-backbone branch too)
#
# Everything else (class composition target, input resolution,
# tile params, epochs, schedule) matches the 1-GPU ablation so
# the two runs differ on exactly the two levers we want to test:
# capacity + grad stability.
#
# Resource budget per memory feedback_arisia_resource_budgets:
# arisia is shared; we request 2 CPU and 24G per GPU for the
# hgnetv2_s family. Override via env when submitting if the kit's
# pinned scaling underbids.
#
# Prerequisite: pretrained checkpoint on disk. Fetch with:
#   bash projects/viame_sealions_2026/scripts/fetch_pretrained.sh deimv2_hgnetv2_s
#
# Submit:
#   bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_hgnetv2_s_2gpu_arisia_gen004_balanced.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion
export KCD_VARIANT=deimv2_hgnetv2_s
export KCD_NUM_GPUS=2
export KCD_PER_GPU_BATCH=16          # total batch = 32
export KCD_VAL_BATCH_MULT=1
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[320, 320]'
export KCD_TRAIN_POLICY=fixed
# sqrt(2) batch scaling from gen003 baseline (5.66e-4 @ batch 16).
export KCD_LR=8.00e-4
export KCD_BACKBONE_LR=4.00e-4
export KCD_USE_AMP=true

# ============================================================
# Tile params — unchanged from gen003 (universal tile cache shared)
# ============================================================
export KCD_TILE_SIZE=320
export KCD_TILE_SOURCE_SCALES=1.0
export KCD_TILE_STRIDE_FRAC=0.5
export KCD_TILE_MIN_GT_AREA_FRAC=0.0005
export KCD_TILE_MIN_KEEP_FRACTION=0.20
export KCD_TILE_OVERSIZE_FACTOR=1.2
export KCD_TILE_KEEP_NEGATIVE=true

# ============================================================
# Backend: JPEG CocoDetection (matches the 1-GPU ablation)
# ============================================================
export KCD_USE_WEBDATASET=0

# ============================================================
# Class balance — IDENTICAL to the 1-GPU ablation
# ============================================================
# Keeping the same target so the n-vs-s and 1-vs-2-GPU comparison
# isolates capacity + batch effects from data composition.
export KCD_BALANCE_TARGET_JSON='{"<empty>": 0.4, "pup": 0.2, "nonpup_sealion": 0.4}'

# ============================================================
# Run identity
# ============================================================
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
