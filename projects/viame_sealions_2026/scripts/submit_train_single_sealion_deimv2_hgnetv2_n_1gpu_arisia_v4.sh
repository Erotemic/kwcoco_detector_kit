#!/usr/bin/env bash
# Lightweight 1-GPU baseline (mobile-class HGNetv2-N, COCO-init).
#
#   scheme:   single_sealion (P0 — 1-class localization-only baseline)
#   variant:  deimv2_hgnetv2_n
#   gpus:     1 (arisia)
#   version:  v4
#
# Sibling of submit_train_pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_v6.sh;
# only scheme + category_names differ. 1-class scheme is the lightest
# of the three on matcher cost (no class dimension to multiply
# num_queries x num_targets by), so batch=24 should be safe.
#
# Changes vs v3:
#   - batch=32 -> 24. Same Mosaic cliff at epoch 4 as the pup runs;
#     1-class lets us keep a slightly higher batch than the 2-cls
#     pup_v6 floor of 16.
#   - LR scaled to sqrt(24/32)*v1's 8e-4.
#
# Submit (from kit root):
#   bash projects/viame_sealions_2026/scripts/submit_train_single_sealion_deimv2_hgnetv2_n_1gpu_arisia_v4.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=single_sealion
export KCD_CATEGORY_NAMES=sealion
export KCD_VARIANT=deimv2_hgnetv2_n
export KCD_NUM_GPUS=1
export KCD_PER_GPU_BATCH=24
export KCD_VAL_BATCH_MULT=1
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[320, 320]'
export KCD_TRAIN_POLICY=fixed
export KCD_LR=6.93e-4
export KCD_BACKBONE_LR=3.46e-4
export KCD_USE_AMP=true

# Tile params — shared with the other baselines.
export KCD_TILE_SIZE=640
export KCD_TILE_SOURCE_SCALES=1.0,0.5,0.25,0.125
export KCD_TILE_STRIDE_FRAC=0.5
export KCD_TILE_MIN_GT_AREA_FRAC=0.0005
export KCD_TILE_MIN_KEEP_FRACTION=0.20
export KCD_TILE_OVERSIZE_FACTOR=1.2
export KCD_TILE_KEEP_NEGATIVE=true

# ============================================================
# Run identity
# ============================================================
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
