#!/usr/bin/env bash
# Lightweight 1-GPU baseline (mobile-class HGNetv2-N, COCO-init).
#
#   scheme:   single_sealion (P0 — 1-class localization-only baseline)
#   variant:  deimv2_hgnetv2_n
#   gpus:     1 (arisia)
#   version:  v5
#
# Changes vs v4:
#   - batch=24 -> 16. v4 OOM'd at epoch 9 (Mosaic cliff with batch=24
#     wasn't safe even for 1-class — my class-cost intuition was wrong).
#     v6 of the pup scheme survived at batch=16 with the same Mosaic
#     schedule, confirming 16 is the universal A6000 ceiling at 320x320.
#   - LR: 6.93e-4 -> 5.66e-4 (sqrt(16/32) * v1's 8e-4).
#   - Backbone LR: 3.46e-4 -> 2.83e-4.
#
# This run is the matched-config sibling of
# submit_train_pup_vs_nonpup_..._v6.sh — same batch + LR + tile params
# means cross-scheme AP comparisons are now apples-to-apples.
#
# Submit (from kit root):
#   bash projects/viame_sealions_2026/scripts/submit_train_single_sealion_deimv2_hgnetv2_n_1gpu_arisia_v5.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=single_sealion
export KCD_CATEGORY_NAMES=sealion
export KCD_VARIANT=deimv2_hgnetv2_n
export KCD_NUM_GPUS=1
export KCD_PER_GPU_BATCH=16
export KCD_VAL_BATCH_MULT=1
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[320, 320]'
export KCD_TRAIN_POLICY=fixed
export KCD_LR=5.66e-4
export KCD_BACKBONE_LR=2.83e-4
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
