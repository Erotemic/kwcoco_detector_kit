#!/usr/bin/env bash
# Lightweight 1-GPU baseline (mobile-class HGNetv2-N, COCO-init).
#
#   scheme:   pup_vs_nonpup (P1 — 2-class)
#   variant:  deimv2_hgnetv2_n
#   gpus:     1 (arisia)
#   version:  v6
#
# Changes vs v5:
#   - Batch per-GPU: 32 -> 16. v5 trained stably epochs 0-4 at max_mem
#     ~31-36 GB, then OOM'd at epoch 5 iter ~6000 trying to allocate
#     12.57 GiB in one shot. Root cause: Mosaic + MixUp + CopyBlend
#     turn on at epoch 4 (per DEIMv2's transforms epoch schedule),
#     and a 4x-mosaic'd batch item carries ~4x the annotations of a
#     normal tile. With sea-lion crowd tiles, that's hundreds of GT
#     boxes -> Hungarian matcher cost matrix blows up.
#   - LR: 8e-4 -> 5.66e-4 (1/sqrt(2) * v1).
#   - Backbone LR: 4e-4 -> 2.83e-4.
#
# At batch=16, a 4x-mosaic'd batch is effectively ~64 GT-density,
# below v4's batch=48 ceiling (which trained successfully epochs 0-3
# before the variance OOM).
#
# Submit (from kit root):
#   bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_v6.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion
export KCD_VARIANT=deimv2_hgnetv2_n
export KCD_NUM_GPUS=1
export KCD_PER_GPU_BATCH=16                    # v5 OOM'd at 32 once Mosaic kicked in (epoch 4)
export KCD_VAL_BATCH_MULT=1
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[320, 320]'
export KCD_TRAIN_POLICY=fixed
export KCD_LR=5.66e-4
export KCD_BACKBONE_LR=2.83e-4
export KCD_USE_AMP=true

# Tile params: shared with the other baselines.
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
