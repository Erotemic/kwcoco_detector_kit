#!/usr/bin/env bash
# Lightweight 1-GPU baseline (mobile-class HGNetv2-N, COCO-init).
#
#   scheme:   lifestage_6cls (P2 — full age-sex classifier + NFS distractor)
#   variant:  deimv2_hgnetv2_n (3.6M params, 43.0 COCO AP)
#   gpus:     1 (arisia)
#   version:  v3
#
# Sibling of submit_train_pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_v5.sh;
# only the scheme + category_names differ.
#
# Changes vs v2:
#   - batch=48 -> 32 (pup_v4 OOM'd at iter ~2050 from box-count variance).
#   - LR: 9.8e-4 -> 8e-4 (back to v1 baseline; batch=32 == v1).
#   - Backbone LR: 4.9e-4 -> 4e-4.
#   - val_batch_mult: defaults to 1x.
#
# Note: 6-cls scheme could have somewhat higher memory than pup_vs_nonpup
# (one extra classifier head per query, plus more positives per image
# spread across classes). Watch max_mem closely on the first 1000 iters;
# if it climbs above 35 GB, consider v4 at batch=24.
#
# Submit (from kit root):
#   bash projects/viame_sealions_2026/scripts/submit_train_lifestage_6cls_deimv2_hgnetv2_n_1gpu_arisia_v3.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=lifestage_6cls
export KCD_CATEGORY_NAMES=bull,subadult_male,female,juvenile,pup,northern_fur_seal
export KCD_VARIANT=deimv2_hgnetv2_n
export KCD_NUM_GPUS=1
export KCD_PER_GPU_BATCH=32
export KCD_VAL_BATCH_MULT=1
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[320, 320]'
export KCD_TRAIN_POLICY=fixed
export KCD_LR=8e-4
export KCD_BACKBONE_LR=4e-4
export KCD_USE_AMP=true

# Tile params — shared with pup_vs_nonpup + single_sealion baselines.
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
