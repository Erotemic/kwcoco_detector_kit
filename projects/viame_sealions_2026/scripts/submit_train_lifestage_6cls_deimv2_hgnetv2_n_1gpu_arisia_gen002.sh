#!/usr/bin/env bash
# Generation 2 — full-resolution tiles + WebDataset.
#
#   scheme:   lifestage_6cls (P2)
#   variant:  deimv2_hgnetv2_n
#   gpus:     1 (arisia)
#   gen:      002
#
# Sibling of submit_train_pup_vs_nonpup_..._gen002.sh; only scheme +
# category_names + per-GPU batch differ. See that file's preamble for
# the gen001->gen002 rationale (full-res tiles + WebDataset IO).
#
# Note: 6-cls scheme adds matcher-cost overhead vs the 2-cls schemes
# (each class extra to discriminate at every query slot). v4 OOM'd at
# batch=32 with the OLD downsampled-tile bundle; gen001's batch=12 was
# the safe ceiling. At full-resolution tiles the per-batch GT density
# is higher (no decimated views to dilute crowds), so we keep batch=12.
#
# Submit (from kit root):
#   bash projects/viame_sealions_2026/scripts/submit_train_lifestage_6cls_deimv2_hgnetv2_n_1gpu_arisia_gen002.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=lifestage_6cls
export KCD_CATEGORY_NAMES=bull,subadult_male,female,juvenile,pup,northern_fur_seal
export KCD_VARIANT=deimv2_hgnetv2_n
export KCD_NUM_GPUS=1
export KCD_PER_GPU_BATCH=12
export KCD_VAL_BATCH_MULT=1
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[320, 320]'
export KCD_TRAIN_POLICY=fixed
export KCD_LR=4.9e-4
export KCD_BACKBONE_LR=2.45e-4
export KCD_USE_AMP=true

# ============================================================
# gen002 tile params — FULL RESOLUTION ONLY
# ============================================================
export KCD_TILE_SIZE=320
export KCD_TILE_SOURCE_SCALES=1.0
export KCD_TILE_STRIDE_FRAC=0.5
export KCD_TILE_MIN_GT_AREA_FRAC=0.0005
export KCD_TILE_MIN_KEEP_FRACTION=0.20
export KCD_TILE_OVERSIZE_FACTOR=1.2
export KCD_TILE_KEEP_NEGATIVE=true

export KCD_USE_WEBDATASET=1

# ============================================================
# Run identity
# ============================================================
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
