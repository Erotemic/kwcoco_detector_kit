#!/usr/bin/env bash
# Lightweight 1-GPU baseline (mobile-class HGNetv2-N, COCO-init).
#
#   scheme:   lifestage_6cls (P2 — full age-sex classifier + NFS distractor)
#   variant:  deimv2_hgnetv2_n (3.6M params, 43.0 COCO AP)
#   gpus:     1 (arisia)
#   version:  v1
#
# Sibling of submit_train_pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_v2.sh;
# only the scheme + category_names differ. All other hyperparameters
# match so cross-scheme comparisons are meaningful.
#
# This is the P2 operational scheme per docs/research_plan.md — six
# classes: bull, subadult_male, female, juvenile, pup, northern_fur_seal.
# subadult_male is the weakest class (3,202 training instances); watch
# its per-class AP. NFS provides a species-boundary signal even though
# it's not the operational target.
#
# Submit (from kit root):
#   bash projects/viame_sealions_2026/scripts/submit_train_lifestage_6cls_deimv2_hgnetv2_n_1gpu_arisia_v1.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=lifestage_6cls
# Must match docs/class_schemes.yaml::schemes.lifestage_6cls.target_order.
export KCD_CATEGORY_NAMES=bull,subadult_male,female,juvenile,pup,northern_fur_seal
export KCD_VARIANT=deimv2_hgnetv2_n
export KCD_NUM_GPUS=1
export KCD_PER_GPU_BATCH=128
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[320, 320]'
export KCD_TRAIN_POLICY=fixed
export KCD_LR=1.6e-3
export KCD_BACKBONE_LR=8e-4
export KCD_USE_AMP=true

# Tile params — identical across the three pup_vs_nonpup / single_sealion
# / lifestage_6cls baselines so they share one universal tile cache.
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
