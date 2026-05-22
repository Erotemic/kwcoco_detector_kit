#!/usr/bin/env bash
# Production 4-GPU operational run.
#
#   scheme:   pup_vs_nonpup (P1 — 2-class pup / nonpup_sealion)
#   variant:  deimv2_dinov3_s (DINOv3-S foundation backbone)
#   gpus:     4 (arisia's RTX A6000 cluster)
#   version:  v1 — initial hyperparams from research_plan phase 4
#
# All hyperparameters are declared explicitly below. Tweaks → new vN
# file in this directory; the boilerplate scripts (_submit_train.sh,
# _sbatch_train.sh, _launch_train.sh) stay put.
#
# Submit (from kit root):
#   bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_dinov3_s_4gpu_arisia_v1.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion   # must match scheme target_classes order
export KCD_VARIANT=deimv2_dinov3_s
export KCD_NUM_GPUS=4
export KCD_PER_GPU_BATCH=16
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW=640,640
export KCD_TRAIN_POLICY=multiscale_512_768
export KCD_LR=5e-4
export KCD_BACKBONE_LR=2.5e-5
export KCD_USE_AMP=true
# KCD_INIT_CHECKPOINT auto-resolves from variant (DEIMv2_DINOv3_S_COCO)

# Tile params (per-scheme, model-independent — would only change if
# the scheme's image dist. changes meaningfully).
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
