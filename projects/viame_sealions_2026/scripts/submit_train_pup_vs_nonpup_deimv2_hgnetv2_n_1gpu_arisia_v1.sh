#!/usr/bin/env bash
# Lightweight 1-GPU baseline (mobile-class HGNetv2-N, COCO-init).
#
#   scheme:   pup_vs_nonpup (P1 — 2-class)
#   variant:  deimv2_hgnetv2_n (3.6M params, 43.0 COCO AP)
#   gpus:     1 (arisia, less preemption risk than 4-GPU jobs)
#   version:  v1 — initial mobile-tier baseline
#
# Provides a floor mAP for direct comparison against the 4-GPU
# dinov3_s run (same scheme, same epochs, same dataloader).
#
# Submit (from kit root):
#   bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_v1.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=pup_vs_nonpup
export KCD_VARIANT=deimv2_hgnetv2_n
export KCD_NUM_GPUS=1
export KCD_PER_GPU_BATCH=32
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW=320,320       # HGNetv2 doesn't support dynamic input; native 320
export KCD_TRAIN_POLICY=fixed
export KCD_LR=8e-4                # from deimv2_hgnetv2_n_coco.yml
export KCD_BACKBONE_LR=4e-4
export KCD_USE_AMP=true
# KCD_INIT_CHECKPOINT auto-resolves from variant (DEIMv2_HGNetv2_N_COCO)

# Tile params: identical to the 4-GPU run so both share the per-scheme
# tile cache (tile is model-independent — same 640px multiscale tiles
# get downsampled to 320 in the dataloader for hgnetv2).
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
