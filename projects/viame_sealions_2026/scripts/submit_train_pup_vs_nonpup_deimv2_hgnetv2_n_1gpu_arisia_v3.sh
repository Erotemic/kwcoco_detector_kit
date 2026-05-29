#!/usr/bin/env bash
# Lightweight 1-GPU baseline (mobile-class HGNetv2-N, COCO-init).
#
#   scheme:   pup_vs_nonpup (P1 — 2-class)
#   variant:  deimv2_hgnetv2_n (3.6M params, 43.0 COCO AP)
#   gpus:     1 (arisia)
#   version:  v3
#
# Changes vs v2:
#   - Batch per-GPU: 128 -> 64. v2 OOM'd on the A6000 partway through
#     the first epoch ("Tried to allocate 10.04 GiB", 4.87 GiB free of
#     47.40 GiB total). Activations + optimizer state at batch=128 +
#     320x320 + transformer stack overshoots 48 GB. 64 gives ~2x
#     headroom; still 2x v1's 32.
#   - LR + backbone LR: pulled back to sqrt-2 of v1 (4x batch -> 2x LR).
#     v2's LR was tuned for batch=128; at batch=64 (2x v1) we want only
#     sqrt-2 of v1's LR.
#
# Submit (from kit root):
#   bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_v3.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion   # must match scheme target_classes order
export KCD_VARIANT=deimv2_hgnetv2_n
export KCD_NUM_GPUS=1
export KCD_PER_GPU_BATCH=64                    # v2 OOM'd at 128; 64 is the safe headroom
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[320, 320]'               # HGNetv2 native 320, no dynamic
export KCD_TRAIN_POLICY=fixed
export KCD_LR=1.13e-3                          # sqrt(2) * v1's 8e-4
export KCD_BACKBONE_LR=5.66e-4                 # sqrt(2) * v1's 4e-4
export KCD_USE_AMP=true
# KCD_INIT_CHECKPOINT auto-resolves from variant (DEIMv2_HGNetv2_N_COCO)

# Tile params: identical to v1/v2 so the universal tile cache is shared.
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
