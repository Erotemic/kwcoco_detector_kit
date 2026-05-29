#!/usr/bin/env bash
# Lightweight 1-GPU baseline (mobile-class HGNetv2-N, COCO-init).
#
#   scheme:   pup_vs_nonpup (P1 — 2-class)
#   variant:  deimv2_hgnetv2_n (3.6M params, 43.0 COCO AP)
#   gpus:     1 (arisia)
#   version:  v5
#
# Changes vs v4:
#   - Batch per-GPU: 48 -> 32. v4 hit max_mem=43.8 GB at iter 1500 then
#     OOM'd at iter ~2050 trying to allocate +6.27 GiB. Box-count
#     variance drives a much larger memory delta than expected — the
#     aux losses on 6 decoder layers + DN branches scale with
#     num_queries x num_targets per layer. batch=32 (v1 floor) gives
#     ~29 GB nominal peak with ~10 GB headroom for variance spikes.
#   - val_batch_mult: defaults to 1x (=32) instead of the launcher's
#     prior 2x default. Eval has no backward pass but still allocates
#     activations + matcher intermediates; a 2x val batch can OOM mid-
#     epoch eval after a stable train.
#   - LR: 9.8e-4 -> 8e-4 (back to v1 — batch=32 matches v1 exactly).
#   - Backbone LR: 4.9e-4 -> 4e-4 (same).
#
# Submit (from kit root):
#   bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_v5.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion
export KCD_VARIANT=deimv2_hgnetv2_n
export KCD_NUM_GPUS=1
export KCD_PER_GPU_BATCH=32                    # v4 OOM'd at 48; back to v1's proven floor
export KCD_VAL_BATCH_MULT=1                    # default was 2x; pull to 1x for headroom
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[320, 320]'
export KCD_TRAIN_POLICY=fixed
export KCD_LR=8e-4                             # batch=32 == v1; revert to v1 LR
export KCD_BACKBONE_LR=4e-4
export KCD_USE_AMP=true

# Tile params: identical to v1..v4 so the universal tile cache is shared.
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
