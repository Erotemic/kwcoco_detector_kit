#!/usr/bin/env bash
# Lightweight 1-GPU baseline (mobile-class HGNetv2-N, COCO-init).
#
#   scheme:   pup_vs_nonpup (P1 — 2-class)
#   variant:  deimv2_hgnetv2_n (3.6M params, 43.0 COCO AP)
#   gpus:     1 (arisia)
#   version:  v4
#
# Changes vs v3:
#   - Batch per-GPU: 64 -> 48. v3 sat at 44/47 GB (94% util) with no
#     headroom for variance; a denser-than-average batch will OOM
#     mid-epoch. 48 should put peak around 33-35 GB.
#   - Dataloader workers: planned 4 -> 8 (train), 2 -> 4 (val) but
#     temporarily disabled — the deployed docker image's pareto_sweep
#     doesn't accept the new CLI flag. Reactivate after the next image
#     rebuild (see _launch_train.sh TODO). v3 was dataloader-bound
#     (10-99% GPU util) so this is a throughput win for a later cycle,
#     not a stability requirement.
#
# Submit (from kit root):
#   bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_v4.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion   # must match scheme target_classes order
export KCD_VARIANT=deimv2_hgnetv2_n
export KCD_NUM_GPUS=1
export KCD_PER_GPU_BATCH=48                    # v3 ran 44/47 GB at 64; back off for headroom
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[320, 320]'               # HGNetv2 native 320, no dynamic
export KCD_TRAIN_POLICY=fixed
export KCD_LR=9.8e-4                           # sqrt(48/32) * v1's 8e-4
export KCD_BACKBONE_LR=4.9e-4                  # same ratio
export KCD_USE_AMP=true
export KCD_TRAIN_NUM_WORKERS=8                 # v3 was dataloader-bound; double it
export KCD_VAL_NUM_WORKERS=4
# KCD_INIT_CHECKPOINT auto-resolves from variant (DEIMv2_HGNetv2_N_COCO)

# Tile params: identical to v1/v2/v3 so the universal tile cache is shared.
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
