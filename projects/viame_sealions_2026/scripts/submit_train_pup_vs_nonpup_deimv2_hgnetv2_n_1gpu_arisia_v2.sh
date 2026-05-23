#!/usr/bin/env bash
# Lightweight 1-GPU baseline (mobile-class HGNetv2-N, COCO-init).
#
#   scheme:   pup_vs_nonpup (P1 — 2-class)
#   variant:  deimv2_hgnetv2_n (3.6M params, 43.0 COCO AP)
#   gpus:     1 (arisia, less preemption risk than 4-GPU jobs)
#   version:  v2
#
# Changes vs v1:
#   - Batch per-GPU: 32 -> 128 (4x). v1 used 4.9/47 GB of A6000; with
#     ~5 MB/sample activations the bump puts us around ~20 GB, plenty
#     of headroom remaining.
#   - LR + backbone LR: sqrt-2 scaling (~2x) per the AdamW + warmup
#     convention. Linear scaling (4x) is too aggressive for transformer-
#     style detectors; sqrt avoids the "huge batch + huge LR → diverges
#     in the first hundred iters" failure mode.
#
# Submit (from kit root):
#   bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_v2.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion   # must match scheme target_classes order
export KCD_VARIANT=deimv2_hgnetv2_n
export KCD_NUM_GPUS=1
export KCD_PER_GPU_BATCH=128                   # was 32 in v1; A6000 has tons of headroom
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[320, 320]'               # HGNetv2 native 320, no dynamic
export KCD_TRAIN_POLICY=fixed
export KCD_LR=1.6e-3                           # was 8e-4 in v1; sqrt-2 scaling for 4x batch
export KCD_BACKBONE_LR=8e-4                    # was 4e-4 in v1; same sqrt-2 scaling
export KCD_USE_AMP=true
# KCD_INIT_CHECKPOINT auto-resolves from variant (DEIMv2_HGNetv2_N_COCO)

# Tile params: identical to v1 so we share the per-scheme tile cache.
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
