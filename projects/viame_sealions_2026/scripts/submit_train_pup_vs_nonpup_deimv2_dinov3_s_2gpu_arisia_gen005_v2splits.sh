#!/usr/bin/env bash
# Generation 5 — pup_vs_nonpup on the v2 corpus splits.
#
#   scheme:   pup_vs_nonpup (P1, binding constraint)
#   variant:  deimv2_dinov3_s
#   gpus:     2 (arisia)
#   gen:      005
#
# WHAT'S NEW vs gen004 balanced:
#
#   1. Splits.    v1 (1314 train / NFS=0 in test, the wrong subset)
#                 -> v2 (5997 train / NFS=31 in test across 5 indep
#                 clusters / all 9 classes). See
#                 docs/journals/2026-06-05_splits_v2_design.md.
#
#   2. Corpus.    1314 -> 5997 train imgs (4.6x). Years 2007-2024
#                 instead of 2021-2024.
#
#   3. Categories. Apply_scheme now reads category names directly
#                  from the v2 *_norm bundles (no source_category
#                  letter codes). class_schemes.yaml rewritten with
#                  full-name keys. See
#                  docs/journals/2026-06-05_phase2_change_inventory.md.
#
#   4. Tile cache. Built by submit_build_tiles.sh as a separate job.
#                  Cache scales reduced from [1.0,0.5,0.25,0.125] to
#                  [1.0,0.5] (the deeper scales dropped 100% of pups
#                  via min_gt_area_frac; see same journal).
#
# UNCHANGED from gen004 balanced (these worked):
#   * dinov3_s + class balance + fixed 640 + AMP + per_gpu_batch=8
#   * max_oversample=1 (pup repetition cap)
#   * LR 5e-4 head / 2.5e-5 backbone
#   * 30 epochs
#
# Pre-flight:
#   1. Tile cache must exist:
#        bash projects/viame_sealions_2026/scripts/submit_build_tiles.sh
#      (or this job will fail fast inside docker with a pointer at it)
#   2. v2 *_norm bundles must be on arisia (rsync_to_arisia.sh).
#   3. Docker image rebuilt with the latest Phase-2 scheme tooling.
#
# Submit:
#   bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_dinov3_s_2gpu_arisia_gen005_v2splits.sh
#
# Chain to the tile-build job:
#   KCD_DEPENDS_ON=<build_tiles_jobid> \\
#     bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_dinov3_s_2gpu_arisia_gen005_v2splits.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters (identical to gen004 balanced)
# ============================================================
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion
export KCD_VARIANT=deimv2_dinov3_s
export KCD_NUM_GPUS=2
export KCD_PER_GPU_BATCH=8           # total batch = 16
export KCD_VAL_BATCH_MULT=1
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[640, 640]'
export KCD_TRAIN_POLICY=fixed
export KCD_LR=5e-4
export KCD_BACKBONE_LR=2.5e-5
export KCD_USE_AMP=true

# ============================================================
# Tile params — left at paths.sh canonical defaults
# ============================================================
# Defaults in paths.sh now point at the v2-corpus-tuned values:
#   KCD_TILE_SIZE=640
#   KCD_TILE_SOURCE_SCALES=1.0,0.5       (was 1.0,0.5,0.25,0.125)
#   KCD_TILE_STRIDE_FRAC=0.5
#   KCD_TILE_MIN_GT_AREA_FRAC=0.0005
#   KCD_TILE_MIN_KEEP_FRACTION=0.20
#   KCD_TILE_OVERSIZE_FACTOR=1.2
#   KCD_TILE_KEEP_NEGATIVE=true
#   KCD_TILE_CATEGORY_NAMES=<9-name v2 union>
#   KCD_TILE_MODE=multiscale
# Don't override unless you specifically want a different cache key.

# ============================================================
# Backend: JPEG CocoDetection
# ============================================================
export KCD_USE_WEBDATASET=0

# ============================================================
# Class balance — same composition as gen004 balanced
# ============================================================
export KCD_BALANCE_TARGET_JSON='{"<empty>": 0.4, "pup": 0.2, "nonpup_sealion": 0.4}'
export KCD_BALANCE_MAX_OVERSAMPLE=1

# ============================================================
# Slurm resource budget
# ============================================================
# Right-sized for dinov3_s + 640x640 + 2-GPU JPEG usage. Same as
# gen004 since the per-GPU footprint hasn't changed (the corpus is
# larger but each iteration's memory profile is identical — only the
# epoch count of unique tiles grows).
export KCD_CPUS_PER_TASK="${KCD_CPUS_PER_TASK:-4}"
export KCD_MEM="${KCD_MEM:-32G}"
export KCD_TRAIN_NUM_WORKERS="${KCD_TRAIN_NUM_WORKERS:-2}"
export KCD_VAL_NUM_WORKERS="${KCD_VAL_NUM_WORKERS:-1}"

# ============================================================
# Run identity
# ============================================================
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
