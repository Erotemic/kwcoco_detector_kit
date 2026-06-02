#!/usr/bin/env bash
# Generation 4 — class-balanced JPEG backend, single-GPU ablation.
#
#   scheme:   pup_vs_nonpup (P1, binding constraint per
#             project_pup_is_binding_constraint memory)
#   variant:  deimv2_hgnetv2_n
#   gpus:     1 (arisia)
#   gen:      004
#
# Hypothesis: gen002 pup_vs_nonpup AP was 0.025 because pup is rare
# (~1-2% of tiles) so the matcher rarely sees pup gradient. Bumping
# pup to 20% of training samples via on-disk MSCOCO duplication
# should unlock pup features. Empty + nonpup share the rest so the
# negative gradient and adult-sealion features stay present.
#
# This is the ABLATION: same model, same input resolution, same
# LR, same epochs as gen003. The ONLY change is class composition.
# If gen004-balanced beats gen002's 0.025 and ideally gen003's
# unbalanced run, the lift is attributable to balancing alone.
#
# Backend switch: JPEG (CocoDetection) instead of WDS. The JPEG
# path has no runtime bucket_weights equivalent; the kit-side
# balance_mscoco CLI duplicates MSCOCO image entries at on-disk
# composition time. See journal 2026-06-01 for the audit.
#
# What changed from gen003 pup_vs_nonpup:
#   1. KCD_USE_WEBDATASET=0 (switch to JPEG / CocoDetection).
#   2. KCD_BALANCE_TARGET_JSON oversamples pup and undersamples
#      empties.
#   3. KCD_WDS_* vars all removed — they don't apply to the JPEG
#      path.
#
# Everything else (model, LR, batch, augmentation, schedule,
# epochs) matches gen003 so the change in result is attributable
# to the class-balance fix.
#
# Submit:
#   bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_gen004_balanced.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters — match gen003 pup_vs_nonpup exactly
# ============================================================
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion
export KCD_VARIANT=deimv2_hgnetv2_n
export KCD_NUM_GPUS=1
export KCD_PER_GPU_BATCH=16
export KCD_VAL_BATCH_MULT=1
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[320, 320]'
export KCD_TRAIN_POLICY=fixed
export KCD_LR=5.66e-4
export KCD_BACKBONE_LR=2.83e-4
export KCD_USE_AMP=true

# ============================================================
# Tile params — unchanged from gen003 (universal tile cache is shared)
# ============================================================
export KCD_TILE_SIZE=320
export KCD_TILE_SOURCE_SCALES=1.0
export KCD_TILE_STRIDE_FRAC=0.5
export KCD_TILE_MIN_GT_AREA_FRAC=0.0005
export KCD_TILE_MIN_KEEP_FRACTION=0.20
export KCD_TILE_OVERSIZE_FACTOR=1.2
export KCD_TILE_KEEP_NEGATIVE=true

# ============================================================
# Backend: JPEG CocoDetection (not WDS)
# ============================================================
# The WDS path's bucket_weights re-weights at sample-pick time;
# the JPEG path's balance_mscoco duplicates image entries in the
# on-disk MSCOCO before training. Either gives the model the
# same gradient distribution, but JPEG is what we just shipped
# tests for (commit 4c338c8) and what arisia's SSD favors per
# the user's earlier perf note ("the simple jpeg loader is
# faster than webdataset for SSDs").
export KCD_USE_WEBDATASET=0

# ============================================================
# gen004-specific knobs — these define the experiment
# ============================================================
# Class-balance target. Bucket keys:
#   "<empty>" — images with no annotations after apply_scheme
#   "pup", "nonpup_sealion" — target_category names from the scheme
#
# Source composition (per gen003 corpus stats): empties ~80%,
# pup ~1-2%, nonpup the rest. After balance:
#   empty       0.4   (still a strong neg gradient, but yields
#                      room for positives)
#   nonpup      0.4   (matches a moderately-common class — the
#                      easier positive — so the matcher learns it)
#   pup         0.2   (~10-20x bump vs natural rate; pup tiles
#                      will repeat within an epoch but stochastic
#                      augmentations differentiate them)
#
# Reproducibility: this string IS the experiment record. Don't
# pass via env override; edit the script for a new run.
export KCD_BALANCE_TARGET_JSON='{"<empty>": 0.4, "pup": 0.2, "nonpup_sealion": 0.4}'

# Match input size = output size (per user decision 2026-06-01).
# Default is unset; balance_mscoco uses len(src.images) when
# KCD_BALANCE_TARGET_SIZE is unset.
# export KCD_BALANCE_TARGET_SIZE=

# ============================================================
# Run identity
# ============================================================
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
