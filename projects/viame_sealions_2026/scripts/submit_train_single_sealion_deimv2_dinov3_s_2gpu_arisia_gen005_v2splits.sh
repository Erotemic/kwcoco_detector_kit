#!/usr/bin/env bash
# Generation 5 — single_sealion on the v2 corpus splits.
#
# Sibling to submit_train_pup_vs_nonpup_..._gen005_v2splits.sh. Same
# recipe, different scheme:
#   * KCD_SCHEME=single_sealion (was pup_vs_nonpup)
#   * KCD_CATEGORY_NAMES=sealion (was pup,nonpup_sealion)
#   * KCD_BALANCE_TARGET_JSON shifted to single-class targets
#
# Why run this alongside pup_vs_nonpup gen005:
#   1. Validates Phase 2 (the apply_scheme rewrite) across multiple
#      schemes simultaneously.
#   2. Establishes the v2 corpus's localization-only AP ceiling -
#      single_sealion is the P0 baseline per research_plan.md. The
#      pup_vs_nonpup gen005 number isn't directly comparable to gen004
#      anymore (different corpus), and we need a per-corpus reference
#      to interpret it.
#   3. Reuses the SAME tile cache (scheme-agnostic _universal/b9540ace)
#      so zero extra tile-build cost - just a second apply_scheme +
#      sweep.
#
# Pre-flight (identical to the pup_vs_nonpup gen005):
#   * tile cache built and present at $KCD_TILE_CACHE_DPATH/_universal/
#   * v2 *_norm bundles on arisia
#   * docker image rebuilt with Phase-2 scheme tooling
#
# Submit:
#   bash projects/viame_sealions_2026/scripts/submit_train_single_sealion_deimv2_dinov3_s_2gpu_arisia_gen005_v2splits.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters (identical to pup_vs_nonpup gen005)
# ============================================================
export KCD_SCHEME=single_sealion
export KCD_CATEGORY_NAMES=sealion
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

# Tile params left at paths.sh canonical defaults (same cache as
# pup_vs_nonpup).

# JPEG backend (matches pup_vs_nonpup)
export KCD_USE_WEBDATASET=0

# ============================================================
# Class balance — single-class composition
# ============================================================
# Same empty:positive ratio as pup_vs_nonpup but collapsed to one
# positive class. Keeps the "balance lever" knob consistent so
# scheme comparisons are clean.
export KCD_BALANCE_TARGET_JSON='{"<empty>": 0.4, "sealion": 0.6}'
export KCD_BALANCE_MAX_OVERSAMPLE=1

# Slurm resources (identical to pup_vs_nonpup gen005)
export KCD_CPUS_PER_TASK="${KCD_CPUS_PER_TASK:-4}"
export KCD_MEM="${KCD_MEM:-32G}"
export KCD_TRAIN_NUM_WORKERS="${KCD_TRAIN_NUM_WORKERS:-2}"
export KCD_VAL_NUM_WORKERS="${KCD_VAL_NUM_WORKERS:-1}"

# Run identity
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
