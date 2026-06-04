#!/usr/bin/env bash
# gen004 single_sealion + dinov3_s + balance — 2 GPU.
#
# Companion to the pup_vs_nonpup gen004 resume. Same recipe
# (dinov3_s + class-balanced JPEG + max_oversample=1 + fixed 640
# + AMP), applied to the second binding-constraint scheme.
#
# Why now: gen003 single_sealion 2565 (skip_empty=False, WDS,
# no balance) reached in-train mAP 0.000 after 11 epochs
# ([[2026-06-04_gen004_forensic_and_resume]]). The matcher
# converged to "predict nothing" because positives were ~20% of
# the stream — exactly the failure mode the balance treatment is
# designed to fix.
#
# gen004 pup_vs_nonpup 2577 with the same recipe reached
# in-train mAP 0.161 in 5 epochs (6.4× gen002's 0.025). Strong
# prior that single_sealion + same recipe will work.
#
# Differences vs the pup_vs_nonpup resume:
#   * Scheme: single_sealion (one target class: "sealion")
#   * Balance target: 2-bucket {empty: 0.5, sealion: 0.5}
#   * Start from scratch (init_checkpoint = dinov3_s COCO pretrained)
#     — no prior dinov3_s single_sealion checkpoint exists
#
# Resource budget matches pup resume; same model + scale + batch.
#
# Submit:
#   bash projects/viame_sealions_2026/scripts/submit_train_single_sealion_deimv2_dinov3_s_2gpu_arisia_gen004_balanced.sh
#
# Or queue behind the pup_vs_nonpup resume so they run back-to-back
# on the same 2 GPUs without a wallclock gap. The kit's
# KCD_DEPENDS_ON env hooks slurm --dependency. Use afterany so the
# single_sealion job runs regardless of whether the pup resume
# succeeded or hit walltime / OOM (we want single_sealion either
# way; afterok would block on a pup failure).
#
#   PUP_JOB=2579   # whatever sbatch returns for the pup resume
#   KCD_DEPENDS_ON="afterany:$PUP_JOB" \
#     bash projects/viame_sealions_2026/scripts/submit_train_single_sealion_deimv2_dinov3_s_2gpu_arisia_gen004_balanced.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters — mirror the dinov3_s pup gen004 recipe
# ============================================================
export KCD_SCHEME=single_sealion
export KCD_CATEGORY_NAMES=sealion
export KCD_VARIANT=deimv2_dinov3_s
export KCD_NUM_GPUS=2
export KCD_PER_GPU_BATCH=16
export KCD_VAL_BATCH_MULT=1
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[640, 640]'
# Fixed 640, NOT multiscale_512_768 — see resume journal for why
# (768x768 batches OOM at this model size + batch).
export KCD_TRAIN_POLICY=fixed
export KCD_LR=5e-4
export KCD_BACKBONE_LR=2.5e-5
export KCD_USE_AMP=true

# ============================================================
# Tile params — DINOv3 anchor (640 + multi-scale, same hash as
# the pup_vs_nonpup gen004 tile cache so it reuses)
# ============================================================
export KCD_TILE_SIZE=640
export KCD_TILE_SOURCE_SCALES=1.0,0.5,0.25,0.125
export KCD_TILE_STRIDE_FRAC=0.5
export KCD_TILE_MIN_GT_AREA_FRAC=0.0005
export KCD_TILE_MIN_KEEP_FRACTION=0.20
export KCD_TILE_OVERSIZE_FACTOR=1.2
export KCD_TILE_KEEP_NEGATIVE=true

# ============================================================
# Backend: JPEG CocoDetection
# ============================================================
export KCD_USE_WEBDATASET=0

# ============================================================
# Class balance — single_sealion is 2-bucket {empty, sealion}
# ============================================================
# 50/50 is the symmetric upweight: maintains a strong negative
# gradient (half the stream is background) while making sealion
# positives the dominant signal in the other half. v5's natural
# corpus was ~79% empty / 21% positive and reached kit AP 0.177
# without balance; we expect balance to push that significantly
# higher by giving the matcher more positive gradient signal.
export KCD_BALANCE_TARGET_JSON='{"<empty>": 0.5, "sealion": 0.5}'
# Each sealion tile seen once per epoch; empties subsampled to
# match. See balance_mscoco.py:max_oversample help for the
# rationale.
export KCD_BALANCE_MAX_OVERSAMPLE=1

# ============================================================
# Resource budget — same as the pup resume
# ============================================================
# 2-GPU dinov3_s + AMP + balanced (no peak-scale 768 batches at
# fixed policy) → ~30 GB / GPU realistic peak. 24 GB total
# reservation per [[feedback-arisia-resource-budgets]] revised
# targets; bump via env if a profile shows pressure.
export KCD_CPUS_PER_TASK="${KCD_CPUS_PER_TASK:-4}"
export KCD_MEM="${KCD_MEM:-24G}"
export KCD_TRAIN_NUM_WORKERS="${KCD_TRAIN_NUM_WORKERS:-2}"
export KCD_VAL_NUM_WORKERS="${KCD_VAL_NUM_WORKERS:-1}"

# ============================================================
# Run identity
# ============================================================
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
