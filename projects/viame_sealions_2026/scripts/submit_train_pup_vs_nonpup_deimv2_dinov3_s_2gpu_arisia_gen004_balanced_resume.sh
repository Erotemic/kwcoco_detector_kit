#!/usr/bin/env bash
# Resume gen004 dinov3_s + balance from checkpoint0004.pth.
#
# Job 2577 reached in-train mAP 0.161 at epoch 5 before OOMing
# at epoch 6 (45.76 GB / 47.4 GB on GPU 1) and getting walltime-
# cancelled. That run used the LEGACY oversample mode (no
# max_oversample): 229k samples/epoch -> 6 epochs in 48h.
#
# This resume changes two things; everything else matches 2577:
#
#   1. max_oversample=1 (per kit commit bfbed6b). Each pup tile
#      seen once per epoch; epoch length drops from ~7175 iters
#      to whatever the rarest bucket fits. ~10x faster epochs.
#      Stochastic augmentation handles cross-epoch diversity.
#
#   2. AMP actually enabled (per kit commit fixing the
#      argparse-default-False bug). 2577's runtime had
#      use_amp: False despite KCD_USE_AMP=true in the submit
#      script -- DEIMv2's train.py --use-amp arg with
#      action='store_true' was overriding the YAML's
#      use_amp: true. The kit now passes --use-amp to train.py
#      when the YAML says use_amp: true. With AMP on, activation
#      memory halves; the 2577 OOM goes away.
#
# Why checkpoint0004 not best_stg1:
#   * checkpoint0004 = epoch 4 (mAP 0.157), known healthy state.
#   * best_stg1 = epoch 5 (mAP 0.161) but was saved right before
#     the OOM cascade; resume from the more conservative epoch.
#   * 1.5% mAP delta is recoverable in 1-2 fresh epochs anyway.
#
# Resume vs init_checkpoint:
#   * --resume restores ALL state (model + optimizer + scheduler +
#     EMA + epoch number). Training continues from epoch 5.
#   * --init_checkpoint loads model weights only, resets optimizer
#     and starts at epoch 0.
# We want --resume so the LR schedule (FlatCosine warmup +
# flat + cosine) keeps its position.
#
# Submit:
#   bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_dinov3_s_2gpu_arisia_gen004_balanced_resume.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters - identical to 2577's submit script
# ============================================================
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion
export KCD_VARIANT=deimv2_dinov3_s
export KCD_NUM_GPUS=2
export KCD_PER_GPU_BATCH=16
export KCD_VAL_BATCH_MULT=1
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[640, 640]'
export KCD_TRAIN_POLICY=multiscale_512_768
export KCD_LR=5e-4
export KCD_BACKBONE_LR=2.5e-5
export KCD_USE_AMP=true

# ============================================================
# Tile params - same hash as 2577 (reuses the 640 multi-scale cache)
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
# Class balance - SAME target as 2577 but with max_oversample=1
# ============================================================
export KCD_BALANCE_TARGET_JSON='{"<empty>": 0.4, "pup": 0.2, "nonpup_sealion": 0.4}'
export KCD_BALANCE_MAX_OVERSAMPLE=1
# Force-rebalance to get the new max_oversample=1 output (the
# cached train_balanced.mscoco.json from 2577 used the legacy
# mode and has the wrong size).
export KCD_FORCE_REBALANCE=1

# ============================================================
# Resume state from job 2577
# ============================================================
# DEIMv2 --resume restores model + optimizer + LR scheduler + EMA
# + epoch counter. checkpoint0004 = epoch 4 (in-train mAP 0.157,
# the last clean checkpoint before the OOM cascade at epoch 6).
export KCD_RESUME_CKPT=/data/users/jon.crall/kcd_sealion/runs/pup_vs_nonpup_deimv2_dinov3_s_2gpu_arisia_gen004_balanced/runs/deimv2_dinov3_s_640x640_multiscale_512_768/checkpoint0004.pth

# ============================================================
# Dev mount: load today's kit changes without an image rebuild
# ============================================================
# The AMP fix (kit commit c43bf8f) and max_oversample (bfbed6b)
# live in kit Python, which is baked into the docker image at
# build time (pip install -e /opt/kwcoco_detector_kit). Without
# this mount, the container runs the STALE kit code from the
# image -- the --use-amp flag wouldn't be appended to train.py,
# and balance_mscoco wouldn't recognise --max_oversample.
#
# Remove this line after rebuilding the image
# (bash docker/opengroundingdino/build.sh in the kit root).

# ============================================================
# Resource budget - kept tight per memory feedback_arisia_resource_budgets
# ============================================================
# With AMP on, peak memory is ~half of 2577's: ~12 GB per GPU
# train state + 4 GB workers + 4 GB COCO eval = ~16 GB/GPU.
# 24 GB total reservation leaves headroom for the matcher peaks
# without inviting the OOM that killed 2577.
export KCD_CPUS_PER_TASK="${KCD_CPUS_PER_TASK:-4}"
export KCD_MEM="${KCD_MEM:-24G}"
export KCD_TRAIN_NUM_WORKERS="${KCD_TRAIN_NUM_WORKERS:-2}"
export KCD_VAL_NUM_WORKERS="${KCD_VAL_NUM_WORKERS:-1}"

# ============================================================
# Run identity - DIFFERENT KCD_RUN_NAME so we don't clobber 2577
# ============================================================
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
