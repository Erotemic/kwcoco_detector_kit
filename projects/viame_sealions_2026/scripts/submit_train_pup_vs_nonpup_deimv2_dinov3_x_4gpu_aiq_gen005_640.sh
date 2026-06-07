#!/usr/bin/env bash
# Generation 5 — pup_vs_nonpup, 640px, dinov3_X backbone, aiq-gpu Blackwell.
#
#   scheme:   pup_vs_nonpup
#   variant:  deimv2_dinov3_x   (50.3M params, 57.8 COCO AP)
#   gpus:     4 (aiq-gpu: 4x RTX PRO 6000 Blackwell, 96GB each)
#   res:      640 (REUSES the existing b9540ace tile cache — no rebuild)
#   launcher: standalone docker (no slurm) -> KCD_NO_SLURM=1
#
# THE LEVER: model capacity. Identical to the validated dinov3_S aiq run
# (submit_train_pup_vs_nonpup_deimv2_dinov3_s_4gpu_aiq_gen005_640.sh) except
# the backbone steps S(9.7M) -> X(50.3M). One lever at a time: tiles stay
# at 640 (the 1280 cache exists but is a separate experiment). Tiled eval
# is on (now the paths.sh default).
#
# Question: does 5x capacity move overall/pup AP above the S result
# (overall 0.858 / pup 0.838 / nonpup 0.881, tiled)? pup is already
# converged at ~0.84, so watch whether X mostly helps the harder cases.
#
# per_gpu_batch=4 (total 16, == arisia's effective batch, so LR 5e-4 holds)
# is conservative for a first run at this size — 96GB has plenty of room,
# raise KCD_PER_GPU_BATCH (and scale LR) once it proves out.
#
# Pre-flight: fetch the X checkpoint first (the host pre-flight will catch
# it if missing):
#   KCD_IMAGE=kwcoco-detector-kit:ogdino-cu132-aiq \
#     bash projects/viame_sealions_2026/scripts/fetch_pretrained.sh deimv2_dinov3_x
#
# Run it in tmux, capture with tee:
#   KCD_IMAGE=kwcoco-detector-kit:ogdino-cu132-aiq \
#   KCD_NO_SLURM=1 KCD_DEV_MOUNT_KIT=1 KCD_DEV_MOUNT_DEIMV2=1 \
#   KCD_TILE_CACHE_DPATH=/data/users/jon.crall/kcd_sealion/ssd-data/tile_cache \
#     bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_dinov3_x_4gpu_aiq_gen005_640.sh \
#     2>&1 | tee /data/users/jon.crall/kcd_sealion/aiq_pup_x_640.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Hyperparameters (== aiq dinov3_s run; only the backbone changes) ---
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion
export KCD_VARIANT=deimv2_dinov3_x
export KCD_NUM_GPUS="${KCD_NUM_GPUS:-4}"
export KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-4}"    # total = 4 * 4 = 16
export KCD_VAL_BATCH_MULT="${KCD_VAL_BATCH_MULT:-1}"
export KCD_NUM_EPOCHS="${KCD_NUM_EPOCHS:-30}"
export KCD_INPUT_HW="${KCD_INPUT_HW:-[640, 640]}"
export KCD_TRAIN_POLICY=fixed
export KCD_LR="${KCD_LR:-5e-4}"
export KCD_BACKBONE_LR="${KCD_BACKBONE_LR:-2.5e-5}"
export KCD_USE_AMP=true

# ---- Backend + balance (identical to gen005) ---------------------------
export KCD_USE_WEBDATASET=0
export KCD_BALANCE_TARGET_JSON='{"<empty>": 0.4, "pup": 0.2, "nonpup_sealion": 0.4}'
export KCD_BALANCE_MAX_OVERSAMPLE=1

# ---- Tile params: 640 defaults (existing b9540ace cache) ---------------
# Left at paths.sh canonical defaults (KCD_TILE_SIZE=640, scales 1.0,0.5).

# ---- Eval: tiled (now the paths.sh default; explicit here for clarity) --
export KCD_TILED_EVAL="${KCD_TILED_EVAL:-True}"
export KCD_EVAL_DEVICE="${KCD_EVAL_DEVICE:-cuda}"

# ---- Standalone docker on a dedicated box ------------------------------
export KCD_NO_SLURM="${KCD_NO_SLURM:-1}"
export KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-aiq}"
export KCD_TRAIN_NUM_WORKERS="${KCD_TRAIN_NUM_WORKERS:-8}"
export KCD_VAL_NUM_WORKERS="${KCD_VAL_NUM_WORKERS:-4}"

# ---- Run identity ------------------------------------------------------
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
