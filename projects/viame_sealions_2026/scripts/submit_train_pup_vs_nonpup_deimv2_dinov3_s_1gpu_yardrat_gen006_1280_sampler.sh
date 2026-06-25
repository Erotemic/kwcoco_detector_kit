#!/usr/bin/env bash
# Generation 6 — pup_vs_nonpup, 1280px, dinov3_S, yardrat (RTX 8000, 49GB).
#
#   scheme:   pup_vs_nonpup
#   variant:  deimv2_dinov3_s
#   gpus:     1 (yardrat: GPU 0, Quadro RTX 8000, 49 GB, Turing sm_75)
#   res:      1280 (requires a tile cache build — see setup below)
#   launcher: standalone docker (KCD_NO_SLURM=1 — yardrat has no slurm)
#
# GPU NOTE: yardrat has two GPUs — RTX 8000 (49GB, GPU 0) and RTX 5000
# (16GB, GPU 1, display). We use GPU 0 only. The 49GB VRAM allows
# dinov3_s@1280px at batch=2 comfortably (gen005 S@1280 used batch=2 at
# 96GB; headroom here permits batch=4 but start conservative).
#
# ABLATIVE QUESTION: Does 1280px input resolution improve pup tiled AP
# over 640px (namek gen006 S@640 sampler)?  Same backbone, same balance
# mode — single-variable comparison.
#
#   namek:   S@640  sampler 1-GPU   (running)
#   yardrat: S@1280 sampler 1-GPU   (this run)
#
# Both machines use 1 GPU and sampler balance; the only change is tile size.
#
# BALANCE EPOCH LENGTH: The 1280px tile cache has fewer tiles than 640px
# (larger tiles cover more ground). After the tile cache is built, run:
#   python3 -c "
#   import kwcoco
#   d = kwcoco.CocoDataset('$KCD_TILE_CACHE_DPATH/_universal/<hash>/tiles.kwcoco.zip')
#   print('images:', d.n_images)
#   "
# then set KCD_BALANCE_EPOCH_LENGTH = (pup_tile_count / 0.2) rounded to the
# nearest 10k. The value below (140000) is an estimate; override if needed.
#
# Setup sequence (run once on yardrat, from ~/code/kwcoco_detector_kit):
#
#   ## yardrat
#   # 1. Sync latest kit
#   git pull --ff-only
#
#   # 2. Build Docker image (auto-detects CUDA 13.0 -> cu130 profile)
#   bash docker/opengroundingdino/build_auto.sh \
#     2>&1 | tee /data/users/jon.crall/kcd_sealion/build_image_yardrat.log
#
#   # 3. Build 1280px tile cache (CPU job, ~4-6 hours; run in tmux)
#   KCD_NO_SLURM=1 KCD_TILE_SIZE=1280 \
#     bash projects/viame_sealions_2026/scripts/submit_build_tiles.sh \
#     2>&1 | tee /data/users/jon.crall/kcd_sealion/build_tiles_1280_yardrat.log
#
#   # 4. Launch training (after tile cache completes)
#   bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_dinov3_s_1gpu_yardrat_gen006_1280_sampler.sh \
#     2>&1 | tee /data/users/jon.crall/kcd_sealion/pup_vs_nonpup_deimv2_dinov3_s_1gpu_yardrat_gen006_1280_sampler.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion
export KCD_VARIANT=deimv2_dinov3_s
export KCD_NUM_GPUS=1
# 49GB allows more headroom than gen005 S@1280 (96GB, batch=2). Start at 2;
# raise to 4 if max-mem over epoch 0-2 shows space (Mosaic climbs gradually).
export KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-2}"
export KCD_VAL_BATCH_MULT=1
export KCD_NUM_EPOCHS="${KCD_NUM_EPOCHS:-30}"
export KCD_INPUT_HW='[1280, 1280]'
export KCD_TRAIN_POLICY=fixed
# Same LR as gen005 S@1280 (total batch=8; here total batch=2, but keep 4e-4
# as a stable floor — DETR is less LR-sensitive than classification).
export KCD_LR=4e-4
export KCD_BACKBONE_LR=2e-5
export KCD_USE_AMP=true

# ============================================================
# Tile params — 1280 (must match the cache built above)
# ============================================================
export KCD_TILE_SIZE=1280
export KCD_TILE_CACHE_DPATH=/data/users/jon.crall/kcd_sealion/ssd-data/tile_cache

# ============================================================
# Backend: JPEG CocoDetection
# ============================================================
export KCD_USE_WEBDATASET=0

# ============================================================
# Class balance — SAMPLER MODE
# ============================================================
export KCD_BALANCE_MODE=sampler
export KCD_BALANCE_TARGET_JSON='{"<empty>": 0.4, "pup": 0.2, "nonpup_sealion": 0.4}'
# Estimated natural fit for 1280px (pup_tile_count / 0.2 ≈ 140k).
# Verify after tile cache build and adjust if far off.
export KCD_BALANCE_EPOCH_LENGTH="${KCD_BALANCE_EPOCH_LENGTH:-140000}"
export KCD_BALANCE_MAX_OVERSAMPLE=1

# ============================================================
# Standalone (no slurm) — yardrat-specific
# ============================================================
export KCD_NO_SLURM=1
# Auto-profile image: build_auto.sh detects CUDA 13.0 -> cu130.
export KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-auto}"
# Use only GPU 0 (RTX 8000, 49GB). GPU 1 (RTX 5000, 16GB) is for display.
export KCD_GPUS="device=0"

# ============================================================
# Workers
# ============================================================
export KCD_TRAIN_NUM_WORKERS="${KCD_TRAIN_NUM_WORKERS:-8}"
export KCD_VAL_NUM_WORKERS="${KCD_VAL_NUM_WORKERS:-4}"

# ============================================================
# Eval: tiled at 1280
# ============================================================
export KCD_TILED_EVAL="${KCD_TILED_EVAL:-True}"
export KCD_EVAL_DEVICE="${KCD_EVAL_DEVICE:-cuda}"

# ============================================================
# Run identity
# ============================================================
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
