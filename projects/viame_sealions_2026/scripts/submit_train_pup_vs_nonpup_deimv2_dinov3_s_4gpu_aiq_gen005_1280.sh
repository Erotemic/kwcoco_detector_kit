#!/usr/bin/env bash
# Generation 5 — pup_vs_nonpup, 1280px, dinov3_S backbone, aiq-gpu Blackwell.
#
#   scheme:   pup_vs_nonpup
#   variant:  deimv2_dinov3_s
#   gpus:     4 (aiq-gpu: 4x RTX PRO 6000 Blackwell, 96GB each)
#   res:      1280 (uses the prebuilt 0441d89e tile cache — no rebuild)
#   launcher: slurm (aiq has slurm — always slurm) -> KCD_NO_SLURM=0
#
# THE LEVER: input resolution. Identical recipe to the validated dinov3_S
# aiq 640 run (submit_train_pup_vs_nonpup_deimv2_dinov3_s_4gpu_aiq_gen005_640.sh)
# except tile_size and input_hw step 640 -> 1280. One lever at a time: the
# backbone stays S (memory-safe; X is OOM-fragile and would confound), only
# the resolution the model SEES changes.
#
# WHY: capacity is spent — dinov3_X (50M) gave only a uniform ~+0.03 over S,
# not a pup unlock. The remaining identified lever for small objects (pups)
# is resolution (the small-object floor is a train/eval resolution effect).
# Question: does 1280 lift pup tiled AP above the S@640 result (pup 0.840,
# overall 0.858)? If yes, the follow-up is X@1280 (stack both winners).
#
# MEMORY NOTE: 1280 is 4x the pixels of 640, and ViT self-attention scales
# ~16x in token count, so this is a genuinely new memory regime. Starts
# CONSERVATIVE: per_gpu_batch=2 (total 8) at LR 4e-4. Watch `max mem` over
# the first epochs (it climbs as Mosaic kicks in); if there's headroom, raise
# KCD_PER_GPU_BATCH (and scale LR ~linearly). If it OOMs, you're already at
# the floor — drop tile scales or input_hw, not batch.
#
# Run it in tmux, capture with tee (NOT nohup):
#   KCD_IMAGE=kwcoco-detector-kit:ogdino-cu132-aiq \
#   (slurm; gres/partition from your shell rc) \
#   KCD_TILE_CACHE_DPATH=/data/users/jon.crall/kcd_sealion/tile_cache \
#     bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_dinov3_s_4gpu_aiq_gen005_1280.sh \
#     2>&1 | tee /data/users/jon.crall/kcd_sealion/aiq_pup_s_1280.log
#
# Pre-flight: the 0441d89e (1280) tile cache must be present. The train
# script computes the tile hash from KCD_TILE_SIZE below and fails fast if
# the cache is missing — so a wrong tile param can't silently train on the
# wrong data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Hyperparameters (== aiq dinov3_s 640 run; only resolution changes) -
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion
export KCD_VARIANT=deimv2_dinov3_s
export KCD_NUM_GPUS="${KCD_NUM_GPUS:-4}"
# Conservative for the new 1280 regime; raise if `max mem` shows headroom.
export KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-2}"     # total = 4 * 2 = 8
export KCD_VAL_BATCH_MULT="${KCD_VAL_BATCH_MULT:-1}"
export KCD_NUM_EPOCHS="${KCD_NUM_EPOCHS:-30}"
export KCD_INPUT_HW="${KCD_INPUT_HW:-[1280, 1280]}"     # the resolution lever
export KCD_TRAIN_POLICY=fixed
export KCD_LR="${KCD_LR:-4e-4}"                          # total-batch-8 scaled
export KCD_BACKBONE_LR="${KCD_BACKBONE_LR:-2e-5}"
export KCD_USE_AMP=true

# ---- Backend + balance (identical composition to the 640 run) ----------
export KCD_USE_WEBDATASET=0
export KCD_BALANCE_TARGET_JSON='{"<empty>": 0.4, "pup": 0.2, "nonpup_sealion": 0.4}'
export KCD_BALANCE_MAX_OVERSAMPLE=1

# ---- Tile params: 1280 (prebuilt 0441d89e cache) -----------------------
# Only tile_size changes vs the canonical 640 defaults (scales 1.0,0.5 etc).
# This must hash to 0441d89e; the train pre-flight verifies the cache exists.
export KCD_TILE_SIZE="${KCD_TILE_SIZE:-1280}"

# ---- Eval: tiled (windowed) at the trained 1280 window -----------------
export KCD_TILED_EVAL="${KCD_TILED_EVAL:-True}"
export KCD_EVAL_DEVICE="${KCD_EVAL_DEVICE:-cuda}"

# ---- Slurm on aiq (aiq has slurm — always slurm; gres/partition from rc) -
export KCD_NO_SLURM="${KCD_NO_SLURM:-0}"
export KCD_DOCKER_GPU_MODE="${KCD_DOCKER_GPU_MODE:-gpus}"
export KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-aiq}"
export KCD_TRAIN_NUM_WORKERS="${KCD_TRAIN_NUM_WORKERS:-8}"
export KCD_VAL_NUM_WORKERS="${KCD_VAL_NUM_WORKERS:-4}"

# ---- Run identity ------------------------------------------------------
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
