#!/usr/bin/env bash
# Generation 6 — pup_vs_nonpup, 1280px, dinov3_X, aiq-gpu Blackwell.
#
#   scheme:   pup_vs_nonpup
#   variant:  deimv2_dinov3_x   (50.3M params)
#   gpus:     4 (aiq-gpu: 4x RTX PRO 6000 Blackwell, 96GB each)
#   res:      1280 (uses the prebuilt 0441d89e tile cache)
#   launcher: slurm
#
# ABLATIVE QUESTION: Does stacking BOTH gen005 winners (X backbone + 1280
# resolution) unlock further pup AP gains beyond either lever alone?
#
# gen005 baseline results (tiled AP):
#   S@640  (arisia 2-GPU):  pup ~0.840, overall ~0.858
#   X@640  (aiq 4-GPU):     pup ~0.864, overall ~0.892  (+0.03 uniform)
#   S@1280 (aiq 4-GPU):     pending
#
# If the resolution and backbone gains are additive, X@1280 should exceed
# both. If not, the interaction tells us which lever dominates.
#
# MEMORY: X at 1280px is the hardest memory regime we've run. ViT attention
# at 1280px / 14px patch = 8281 tokens; X has 50M params. Starting with
# per_gpu_batch=1 (total 4) — conservative for the first run. Watch `max mem`
# over epochs 0-2 (Mosaic kicks in at epoch 2); raise KCD_PER_GPU_BATCH
# to 2 if headroom permits, and scale LR proportionally.
#
# BALANCE: file mode (same as all gen005 1280 runs) — one lever at a time.
# The sampler-vs-file ablation is covered by namek + arisia gen006.
#
# Submit (slurm writes to slurm_logs; follow with follow_job.sh <jobid>):
#   KCD_IMAGE=kwcoco-detector-kit:ogdino-cu132-aiq \
#   KCD_TILE_CACHE_DPATH=/data/users/jon.crall/kcd_sealion/ssd-data/tile_cache \
#     bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_dinov3_x_4gpu_aiq_gen006_1280.sh
#
# Pre-flight: 0441d89e (1280px) tile cache must exist on aiq.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion
export KCD_VARIANT=deimv2_dinov3_x
export KCD_NUM_GPUS="${KCD_NUM_GPUS:-4}"
# Conservative for X@1280 — raise to 2 if max-mem shows headroom after ep0.
export KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-1}"    # total = 4 * 1 = 4
export KCD_VAL_BATCH_MULT="${KCD_VAL_BATCH_MULT:-1}"
export KCD_NUM_EPOCHS="${KCD_NUM_EPOCHS:-30}"
export KCD_INPUT_HW="${KCD_INPUT_HW:-[1280, 1280]}"
export KCD_TRAIN_POLICY=fixed
# LR scaled from 5e-4 @ total-batch-32 (gen005 X@640) to total-batch-4:
# keeping the gen005 S@1280 4e-4 value (same total batch of 8 → 4 is even
# smaller; hold at 4e-4 as a floor and tune up if training is unstable).
export KCD_LR="${KCD_LR:-4e-4}"
export KCD_BACKBONE_LR="${KCD_BACKBONE_LR:-2e-5}"
export KCD_USE_AMP=true

# ============================================================
# Backend + balance — file mode (same as gen005 1280 for comparability)
# ============================================================
export KCD_USE_WEBDATASET=0
export KCD_BALANCE_TARGET_JSON='{"<empty>": 0.4, "pup": 0.2, "nonpup_sealion": 0.4}'
export KCD_BALANCE_MAX_OVERSAMPLE=1

# ============================================================
# Tile params — 1280 (prebuilt 0441d89e cache)
# ============================================================
export KCD_TILE_SIZE="${KCD_TILE_SIZE:-1280}"

# ============================================================
# Eval: tiled at 1280
# ============================================================
export KCD_TILED_EVAL="${KCD_TILED_EVAL:-True}"
export KCD_EVAL_DEVICE="${KCD_EVAL_DEVICE:-cuda}"

# ============================================================
# Slurm on aiq
# ============================================================
export KCD_NO_SLURM="${KCD_NO_SLURM:-0}"
export KCD_DOCKER_GPU_MODE="${KCD_DOCKER_GPU_MODE:-gpus}"
export KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-aiq}"
export KCD_TRAIN_NUM_WORKERS="${KCD_TRAIN_NUM_WORKERS:-8}"
export KCD_VAL_NUM_WORKERS="${KCD_VAL_NUM_WORKERS:-4}"

# ============================================================
# Run identity
# ============================================================
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
