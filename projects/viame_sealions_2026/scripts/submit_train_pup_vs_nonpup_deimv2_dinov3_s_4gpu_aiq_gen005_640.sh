#!/usr/bin/env bash
# Generation 5 — pup_vs_nonpup, 640px, on the aiq-gpu Blackwell box.
#
#   scheme:   pup_vs_nonpup
#   variant:  deimv2_dinov3_s
#   gpus:     4 (aiq-gpu: 4x RTX PRO 6000 Blackwell, 96GB each)
#   res:      640 (REUSES the existing b9540ace tile cache — no rebuild)
#   launcher: slurm (aiq has slurm — always slurm) -> KCD_NO_SLURM=0
#
# PURPOSE: a shakedown + Blackwell baseline. Validates the full training
# path on sm_120 (MultiScaleDeformableAttention ops, AMP, 4-GPU DDP) and
# the new tiled-eval, on the existing 640 tiles, while the 1280 tile
# rebuild runs in parallel on CPU. Recipe stays close to arisia gen005 so
# the number is comparable; this is NOT yet the high-res experiment.
#
# Conservative on purpose: per_gpu_batch=8 (total 32, 2x arisia's 16) at
# LR 5e-4 — validates 4-GPU DDP without an aggressive LR jump. The 96GB
# cards have huge headroom; raise KCD_PER_GPU_BATCH (and scale LR) once
# this run proves the path.
#
# Submit (slurm writes the log to slurm_logs; follow with follow_job.sh <jobid>):
#   KCD_IMAGE=kwcoco-detector-kit:ogdino-cu132-aiq \
#     bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_dinov3_s_4gpu_aiq_gen005_640.sh
#
# Pre-flight on aiq-gpu:
#   * b9540ace tile cache present (set KCD_TILE_CACHE_DPATH to the SSD path,
#     or symlink it like arisia).
#   * v2 *_norm bundles + imagery present under $KCD_DATA_DPATH/unpacked.
#   * Blackwell image built: docker/opengroundingdino/build_aiq_cuda132_blackwell.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Hyperparameters (close to arisia gen005; bigger batch via 4 GPUs) --
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion
export KCD_VARIANT=deimv2_dinov3_s
export KCD_NUM_GPUS="${KCD_NUM_GPUS:-4}"
export KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-8}"     # total = 4 * 8 = 32
export KCD_VAL_BATCH_MULT="${KCD_VAL_BATCH_MULT:-1}"
export KCD_NUM_EPOCHS="${KCD_NUM_EPOCHS:-30}"
export KCD_INPUT_HW="${KCD_INPUT_HW:-[640, 640]}"
export KCD_TRAIN_POLICY=fixed
export KCD_LR="${KCD_LR:-5e-4}"
export KCD_BACKBONE_LR="${KCD_BACKBONE_LR:-2.5e-5}"
export KCD_USE_AMP=true

# ---- Backend + balance (identical composition to arisia gen005) --------
export KCD_USE_WEBDATASET=0
export KCD_BALANCE_TARGET_JSON='{"<empty>": 0.4, "pup": 0.2, "nonpup_sealion": 0.4}'
export KCD_BALANCE_MAX_OVERSAMPLE=1

# ---- Tile params: 640 defaults (existing b9540ace cache) ---------------
# Left at paths.sh canonical defaults (KCD_TILE_SIZE=640, scales 1.0,0.5).
# Don't override or it'll look for a different (nonexistent) cache.

# ---- Eval: tiled (windowed) on GPU, batched + prefetched ---------------
# The whole point of the kit work this session — closes the train/eval
# resolution gap for pups. Override to False for a plain whole-image eval.
# Train-only: aiq is a training box; eval/export/bench run on namek via the
# rescore pipeline (rescore_all.sh) so a post-train bug can't derail training.
export KCD_DO_EXPORT="${KCD_DO_EXPORT:-False}"
export KCD_DO_EVAL="${KCD_DO_EVAL:-False}"
export KCD_DO_BENCH="${KCD_DO_BENCH:-False}"
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
