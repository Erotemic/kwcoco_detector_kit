#!/usr/bin/env bash
# Generation 4 — class-balanced JPEG backend, 2-GPU + DINOv3-S backbone.
#
#   scheme:   pup_vs_nonpup (P1, binding constraint)
#   variant:  deimv2_dinov3_s  (foundation backbone, 9.7M params,
#                               50.9 COCO AP — DEIMv2's S tier)
#   gpus:     2 (arisia)
#   gen:      004
#
# Bigger-leap companion to the 1-GPU ablation
# (submit_train_pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_gen004_balanced.sh).
# Same class-balance target so the comparison isolates "bigger
# model + 2-GPU" vs the 1-GPU baseline.
#
# Why DINOv3-S instead of an HGNetv2 step-up: DEIMv2's public
# model zoo's S/M/L/X tiers all use the DINOv3 foundation backbone;
# there's no HGNetv2-S COCO checkpoint upstream. DINOv3-S is the
# correct "next tier" with a real pretrained init.
#
# Compared to the 1-GPU ablation:
#   1. KCD_VARIANT    deimv2_hgnetv2_n -> deimv2_dinov3_s
#                     (3.6M -> 9.7M params; foundation backbone)
#   2. KCD_NUM_GPUS   1 -> 2 (effective batch 32 vs 16)
#   3. KCD_INPUT_HW   320x320 -> 640x640 (DINOv3-S anchor)
#   4. Tile params    320 / scale=1.0 -> 640 / scale=1.0,0.5,0.25,0.125
#                     (DIFFERENT tile cache; one-time re-tile cost)
#   5. KCD_LR         5.66e-4 -> 5e-4 (matches dinov3_s v1 recipe)
#   6. KCD_BACKBONE_LR 2.83e-4 -> 2.5e-5 (DINOv3 backbone freezing
#                     pattern; backbone LR much lower than head LR)
#
# LR + tile params are copied from the v1 dinov3_s 4-GPU recipe
# (submit_train_pup_vs_nonpup_deimv2_dinov3_s_4gpu_arisia_v1.sh)
# since we know those hyperparams converge.
#
# Tile cache prep: this needs the 640-tile multiscale cache built
# (DIFFERENT hash from the 320 cache the 1-GPU run uses). Either
# the kit will tile on first run (~hours), or pre-warm with:
#   KCD_DATA_PREP_ONLY=1 bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_dinov3_s_2gpu_arisia_gen004_balanced.sh
# (the prep-only invocation tiles + exits without slurm). Once the
# cache exists it's reusable for every future dinov3 run.
#
# Submit:
#   bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_dinov3_s_2gpu_arisia_gen004_balanced.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion
export KCD_VARIANT=deimv2_dinov3_s
export KCD_NUM_GPUS=2
export KCD_PER_GPU_BATCH=16          # total batch = 32
export KCD_VAL_BATCH_MULT=1
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[640, 640]'
export KCD_TRAIN_POLICY=multiscale_512_768
export KCD_LR=5e-4
export KCD_BACKBONE_LR=2.5e-5
export KCD_USE_AMP=true

# ============================================================
# Tile params — DINOv3 anchor (640 + multi-scale)
# ============================================================
# DIFFERENT hash from gen004 1-GPU's 320/1.0 cache. First run will
# tile to build this cache; reuses on subsequent runs.
export KCD_TILE_SIZE=640
export KCD_TILE_SOURCE_SCALES=1.0,0.5,0.25,0.125
export KCD_TILE_STRIDE_FRAC=0.5
export KCD_TILE_MIN_GT_AREA_FRAC=0.0005
export KCD_TILE_MIN_KEEP_FRACTION=0.20
export KCD_TILE_OVERSIZE_FACTOR=1.2
export KCD_TILE_KEEP_NEGATIVE=true

# ============================================================
# Backend: JPEG CocoDetection (matches the 1-GPU ablation)
# ============================================================
export KCD_USE_WEBDATASET=0

# ============================================================
# Class balance — IDENTICAL to the 1-GPU ablation
# ============================================================
# Keeping the same target so the n-vs-s and 1-vs-2-GPU comparison
# isolates capacity + batch + resolution from data composition.
export KCD_BALANCE_TARGET_JSON='{"<empty>": 0.4, "pup": 0.2, "nonpup_sealion": 0.4}'
# Cap pup repetition at 1× per epoch (see Run 1 comment for
# rationale). Matches the ablation so the comparison between
# Run 1 and Run 2 isolates capacity / batch / resolution from
# data composition.
export KCD_BALANCE_MAX_OVERSAMPLE=1

# ============================================================
# Slurm resource budget (performance-only — env-overridable)
# ============================================================
# Right-sized for dinov3_s + 640x640 + 2-GPU JPEG usage:
#   Model + opt + AMP grads (9.7M params, 2 ranks)  ~ 4 GB
#   4 dataloader workers (640x640 = 4x decode mem)  ~12 GB
#   Main process + COCO eval on rank 0              ~ 4 GB
#   DDP overhead + headroom                         ~ 4 GB
#   Peak realistic                                  ~24 GB
# 32 GB total for 2 GPUs = 16 GB/GPU; half the kit default and
# still ~8 GB headroom. Bump via env if profile shows pressure.
#
# arisia is shared; minimizing the reservation lets co-scheduled
# jobs run instead of queuing on (Resources).
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
