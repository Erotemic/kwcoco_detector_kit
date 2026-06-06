#!/usr/bin/env bash
# Generation 5 — pup_vs_nonpup batch-16 ablation on the v2 corpus.
#
# Sibling to submit_train_pup_vs_nonpup_..._gen005_v2splits.sh, same
# everything EXCEPT per_gpu_batch doubled (8 -> 16). The gen004 OOM
# that forced batch=8 was at multiscale_512_768; at fixed 640 we
# have ~7-10 GB of unused VRAM headroom on A6000 per the gen004
# forensic journal. Worth a clean test on the v2 corpus.
#
# What we're testing:
#   * Whether fixed 640 (no multiscale) actually leaves the headroom
#     the forensic math implied.
#   * Whether 2x effective batch (32 vs 16) materially changes the
#     gen005 convergence curve - bigger batch usually means cleaner
#     gradients per step and faster wallclock per epoch, but small-
#     object detection (pup) sometimes benefits from smaller batches
#     because the matcher gets less averaged.
#
# Compared to the baseline pup_vs_nonpup gen005:
#   * KCD_PER_GPU_BATCH  8 -> 16 (total batch 16 -> 32)
#   * KCD_VAL_BATCH_MULT 1 -> 1 (val batch same as train)
# Everything else (LR, balance, epochs, tile params) unchanged.
#
# If this OOMs at epoch N, the gen004 journal's "27 GB peak at 640
# fixed" estimate was wrong. Cancel and we know per_gpu_batch=8 is
# the ceiling at fixed 640 + dinov3_s.
#
# Submit:
#   bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_dinov3_s_2gpu_arisia_gen005_v2splits_batch16.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Hyperparameters
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion
export KCD_VARIANT=deimv2_dinov3_s
export KCD_NUM_GPUS=2
export KCD_PER_GPU_BATCH=16          # << doubled from baseline; total = 32
export KCD_VAL_BATCH_MULT=1
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[640, 640]'
export KCD_TRAIN_POLICY=fixed
export KCD_LR=5e-4
export KCD_BACKBONE_LR=2.5e-5
export KCD_USE_AMP=true

# Tile params at canonical defaults (shared cache w/ baseline).

# JPEG backend (matches baseline)
export KCD_USE_WEBDATASET=0

# Class balance (identical to baseline)
export KCD_BALANCE_TARGET_JSON='{"<empty>": 0.4, "pup": 0.2, "nonpup_sealion": 0.4}'
export KCD_BALANCE_MAX_OVERSAMPLE=1

# Slurm resources - bump memory headroom for the 2x batch since
# dataloader workers' per-image buffers stack.
export KCD_CPUS_PER_TASK="${KCD_CPUS_PER_TASK:-4}"
export KCD_MEM="${KCD_MEM:-40G}"   # baseline = 32G
export KCD_TRAIN_NUM_WORKERS="${KCD_TRAIN_NUM_WORKERS:-2}"
export KCD_VAL_NUM_WORKERS="${KCD_VAL_NUM_WORKERS:-1}"

RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
