#!/usr/bin/env bash
# Generation 6 — full_8cls, 640px, dinov3_S, arisia 2-GPU, SAMPLER balance.
#
#   scheme:   full_8cls  (juvenile,bull,female,subadult_male,pup,dead_pup,
#                         northern_fur_seal,dead_nonpup — 8 classes)
#   variant:  deimv2_dinov3_s
#   gpus:     2 (arisia)
#   res:      640 (reuses the existing b9540ace tile cache — no rebuild)
#   launcher: slurm
#
# ABLATIVE QUESTION: Does dataloader-level sampler balance (per-epoch
# weighted resampling) fix the full_8cls balance collapse that crushed gen005?
#
# The gen005 full_8cls collapse: file balance with dead_nonpup (261 tiles)
# at 5% target → natural fit = 261/0.05 = 5220 tiles → 8 steps/epoch and
# a useless checkpoint.  The gen005 fix was to EXCLUDE dead classes from the
# target JSON entirely.  Sampler mode with max_oversample=1 provides the
# same protection automatically: the cap (1/N per index) prevents any
# data-starved stratum from dominating the epoch, even if it appears in
# the target JSON.
#
# The binding class for full_8cls is NFS (8,727 tiles at 10% target):
#   natural fit = 8727 / 0.10 = 87,270 tiles/epoch
# With max_oversample=1, NFS tiles appear ~1x per epoch; dead_nonpup tiles
# appear at their natural frequency (~261/87270 ≈ 0.3% per epoch) without
# collapsing the set.
#
# WHAT'S NEW vs gen005 full_8cls (aiq dinov3_X file balance):
#   * dinov3_s (cheaper — isolates balance effect, not backbone)
#   * arisia 2-GPU (comparable to gen005 arisia pup_vs_nonpup runs)
#   * KCD_BALANCE_MODE=sampler + KCD_BALANCE_EPOCH_LENGTH=87000
#   * KCD_BALANCE_MAX_OVERSAMPLE=1 (cap prevents dead-class collapse)
#
# UNCHANGED from gen005 full_8cls:
#   * 8-class scheme, category order (comparison contract)
#   * Dead classes NOT in target JSON (still train as output classes at
#     natural frequency — same policy as gen005)
#   * 640px tile cache (b9540ace), 30 epochs, AMP
#
# Submit:
#   bash projects/viame_sealions_2026/scripts/submit_train_full_8cls_deimv2_dinov3_s_2gpu_arisia_gen006_sampler.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=full_8cls
# Order MUST equal full_8cls.target_order (the comparison contract).
export KCD_CATEGORY_NAMES=juvenile,bull,female,subadult_male,pup,dead_pup,northern_fur_seal,dead_nonpup
export KCD_DISTRACTOR_CLASSES="${KCD_DISTRACTOR_CLASSES:-northern_fur_seal}"
export KCD_VARIANT=deimv2_dinov3_s
export KCD_NUM_GPUS=2
export KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-8}"   # total = 16
export KCD_VAL_BATCH_MULT=1
export KCD_NUM_EPOCHS="${KCD_NUM_EPOCHS:-30}"
export KCD_INPUT_HW='[640, 640]'
export KCD_TRAIN_POLICY=fixed
export KCD_LR=5e-4
export KCD_BACKBONE_LR=2.5e-5
export KCD_USE_AMP=true

# ============================================================
# Backend: JPEG CocoDetection
# ============================================================
export KCD_USE_WEBDATASET=0

# ============================================================
# Class balance — SAMPLER MODE
# ============================================================
# Dead classes (dead_pup, dead_nonpup) excluded from target JSON:
# they ride at natural frequency without collapsing the epoch.
# NFS is the binding class: 8727 tiles / 0.10 target = 87,270 natural fit.
export KCD_BALANCE_MODE=sampler
export KCD_BALANCE_TARGET_JSON='{"<empty>": 0.25, "pup": 0.20, "northern_fur_seal": 0.10, "bull": 0.12, "subadult_male": 0.11, "female": 0.11, "juvenile": 0.11}'
export KCD_BALANCE_EPOCH_LENGTH=87000
export KCD_BALANCE_MAX_OVERSAMPLE=1

# ============================================================
# Tile params — canonical 640 defaults (reuse b9540ace cache)
# ============================================================

# ============================================================
# Eval: tiled
# ============================================================
export KCD_TILED_EVAL="${KCD_TILED_EVAL:-True}"
export KCD_EVAL_DEVICE="${KCD_EVAL_DEVICE:-cuda}"

# ============================================================
# Slurm on arisia
# ============================================================
export KCD_CPUS_PER_TASK="${KCD_CPUS_PER_TASK:-4}"
export KCD_MEM="${KCD_MEM:-32G}"
export KCD_TRAIN_NUM_WORKERS="${KCD_TRAIN_NUM_WORKERS:-4}"
export KCD_VAL_NUM_WORKERS="${KCD_VAL_NUM_WORKERS:-2}"

# ============================================================
# Run identity
# ============================================================
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
