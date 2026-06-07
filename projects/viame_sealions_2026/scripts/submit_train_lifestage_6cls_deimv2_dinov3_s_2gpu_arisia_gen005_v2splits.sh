#!/usr/bin/env bash
# Generation 5 — lifestage_6cls on the v2 corpus splits.
#
#   scheme:   lifestage_6cls (6-class age/sex + northern_fur_seal distractor)
#   variant:  deimv2_dinov3_s
#   gpus:     2 (arisia)
#   gen:      005
#
# Completes the gen005 scheme set on the corrected v2 corpus
# (single_sealion + pup_vs_nonpup already on v2; this is the third,
# full operational scheme). Same proven dinov3_s / 640 / balanced /
# tiled-eval recipe as gen005 pup_vs_nonpup — the only changes are the
# scheme, its 6-class target order, and the NFS distractor.
#
# Classes (target_order == output class-index order; see
# docs/class_schemes.yaml):
#   0 bull  1 subadult_male  2 female  3 juvenile  4 pup  5 northern_fur_seal
# northern_fur_seal is a DISTRACTOR: the model learns to detect it (to
# discriminate it) but it's excluded from the class-agnostic detection AP
# (KCD_DISTRACTOR_CLASSES -> the eval writes a detect_metrics.northern_fur_seal.json
# sidecar that eligibility selects on). Per-class NFS AP is still reported.
#
# Balance: train-split image coverage is bull 4197 / subadult_male 3470 /
# female 2997 / juvenile 3911 / pup 1965 / NFS 118 (of 5997 imgs). Boost
# the rare-but-critical pup, give NFS a floor so the distractor is
# learnable, keep the rest near-uniform. max_oversample=1 caps repetition
# (so NFS can't dominate off its 118 images). Tune if a class lags.
#
# Submit:
#   bash projects/viame_sealions_2026/scripts/submit_train_lifestage_6cls_deimv2_dinov3_s_2gpu_arisia_gen005_v2splits.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Scheme + classes
# ============================================================
export KCD_SCHEME=lifestage_6cls
export KCD_CATEGORY_NAMES=bull,subadult_male,female,juvenile,pup,northern_fur_seal
export KCD_DISTRACTOR_CLASSES=northern_fur_seal

# ============================================================
# Hyperparameters (identical to gen005 pup_vs_nonpup)
# ============================================================
export KCD_VARIANT=deimv2_dinov3_s
export KCD_NUM_GPUS=2
export KCD_PER_GPU_BATCH=8            # total batch = 16
export KCD_VAL_BATCH_MULT=1
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[640, 640]'
export KCD_TRAIN_POLICY=fixed
export KCD_LR=5e-4
export KCD_BACKBONE_LR=2.5e-5
export KCD_USE_AMP=true

# ============================================================
# Tile params — paths.sh canonical defaults (shared b9540ace cache)
# ============================================================
# Same 640 multiscale cache as the other gen005 dinov3_s runs.

# ============================================================
# Backend + class balance (6-class)
# ============================================================
export KCD_USE_WEBDATASET=0
export KCD_BALANCE_TARGET_JSON='{"<empty>": 0.25, "pup": 0.20, "northern_fur_seal": 0.10, "bull": 0.12, "subadult_male": 0.11, "female": 0.11, "juvenile": 0.11}'
export KCD_BALANCE_MAX_OVERSAMPLE=1

# ============================================================
# Slurm resource budget (same as gen005 pup)
# ============================================================
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
