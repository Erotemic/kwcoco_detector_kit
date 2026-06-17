#!/usr/bin/env bash
# Generation 6 — pup_vs_nonpup, 640px, namek (single 3090), sampler balance.
#
#   scheme:   pup_vs_nonpup
#   variant:  deimv2_dinov3_s
#   gpus:     1 (namek: RTX 3090, 24 GB)
#   res:      640 (reuses the existing tile cache — no rebuild needed)
#   launcher: standalone docker (KCD_NO_SLURM=1 — namek has no slurm)
#
# ABLATIVE QUESTION: Does dataloader-level sampler balance (per-epoch
# weighted resampling) produce equivalent training dynamics to the existing
# file-based balance (static image-row duplication)?  Everything else is
# held constant vs gen005 arisia.
#
# The 1-GPU setup confounds throughput (half the batch throughput vs gen005
# arisia 2-GPU), but does NOT confound balance QUALITY — the sampler draws
# the same class-proportion targets from the same corpus.  The comparison
# is valid for asking "do the loss curves and early-checkpoint AP look
# similar to file-mode at the same number of total gradient steps?"
#
# WHAT'S NEW vs gen005:
#   * KCD_BALANCE_MODE=sampler — BalancedSampleForest.index_weights() is
#     now implemented in kwcoco_dataloader 902a296.  The static balanced
#     MSCOCO file is replaced by per-epoch weighted sampling from the
#     UNBALANCED mscoco (no tile-count bloat, no static distribution lock).
#   * KCD_DEV_MOUNT_* — uses live host code so the image does not need to
#     be rebuilt before this test.  Rebuild the image after the run
#     validates the pipeline.
#   * KCD_NO_SLURM=1 — runs via docker run (foreground, tmux + tee).
#
# UNCHANGED from gen005 arisia:
#   * dinov3_s backbone
#   * 640px fixed tiles, paths.sh canonical tile cache
#   * balance target JSON (same class proportions)
#   * 30 epochs, LR 5e-4 / backbone 2.5e-5, AMP
#
# Pre-flight (run on namek before submitting):
#   1. Tile cache present:
#        ls $KCD_TILE_CACHE_DPATH/_universal/
#      (should show a b9540ace-prefixed dir from the gen005 build)
#   2. v2 *_norm bundles on disk:
#        ls $KCD_DATA_DPATH/unpacked/train_norm_v2.kwcoco.zip
#   3. Kit checkout up to date (this script needs gen006 code):
#        git -C ~/code/kwcoco_detector_kit pull --ff-only
#   4. Docker image present (any recent ogdino-auto build):
#        docker images | grep kwcoco-detector-kit
#      (dev mounts override the stale image code — no rebuild required)
#
# Submit (run in tmux, capture with tee):
#   cd ~/code/kwcoco_detector_kit
#   bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_dinov3_s_1gpu_namek_gen006_sampler.sh \
#     2>&1 | tee /data/users/jon.crall/kcd_sealion/pup_vs_nonpup_deimv2_dinov3_s_1gpu_namek_gen006_sampler.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters — held constant vs gen005 where possible
# ============================================================
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion
export KCD_VARIANT=deimv2_dinov3_s
export KCD_NUM_GPUS=1
export KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-8}"   # 24 GB 3090 fits 8 at 640
export KCD_VAL_BATCH_MULT=1
export KCD_NUM_EPOCHS="${KCD_NUM_EPOCHS:-30}"
export KCD_INPUT_HW='[640, 640]'
export KCD_TRAIN_POLICY=fixed
export KCD_LR=5e-4
export KCD_BACKBONE_LR=2.5e-5
export KCD_USE_AMP=true

# ============================================================
# Tile params — identical to gen005 (reuse existing cache)
# ============================================================
# paths.sh canonical defaults (KCD_TILE_SIZE=640, scales 1.0,0.5).
# Do NOT override — any change produces a different cache key and
# requires a full tile rebuild.

# ============================================================
# Backend: JPEG CocoDetection (no WebDataset — incompatible with sampler)
# ============================================================
export KCD_USE_WEBDATASET=0

# ============================================================
# Class balance — SAMPLER MODE (the ablation variable)
# ============================================================
# Same target proportions as gen005 file-mode, but drawn per epoch from
# the UNBALANCED mscoco.  TRAIN_KWCOCO stays pointed at the unbalanced
# file; the balanced_sampler CLI runs inside the container and writes
# a balance_weights.json sidecar in the run directory.
export KCD_BALANCE_MODE=sampler
export KCD_BALANCE_TARGET_JSON='{"<empty>": 0.4, "pup": 0.2, "nonpup_sealion": 0.4}'
# MAX_OVERSAMPLE applies only to file mode; sampler mode ignores it.
# Leave it set to suppress the "no balance" fast-path in _launch_train.sh.
export KCD_BALANCE_MAX_OVERSAMPLE=1

# ============================================================
# Standalone (no slurm) — namek-specific
# ============================================================
export KCD_NO_SLURM=1
# Use auto-profile image (no cu132 requirement; matches namek's driver).
export KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-auto}"
# Dev-mount host code so we get the sampler patches without rebuilding.
# After this run validates the pipeline, rebuild the image on namek.
export KCD_DEV_MOUNT_KIT=1
export KCD_DEV_MOUNT_DEIMV2=1
export KCD_DEV_MOUNT_KWCOCO_DATALOADER=1

# ============================================================
# Workers — conservative for a single-socket workstation
# ============================================================
export KCD_TRAIN_NUM_WORKERS="${KCD_TRAIN_NUM_WORKERS:-4}"
export KCD_VAL_NUM_WORKERS="${KCD_VAL_NUM_WORKERS:-2}"

# ============================================================
# Eval: tiled (same as gen005 baseline for fair comparison)
# ============================================================
export KCD_TILED_EVAL="${KCD_TILED_EVAL:-True}"
export KCD_EVAL_DEVICE="${KCD_EVAL_DEVICE:-cuda}"

# ============================================================
# Run identity
# ============================================================
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
