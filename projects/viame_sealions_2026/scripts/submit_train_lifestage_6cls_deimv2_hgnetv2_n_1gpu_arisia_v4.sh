#!/usr/bin/env bash
# Lightweight 1-GPU baseline (mobile-class HGNetv2-N, COCO-init).
#
#   scheme:   lifestage_6cls (P2 — full age-sex classifier + NFS distractor)
#   variant:  deimv2_hgnetv2_n
#   gpus:     1 (arisia)
#   version:  v4
#
# Sibling of submit_train_pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_v6.sh;
# only scheme + category_names differ. 6 classes triple the matcher's
# class-dim cost vs the 2-class pup scheme, so batch must be lower
# than pup_v6's 16 to fit the same memory envelope.
#
# Changes vs v3:
#   - batch=32 -> 12. v3 OOM'd at epoch 0 iter ~4000 trying to
#     allocate 6.84 GiB; that's matcher cost matrix for a high-
#     density tile across 6 classes. batch=12 gives ~75% headroom
#     to absorb the worst-case multi-class crowd tile + the Mosaic
#     cliff at epoch 4.
#   - LR scaled to sqrt(12/32)*v1's 8e-4.
#
# Note: subadult_male is the weakest class (3,202 training instances);
# watch its per-class AP. NFS provides species-boundary signal.
#
# Submit (from kit root):
#   bash projects/viame_sealions_2026/scripts/submit_train_lifestage_6cls_deimv2_hgnetv2_n_1gpu_arisia_v4.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_SCHEME=lifestage_6cls
export KCD_CATEGORY_NAMES=bull,subadult_male,female,juvenile,pup,northern_fur_seal
export KCD_VARIANT=deimv2_hgnetv2_n
export KCD_NUM_GPUS=1
export KCD_PER_GPU_BATCH=12
export KCD_VAL_BATCH_MULT=1
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[320, 320]'
export KCD_TRAIN_POLICY=fixed
export KCD_LR=4.9e-4
export KCD_BACKBONE_LR=2.45e-4
export KCD_USE_AMP=true

# Tile params — shared with the other baselines.
export KCD_TILE_SIZE=640
export KCD_TILE_SOURCE_SCALES=1.0,0.5,0.25,0.125
export KCD_TILE_STRIDE_FRAC=0.5
export KCD_TILE_MIN_GT_AREA_FRAC=0.0005
export KCD_TILE_MIN_KEEP_FRACTION=0.20
export KCD_TILE_OVERSIZE_FACTOR=1.2
export KCD_TILE_KEEP_NEGATIVE=true

# ============================================================
# Run identity
# ============================================================
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
