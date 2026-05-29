#!/usr/bin/env bash
# Resume the partially-trained lifestage_6cls v4 run.
#
# Background: job 2504 hit slurm's 72h walltime mid-epoch 29 (the final
# epoch). Best vali coco_eval_bbox = 0.0505 at epoch 23, still climbing
# at the time of the kill. last.pth contains the full training state
# (optimizer, EMA, epoch counter), so DEIMv2 can resume training from
# the exact iteration that was interrupted.
#
# After resume the run will:
#   1. finish epoch 29
#   2. run final vali eval
#   3. trigger the sweep's post-train pipeline (export, test eval, bench)
#
# Outputs land back in v4's run dir; no new dir is created.
#
# Submit (from kit root):
#   bash projects/viame_sealions_2026/scripts/submit_resume_lifestage_6cls_deimv2_hgnetv2_n_1gpu_arisia_v4.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

# Resolve the prior run's last.pth.
RESUME_FROM="$KCD_TRAINING_ROOT/runs/lifestage_6cls_deimv2_hgnetv2_n_1gpu_arisia_v4/runs/deimv2_hgnetv2_n_320x320_fixed/last.pth"
if [ ! -f "$RESUME_FROM" ]; then
    echo "ERROR: cannot find last.pth at $RESUME_FROM" >&2
    echo "  Confirm the prior run dir exists on this host's data drive." >&2
    exit 1
fi
export KCD_RESUME_CKPT="$RESUME_FROM"
# The walltimed v4 run left every stage's artifact on disk
# (best_stg2.pth, .onnx, eval/detect_metrics.json, .bench.json), so
# sweep would otherwise skip *all four* stages. Force all of them so
# the resume actually continues training AND re-runs export+eval+bench
# off the post-resume checkpoint.
export KCD_FORCE_TRAIN=true
export KCD_FORCE_EXPORT=true
export KCD_FORCE_EVAL=true
export KCD_FORCE_BENCH=true

# ============================================================
# Same hyperparameters as the killed v4 run. Keep these in lockstep
# with submit_train_lifestage_..._v4.sh — DEIMv2 requires the resume
# config to match the original config exactly.
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
# NFS is the lifestage_6cls scheme's distractor (see
# docs/class_schemes.yaml::lifestage_6cls.distractor_classes). The
# launcher auto-resolves it and sets KCD_EXCLUDE_EVAL_CLASSES; no need
# to override here. The eval step writes detect_metrics.northern_fur_seal.json
# with NFS pruned from GT+pred, and eligibility selects on that file.

export KCD_TILE_SIZE=640
export KCD_TILE_SOURCE_SCALES=1.0,0.5,0.25,0.125
export KCD_TILE_STRIDE_FRAC=0.5
export KCD_TILE_MIN_GT_AREA_FRAC=0.0005
export KCD_TILE_MIN_KEEP_FRACTION=0.20
export KCD_TILE_OVERSIZE_FACTOR=1.2
export KCD_TILE_KEEP_NEGATIVE=true

# Override KCD_RUN_NAME to point at the EXISTING v4 dir so output lands
# in the same place as the original run (resume semantics).
export KCD_RUN_NAME=lifestage_6cls_deimv2_hgnetv2_n_1gpu_arisia_v4

exec bash "$SCRIPT_DIR/_submit_train.sh"
