#!/usr/bin/env bash
# sealion_aerial end-to-end driver.
#
# Assumes you've already run prepare_kwcoco.py to materialise:
#   $KCD_ROOT/sealion/raw.kwcoco.zip
#
# Then this script:
#   1. Tiles the raw bundle (multi-scale, oversize=1.2 for crop-aug margin)
#   2. Trains deimv2_dinov3_s (Phase 2: real GPU required)
#   3. Exports + evals + benches
#   4. Aggregates into an eligibility manifest
#
# Env knobs:
#   KCD_ROOT          workspace (default ~/data/kcd_sealion)
#   KCD_NUM_GPUS      DDP world size (default 1)
#   KCD_DISTRIBUTED   1 to enable torch.distributed.run (default 0)
#   KCD_TIER          override auto-tier (S/M/L/XL/cluster)
#   PYTHON_BIN        python interpreter (default python)
#   KCD_DEIMV2_REPO_DPATH   required for real DEIMv2 training
set -euo pipefail

KCD_ROOT="${KCD_ROOT:-$HOME/data/kcd_sealion}"
PYTHON_BIN="${PYTHON_BIN:-python}"
KCD_NUM_GPUS="${KCD_NUM_GPUS:-1}"
KCD_DISTRIBUTED="${KCD_DISTRIBUTED:-0}"
KCD_TIER="${KCD_TIER:-L}"
KCD_CATEGORY="${KCD_CATEGORY:-sealion}"

mkdir -p "$KCD_ROOT"
RAW="$KCD_ROOT/sealion/raw.kwcoco.zip"
TILES="$KCD_ROOT/sealion/tiles.kwcoco.zip"

if [ ! -f "$RAW" ]; then
    echo "Missing $RAW — run prepare_kwcoco.py first." >&2
    exit 1
fi

echo
echo "=== 1. Multi-scale tile ==="
"$PYTHON_BIN" -m kwcoco_detector_kit tile \
    "$RAW" "$TILES" \
    --mode multiscale \
    --tile_size 640 \
    --source_scales "1.0,0.5,0.25,0.125" \
    --stride_frac 0.5 \
    --min_gt_area_frac 0.0005 \
    --min_keep_fraction 0.20 \
    --oversize_factor 1.2 \
    --keep_negative true \
    --category_name "$KCD_CATEGORY"

echo
echo "=== 2. Sweep (train + export + eval + bench) ==="
DIST_FLAG=()
if [ "$KCD_DISTRIBUTED" = "1" ]; then
    DIST_FLAG+=("--num_gpus" "$KCD_NUM_GPUS")
fi
"$PYTHON_BIN" -m kwcoco_detector_kit sweep \
    --train_kwcoco "$TILES" \
    --vali_kwcoco  "$TILES" \
    --test_kwcoco  "$RAW" \
    --kcd_root "$KCD_ROOT" \
    --trainer deimv2 \
    --variant deimv2_dinov3_s \
    --input_hw 640,640 \
    --train_policy multiscale_512_768 \
    --num_epochs 30 \
    --batch_size 16 \
    --val_batch_size 32 \
    --num_classes 1 \
    --category_name "$KCD_CATEGORY" \
    --lr 5e-4 \
    --backbone_lr 2.5e-5 \
    --use_amp true \
    --scale_tier "$KCD_TIER" \
    "${DIST_FLAG[@]}"

echo
echo "=== 3. Eligibility manifest ==="
"$PYTHON_BIN" -m kwcoco_detector_kit manifest \
    --auto \
    --kcd_root "$KCD_ROOT" \
    --out      "$KCD_ROOT/manifest.tsv" \
    --out_json "$KCD_ROOT/manifest.json" \
    --max_desktop_ms 250 \
    --allow_missing_desktop_bench true

echo
echo "=== sealion_aerial complete ==="
echo "  manifest: $KCD_ROOT/manifest.tsv"
