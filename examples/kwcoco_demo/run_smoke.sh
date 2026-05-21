#!/usr/bin/env bash
# kwcoco_demo — end-to-end CPU smoke for kwcoco-detector-kit.
#
# Exercises the full pipeline: synth kwcoco → multiscale tile → train
# mock_tiny → ONNX export → kwcoco eval → ONNX bench → eligibility
# manifest. Target: <90 s on a 1-CPU laptop.
#
# Usage:
#   bash examples/kwcoco_demo/run_smoke.sh
#
# Env knobs:
#   KCD_ROOT          workspace root (default: /tmp/kcd_demo_smoke)
#   PYTHON_BIN        Python interpreter (default: python)
#   KCD_CATEGORY      category name (default: widget)
set -euo pipefail

KCD_ROOT="${KCD_ROOT:-/tmp/kcd_demo_smoke}"
PYTHON_BIN="${PYTHON_BIN:-python}"
KCD_CATEGORY="${KCD_CATEGORY:-widget}"

rm -rf "$KCD_ROOT"
mkdir -p "$KCD_ROOT"

echo
echo "=== 1. Synth kwcoco bundle ==="
"$PYTHON_BIN" -m kwcoco_detector_kit demo-data \
    "$KCD_ROOT/raw.kwcoco.zip" \
    --num_images 16 \
    --num_categories 1 \
    --image_size 256,256 \
    --category_names "$KCD_CATEGORY"

echo
echo "=== 2. Tile (multiscale) ==="
"$PYTHON_BIN" -m kwcoco_detector_kit tile \
    "$KCD_ROOT/raw.kwcoco.zip" \
    "$KCD_ROOT/tiles.kwcoco.zip" \
    --mode multiscale \
    --tile_size 256 \
    --source_scales "1.0,0.66" \
    --stride_frac 0.5 \
    --min_gt_area_frac 0.001 \
    --min_source_scale_long_side 64 \
    --min_keep_fraction 0.20 \
    --oversize_factor 1.0 \
    --category_names "$KCD_CATEGORY"

echo
echo "=== 3. Train + export + eval + bench + manifest ==="
"$PYTHON_BIN" -m kwcoco_detector_kit run-all \
    --train_kwcoco "$KCD_ROOT/tiles.kwcoco.zip" \
    --vali_kwcoco  "$KCD_ROOT/tiles.kwcoco.zip" \
    --test_kwcoco  "$KCD_ROOT/raw.kwcoco.zip" \
    --workdir      "$KCD_ROOT" \
    --category_names "$KCD_CATEGORY" \
    --trainer mock_tiny \
    --variant mock_tiny \
    --tier S \
    --input_hw 256,256 \
    --num_epochs 2 \
    --batch_size 2

echo
echo "=== smoke complete ==="
echo "  manifest: $KCD_ROOT/manifest.tsv"
echo "  manifest: $KCD_ROOT/manifest.json"
