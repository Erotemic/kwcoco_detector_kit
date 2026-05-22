#!/usr/bin/env bash
# Generic single-GPU baseline launcher. Train any (variant, scheme)
# combo to establish a lightweight baseline on the same data the big
# multi-GPU runs use.
#
# Driven by env vars (all have defaults in scripts/paths.sh-style):
#
#   KCD_SCHEME            scheme name from docs/class_schemes.yaml
#                         (default: pup_vs_nonpup)
#   KCD_VARIANT           DEIMv2 variant; the kit's _BatchRow table
#                         knows the supported set (default: deimv2_hgnetv2_n)
#   KCD_INIT_CHECKPOINT   absolute path to the .pth COCO init weights.
#                         Auto-resolved from KCD_VARIANT when unset.
#   KCD_INPUT_HW          "H,W" — input resolution. Auto-resolved when
#                         unset: 320,320 for hgnetv2_*, 640,640 for dinov3_*.
#   KCD_TRAIN_POLICY      fixed | multiscale | multiscale_<lo>_<hi>
#                         (default: fixed — hgnetv2 doesn't support dynamic)
#   KCD_CATEGORY_NAMES    comma-separated. Auto-resolved from scheme.
#   KCD_NUM_EPOCHS        default 30 (apples-to-apples vs the dinov3_s run)
#   KCD_PER_GPU_BATCH     per-GPU batch (default: variant native — 32
#                         for hgnetv2_n, 8 for dinov3_s)
#   KCD_TRAIN_FROM_SCRATCH=1   skip init_checkpoint
#
# Outputs land under $KCD_TRAINING_ROOT/baseline_<variant>_<scheme>/.
# Tile bundle is shared per-scheme at the existing big-run kcd_root
# when present (e.g., $KCD_TRAINING_ROOT/<scheme>/tiles.kwcoco.zip);
# otherwise a fresh tile is created inside the baseline workspace.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
KCD_SCHEME="${KCD_SCHEME:-pup_vs_nonpup}"
KCD_VARIANT="${KCD_VARIANT:-deimv2_hgnetv2_n}"
NUM_EPOCHS="${KCD_NUM_EPOCHS:-${NUM_EPOCHS:-30}}"

# Variant -> default init checkpoint + input HW + per-GPU batch.
case "$KCD_VARIANT" in
    deimv2_hgnetv2_n)
        : "${KCD_INIT_CHECKPOINT:=$KCD_DEIMV2_HGNETV2_N_COCO_PTH}"
        : "${KCD_INPUT_HW:=320,320}"
        : "${KCD_PER_GPU_BATCH:=32}"
        : "${KCD_TRAIN_POLICY:=fixed}"
        ;;
    deimv2_dinov3_s)
        : "${KCD_INIT_CHECKPOINT:=$KCD_DEIMV2_DINOV3_S_COCO_PTH}"
        : "${KCD_INPUT_HW:=640,640}"
        : "${KCD_PER_GPU_BATCH:=8}"
        : "${KCD_TRAIN_POLICY:=multiscale_512_768}"
        ;;
    *)
        echo "ERROR: KCD_VARIANT=$KCD_VARIANT not recognized by launch_baseline.sh" >&2
        echo "       Add a case branch with init checkpoint + input_hw + batch defaults" >&2
        echo "       (or set KCD_INIT_CHECKPOINT / KCD_INPUT_HW / KCD_PER_GPU_BATCH explicitly)" >&2
        exit 1
        ;;
esac

# Scheme -> category_names + scheme dir + kwcoco bundles.
SCHEME_DIR="$KCD_SCHEMES_DIR/$KCD_SCHEME"
TRAIN_KWCOCO="$SCHEME_DIR/train.kwcoco.zip"
VALI_KWCOCO="$SCHEME_DIR/vali.kwcoco.zip"
TEST_KWCOCO="$SCHEME_DIR/test.kwcoco.zip"
kcd_require_path "$KCD_SCHEME train.kwcoco.zip" "$TRAIN_KWCOCO"
kcd_require_path "$KCD_SCHEME vali.kwcoco.zip" "$VALI_KWCOCO"
kcd_require_path "$KCD_SCHEME test.kwcoco.zip" "$TEST_KWCOCO"

# Derive category_names from the scheme report if not set explicitly.
if [ -z "${KCD_CATEGORY_NAMES:-}" ]; then
    KCD_CATEGORY_NAMES="$("$PYTHON_BIN" -c "
import json, sys, pathlib
fp = pathlib.Path('$SCHEME_DIR/scheme_report.json')
data = json.loads(fp.read_text())
names = data.get('target_order') or list(data.get('per_target_class', {}).keys())
print(','.join(names))
")"
fi
[ -z "$KCD_CATEGORY_NAMES" ] && {
    echo "ERROR: could not resolve category_names for scheme=$KCD_SCHEME" >&2
    exit 1
}

# Per-experiment kcd_root.
KCD_ROOT="${KCD_ROOT_BASELINE:-$KCD_TRAINING_ROOT/baseline_${KCD_VARIANT}_${KCD_SCHEME}}"

# Init checkpoint sanity (unless training from scratch).
if [ "${KCD_TRAIN_FROM_SCRATCH:-0}" = "1" ]; then
    echo "WARNING: KCD_TRAIN_FROM_SCRATCH=1 — skipping init_checkpoint" >&2
    INIT_FLAG=()
    INIT_CKPT_DISPLAY="<from-scratch>"
else
    kcd_require_path "$KCD_VARIANT COCO pretrained checkpoint" "$KCD_INIT_CHECKPOINT" || {
        echo "  Run: bash projects/viame_sealions_2026/scripts/fetch_pretrained.sh $KCD_VARIANT" >&2
        exit 1
    }
    INIT_FLAG=(--init_checkpoint "$KCD_INIT_CHECKPOINT")
    INIT_CKPT_DISPLAY="$KCD_INIT_CHECKPOINT"
fi

mkdir -p "$KCD_ROOT"
# nccl_traces/ for flight recorder dumps. Single-GPU runs don't use
# NCCL collectives for gradient sync, but the env vars are still set
# by sbatch — keep the dir so torch's writer doesn't error.
mkdir -p "$KCD_ROOT/nccl_traces"

# Shared per-scheme tile cache. The pup_vs_nonpup big-run kcd_root
# already has a tile bundle at the canonical path; reuse it if present
# instead of retiling. Falls back to the baseline workspace otherwise.
SHARED_TILES="$KCD_TRAINING_ROOT/$KCD_SCHEME/tiles.kwcoco.zip"
LOCAL_TILES="$KCD_ROOT/tiles.kwcoco.zip"
if [ -f "$SHARED_TILES" ] && [ "${KCD_FORCE_RETILE:-0}" != "1" ]; then
    TILES="$SHARED_TILES"
else
    TILES="$LOCAL_TILES"
fi

# Disk guard.
KCD_MIN_FREE_GB="${KCD_MIN_FREE_GB:-30}"
free_kb=$(df -k --output=avail "$KCD_TRAINING_ROOT" 2>/dev/null | tail -n1 | tr -d ' ')
if [ -n "$free_kb" ]; then
    free_gb=$(( free_kb / 1024 / 1024 ))
    echo "  free disk:   ${free_gb} GB at $KCD_TRAINING_ROOT (need >= ${KCD_MIN_FREE_GB})"
    if [ "$free_gb" -lt "$KCD_MIN_FREE_GB" ]; then
        echo "ERROR: ${free_gb} GB free; need >= ${KCD_MIN_FREE_GB}" >&2
        exit 1
    fi
fi

echo
echo "=== baseline config ==="
echo "  scheme:       $KCD_SCHEME"
echo "  variant:      $KCD_VARIANT"
echo "  categories:   $KCD_CATEGORY_NAMES"
echo "  input_hw:     $KCD_INPUT_HW"
echo "  train_policy: $KCD_TRAIN_POLICY"
echo "  init_ckpt:    $INIT_CKPT_DISPLAY"
echo "  kcd_root:     $KCD_ROOT"
echo "  tiles:        $TILES $([ "$TILES" = "$SHARED_TILES" ] && echo "(shared with $KCD_SCHEME big-run)")"
echo "  epochs:       $NUM_EPOCHS"
echo "  batch:        $KCD_PER_GPU_BATCH (single GPU)"

# Tile if the chosen tile path doesn't exist (or is suspiciously small).
TILE_VALID=0
if [ -f "$TILES" ]; then
    sz=$(stat -c%s "$TILES" 2>/dev/null || echo 0)
    if [ "$sz" -gt 102400 ] && [ "${KCD_FORCE_RETILE:-0}" != "1" ]; then
        TILE_VALID=1
    fi
fi

echo
echo "=== 1. Multi-scale tile ==="
if [ "$TILE_VALID" = "1" ]; then
    echo "  Reusing $TILES (KCD_FORCE_RETILE=1 to redo)."
else
    mkdir -p "$(dirname "$TILES")"
    "$PYTHON_BIN" -m kwcoco_detector_kit tile \
        "$TRAIN_KWCOCO" "$TILES" \
        --mode multiscale \
        --tile_size 640 \
        --source_scales "1.0,0.5,0.25,0.125" \
        --stride_frac 0.5 \
        --min_gt_area_frac 0.0005 \
        --min_keep_fraction 0.20 \
        --oversize_factor 1.2 \
        --keep_negative true \
        --category_names "$KCD_CATEGORY_NAMES"
fi

echo
echo "=== 2. Sweep (train + export + eval + bench) ==="
"$PYTHON_BIN" -m kwcoco_detector_kit sweep \
    --train_kwcoco "$TILES" \
    --vali_kwcoco  "$VALI_KWCOCO" \
    --test_kwcoco  "$TEST_KWCOCO" \
    --kcd_root "$KCD_ROOT" \
    --trainer deimv2 \
    --variant "$KCD_VARIANT" \
    --input_hw "$KCD_INPUT_HW" \
    --train_policy "$KCD_TRAIN_POLICY" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$KCD_PER_GPU_BATCH" \
    --val_batch_size $(( 2 * KCD_PER_GPU_BATCH )) \
    --category_names "$KCD_CATEGORY_NAMES" \
    --use_amp true \
    --scale_tier "L" \
    --num_gpus 1 \
    "${INIT_FLAG[@]}"

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
echo "=== baseline complete: $KCD_VARIANT on $KCD_SCHEME ==="
echo "  manifest: $KCD_ROOT/manifest.tsv"
