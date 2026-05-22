#!/usr/bin/env bash
# Internal boilerplate. Runs INSIDE the docker container — as root.
# Receives all hyperparams via KCD_* env vars from _sbatch_train.sh.
#
# Pipeline: tile (shared per-scheme cache) -> sweep (train+export+eval+bench)
# -> manifest. Don't invoke directly; called by _sbatch_train.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"

: "${KCD_RUN_NAME:?_launch_train.sh: missing KCD_RUN_NAME}"
: "${KCD_SCHEME:?_launch_train.sh: missing KCD_SCHEME}"
: "${KCD_VARIANT:?_launch_train.sh: missing KCD_VARIANT}"
: "${KCD_NUM_GPUS:?_launch_train.sh: missing KCD_NUM_GPUS}"
: "${KCD_PER_GPU_BATCH:?_launch_train.sh: missing KCD_PER_GPU_BATCH}"
: "${KCD_NUM_EPOCHS:?_launch_train.sh: missing KCD_NUM_EPOCHS}"
: "${KCD_INPUT_HW:?_launch_train.sh: missing KCD_INPUT_HW}"
: "${KCD_TRAIN_POLICY:?_launch_train.sh: missing KCD_TRAIN_POLICY}"
: "${KCD_LR:?_launch_train.sh: missing KCD_LR}"
: "${KCD_BACKBONE_LR:?_launch_train.sh: missing KCD_BACKBONE_LR}"
: "${KCD_USE_AMP:?_launch_train.sh: missing KCD_USE_AMP}"

KCD_ROOT="$KCD_RUNS_DPATH/$KCD_RUN_NAME"
# Tile cache is keyed by (scheme, tile-params-hash). Two runs of the
# same scheme with identical tile params share the cache; different
# tile params get a different sub-dir so we never silently reuse
# mismatched tiles. The hash + a human-readable tile_params.txt are
# computed once tile params are resolved below.

# Scheme -> kwcoco bundles + category_names.
SCHEME_DIR="$KCD_SCHEMES_DIR/$KCD_SCHEME"
TRAIN_KWCOCO="$SCHEME_DIR/train.kwcoco.zip"
VALI_KWCOCO="$SCHEME_DIR/vali.kwcoco.zip"
TEST_KWCOCO="$SCHEME_DIR/test.kwcoco.zip"
kcd_require_path "$KCD_SCHEME train.kwcoco.zip" "$TRAIN_KWCOCO"
kcd_require_path "$KCD_SCHEME vali.kwcoco.zip" "$VALI_KWCOCO"
kcd_require_path "$KCD_SCHEME test.kwcoco.zip" "$TEST_KWCOCO"

if [ -z "${KCD_CATEGORY_NAMES:-}" ]; then
    # build_scheme_kwcoco.py writes `target_classes` (list) at the top
    # of scheme_report.json. Older schemes might use `target_order`;
    # check both.
    KCD_CATEGORY_NAMES="$("$PYTHON_BIN" -c "
import json, pathlib, sys
fp = pathlib.Path('$SCHEME_DIR/scheme_report.json')
if not fp.exists():
    sys.exit(f'scheme_report.json missing at {fp}')
data = json.loads(fp.read_text())
names = data.get('target_classes') or data.get('target_order') or []
if not names:
    # Last-ditch fallback: read the train split's per_target_class.
    names = list(data.get('splits', {}).get('train', {}).get('per_target_class', {}).keys())
if not names:
    sys.exit(f'no target_classes / target_order / per_target_class in {fp}')
print(','.join(names))
")"
fi
[ -z "$KCD_CATEGORY_NAMES" ] && {
    echo "ERROR: could not resolve category_names for scheme=$KCD_SCHEME" >&2
    echo "       Set KCD_CATEGORY_NAMES explicitly in the submit script, or" >&2
    echo "       rebuild the scheme bundle (scripts/build_scheme_kwcoco.py)" >&2
    exit 1
}

# Variant -> init checkpoint (when not explicitly set).
if [ -z "${KCD_INIT_CHECKPOINT:-}" ] && [ "${KCD_TRAIN_FROM_SCRATCH:-0}" != "1" ]; then
    case "$KCD_VARIANT" in
        deimv2_dinov3_s)  KCD_INIT_CHECKPOINT="$KCD_DEIMV2_DINOV3_S_COCO_PTH" ;;
        deimv2_hgnetv2_n) KCD_INIT_CHECKPOINT="$KCD_DEIMV2_HGNETV2_N_COCO_PTH" ;;
        *) ;;
    esac
fi
if [ "${KCD_TRAIN_FROM_SCRATCH:-0}" = "1" ]; then
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

# Tile params with sensible defaults — per-scheme, model-independent.
KCD_TILE_SIZE="${KCD_TILE_SIZE:-640}"
KCD_TILE_SOURCE_SCALES="${KCD_TILE_SOURCE_SCALES:-1.0,0.5,0.25,0.125}"
KCD_TILE_STRIDE_FRAC="${KCD_TILE_STRIDE_FRAC:-0.5}"
KCD_TILE_MIN_GT_AREA_FRAC="${KCD_TILE_MIN_GT_AREA_FRAC:-0.0005}"
KCD_TILE_MIN_KEEP_FRACTION="${KCD_TILE_MIN_KEEP_FRACTION:-0.20}"
KCD_TILE_OVERSIZE_FACTOR="${KCD_TILE_OVERSIZE_FACTOR:-1.2}"
KCD_TILE_KEEP_NEGATIVE="${KCD_TILE_KEEP_NEGATIVE:-true}"

# Auto-pick scale tier if not set.
if [ -z "${KCD_SCALE_TIER:-}" ]; then
    if   [ "$KCD_NUM_GPUS" -ge 5 ]; then KCD_SCALE_TIER=cluster
    elif [ "$KCD_NUM_GPUS" -ge 2 ]; then KCD_SCALE_TIER=2-4xL
    else KCD_SCALE_TIER=L
    fi
fi

TOTAL_BATCH=$(( KCD_PER_GPU_BATCH * KCD_NUM_GPUS ))
TOTAL_VAL_BATCH=$(( 2 * KCD_PER_GPU_BATCH * KCD_NUM_GPUS ))

# Tile-cache key. Hash the actual tile params (post-default-resolution)
# so two runs with the same effective params share, and different ones
# don't silently collide. sha1 truncated to 8 hex chars is plenty here.
TILE_PARAMS_BODY=$(printf '%s\n' \
    "tile_size=$KCD_TILE_SIZE" \
    "source_scales=$KCD_TILE_SOURCE_SCALES" \
    "stride_frac=$KCD_TILE_STRIDE_FRAC" \
    "min_gt_area_frac=$KCD_TILE_MIN_GT_AREA_FRAC" \
    "min_keep_fraction=$KCD_TILE_MIN_KEEP_FRACTION" \
    "oversize_factor=$KCD_TILE_OVERSIZE_FACTOR" \
    "keep_negative=$KCD_TILE_KEEP_NEGATIVE" \
    "category_names=$KCD_CATEGORY_NAMES")
TILE_HASH=$(printf '%s' "$TILE_PARAMS_BODY" | sha1sum | cut -c1-8)
TILE_DIR="$KCD_TILE_CACHE_DPATH/$KCD_SCHEME/$TILE_HASH"
TILES="$TILE_DIR/tiles.kwcoco.zip"

mkdir -p "$KCD_ROOT" "$KCD_ROOT/nccl_traces" "$TILE_DIR"
# Stash the human-readable params next to the bundle so the hash is
# invertible by inspection.
printf '%s\n' "$TILE_PARAMS_BODY" > "$TILE_DIR/tile_params.txt"

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
echo "=== run config ==="
echo "  run_name:     $KCD_RUN_NAME"
echo "  scheme:       $KCD_SCHEME"
echo "  variant:      $KCD_VARIANT"
echo "  categories:   $KCD_CATEGORY_NAMES"
echo "  input_hw:     $KCD_INPUT_HW"
echo "  train_policy: $KCD_TRAIN_POLICY"
echo "  init_ckpt:    $INIT_CKPT_DISPLAY"
echo "  kcd_root:     $KCD_ROOT"
echo "  tiles:        $TILES"
echo "                (cache key = $KCD_SCHEME/$TILE_HASH; see tile_params.txt)"
echo "  gpus:         $KCD_NUM_GPUS  (scale_tier=$KCD_SCALE_TIER)"
echo "  batch:        total=$TOTAL_BATCH  per_gpu=$KCD_PER_GPU_BATCH  val_total=$TOTAL_VAL_BATCH"
echo "  epochs:       $KCD_NUM_EPOCHS"
echo "  lr:           head=$KCD_LR  backbone=$KCD_BACKBONE_LR"
echo "  use_amp:      $KCD_USE_AMP"

echo
echo "=== 1. Multi-scale tile (shared per-scheme cache) ==="
TILE_VALID=0
if [ -f "$TILES" ] && [ "${KCD_FORCE_RETILE:-0}" != "1" ]; then
    sz=$(stat -c%s "$TILES" 2>/dev/null || echo 0)
    [ "$sz" -gt 102400 ] && TILE_VALID=1
fi
if [ "$TILE_VALID" = "1" ]; then
    echo "  Reusing $TILES (KCD_FORCE_RETILE=1 to redo)."
else
    "$PYTHON_BIN" -m kwcoco_detector_kit tile \
        "$TRAIN_KWCOCO" "$TILES" \
        --mode multiscale \
        --tile_size "$KCD_TILE_SIZE" \
        --source_scales "$KCD_TILE_SOURCE_SCALES" \
        --stride_frac "$KCD_TILE_STRIDE_FRAC" \
        --min_gt_area_frac "$KCD_TILE_MIN_GT_AREA_FRAC" \
        --min_keep_fraction "$KCD_TILE_MIN_KEEP_FRACTION" \
        --oversize_factor "$KCD_TILE_OVERSIZE_FACTOR" \
        --keep_negative "$KCD_TILE_KEEP_NEGATIVE" \
        --category_names "$KCD_CATEGORY_NAMES"
fi

echo
echo "=== 2. Sweep (train + export + eval + bench) ==="
DIST_FLAG=(--num_gpus "$KCD_NUM_GPUS")
[ "$KCD_NUM_GPUS" -gt 1 ] && DIST_FLAG+=(--distributed true)

"$PYTHON_BIN" -m kwcoco_detector_kit sweep \
    --train_kwcoco "$TILES" \
    --vali_kwcoco  "$VALI_KWCOCO" \
    --test_kwcoco  "$TEST_KWCOCO" \
    --kcd_root "$KCD_ROOT" \
    --trainer deimv2 \
    --variant "$KCD_VARIANT" \
    --input_hw "$KCD_INPUT_HW" \
    --train_policy "$KCD_TRAIN_POLICY" \
    --num_epochs "$KCD_NUM_EPOCHS" \
    --batch_size "$TOTAL_BATCH" \
    --val_batch_size "$TOTAL_VAL_BATCH" \
    --category_names "$KCD_CATEGORY_NAMES" \
    --lr "$KCD_LR" \
    --backbone_lr "$KCD_BACKBONE_LR" \
    --use_amp "$KCD_USE_AMP" \
    --scale_tier "$KCD_SCALE_TIER" \
    "${INIT_FLAG[@]}" \
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
echo "=== run complete: $KCD_RUN_NAME ==="
echo "  manifest: $KCD_ROOT/manifest.tsv"
echo
echo "Register the result back in docs/training_runs.yaml:"
echo "  python3 $KCD_REPO_ROOT/scripts/training_registry.py update <run-id> \\"
echo "      --status done \\"
echo "      --metric vali_map=<num> --metric vali_map50=<num> \\"
echo "      --artifact detect_metrics_json=$KCD_ROOT/eval/<candidate_id>/eval/detect_metrics.json"
