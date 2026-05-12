#!/usr/bin/env bash
# Start a first-pass VIAME sea-lion detector tune.
#
# Defaults are intentionally conservative: positive-only tiles, batch=2,
# no ONNX/export/bench stage, and no eval stage unless KCD_DO_EVAL=1.
# Override NUM_GPUS=4 KCD_DISTRIBUTED=1 SCALE_TIER=2-4xL for a 4x A6000 run.
set -euo pipefail

KIT_DPATH="${KIT_DPATH:-/home/joncrall/code/kwcoco_detector_kit}"
DATA_DPATH="${DATA_DPATH:-/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026}"
KCD_ROOT="${KCD_ROOT:-$DATA_DPATH/training_runs/ogdino_swint_3090}"
KCD_CACHE_ROOT="${KCD_CACHE_ROOT:-$DATA_DPATH/training_runs/cache/ogdino_swint}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RAW_DPATH="${RAW_DPATH:-$DATA_DPATH/training_ready_v1}"
TRAIN_RAW="${TRAIN_RAW:-$RAW_DPATH/train.kwcoco.zip}"
VALI_RAW="${VALI_RAW:-$RAW_DPATH/vali.kwcoco.zip}"
TEST_RAW="${TEST_RAW:-$RAW_DPATH/test.kwcoco.zip}"

TILE_CACHE_NAME="tiles_t${TILE_SIZE:-800}_scales_${SOURCE_SCALES:-1.0,0.5,0.25}_stride_${STRIDE_FRAC:-0.75}_pos"
TILE_CACHE_NAME="${TILE_CACHE_NAME//[^A-Za-z0-9._-]/_}"
if [ -z "${TILE_DPATH+x}" ]; then
    TILE_DPATH="$KCD_CACHE_ROOT/$TILE_CACHE_NAME"
    LEGACY_TILE_DPATH="$DATA_DPATH/training_runs/ogdino_swint_3090/tiles"
    if [ ! -f "$TILE_DPATH/train_tiles_pos.kwcoco.zip" ] && [ -f "$LEGACY_TILE_DPATH/train_tiles_pos.kwcoco.zip" ]; then
        TILE_DPATH="$LEGACY_TILE_DPATH"
    fi
fi
TRAIN_TILES="${TRAIN_TILES:-$TILE_DPATH/train_tiles_pos.kwcoco.zip}"
VALI_TILES="${VALI_TILES:-$TILE_DPATH/vali_tiles_pos.kwcoco.zip}"
TEST_TILES="${TEST_TILES:-$TILE_DPATH/test_tiles_pos.kwcoco.zip}"

SMOKE="${SMOKE:-0}"
SMOKE_TRAIN_IMAGES="${SMOKE_TRAIN_IMAGES:-512}"
SMOKE_VALI_IMAGES="${SMOKE_VALI_IMAGES:-128}"

TILE_SIZE="${TILE_SIZE:-800}"
SOURCE_SCALES="${SOURCE_SCALES:-1.0,0.5,0.25}"
STRIDE_FRAC="${STRIDE_FRAC:-0.75}"
MIN_GT_AREA_FRAC="${MIN_GT_AREA_FRAC:-0.0001}"
MIN_KEEP_FRACTION="${MIN_KEEP_FRACTION:-0.20}"

NUM_EPOCHS="${NUM_EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
LR="${LR:-1e-4}"
BACKBONE_LR="${BACKBONE_LR:-1e-5}"
KCD_DO_EVAL="${KCD_DO_EVAL:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
KCD_DISTRIBUTED="${KCD_DISTRIBUTED:-0}"
SCALE_TIER="${SCALE_TIER:-M}"

export KCD_ROOT
export KCD_CACHE_ROOT
export PYTHON_BIN
export KCD_OPENGROUNDINGDINO_REPO_DPATH="${KCD_OPENGROUNDINGDINO_REPO_DPATH:-$KIT_DPATH/tpl/Open-GroundingDino}"
if [ -z "${PRETRAIN_MODEL_PATH+x}" ]; then
    PRETRAIN_MODEL_PATH="$KCD_CACHE_ROOT/pretrained/groundingdino_swint_ogc.pth"
    LEGACY_PRETRAIN_MODEL_PATH="$DATA_DPATH/training_runs/ogdino_swint_3090/pretrained/groundingdino_swint_ogc.pth"
    if [ ! -f "$PRETRAIN_MODEL_PATH" ] && [ -f "$LEGACY_PRETRAIN_MODEL_PATH" ]; then
        PRETRAIN_MODEL_PATH="$LEGACY_PRETRAIN_MODEL_PATH"
    fi
fi
export PRETRAIN_MODEL_PATH
export TEXT_ENCODER_TYPE="${TEXT_ENCODER_TYPE:-bert-base-uncased}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$KCD_ROOT" "$KCD_CACHE_ROOT/pretrained" "$TILE_DPATH"

if [ ! -f "$TRAIN_RAW" ] || [ ! -f "$VALI_RAW" ] || [ ! -f "$TEST_RAW" ]; then
    echo "Missing prepared raw splits under $RAW_DPATH" >&2
    echo "Run examples/viame_sealions_2026/prepare_training_kwcoco.py first." >&2
    exit 1
fi
if [ ! -f "$PRETRAIN_MODEL_PATH" ]; then
    echo "Missing PRETRAIN_MODEL_PATH=$PRETRAIN_MODEL_PATH" >&2
    echo "Run examples/viame_sealions_2026/setup_host_env.sh first." >&2
    exit 1
fi

echo "KCD_ROOT=$KCD_ROOT"
echo "KCD_CACHE_ROOT=$KCD_CACHE_ROOT"
echo "TILE_DPATH=$TILE_DPATH"
echo "PRETRAIN_MODEL_PATH=$PRETRAIN_MODEL_PATH"

tile_one() {
    local src="$1"
    local dst="$2"
    if [ -f "$dst" ]; then
        echo "Reusing tiles: $dst"
        return
    fi
    "$PYTHON_BIN" -m kwcoco_detector_kit tile \
        "$src" "$dst" \
        --mode multiscale \
        --category_name sealion \
        --tile_size "$TILE_SIZE" \
        --source_scales "$SOURCE_SCALES" \
        --stride_frac "$STRIDE_FRAC" \
        --min_gt_area_frac "$MIN_GT_AREA_FRAC" \
        --min_keep_fraction "$MIN_KEEP_FRACTION" \
        --keep_negative false \
        --oversize_factor 1.0 \
        --jpeg_quality 90
}

echo
echo "=== 1. Positive-only tiling ==="
tile_one "$TRAIN_RAW" "$TRAIN_TILES"
tile_one "$VALI_RAW" "$VALI_TILES"
tile_one "$TEST_RAW" "$TEST_TILES"

if [ "$SMOKE" = "1" ]; then
    SMOKE_DPATH="$KCD_ROOT/smoke_tiles"
    mkdir -p "$SMOKE_DPATH"
    TRAIN_TILES="$SMOKE_DPATH/train_tiles_pos_${SMOKE_TRAIN_IMAGES}.kwcoco.zip"
    VALI_TILES="$SMOKE_DPATH/vali_tiles_pos_${SMOKE_VALI_IMAGES}.kwcoco.zip"
    TEST_TILES="$SMOKE_DPATH/test_tiles_pos_${SMOKE_VALI_IMAGES}.kwcoco.zip"
    subset_abs_one() {
        local src="$1"
        local dst="$2"
        local n="$3"
        "$PYTHON_BIN" - "$src" "$dst" "$n" <<'PY'
import sys, kwcoco
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
dset = kwcoco.CocoDataset(src)
gids = sorted(dset.index.imgs)[:n]
abs_fpaths = {gid: dset.get_image_fpath(gid) for gid in gids}
sub = dset.subset(gids, copy=True)
for img in sub.dataset['images']:
    img['file_name'] = str(abs_fpaths[img['id']])
sub.fpath = dst
sub.dump(dst, newlines=True)
print(f'wrote smoke subset {dst}: {sub.n_images} images, {sub.n_annots} annots')
PY
    }
    subset_abs_one "$TILE_DPATH/train_tiles_pos.kwcoco.zip" "$TRAIN_TILES" "$SMOKE_TRAIN_IMAGES"
    subset_abs_one "$TILE_DPATH/vali_tiles_pos.kwcoco.zip" "$VALI_TILES" "$SMOKE_VALI_IMAGES"
    subset_abs_one "$TILE_DPATH/test_tiles_pos.kwcoco.zip" "$TEST_TILES" "$SMOKE_VALI_IMAGES"
fi

EVAL_FLAG=(--no-do_eval)
if [ "$KCD_DO_EVAL" = "1" ]; then
    EVAL_FLAG=(--do_eval true)
fi
DISTRIBUTED_FLAG=(--no-distributed)
if [ "$KCD_DISTRIBUTED" = "1" ] || [ "$KCD_DISTRIBUTED" = "true" ]; then
    DISTRIBUTED_FLAG=(--distributed)
fi

echo
echo "=== 2. OpenGroundingDINO Swin-T tune ==="
"$PYTHON_BIN" -m kwcoco_detector_kit sweep \
    --train_kwcoco "$TRAIN_TILES" \
    --vali_kwcoco "$VALI_TILES" \
    --test_kwcoco "$TEST_TILES" \
    --kcd_root "$KCD_ROOT" \
    --trainer opengroundingdino \
    --variant opengroundingdino_swint \
    --input_hw "$TILE_SIZE,$TILE_SIZE" \
    --train_policy fixed \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --val_batch_size "$VAL_BATCH_SIZE" \
    --num_classes 1 \
    --category_name sealion \
    --lr "$LR" \
    --backbone_lr "$BACKBONE_LR" \
    --use_amp true \
    --scale_tier "$SCALE_TIER" \
    --num_gpus "$NUM_GPUS" \
    "${DISTRIBUTED_FLAG[@]}" \
    --no-do_export \
    --no-do_bench \
    "${EVAL_FLAG[@]}"

echo
echo "Done. KCD_ROOT=$KCD_ROOT"
