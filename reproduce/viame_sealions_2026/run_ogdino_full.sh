#!/usr/bin/env bash
# Real-data full-splits OpenGroundingDINO run for VIAME Sea Lions 2021-2024.
#
# Run inside the kwcoco-detector-kit Docker image (cu13x). The Slurm
# scaffolding lives next to this file in slurm/.
#
# Pipeline:
#   1. Restage the prepared train/vali/test kwcoco bundles so the absolute
#      file_name entries resolve here (laptop path -> $DATA_DPATH via
#      $KCD_PATH_REWRITE_PREFIXES).
#   2. Positive-only multiscale tile at $INPUT_SIZE.
#   3. Sweep OpenGroundingDINO Swin-T on the tiles, distributed over
#      $NUM_GPUS GPUs.
#
# Most knobs are env-var overridable. The defaults below target 4x A6000
# (47 GB each) at input_size=800. At b=8 the 3090 saw ~13.6 GB; A6000
# has ~3.5x the headroom, so b=16 fits comfortably and leaves room for
# the validation pass. Bump BATCH_SIZE as you confirm memory usage.
set -euo pipefail

THIS_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_DPATH="${KIT_DPATH:-$(cd "$THIS_DPATH/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_DPATH="${DATA_DPATH:?DATA_DPATH must point at the viame_sealions_2026 data root accessible here}"
RAW_DPATH="${RAW_DPATH:-$DATA_DPATH/training_ready_v1}"
TRAIN_RAW="${TRAIN_RAW:-$RAW_DPATH/train.kwcoco.zip}"
VALI_RAW="${VALI_RAW:-$RAW_DPATH/vali.kwcoco.zip}"
TEST_RAW="${TEST_RAW:-$RAW_DPATH/test.kwcoco.zip}"

KCD_REPRODUCE_ROOT="${KCD_REPRODUCE_ROOT:?KCD_REPRODUCE_ROOT must be set (per-run outputs)}"
KCD_CACHE_ROOT="${KCD_CACHE_ROOT:?KCD_CACHE_ROOT must be set (reusable tiles + pretrained)}"

CATEGORY_NAME="${CATEGORY_NAME:-sealion}"
VARIANT="${VARIANT:-opengroundingdino_swint}"
INPUT_SIZE="${INPUT_SIZE:-800}"
SOURCE_SCALES="${SOURCE_SCALES:-1.0,0.5,0.25}"
STRIDE_FRAC="${STRIDE_FRAC:-0.75}"
MIN_GT_AREA_FRAC="${MIN_GT_AREA_FRAC:-0.0001}"
MIN_KEEP_FRACTION="${MIN_KEEP_FRACTION:-0.20}"

NUM_GPUS="${NUM_GPUS:-4}"
# A6000 (47 GB) at input=800: epoch-0 telemetry from b=16 showed max
# mem ~25 GB/GPU and loss dropping cleanly. b=24 should sit near 38 GB,
# still leaving headroom for the val pass. Push higher (28-32) only
# after confirming val memory stays under ~44 GB.
BATCH_SIZE="${BATCH_SIZE:-24}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-24}"
# Loss curve at b=16 e=12 was still descending end-of-epoch-0 — 16
# epochs gives more time to converge. Per-epoch wall clock ~27 min
# at b=16 / 539 steps; at b=24 fewer steps, still well under 24h cap.
NUM_EPOCHS="${NUM_EPOCHS:-16}"
LR="${LR:-1e-4}"
BACKBONE_LR="${BACKBONE_LR:-1e-5}"
SCALE_TIER="${SCALE_TIER:-2-4xL}"
KCD_DO_EVAL="${KCD_DO_EVAL:-0}"
KCD_DISTRIBUTED="${KCD_DISTRIBUTED:-1}"

PRETRAIN_MODEL_PATH="${PRETRAIN_MODEL_PATH:-$KCD_CACHE_ROOT/pretrained/groundingdino_swint_ogc.pth}"

export KIT_DPATH PYTHON_BIN
export KCD_CACHE_ROOT
export PRETRAIN_MODEL_PATH
export KCD_OPENGROUNDINGDINO_REPO_DPATH="${KCD_OPENGROUNDINGDINO_REPO_DPATH:-$KIT_DPATH/tpl/Open-GroundingDino}"
export TEXT_ENCODER_TYPE="${TEXT_ENCODER_TYPE:-bert-base-uncased}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
# Silence "huggingface/tokenizers: got forked after parallelism was
# used" warning — BERT tokenizer fires before dataloader forks workers.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "$KCD_REPRODUCE_ROOT" "$KCD_CACHE_ROOT/pretrained"

echo "=== Reproduce: VIAME Sea Lions OGDino full run ==="
echo "KIT_DPATH=$KIT_DPATH"
echo "DATA_DPATH=$DATA_DPATH"
echo "RAW_DPATH=$RAW_DPATH"
echo "KCD_REPRODUCE_ROOT=$KCD_REPRODUCE_ROOT"
echo "KCD_CACHE_ROOT=$KCD_CACHE_ROOT"
echo "PRETRAIN_MODEL_PATH=$PRETRAIN_MODEL_PATH"
echo "NUM_GPUS=$NUM_GPUS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "VAL_BATCH_SIZE=$VAL_BATCH_SIZE"
echo "NUM_EPOCHS=$NUM_EPOCHS"
echo "INPUT_SIZE=$INPUT_SIZE"
echo "KCD_PATH_REWRITE_PREFIXES=${KCD_PATH_REWRITE_PREFIXES:-<unset>}"

"$PYTHON_BIN" - <<'PY'
import shutil, subprocess
try:
    import torch
    print(f"torch={torch.__version__} cuda={torch.version.cuda} "
          f"available={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
except Exception as ex:
    print(f"torch probe failed: {ex}")
nvcc = shutil.which("nvcc")
print(f"nvcc={nvcc}")
if nvcc:
    print(subprocess.check_output([nvcc, "--version"], text=True).strip())
PY
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
fi

for raw in "$TRAIN_RAW" "$VALI_RAW" "$TEST_RAW"; do
    if [ ! -f "$raw" ]; then
        echo "ERROR: missing prepared split: $raw" >&2
        exit 1
    fi
done

ensure_pretrain() {
    if [ -f "$PRETRAIN_MODEL_PATH" ]; then
        echo "Reusing PRETRAIN_MODEL_PATH=$PRETRAIN_MODEL_PATH"
        return
    fi
    mkdir -p "$(dirname "$PRETRAIN_MODEL_PATH")"
    local url="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    if command -v curl >/dev/null 2>&1; then
        curl -L -o "$PRETRAIN_MODEL_PATH" "$url"
    else
        wget -O "$PRETRAIN_MODEL_PATH" "$url"
    fi
}
ensure_pretrain

# -- 1. Restage parent kwcoco bundles with rewritten paths --------------
# The prepared splits encode the laptop's absolute image paths
# (/media/joncrall/raid/...). Inside this container the same files live
# under $DATA_DPATH (mounted RO). We write rewritten copies under
# $KCD_REPRODUCE_ROOT/staged/ so the literal paths resolve here.
STAGE_DPATH="$KCD_REPRODUCE_ROOT/staged"
mkdir -p "$STAGE_DPATH"

restage_kwcoco() {
    local src="$1"
    local dst="$2"
    if [ -f "$dst" ]; then
        if "$PYTHON_BIN" - "$dst" <<'PY'
import sys
from pathlib import Path
import kwcoco

dset = kwcoco.CocoDataset.coerce(sys.argv[1])
gids = list(dset.index.imgs)[: min(8, dset.n_images)]
for gid in gids:
    if not Path(dset.get_image_fpath(gid)).exists():
        raise SystemExit(1)
PY
        then
            echo "Reusing staged kwcoco: $dst"
            return
        fi
        echo "Staged kwcoco stale, rebuilding: $dst"
        rm -f "$dst"
    fi
    "$PYTHON_BIN" - "$src" "$dst" <<'PY'
import os
import sys
from pathlib import Path
import kwcoco


def rewrite_path(path):
    path = Path(path)
    if path.exists():
        return path
    pairs = os.environ.get("KCD_PATH_REWRITE_PREFIXES", "")
    for pair in [p for p in pairs.split(";") if p]:
        if "=" not in pair:
            continue
        old, new = pair.split("=", 1)
        oldp = Path(old)
        try:
            rel = path.relative_to(oldp)
        except ValueError:
            continue
        cand = Path(new) / rel
        if cand.exists():
            return cand
    data_root = os.environ.get("DATA_DPATH")
    if data_root:
        marker = Path(data_root).name
        parts = path.parts
        if marker in parts:
            idx = parts.index(marker)
            cand = Path(data_root).joinpath(*parts[idx + 1:])
            if cand.exists():
                return cand
    return path


src, dst = sys.argv[1], sys.argv[2]
dset = kwcoco.CocoDataset.coerce(src)
missing = []
for img in dset.dataset.get("images", []):
    fpath = rewrite_path(dset.get_image_fpath(img["id"]))
    if not Path(fpath).exists():
        missing.append(str(fpath))
        if len(missing) >= 8:
            break
    img["file_name"] = str(fpath)
if missing:
    raise FileNotFoundError("missing rewritten image paths:\n" + "\n".join(missing))
dset.fpath = dst
dset.dump()
print(f"wrote {dst}: {dset.n_images} images, {dset.n_annots} annotations")
PY
}

restage_kwcoco "$TRAIN_RAW" "$STAGE_DPATH/train.kwcoco.zip"
restage_kwcoco "$VALI_RAW"  "$STAGE_DPATH/vali.kwcoco.zip"
restage_kwcoco "$TEST_RAW"  "$STAGE_DPATH/test.kwcoco.zip"

TRAIN_STAGED="$STAGE_DPATH/train.kwcoco.zip"
VALI_STAGED="$STAGE_DPATH/vali.kwcoco.zip"
TEST_STAGED="$STAGE_DPATH/test.kwcoco.zip"

# -- 2. Multiscale positive-only tiling --------------------------------
TILE_CACHE_NAME="tiles_t${INPUT_SIZE}_scales_${SOURCE_SCALES}_stride_${STRIDE_FRAC}_pos"
TILE_CACHE_NAME="${TILE_CACHE_NAME//[^A-Za-z0-9._-]/_}"
TILE_DPATH="${TILE_DPATH:-$KCD_CACHE_ROOT/$TILE_CACHE_NAME}"
mkdir -p "$TILE_DPATH"
TRAIN_TILES="$TILE_DPATH/train_tiles_pos.kwcoco.zip"
VALI_TILES="$TILE_DPATH/vali_tiles_pos.kwcoco.zip"
TEST_TILES="$TILE_DPATH/test_tiles_pos.kwcoco.zip"
echo "TILE_DPATH=$TILE_DPATH"

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
        --category_name "$CATEGORY_NAME" \
        --tile_size "$INPUT_SIZE" \
        --source_scales "$SOURCE_SCALES" \
        --stride_frac "$STRIDE_FRAC" \
        --min_gt_area_frac "$MIN_GT_AREA_FRAC" \
        --min_keep_fraction "$MIN_KEEP_FRACTION" \
        --keep_negative false \
        --oversize_factor 1.0 \
        --jpeg_quality 90
}

tile_one "$TRAIN_STAGED" "$TRAIN_TILES"
tile_one "$VALI_STAGED"  "$VALI_TILES"
tile_one "$TEST_STAGED"  "$TEST_TILES"

# -- 3. OpenGroundingDINO sweep ----------------------------------------
RUN_ROOT="$KCD_REPRODUCE_ROOT/run"
mkdir -p "$RUN_ROOT"

EVAL_FLAG=(--no-do_eval)
if [ "$KCD_DO_EVAL" = "1" ] || [ "$KCD_DO_EVAL" = "true" ]; then
    EVAL_FLAG=(--do_eval true)
fi
DISTRIBUTED_FLAG=(--no-distributed)
if [ "$KCD_DISTRIBUTED" = "1" ] || [ "$KCD_DISTRIBUTED" = "true" ]; then
    DISTRIBUTED_FLAG=(--distributed)
fi

"$PYTHON_BIN" -m kwcoco_detector_kit sweep \
    --train_kwcoco "$TRAIN_TILES" \
    --vali_kwcoco "$VALI_TILES" \
    --test_kwcoco "$TEST_TILES" \
    --kcd_root "$RUN_ROOT" \
    --trainer opengroundingdino \
    --variant "$VARIANT" \
    --input_hw "$INPUT_SIZE,$INPUT_SIZE" \
    --train_policy fixed \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --val_batch_size "$VAL_BATCH_SIZE" \
    --num_classes 1 \
    --category_name "$CATEGORY_NAME" \
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
echo "PASS: VIAME Sea Lions OGDino full run"
echo "RUN_ROOT=$RUN_ROOT"
