#!/usr/bin/env bash
# Shared helpers for the OpenGroundingDINO / DINOv2 4-GPU smoke ladder.
set -euo pipefail

THIS_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_DPATH="${KIT_DPATH:-$(cd "$THIS_DPATH/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

KCD_SMOKE_ROOT="${KCD_SMOKE_ROOT:-/tmp/kcd_smoketests/dino_v2_4x}"
KCD_CACHE_ROOT="${KCD_CACHE_ROOT:-/tmp/kcd_smoketests/cache/opengroundingdino}"
KCD_OPENGROUNDINGDINO_REPO_DPATH="${KCD_OPENGROUNDINGDINO_REPO_DPATH:-$KIT_DPATH/tpl/Open-GroundingDino}"

CATEGORY_NAME="${CATEGORY_NAME:-widget}"
VARIANT="${VARIANT:-opengroundingdino_swint}"
INPUT_SIZE="${INPUT_SIZE:-320}"
PRETRAIN_MODEL_PATH="${PRETRAIN_MODEL_PATH:-$KCD_CACHE_ROOT/pretrained/groundingdino_swint_ogc.pth}"

export PYTHON_BIN
export KCD_CACHE_ROOT
export KCD_OPENGROUNDINGDINO_REPO_DPATH
export PRETRAIN_MODEL_PATH
export TEXT_ENCODER_TYPE="${TEXT_ENCODER_TYPE:-bert-base-uncased}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

mkdir -p "$KCD_SMOKE_ROOT" "$KCD_CACHE_ROOT/pretrained"

log_section() {
    echo
    echo "=== $* ==="
}

run_cmd() {
    echo
    printf '+'
    printf ' %q' "$@"
    echo
    "$@"
}

require_file() {
    local fpath="$1"
    local msg="${2:-missing required file}"
    if [ ! -f "$fpath" ]; then
        echo "$msg: $fpath" >&2
        exit 1
    fi
}

ensure_pretrain() {
    if [ -f "$PRETRAIN_MODEL_PATH" ]; then
        echo "Reusing PRETRAIN_MODEL_PATH=$PRETRAIN_MODEL_PATH"
        return
    fi
    log_section "Download OpenGroundingDINO Swin-T checkpoint"
    mkdir -p "$(dirname "$PRETRAIN_MODEL_PATH")"
    local url="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    if command -v curl >/dev/null 2>&1; then
        run_cmd curl -L -o "$PRETRAIN_MODEL_PATH" "$url"
    else
        run_cmd wget -O "$PRETRAIN_MODEL_PATH" "$url"
    fi
}

print_env_summary() {
    log_section "Environment summary"
    echo "KIT_DPATH=$KIT_DPATH"
    echo "KCD_SMOKE_ROOT=$KCD_SMOKE_ROOT"
    echo "KCD_CACHE_ROOT=$KCD_CACHE_ROOT"
    echo "KCD_OPENGROUNDINGDINO_REPO_DPATH=$KCD_OPENGROUNDINGDINO_REPO_DPATH"
    echo "PRETRAIN_MODEL_PATH=$PRETRAIN_MODEL_PATH"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    "$PYTHON_BIN" - <<'PY'
import shutil
import subprocess
try:
    import torch
    print(f"torch={torch.__version__}")
    print(f"torch.version.cuda={torch.version.cuda}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count={torch.cuda.device_count()}")
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
}

make_demo_splits() {
    local dpath="$1"
    mkdir -p "$dpath"
    if [ -f "$dpath/train.kwcoco.zip" ] && \
       [ -f "$dpath/vali.kwcoco.zip" ] && \
       [ -f "$dpath/test.kwcoco.zip" ]; then
        echo "Reusing demo splits under $dpath"
        return
    fi
    log_section "Create tiny kwcoco demo splits"
    run_cmd "$PYTHON_BIN" -m kwcoco_detector_kit demo-data \
        "$dpath/train.kwcoco.zip" \
        --num_images "${DEMO_TRAIN_IMAGES:-8}" \
        --image_size "${DEMO_IMAGE_SIZE:-256,256}" \
        --category_name "$CATEGORY_NAME"
    run_cmd "$PYTHON_BIN" -m kwcoco_detector_kit demo-data \
        "$dpath/vali.kwcoco.zip" \
        --num_images "${DEMO_VALI_IMAGES:-4}" \
        --image_size "${DEMO_IMAGE_SIZE:-256,256}" \
        --seed 1 \
        --category_name "$CATEGORY_NAME"
    run_cmd "$PYTHON_BIN" -m kwcoco_detector_kit demo-data \
        "$dpath/test.kwcoco.zip" \
        --num_images "${DEMO_TEST_IMAGES:-4}" \
        --image_size "${DEMO_IMAGE_SIZE:-256,256}" \
        --seed 2 \
        --category_name "$CATEGORY_NAME"
}

make_subset_abs() {
    local src="$1"
    local dst="$2"
    local n="$3"
    if [ -f "$dst" ]; then
        echo "Reusing subset: $dst"
        return
    fi
    mkdir -p "$(dirname "$dst")"
    run_cmd "$PYTHON_BIN" - "$src" "$dst" "$n" <<'PY'
import sys
import kwcoco

src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
dset = kwcoco.CocoDataset.coerce(src)
gids = sorted(dset.index.imgs)[:n]
abs_fpaths = {gid: dset.get_image_fpath(gid) for gid in gids}
sub = dset.subset(gids, copy=True)
for img in sub.dataset.get("images", []):
    img["file_name"] = str(abs_fpaths[img["id"]])
sub.fpath = dst
sub.dump()
print(f"wrote {dst}: {sub.n_images} images, {sub.n_annots} annotations")
PY
}

run_mock_sweep() {
    local root="$1"
    local train="$2"
    local vali="$3"
    local test="$4"
    run_cmd "$PYTHON_BIN" -m kwcoco_detector_kit sweep \
        --train_kwcoco "$train" \
        --vali_kwcoco "$vali" \
        --test_kwcoco "$test" \
        --kcd_root "$root" \
        --trainer mock_tiny \
        --variant mock_tiny \
        --input_hw 128,128 \
        --train_policy fixed \
        --num_epochs 1 \
        --batch_size 2 \
        --val_batch_size 2 \
        --num_classes 1 \
        --category_name "$CATEGORY_NAME" \
        --scale_tier S \
        --no-do_export \
        --no-do_eval \
        --no-do_bench
}

run_ogdino_sweep() {
    local root="$1"
    local train="$2"
    local vali="$3"
    local test="$4"
    local num_gpus="$5"
    local distributed="$6"
    local batch_size="$7"
    local scale_tier="$8"

    local distributed_flag=(--no-distributed)
    if [ "$distributed" = "1" ]; then
        distributed_flag=(--distributed)
    fi

    run_cmd "$PYTHON_BIN" -m kwcoco_detector_kit sweep \
        --train_kwcoco "$train" \
        --vali_kwcoco "$vali" \
        --test_kwcoco "$test" \
        --kcd_root "$root" \
        --trainer opengroundingdino \
        --variant "$VARIANT" \
        --input_hw "$INPUT_SIZE,$INPUT_SIZE" \
        --train_policy fixed \
        --num_epochs "${NUM_EPOCHS:-1}" \
        --batch_size "$batch_size" \
        --val_batch_size "$batch_size" \
        --num_classes 1 \
        --category_name "$CATEGORY_NAME" \
        --lr "${LR:-1e-4}" \
        --backbone_lr "${BACKBONE_LR:-1e-5}" \
        --use_amp true \
        --scale_tier "$scale_tier" \
        --num_gpus "$num_gpus" \
        "${distributed_flag[@]}" \
        --no-do_export \
        --no-do_eval \
        --no-do_bench
}
