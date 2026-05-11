#!/usr/bin/env bash
# kwcoco-detector-kit — GPU smoke (single GPU, synthetic kwcoco data).
#
# Exercises the main codepaths on real hardware:
#   1. env + tier probes
#   2. CPU baseline (mock_tiny)
#   3. DEIMv2 hgnetv2_atto on a single GPU
#   4. round-loop (mock_tiny) to exercise mine + merge
#   5. unified eligibility manifest
#
# Designed to keep VRAM, disk, and time well under stress thresholds —
# 8 synth images, batch 4, 1 epoch. Total wall time ~60-90s on a warm host.
#
# Env knobs:
#   KCD_ROOT                 workspace (default /tmp/kcd_gpu_smoke)
#   CUDA_VISIBLE_DEVICES     GPU index (default 1)
#   PYTHON_BIN               python interpreter (default python)
#   KCD_DEIMV2_REPO_DPATH    DEIMv2 checkout (default $HOME/code/shitspotter/tpl/DEIMv2)
#   SKIP_DEIMV2=1            skip the DEIMv2 GPU stage
#   SKIP_ROUND_LOOP=1        skip the round-loop CPU stage
#   SKIP_CPU_BASELINE=1      skip the mock_tiny CPU baseline
#   KEEP_WORKDIR=1           don't rm -rf $KCD_ROOT at start
set -euo pipefail

KCD_ROOT="${KCD_ROOT:-/tmp/kcd_gpu_smoke}"
PYTHON_BIN="${PYTHON_BIN:-python}"
KCD_DEIMV2_REPO_DPATH="${KCD_DEIMV2_REPO_DPATH:-$HOME/code/shitspotter/tpl/DEIMv2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SKIP_DEIMV2="${SKIP_DEIMV2:-0}"
SKIP_ROUND_LOOP="${SKIP_ROUND_LOOP:-0}"
SKIP_CPU_BASELINE="${SKIP_CPU_BASELINE:-0}"
KEEP_WORKDIR="${KEEP_WORKDIR:-0}"

export CUDA_VISIBLE_DEVICES
export KCD_DEIMV2_REPO_DPATH

declare -i N_PASS=0
declare -i N_FAIL=0
declare -a STAGES_FAILED=()

_stage() {
    local label="$1"
    echo
    echo "=================================================================="
    echo "  $label"
    echo "=================================================================="
}

_status() {
    # _status <label> <exit_code>
    local label="$1"
    local rc="$2"
    if [ "$rc" -eq 0 ]; then
        echo "  [PASS]  $label"
        N_PASS=$((N_PASS + 1))
    else
        echo "  [FAIL]  $label (exit $rc)"
        N_FAIL=$((N_FAIL + 1))
        STAGES_FAILED+=("$label")
    fi
}

# ---- 0. workspace ----
if [ "$KEEP_WORKDIR" != "1" ]; then
    rm -rf "$KCD_ROOT"
fi
mkdir -p "$KCD_ROOT"

# ---- 1. env probe ----
_stage "Stage 1 — env probe"
echo "  KCD_ROOT              = $KCD_ROOT"
echo "  PYTHON_BIN            = $PYTHON_BIN"
echo "  CUDA_VISIBLE_DEVICES  = $CUDA_VISIBLE_DEVICES"
echo "  KCD_DEIMV2_REPO_DPATH = $KCD_DEIMV2_REPO_DPATH"
echo
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.free,pcie.link.width.current \
               --format=csv,noheader 2>&1 | head -10
else
    echo "  (no nvidia-smi on PATH)"
fi
echo
"$PYTHON_BIN" --version
"$PYTHON_BIN" -c "
import torch
print(f'  torch                {torch.__version__}')
print(f'  cuda available       {torch.cuda.is_available()}')
print(f'  device count         {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f'  device {i}             {torch.cuda.get_device_name(i)} '
              f'({total/1024**3:.1f} GB total, {free/1024**3:.1f} GB free)')
"
"$PYTHON_BIN" -c "import kwcoco_detector_kit; print(f'  kwcoco_detector_kit  {kwcoco_detector_kit.__version__}')"
echo
echo "  --- check-env (strict import for deimv2 group to catch version conflicts) ---"
set +e
"$PYTHON_BIN" -m kwcoco_detector_kit check-env --groups core,onnx,deimv2 2>&1 | tail -25
ENV_RC=$?
set -e
if [ "$ENV_RC" -ne 0 ]; then
    echo "  WARNING: check-env reported missing or broken deps (rc=$ENV_RC)."
    echo "  Fix what's missing before relying on Stage 2/3 results."
fi
echo
echo "  --- detect_tier ---"
"$PYTHON_BIN" -c "
from kwcoco_detector_kit.trainers._tier import detect_tier
info = detect_tier()
print(f'  tier={info.tier} aggregate_vram={info.aggregate_vram_gb:.1f}GB '
      f'visible_gpus={info.num_visible_gpus}')
if info.pcie_warning:
    print('  pcie_warning:', info.pcie_warning)
"
_status "Stage 1 — env probe" 0

# ---- 2. CPU baseline (mock_tiny) ----
if [ "$SKIP_CPU_BASELINE" != "1" ]; then
    _stage "Stage 2 — CPU baseline (mock_tiny)"
    set +e
    "$PYTHON_BIN" -m kwcoco_detector_kit demo-data \
        "$KCD_ROOT/raw.kwcoco.zip" \
        --num_images 8 --num_categories 1 \
        --image_size 256,256 --category_name widget
    rc1=$?
    "$PYTHON_BIN" -m kwcoco_detector_kit tile \
        "$KCD_ROOT/raw.kwcoco.zip" "$KCD_ROOT/tiles.kwcoco.zip" \
        --mode multiscale --tile_size 128 --source_scales "1.0,0.5" \
        --stride_frac 1.0 --min_gt_area_frac 0.001 \
        --min_source_scale_long_side 32 --keep_negative true \
        --category_name widget
    rc2=$?
    "$PYTHON_BIN" -m kwcoco_detector_kit sweep \
        --train_kwcoco "$KCD_ROOT/tiles.kwcoco.zip" \
        --vali_kwcoco  "$KCD_ROOT/tiles.kwcoco.zip" \
        --test_kwcoco  "$KCD_ROOT/raw.kwcoco.zip" \
        --kcd_root "$KCD_ROOT" \
        --trainer mock_tiny --variant mock_tiny \
        --input_hw 128,128 --train_policy fixed \
        --num_epochs 1 --batch_size 2 --val_batch_size 2 \
        --num_classes 1 --category_name widget \
        --scale_tier S --num_gpus 1
    rc3=$?
    set -e
    if [ $((rc1+rc2+rc3)) -eq 0 ]; then _status "Stage 2 — mock_tiny CPU" 0
    else                                 _status "Stage 2 — mock_tiny CPU" 1
    fi
else
    echo "  [SKIP]  Stage 2 — mock_tiny CPU baseline"
fi

# ---- 3. DEIMv2 GPU cell ----
if [ "$SKIP_DEIMV2" != "1" ]; then
    if [ ! -d "$KCD_DEIMV2_REPO_DPATH" ]; then
        echo "  [SKIP]  Stage 3 — DEIMv2 GPU cell (missing $KCD_DEIMV2_REPO_DPATH)"
    else
        _stage "Stage 3 — DEIMv2 hgnetv2_atto on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        # Smallest variant, 256x256 input, batch 4, 1 epoch.
        # Disable AMP for the smoke (HGNetv2 atto at small input is fast either way).
        set +e
        "$PYTHON_BIN" -m kwcoco_detector_kit sweep \
            --train_kwcoco "$KCD_ROOT/tiles.kwcoco.zip" \
            --vali_kwcoco  "$KCD_ROOT/tiles.kwcoco.zip" \
            --test_kwcoco  "$KCD_ROOT/raw.kwcoco.zip" \
            --kcd_root "$KCD_ROOT" \
            --trainer deimv2 --variant deimv2_hgnetv2_atto \
            --input_hw 256,256 --train_policy fixed \
            --num_epochs 1 --batch_size 4 --val_batch_size 4 \
            --num_classes 1 --category_name widget \
            --lr 5e-4 --backbone_lr 2.5e-5 --use_amp false \
            --scale_tier S --num_gpus 1
        rc=$?
        set -e
        _status "Stage 3 — DEIMv2 GPU" "$rc"
    fi
else
    echo "  [SKIP]  Stage 3 — DEIMv2 GPU (SKIP_DEIMV2=1)"
fi

# ---- 4. Round-loop (mock_tiny CPU) ----
if [ "$SKIP_ROUND_LOOP" != "1" ]; then
    _stage "Stage 4 — round_loop (mock_tiny CPU) — exercises mine + merge"
    # The round_loop driver expects positive + negative tile bundles.
    # We can derive them from the multiscale tile output by splitting by tile_role.
    "$PYTHON_BIN" -c "
import kwcoco
from pathlib import Path
src = kwcoco.CocoDataset.coerce('$KCD_ROOT/tiles.kwcoco.zip')

def split(role, dst):
    out = kwcoco.CocoDataset()
    out.fpath = str(dst)
    cid = out.add_category(name='widget')
    src_cid = {c['name']: c['id'] for c in src.dataset['categories']}.get('widget')
    new_gids = {}
    for img in src.images().objs:
        if img.get('tile_role') != role:
            continue
        new_img = {k: v for k, v in img.items() if k != 'id'}
        new_gids[img['id']] = out.add_image(**new_img)
    for ann in src.dataset.get('annotations', []):
        gid = ann.get('image_id')
        if gid not in new_gids:
            continue
        if ann.get('category_id') != src_cid:
            continue
        out.add_annotation(
            image_id=new_gids[gid], category_id=cid,
            bbox=list(ann['bbox']), area=float(ann.get('area', 0.0)),
            iscrowd=int(ann.get('iscrowd', 0)),
        )
    out.dump()
    print(f'  split role={role}: {out.n_images} imgs, {out.n_annots} anns -> {dst}')

split('positive', '$KCD_ROOT/pos.kwcoco.zip')
split('negative', '$KCD_ROOT/neg.kwcoco.zip')
"
    set +e
    "$PYTHON_BIN" -m kwcoco_detector_kit round-loop \
        --pos_tiles_kwcoco "$KCD_ROOT/pos.kwcoco.zip" \
        --neg_tiles_kwcoco "$KCD_ROOT/neg.kwcoco.zip" \
        --vali_kwcoco "$KCD_ROOT/tiles.kwcoco.zip" \
        --test_kwcoco "$KCD_ROOT/raw.kwcoco.zip" \
        --kcd_root "$KCD_ROOT" \
        --trainer mock_tiny --variant mock_tiny \
        --input_hw 128,128 --train_policy fixed \
        --num_rounds 2 --round0_neg_over_pos 1.0 \
        --mine_score_thresh 0.05 --max_hard_per_round 10 \
        --num_epochs 1 --batch_size 2 --val_batch_size 2 \
        --num_classes 1 --category_name widget \
        --scale_tier S --num_gpus 1
    rc=$?
    set -e
    _status "Stage 4 — round_loop" "$rc"
else
    echo "  [SKIP]  Stage 4 — round_loop"
fi

# ---- 5. Unified manifest ----
_stage "Stage 5 — unified eligibility manifest"
set +e
"$PYTHON_BIN" -m kwcoco_detector_kit manifest \
    --auto --kcd_root "$KCD_ROOT" \
    --out "$KCD_ROOT/manifest.tsv" \
    --out_json "$KCD_ROOT/manifest.json" \
    --max_desktop_ms 500 \
    --allow_missing_desktop_bench true \
    --include_smoke_models true \
    --print_winner true
rc=$?
set -e
_status "Stage 5 — manifest" "$rc"

# ---- Final summary ----
echo
echo "=================================================================="
echo "  GPU smoke summary"
echo "=================================================================="
echo "  passed: $N_PASS"
echo "  failed: $N_FAIL"
if [ "$N_FAIL" -gt 0 ]; then
    echo "  failed stages: ${STAGES_FAILED[*]}"
fi
echo "  manifest: $KCD_ROOT/manifest.tsv"
echo "  workdir:  $KCD_ROOT"
exit "$N_FAIL"
