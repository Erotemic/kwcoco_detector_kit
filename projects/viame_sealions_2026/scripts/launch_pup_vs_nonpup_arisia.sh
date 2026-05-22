#!/usr/bin/env bash
# Launch the pup_vs_nonpup 2-class baseline.
#
# All paths come from scripts/paths.sh. Override any KCD_* variable in
# your shell to redirect — never edit this script for path tweaks.
#
# Prereqs (one-time per host):
#   1. bash scripts/fetch_pretrained.sh
#        -> writes $KCD_DEIMV2_DINOV3_S_COCO_PTH
#   2. python scripts/build_scheme_kwcoco.py --scheme pup_vs_nonpup
#        -> writes $KCD_SCHEMES_DIR/pup_vs_nonpup/{train,vali,test}.kwcoco.zip
#   3. Ensure $KCD_DEIMV2_REPO_DPATH is set, or use the docker image
#      (recommended — see docker/opengroundingdino/README.md).
#
# Per-run env knobs (sensible defaults, override only when needed):
#   KCD_NUM_GPUS      DDP world size. Auto-detected from `nvidia-smi -L`;
#                     defaults to all visible GPUs.
#   KCD_DISTRIBUTED   torchrun on/off. Auto-on when num_gpus > 1.
#   KCD_TIER          scale tier (S/M/L/XL/2-4xL/cluster). Auto-picked
#                     from num_gpus.
#   KCD_PER_GPU_BATCH per-GPU batch size (default 16). Total batch =
#                     num_gpus * per_gpu_batch (DEIMv2 takes total_batch).
#   PYTHON_BIN        python (default python)
#   NUM_EPOCHS        default 30
#   KCD_TRAIN_FROM_SCRATCH=1   skip init_checkpoint (expect 5-10 AP loss)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_EPOCHS="${NUM_EPOCHS:-30}"
KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-16}"

# Auto-detect visible GPUs. `nvidia-smi -L` works inside docker when
# --gpus is set; lists one "GPU N:" line per device.
if [ -z "${KCD_NUM_GPUS:-}" ]; then
    KCD_NUM_GPUS="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 0)"
fi
[ -z "$KCD_NUM_GPUS" ] || [ "$KCD_NUM_GPUS" -lt 1 ] && KCD_NUM_GPUS=1

# Distributed flips on automatically when num_gpus > 1.
if [ "$KCD_NUM_GPUS" -gt 1 ]; then
    KCD_DISTRIBUTED="${KCD_DISTRIBUTED:-1}"
else
    KCD_DISTRIBUTED="${KCD_DISTRIBUTED:-0}"
fi

# Scale tier defaults from gpu count when not explicitly forced.
if [ -z "${KCD_TIER:-}" ]; then
    if [ "$KCD_NUM_GPUS" -ge 5 ]; then
        KCD_TIER="cluster"
    elif [ "$KCD_NUM_GPUS" -ge 2 ]; then
        KCD_TIER="2-4xL"
    else
        KCD_TIER="L"
    fi
fi

# DEIMv2 takes `batch_size` as the TOTAL batch across all GPUs (it's
# forwarded as total_batch_size in the DEIMv2 YAML). Scale linearly so
# per-GPU work stays at KCD_PER_GPU_BATCH regardless of GPU count.
TOTAL_BATCH=$(( KCD_PER_GPU_BATCH * KCD_NUM_GPUS ))
TOTAL_VAL_BATCH=$(( 2 * KCD_PER_GPU_BATCH * KCD_NUM_GPUS ))

SCHEME_DIR="$KCD_SCHEMES_DIR/pup_vs_nonpup"
TRAIN_KWCOCO="$SCHEME_DIR/train.kwcoco.zip"
VALI_KWCOCO="$SCHEME_DIR/vali.kwcoco.zip"
TEST_KWCOCO="$SCHEME_DIR/test.kwcoco.zip"
KCD_ROOT="$KCD_ROOT_PUP_VS_NONPUP"

kcd_require_path "pup_vs_nonpup train.kwcoco.zip" "$TRAIN_KWCOCO"
kcd_require_path "pup_vs_nonpup vali.kwcoco.zip" "$VALI_KWCOCO"
kcd_require_path "pup_vs_nonpup test.kwcoco.zip" "$TEST_KWCOCO"

if [ "${KCD_TRAIN_FROM_SCRATCH:-0}" = "1" ]; then
    echo "WARNING: KCD_TRAIN_FROM_SCRATCH=1 — skipping init_checkpoint" >&2
    INIT_FLAG=()
    INIT_CKPT_DISPLAY="<from-scratch>"
else
    kcd_require_path "DEIMv2+DINOv3 COCO pretrained checkpoint" "$KCD_DEIMV2_DINOV3_S_COCO_PTH" || {
        echo "  Run: bash scripts/fetch_pretrained.sh" >&2
        exit 1
    }
    INIT_FLAG=(--init_checkpoint "$KCD_DEIMV2_DINOV3_S_COCO_PTH")
    INIT_CKPT_DISPLAY="$KCD_DEIMV2_DINOV3_S_COCO_PTH"
fi

mkdir -p "$KCD_ROOT"
# nccl_traces/ holds per-rank flight-recorder dumps when the NCCL
# watchdog timeout fires. Created here (inside docker, root) because
# the submit-side user typically can't write under $KCD_ROOT.
mkdir -p "$KCD_ROOT/nccl_traces"
TILES="$KCD_ROOT/tiles.kwcoco.zip"

# Disk guard. The tile step + checkpoints + sweep eval artifacts can
# easily land 20-30 GB under $KCD_ROOT. Fail fast (and clearly) when
# the filesystem doesn't have headroom — prevents the disk-full mid-
# epoch crash that lost the previous run.
KCD_MIN_FREE_GB="${KCD_MIN_FREE_GB:-30}"
free_kb=$(df -k --output=avail "$KCD_TRAINING_ROOT" 2>/dev/null | tail -n1 | tr -d ' ')
if [ -n "$free_kb" ]; then
    free_gb=$(( free_kb / 1024 / 1024 ))
    echo "  free disk:   ${free_gb} GB at $KCD_TRAINING_ROOT (need >= ${KCD_MIN_FREE_GB})"
    if [ "$free_gb" -lt "$KCD_MIN_FREE_GB" ]; then
        echo "ERROR: only ${free_gb} GB free on the filesystem hosting $KCD_TRAINING_ROOT" >&2
        echo "  Expected at least ${KCD_MIN_FREE_GB} GB (tiles ~15GB + checkpoints ~5GB + eval ~5GB)." >&2
        echo "  Free space or set KCD_MIN_FREE_GB to a lower value to bypass (risky)." >&2
        exit 1
    fi
fi

echo
echo "=== config ==="
echo "  repo:        $KCD_REPO_ROOT"
echo "  scheme_dir:  $SCHEME_DIR"
echo "  kcd_root:    $KCD_ROOT"
echo "  init_ckpt:   $INIT_CKPT_DISPLAY"
echo "  tier:        $KCD_TIER  gpus: $KCD_NUM_GPUS  distributed: $KCD_DISTRIBUTED"
echo "  batch:       total=$TOTAL_BATCH  per_gpu=$KCD_PER_GPU_BATCH  val_total=$TOTAL_VAL_BATCH"
echo "  epochs:      $NUM_EPOCHS"

echo
echo "=== 1. Multi-scale tile ==="
# Tiling is the slow step (~15 min for the full 1314-image train set).
# Skip if the cached tiles bundle already exists and is non-trivial
# (the size guard rejects the zero-tile artifacts from the prior
# broken-paths bug). Force a fresh tile with KCD_FORCE_RETILE=1, or
# `rm $KCD_ROOT/tiles.kwcoco.zip` to invalidate.
TILE_VALID=0
if [ -f "$TILES" ] && [ "${KCD_FORCE_RETILE:-0}" != "1" ]; then
    sz=$(stat -c%s "$TILES" 2>/dev/null || echo 0)
    if [ "$sz" -gt 102400 ]; then
        TILE_VALID=1
    fi
fi
if [ "$TILE_VALID" = "1" ]; then
    echo "  Reusing $TILES (KCD_FORCE_RETILE=1 to redo)."
else
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
        --category_names "pup,nonpup_sealion"
fi

echo
echo "=== 2. Sweep (train + export + eval + bench) ==="
DIST_FLAG=(--num_gpus "$KCD_NUM_GPUS")
if [ "$KCD_DISTRIBUTED" = "1" ]; then
    DIST_FLAG+=(--distributed true)
fi
"$PYTHON_BIN" -m kwcoco_detector_kit sweep \
    --train_kwcoco "$TILES" \
    --vali_kwcoco  "$VALI_KWCOCO" \
    --test_kwcoco  "$TEST_KWCOCO" \
    --kcd_root "$KCD_ROOT" \
    --trainer deimv2 \
    --variant deimv2_dinov3_s \
    --input_hw 640,640 \
    --train_policy multiscale_512_768 \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$TOTAL_BATCH" \
    --val_batch_size "$TOTAL_VAL_BATCH" \
    --category_names "pup,nonpup_sealion" \
    --lr 5e-4 \
    --backbone_lr 2.5e-5 \
    --use_amp true \
    --scale_tier "$KCD_TIER" \
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
echo "=== pup_vs_nonpup complete ==="
echo "  manifest: $KCD_ROOT/manifest.tsv"
echo
echo "Register the result back in docs/training_runs.yaml:"
echo "  python3 $KCD_REPO_ROOT/scripts/training_registry.py update <run-id> \\"
echo "      --status done \\"
echo "      --metric vali_map=<num> --metric vali_map50=<num> \\"
echo "      --artifact detect_metrics_json=$KCD_ROOT/eval/<candidate_id>/eval/detect_metrics.json"
