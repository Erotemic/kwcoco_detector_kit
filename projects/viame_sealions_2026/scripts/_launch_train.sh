#!/usr/bin/env bash
# Internal boilerplate. Runs INSIDE the docker container — as root.
# Receives all hyperparams via KCD_* env vars from _sbatch_train.sh.
#
# Pipeline: tile (shared per-scheme cache) -> sweep (train+export+eval+bench)
# -> manifest. Don't invoke directly; called by _sbatch_train.sh.
set -euo pipefail

# Group-writable outputs so the user (host UID) can clean / mutate
# anything this container (root inside) writes. Files 0664, dirs 0775.
umask 002

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
# Tile cache is keyed by tile-geometry hash and is SCHEME-AGNOSTIC.
# Since 2026-05-22 we tile from the universal (sealion-collapsed)
# source bundle with source_category preserved on every annotation,
# then apply the scheme's class-collapse as a fast post-step into
# $KCD_ROOT/scheme_applied/. That way every scheme reuses one tile
# bundle — no per-scheme tile duplication on disk.

# Universal source: single-category bundle with source_category on
# each ann. Tiles come from train; vali/test stay un-tiled because
# they're already small full-image bundles and the model's eval-time
# transforms handle the resize.
UNIVERSAL_DIR="$KCD_TRAINING_READY_DIR"
# Universal sources: post-2026-06-05 default to the v2 *_norm bundles
# (unpacked/{train,vali,test}_norm_v2.kwcoco.zip) via paths.sh. The
# fall-back to KCD_TRAINING_READY_DIR/<split>.kwcoco.zip stays for
# legacy sandboxes; override KCD_UNIVERSAL_*_KWCOCO to point at any
# 9-cat norm bundle.
UNIVERSAL_TRAIN_KWCOCO="${KCD_UNIVERSAL_TRAIN_KWCOCO:-$UNIVERSAL_DIR/train.kwcoco.zip}"
UNIVERSAL_VALI_KWCOCO="${KCD_UNIVERSAL_VALI_KWCOCO:-$UNIVERSAL_DIR/vali.kwcoco.zip}"
UNIVERSAL_TEST_KWCOCO="${KCD_UNIVERSAL_TEST_KWCOCO:-$UNIVERSAL_DIR/test.kwcoco.zip}"
kcd_require_path "universal train.kwcoco.zip" "$UNIVERSAL_TRAIN_KWCOCO"
kcd_require_path "universal vali.kwcoco.zip" "$UNIVERSAL_VALI_KWCOCO"
kcd_require_path "universal test.kwcoco.zip" "$UNIVERSAL_TEST_KWCOCO"

SCHEME_DIR="$KCD_SCHEMES_DIR/$KCD_SCHEME"

if [ -z "${KCD_CATEGORY_NAMES:-}" ]; then
    # Resolve from the scheme YAML directly. The scheme's target_order
    # is the canonical class-index order — it's what apply_scheme uses
    # to assign category_ids in the output bundle and what the sweep
    # must agree on at train time.
    KCD_CATEGORY_NAMES="$("$PYTHON_BIN" -c "
import sys, pathlib, yaml
fp = pathlib.Path('$KCD_REPO_ROOT/docs/class_schemes.yaml')
data = yaml.safe_load(fp.read_text()) or {}
scheme = (data.get('schemes') or {}).get('$KCD_SCHEME')
if not scheme:
    sys.exit(f'scheme $KCD_SCHEME not found in {fp}')
names = scheme.get('target_order') or scheme.get('target_classes') or []
if not names:
    # First-seen order over the scheme's mapping.values() fallback.
    seen = set(); ordered = []
    for tgt in (scheme.get('mapping') or {}).values():
        if tgt not in seen:
            seen.add(tgt); ordered.append(tgt)
    names = ordered
if not names:
    sys.exit(f'no target_order in scheme $KCD_SCHEME')
print(','.join(names))
")"
fi
[ -z "$KCD_CATEGORY_NAMES" ] && {
    echo "ERROR: could not resolve category_names for scheme=$KCD_SCHEME" >&2
    echo "       Set KCD_CATEGORY_NAMES explicitly in the submit script, or" >&2
    echo "       verify docs/class_schemes.yaml has target_order for this scheme" >&2
    exit 1
}

# Auto-resolve distractor_classes from the scheme YAML when the env var
# wasn't set explicitly. Distractor classes are kept in the trained
# model's class set (so it learns to discriminate them) but excluded
# from the class-agnostic detection AP at eval time. The submit script
# CAN override by setting KCD_DISTRACTOR_CLASSES directly.
if [ -z "${KCD_DISTRACTOR_CLASSES:-}" ]; then
    KCD_DISTRACTOR_CLASSES="$("$PYTHON_BIN" -c "
import pathlib, yaml
fp = pathlib.Path('$KCD_REPO_ROOT/docs/class_schemes.yaml')
data = yaml.safe_load(fp.read_text()) or {}
scheme = (data.get('schemes') or {}).get('$KCD_SCHEME') or {}
names = scheme.get('distractor_classes') or []
print(','.join(names))
")"
    if [ -n "$KCD_DISTRACTOR_CLASSES" ]; then
        export KCD_DISTRACTOR_CLASSES
        echo "  scheme $KCD_SCHEME declares distractor_classes=$KCD_DISTRACTOR_CLASSES"
        echo "  -> eval will write a sidecar metrics file with those classes pruned"
    fi
fi

# Auto-derive KCD_WDS_SOURCE_TO_TARGET from the scheme YAML's mapping.
# WebDataset shards are scheme-AGNOSTIC (they carry source_category per
# annotation as written by the writer); the reader applies this
# raw -> target collapse per sample. Submit scripts CAN override.
if [ -z "${KCD_WDS_SOURCE_TO_TARGET:-}" ]; then
    KCD_WDS_SOURCE_TO_TARGET="$("$PYTHON_BIN" -c "
import json, pathlib, yaml
fp = pathlib.Path('$KCD_REPO_ROOT/docs/class_schemes.yaml')
data = yaml.safe_load(fp.read_text()) or {}
scheme = (data.get('schemes') or {}).get('$KCD_SCHEME') or {}
mapping = scheme.get('mapping') or {}
print(json.dumps(mapping))
" 2>/dev/null)"
    [ "$KCD_WDS_SOURCE_TO_TARGET" = "{}" ] && KCD_WDS_SOURCE_TO_TARGET=""
    export KCD_WDS_SOURCE_TO_TARGET
fi

# Variant -> init checkpoint (when not explicitly set).
if [ -z "${KCD_INIT_CHECKPOINT:-}" ] && [ "${KCD_TRAIN_FROM_SCRATCH:-0}" != "1" ]; then
    case "$KCD_VARIANT" in
        deimv2_dinov3_s)  KCD_INIT_CHECKPOINT="$KCD_DEIMV2_DINOV3_S_COCO_PTH" ;;
        deimv2_hgnetv2_n) KCD_INIT_CHECKPOINT="$KCD_DEIMV2_HGNETV2_N_COCO_PTH" ;;
        deimv2_hgnetv2_s) KCD_INIT_CHECKPOINT="$KCD_DEIMV2_HGNETV2_S_COCO_PTH" ;;
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
# Val batch ratio: 2x train was too aggressive when training memory is
# already tight (v4 OOM cycle). Allow override via env; default keeps
# val == train so a denser-than-average val batch can't kill the run.
KCD_VAL_BATCH_MULT="${KCD_VAL_BATCH_MULT:-1}"
TOTAL_VAL_BATCH=$(( KCD_VAL_BATCH_MULT * KCD_PER_GPU_BATCH * KCD_NUM_GPUS ))

# Tile-cache key — scheme-AGNOSTIC. Hash tile geometry params PLUS a
# fingerprint of the tile writer's passthrough-field whitelist (so a
# kit code change that adds/removes preserved annotation fields auto-
# invalidates older caches). Different geometry or whitelist → different
# sub-dir; we never silently reuse mismatched tiles. The May-24 episode
# (48h * 3 jobs trained on empty targets because the cache predated the
# source_category passthrough) is the cost of NOT having this.
WRITER_FINGERPRINT=$("$PYTHON_BIN" -c "
from kwcoco_detector_kit.data import tile
print('v{}:{}'.format(
    getattr(tile, '_TILE_WRITER_VERSION', 1),
    ','.join(sorted(tile._PASSTHROUGH_ANN_FIELDS)),
))
" 2>/dev/null || echo 'unknown')
TILE_PARAMS_BODY=$(printf '%s\n' \
    "tile_mode=${KCD_TILE_MODE:-multiscale}" \
    "tile_size=$KCD_TILE_SIZE" \
    "source_scales=$KCD_TILE_SOURCE_SCALES" \
    "stride_frac=$KCD_TILE_STRIDE_FRAC" \
    "min_gt_area_frac=$KCD_TILE_MIN_GT_AREA_FRAC" \
    "min_keep_fraction=$KCD_TILE_MIN_KEEP_FRACTION" \
    "oversize_factor=$KCD_TILE_OVERSIZE_FACTOR" \
    "keep_negative=$KCD_TILE_KEEP_NEGATIVE" \
    "category_names=${KCD_TILE_CATEGORY_NAMES:-sealion}" \
    "writer_passthrough=$WRITER_FINGERPRINT")
TILE_HASH=$(printf '%s' "$TILE_PARAMS_BODY" | sha1sum | cut -c1-8)
TILE_DIR="$KCD_TILE_CACHE_DPATH/_universal/$TILE_HASH"
UNIVERSAL_TILES="$TILE_DIR/tiles.kwcoco.zip"

# Per-scheme derivatives go under the run's own dir — fast to build,
# tied to the run for traceability.
SCHEME_APPLIED_DIR="$KCD_ROOT/scheme_applied"
TRAIN_KWCOCO="$SCHEME_APPLIED_DIR/train.kwcoco.zip"
VALI_KWCOCO="$SCHEME_APPLIED_DIR/vali.kwcoco.zip"
TEST_KWCOCO="$SCHEME_APPLIED_DIR/test.kwcoco.zip"

mkdir -p "$KCD_ROOT" "$KCD_ROOT/nccl_traces" "$TILE_DIR" "$SCHEME_APPLIED_DIR"
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
echo "  universal_tiles: $UNIVERSAL_TILES"
echo "                   (scheme-agnostic cache key = $TILE_HASH; see tile_params.txt)"
echo "  scheme_applied:  $SCHEME_APPLIED_DIR/<split>.kwcoco.zip"
echo "  gpus:         $KCD_NUM_GPUS  (scale_tier=$KCD_SCALE_TIER)"
echo "  batch:        total=$TOTAL_BATCH  per_gpu=$KCD_PER_GPU_BATCH  val_total=$TOTAL_VAL_BATCH"
echo "  epochs:       $KCD_NUM_EPOCHS"
echo "  lr:           head=$KCD_LR  backbone=$KCD_BACKBONE_LR"
echo "  use_amp:      $KCD_USE_AMP"

echo
echo "=== 1. Verify universal tile cache exists ==="
# Tile-build is its own slurm+docker job since 2026-06-05. This
# script no longer auto-builds — that path was tangled with training
# resource requests (GPUs you don't need, NCCL config that doesn't
# apply, walltime mismatched to tile-build cost). Fail fast here so
# the user sees the cause and runs the build job.
TILE_VALID=0
if [ -f "$UNIVERSAL_TILES" ]; then
    sz=$(stat -c%s "$UNIVERSAL_TILES" 2>/dev/null || echo 0)
    [ "$sz" -gt 102400 ] && TILE_VALID=1
fi
if [ "$TILE_VALID" != "1" ]; then
    echo "ERROR: universal tile cache missing or corrupt." >&2
    echo "  expected: $UNIVERSAL_TILES" >&2
    echo "  TILE_HASH: $TILE_HASH (params in $TILE_DIR/tile_params.txt)" >&2
    echo "  source bundle: $UNIVERSAL_TRAIN_KWCOCO" >&2
    echo "" >&2
    echo "  Build the cache first:" >&2
    echo "    bash $KCD_REPO_ROOT/scripts/submit_build_tiles.sh" >&2
    echo "" >&2
    echo "  The build runs as its own slurm+docker job (no GPU, ~2-4h)." >&2
    echo "  Submit_train_* jobs can fan out from it via" >&2
    echo "    KCD_DEPENDS_ON=<build_jobid> bash submit_train_<...>.sh" >&2
    exit 1
fi
echo "  Reusing $UNIVERSAL_TILES."

# Optional: build WebDataset shards from the universal tile bundle.
# Scheme-AGNOSTIC — one shard set serves every scheme. The reader
# applies source_to_target collapse per sample. Cache lives next to
# the tile bundle so it's reused across runs with the same tile
# config. Placed BEFORE apply_scheme so a "data-prep only" job (see
# KCD_DATA_PREP_ONLY below) can produce the shared tile + shard
# artifacts for many training jobs to fan out from via slurm deps,
# without needing per-scheme apply_scheme to have run first.
if [ "${KCD_USE_WEBDATASET:-0}" = "1" ]; then
    echo
    echo "=== 1b. WebDataset shards (scheme-agnostic, built from universal tiles) ==="
    SHARDS_DPATH="${KCD_WDS_SHARDS_DPATH:-$TILE_DIR/shards}"
    SHARDS_DONE_MARKER="$SHARDS_DPATH/.build_done"
    if [ -f "$SHARDS_DONE_MARKER" ] && [ -z "${KCD_FORCE_RESHARD:-}" ]; then
        echo "  Reusing $SHARDS_DPATH (KCD_FORCE_RESHARD=1 to redo)."
    else
        mkdir -p "$SHARDS_DPATH"
        echo "  Writing $SHARDS_DPATH from $UNIVERSAL_TILES ..."
        # kwcoco_dataloader doesn't ship a top-level __main__.py;
        # invoke the CLI module path directly.
        "$PYTHON_BIN" -m kwcoco_dataloader.cli.build_detection_webdataset \
            --in_fpath "$UNIVERSAL_TILES" \
            --out_dpath "$SHARDS_DPATH" \
            --bucket_attr dominant_raw_class \
            --maxcount 5000 \
            --maxsize_mb 1024 \
            --jpeg_quality 95 \
            --drop_provenance false \
            --progress false
        touch "$SHARDS_DONE_MARKER"
    fi
    export KCD_WDS_SHARDS_DPATH="$SHARDS_DPATH"
    echo "  -> KCD_WDS_SHARDS_DPATH=$KCD_WDS_SHARDS_DPATH"
fi

# Early-exit when only the shared scheme-agnostic data prep was
# requested. Training jobs that depend on this prep via slurm
# --dependency=afterok pick up the tile bundle + shards from cache
# and skip steps 1/1b on re-entry.
if [ "${KCD_DATA_PREP_ONLY:-0}" = "1" ]; then
    echo
    echo "=== prep complete (KCD_DATA_PREP_ONLY=1) ==="
    echo "  universal_tiles: $UNIVERSAL_TILES"
    [ -n "${KCD_WDS_SHARDS_DPATH:-}" ] && echo "  shards:          $KCD_WDS_SHARDS_DPATH"
    echo "  scheme_applied dir was NOT populated (per-scheme; runs in train jobs)."
    exit 0
fi

echo
echo "=== 2. Apply scheme to tile + vali + test ==="
# Re-collapses the universal source_category fields into the scheme's
# target_classes. Fast — pure JSON rewrite, no image reads.
APPLY_SCHEME="$KCD_REPO_ROOT/scripts/apply_scheme_to_kwcoco.py"
kcd_require_path "apply_scheme_to_kwcoco.py" "$APPLY_SCHEME"
for split in "train:$UNIVERSAL_TILES:$TRAIN_KWCOCO" \
             "vali:$UNIVERSAL_VALI_KWCOCO:$VALI_KWCOCO" \
             "test:$UNIVERSAL_TEST_KWCOCO:$TEST_KWCOCO"; do
    name="${split%%:*}"
    rest="${split#*:}"
    src="${rest%:*}"
    dst="${rest##*:}"
    if [ -f "$dst" ] && [ "${KCD_FORCE_REAPPLY:-0}" != "1" ]; then
        echo "  Reusing $dst (KCD_FORCE_REAPPLY=1 to redo)."
        continue
    fi
    echo "  apply $KCD_SCHEME to $name: $src -> $dst"
    "$PYTHON_BIN" "$APPLY_SCHEME" \
        --src "$src" --dst "$dst" --scheme "$KCD_SCHEME"
done

# Fail-fast guard: empty annotations in scheme_applied/train.kwcoco.zip
# is the silent-killer that burned 48h * 3 jobs on May 24. Either the
# tile cache was built before tile.py's source_category passthrough
# landed, or the scheme's mapping/drop rules wiped every label. Bail
# now instead of training 48h on empty targets.
echo
echo "=== 2b. Sanity check: nonzero annotations after apply_scheme ==="
N_TRAIN_ANNS=$("$PYTHON_BIN" -c "
import kwcoco
print(kwcoco.CocoDataset.coerce('$TRAIN_KWCOCO').n_annots)
" 2>/dev/null || echo 0)
N_VALI_ANNS=$("$PYTHON_BIN" -c "
import kwcoco
print(kwcoco.CocoDataset.coerce('$VALI_KWCOCO').n_annots)
" 2>/dev/null || echo 0)
echo "  scheme_applied/train.kwcoco.zip: $N_TRAIN_ANNS annotations"
echo "  scheme_applied/vali.kwcoco.zip:  $N_VALI_ANNS annotations"
if [ "$N_TRAIN_ANNS" -eq 0 ]; then
    echo "ERROR: scheme_applied/train.kwcoco.zip has 0 annotations." >&2
    echo "  Most likely cause: the universal tile cache was built before" >&2
    echo "  tile.py's source_category passthrough was added (kit commit 5d99545)." >&2
    echo "  Fix:" >&2
    echo "    rm -rf $KCD_TILE_CACHE_DPATH/_universal" >&2
    echo "    rm -rf $SCHEME_APPLIED_DIR" >&2
    echo "    # then resubmit. The tile step will re-build from the patched code." >&2
    echo "  Refusing to start a 48h training run with empty targets." >&2
    exit 1
fi
if [ "$N_VALI_ANNS" -eq 0 ]; then
    echo "WARNING: scheme_applied/vali.kwcoco.zip has 0 annotations." >&2
    echo "  Eval mAP will be meaningless. Continuing anyway." >&2
fi

# =========================================================
# 2c. Optional: class-balance the training MSCOCO
# =========================================================
# gen004 lever for the JPEG (non-WDS) backend. Set in the submit
# script when an experiment wants a different class composition
# than the on-disk MSCOCO provides.
#
#   KCD_BALANCE_TARGET_JSON='{"<empty>":0.4,"pup":0.2,"nonpup_sealion":0.4}'
#   KCD_BALANCE_TARGET_SIZE=80000      # optional; defaults to source size
#
# Mechanism: export apply_scheme'd kwcoco -> MSCOCO once, run
# balance_mscoco to oversample/undersample image entries to hit
# the target distribution, then override TRAIN_KWCOCO so the
# trainer's _ensure_mscoco passes the balanced .mscoco.json
# through unchanged. Assets on disk are NOT modified.
#
# Buckets:
#   * <empty> = images with no annotations after apply_scheme
#   * <target_class_name> = dominant target class per image
# (See kwcoco_detector_kit/data/balance_mscoco.py for the
# bucketing rules.)
if [ -n "${KCD_BALANCE_TARGET_JSON:-}" ]; then
    echo
    echo "=== 2c. Class-balance training MSCOCO ==="
    BALANCE_DIR="$KCD_ROOT/balance"
    mkdir -p "$BALANCE_DIR"
    UNBALANCED_MSCOCO="$BALANCE_DIR/train_unbalanced.mscoco.json"
    BALANCED_MSCOCO="$BALANCE_DIR/train_balanced.mscoco.json"

    if [ ! -f "$UNBALANCED_MSCOCO" ] || [ "${KCD_FORCE_REBALANCE:-0}" = "1" ]; then
        echo "  export $TRAIN_KWCOCO -> $UNBALANCED_MSCOCO"
        "$PYTHON_BIN" -c "
import sys
from kwcoco_detector_kit.data.coco_export import export_mscoco
cat_names = [n.strip() for n in '$KCD_CATEGORY_NAMES'.split(',') if n.strip()]
export_mscoco(src='$TRAIN_KWCOCO', dst='$UNBALANCED_MSCOCO',
              category_names=cat_names,
              include_segmentations=False, category_id_start=0)
"
    else
        echo "  Reusing $UNBALANCED_MSCOCO (KCD_FORCE_REBALANCE=1 to redo)."
    fi

    if [ ! -f "$BALANCED_MSCOCO" ] || [ "${KCD_FORCE_REBALANCE:-0}" = "1" ]; then
        echo "  balance: $KCD_BALANCE_TARGET_JSON"
        "$PYTHON_BIN" -m kwcoco_detector_kit.data.balance_mscoco \
            "$UNBALANCED_MSCOCO" "$BALANCED_MSCOCO" \
            --target_distribution "$KCD_BALANCE_TARGET_JSON" \
            ${KCD_BALANCE_TARGET_SIZE:+--target_size "$KCD_BALANCE_TARGET_SIZE"} \
            ${KCD_BALANCE_MAX_OVERSAMPLE:+--max_oversample "$KCD_BALANCE_MAX_OVERSAMPLE"} \
            --seed "${KCD_BALANCE_SEED:-0}"
    else
        echo "  Reusing $BALANCED_MSCOCO (KCD_FORCE_REBALANCE=1 to redo)."
    fi

    # Repoint the trainer at the balanced MSCOCO. The sweep accepts
    # .mscoco.json directly via _ensure_mscoco.
    TRAIN_KWCOCO="$BALANCED_MSCOCO"
    echo "  -> TRAIN_KWCOCO=$TRAIN_KWCOCO"
fi

echo
echo "=== 3. Sweep (train + export + eval + bench) ==="
DIST_FLAG=(--num_gpus "$KCD_NUM_GPUS")
[ "$KCD_NUM_GPUS" -gt 1 ] && DIST_FLAG+=(--distributed true)

# Debug echo: show whether the force/resume vars actually reached the
# container. Catches the env-passthrough bug that produced job 2510's
# instant ok_resumed (the env file was missing these vars, so the
# launcher's ${VAR:+--flag} expansions all collapsed to nothing).
echo "[_launch_train.sh] resume/force state:"
echo "  KCD_RESUME_CKPT     = ${KCD_RESUME_CKPT:-<unset>}"
echo "  KCD_FORCE_TRAIN     = ${KCD_FORCE_TRAIN:-<unset>}"
echo "  KCD_FORCE_EXPORT    = ${KCD_FORCE_EXPORT:-<unset>}"
echo "  KCD_FORCE_EVAL      = ${KCD_FORCE_EVAL:-<unset>}"
echo "  KCD_FORCE_BENCH     = ${KCD_FORCE_BENCH:-<unset>}"
echo "  KCD_DISTRACTOR_CLASSES = ${KCD_DISTRACTOR_CLASSES:-<unset>}"

"$PYTHON_BIN" -m kwcoco_detector_kit sweep \
    --train_kwcoco "$TRAIN_KWCOCO" \
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
    ${KCD_DO_EXPORT:+--do_export "$KCD_DO_EXPORT"} \
    ${KCD_DO_EVAL:+--do_eval "$KCD_DO_EVAL"} \
    ${KCD_DO_BENCH:+--do_bench "$KCD_DO_BENCH"} \
    ${KCD_RESUME_CKPT:+--resume "$KCD_RESUME_CKPT"} \
    ${KCD_FORCE_TRAIN:+--force_train "$KCD_FORCE_TRAIN"} \
    ${KCD_FORCE_EXPORT:+--force_export "$KCD_FORCE_EXPORT"} \
    ${KCD_FORCE_EVAL:+--force_eval "$KCD_FORCE_EVAL"} \
    ${KCD_FORCE_BENCH:+--force_bench "$KCD_FORCE_BENCH"} \
    ${KCD_DISTRACTOR_CLASSES:+--distractor_classes "$KCD_DISTRACTOR_CLASSES"} \
    ${KCD_TILED_EVAL:+--tiled_eval "$KCD_TILED_EVAL"} \
    ${KCD_TILED_EVAL_WINDOW:+--tiled_eval_window "$KCD_TILED_EVAL_WINDOW"} \
    ${KCD_TILED_EVAL_OVERLAP:+--tiled_eval_overlap "$KCD_TILED_EVAL_OVERLAP"} \
    ${KCD_TILED_EVAL_NMS_THRESH:+--tiled_eval_nms_thresh "$KCD_TILED_EVAL_NMS_THRESH"} \
    ${KCD_EVAL_DEVICE:+--eval_device "$KCD_EVAL_DEVICE"} \
    ${KCD_WDS_SHARDS_DPATH:+--train_wds_shards_dpath "$KCD_WDS_SHARDS_DPATH"} \
    ${KCD_WDS_EPOCH_LENGTH:+--train_wds_epoch_length "$KCD_WDS_EPOCH_LENGTH"} \
    ${KCD_WDS_SOURCE_TO_TARGET:+--train_wds_source_to_target "$KCD_WDS_SOURCE_TO_TARGET"} \
    ${KCD_WDS_BUCKET_WEIGHTS_JSON:+--train_wds_bucket_weights "$KCD_WDS_BUCKET_WEIGHTS_JSON"} \
    ${KCD_WDS_SKIP_EMPTY:+--train_wds_skip_empty "$KCD_WDS_SKIP_EMPTY"} \
    --train_num_workers "${KCD_TRAIN_NUM_WORKERS:-4}" \
    --val_num_workers "${KCD_VAL_NUM_WORKERS:-2}" \
    --category_names "$KCD_CATEGORY_NAMES" \
    --lr "$KCD_LR" \
    --backbone_lr "$KCD_BACKBONE_LR" \
    --use_amp "$KCD_USE_AMP" \
    --scale_tier "$KCD_SCALE_TIER" \
    "${INIT_FLAG[@]}" \
    "${DIST_FLAG[@]}"

echo
echo "=== 4. Eligibility manifest ==="
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
