#!/usr/bin/env bash
# End-to-end demo of the gen002 WebDataset training pipeline using
# kwcoco demo data. Exercises every kit-side stage that a real
# training run hits:
#
#   1. Source kwcoco: generated via kwcoco demo (shapes-8 + vidshapes).
#   2. WebDataset shards: built via build_detection_webdataset CLI.
#   3. Sweep: kit's pareto_sweep runs the DEIMv2 trainer end-to-end
#      with WDS input — train + export + eval + bench.
#   4. Manifest: kit's eligibility manifest TSV is generated.
#
# Intentionally tiny: 3 epochs, batch=2, 2 train workers. Should
# complete in 3-8 minutes on a CPU box, faster on GPU. The point
# is to catch ANY breakage in the WDS path before committing to a
# slurm run.
#
# === Host venv (default) ===
#   bash dev/wds_e2e_demo/run_demo.sh
#
# === Docker image ===
#   docker run --rm --gpus all \
#       -v "$(pwd)":/work -w /work \
#       -e PYTHON_BIN=/opt/venv/bin/python \
#       -e DEMO_OUT=/tmp/wds_e2e_demo \
#       kwcoco-detector-kit:ogdino-auto \
#       bash dev/wds_e2e_demo/run_demo.sh
#
# === Common knobs (set as env vars) ===
#   PYTHON_BIN       python to invoke (auto-detected if unset)
#   DEMO_OUT         output root (default /tmp/wds_e2e_demo)
#   DEMO_KEEP=1      reuse existing source kwcoco + shards
#   DEMO_BATCH       train batch size (default 2)
#   DEMO_EPOCHS      epochs (default 3)
#   DEMO_AMP=true    enable mixed precision
#   DEMO_INPUT_HW    e.g. "[256, 256]" (default "[160, 160]")
#   DEMO_TRAIN_WORKERS  dataloader workers (default 2)
#
# === On GPU host (e.g. yardrat / 3090) ===
#   DEMO_BATCH=8 DEMO_AMP=true DEMO_EPOCHS=5 \
#       DEMO_INPUT_HW="[256, 256]" \
#       bash dev/wds_e2e_demo/run_demo.sh
#
# The demo trains from scratch (no init_checkpoint) so we don't
# need pretrained weights on disk; AP will be ~0 but that's not
# what we're testing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DEMO_OUT="${DEMO_OUT:-/tmp/wds_e2e_demo}"
DEMO_KEEP="${DEMO_KEEP:-0}"

# Auto-detect python: prefer host kit venv; fall back to docker image's
# /opt/venv; fall back to PATH python3. Override via PYTHON_BIN env.
if [ -z "${PYTHON_BIN:-}" ]; then
    for cand in "$REPO_ROOT/.venv/bin/python" /opt/venv/bin/python "$(command -v python3 2>/dev/null || true)"; do
        if [ -x "$cand" ]; then
            PYTHON_BIN="$cand"
            break
        fi
    done
fi
echo "PYTHON_BIN=$PYTHON_BIN"

# Verify the python has the kit + WDS reader installed. Fails fast
# if the venv is stale (common after a git checkout without uv sync).
if ! "$PYTHON_BIN" -c "import kwcoco_detector_kit, kwcoco_dataloader, torch; print('  kit:', kwcoco_detector_kit.__file__); print('  wds reader:', kwcoco_dataloader.__file__); print('  torch:', torch.__version__)" 2>&1; then
    echo "ERROR: $PYTHON_BIN is missing kit deps. Run 'uv sync' or use the docker image (see header)." >&2
    exit 1
fi

# Tunable training knobs (override via env for faster GPU iteration):
DEMO_EPOCHS="${DEMO_EPOCHS:-3}"
DEMO_BATCH="${DEMO_BATCH:-2}"
DEMO_VAL_BATCH="${DEMO_VAL_BATCH:-2}"
DEMO_INPUT_HW="${DEMO_INPUT_HW:-[320, 320]}"
DEMO_AMP="${DEMO_AMP:-false}"
DEMO_LR="${DEMO_LR:-1e-3}"
DEMO_BACKBONE_LR="${DEMO_BACKBONE_LR:-1e-4}"
DEMO_TRAIN_WORKERS="${DEMO_TRAIN_WORKERS:-2}"
DEMO_VAL_WORKERS="${DEMO_VAL_WORKERS:-1}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: $PYTHON_BIN not executable. Set PYTHON_BIN= to the venv python." >&2
    exit 1
fi

if [ "$DEMO_KEEP" != "1" ]; then
    echo "=== cleaning $DEMO_OUT ==="
    rm -rf "$DEMO_OUT"
fi
mkdir -p "$DEMO_OUT"

SRC_DPATH="$DEMO_OUT/src"
SHARDS_DPATH="$DEMO_OUT/shards"
KCD_ROOT="$DEMO_OUT/kcd_root"
mkdir -p "$SRC_DPATH" "$KCD_ROOT"

# ---------- 1. source kwcoco (shapes8 + vidshapes split into train/vali/test)
echo
echo "=== 1. source kwcoco ==="
if [ "$DEMO_KEEP" = "1" ] && [ -f "$SRC_DPATH/train.kwcoco.json" ]; then
    echo "  reusing existing source kwcoco at $SRC_DPATH"
else
    "$PYTHON_BIN" - <<PYEOF
"""Generate train/vali/test kwcoco bundles from demo data.

Each sample's annotation gets a 'source_category' field stamped onto
it (same shape the kit's tile.py produces) — that's what the WDS
writer reads to bucket shards.
"""
from pathlib import Path
import kwcoco
import shutil

src_dpath = Path("$SRC_DPATH")
src_dpath.mkdir(parents=True, exist_ok=True)

def materialize(name, demo_key, asset_subdir):
    bundle_dpath = src_dpath / asset_subdir
    bundle_dpath.mkdir(parents=True, exist_ok=True)
    dset = kwcoco.CocoDataset.demo(demo_key)
    dset.reroot(absolute=True)
    out_fpath = src_dpath / f"{name}.kwcoco.json"
    dset.fpath = str(out_fpath)
    cat_id_to_name = {c['id']: c['name'] for c in dset.dataset['categories']}
    for ann in dset.dataset['annotations']:
        ann['source_category'] = cat_id_to_name[ann['category_id']]
    dset.dump()
    return dset

train = materialize('train', 'shapes16', 'train_assets')
vali = materialize('vali', 'shapes4', 'vali_assets')
test = materialize('test', 'shapes4', 'test_assets')

print(f"  train: {train.n_images} images, {train.n_annots} annotations")
print(f"  vali:  {vali.n_images} images, {vali.n_annots} annotations")
print(f"  test:  {test.n_images} images, {test.n_annots} annotations")
print(f"  categories: {[c['name'] for c in train.dataset['categories']]}")
PYEOF
fi

TRAIN_KWCOCO="$SRC_DPATH/train.kwcoco.json"
VALI_KWCOCO="$SRC_DPATH/vali.kwcoco.json"
TEST_KWCOCO="$SRC_DPATH/test.kwcoco.json"

# ---------- 2. WebDataset shards
echo
echo "=== 2. WebDataset shards ==="
if [ "$DEMO_KEEP" = "1" ] && [ -d "$SHARDS_DPATH" ] && \
   [ -n "$(find "$SHARDS_DPATH" -name '*.tar' -print -quit 2>/dev/null)" ]; then
    echo "  reusing shards at $SHARDS_DPATH"
else
    rm -rf "$SHARDS_DPATH"
    "$PYTHON_BIN" -m kwcoco_dataloader.cli.build_detection_webdataset \
        --in_fpath  "$TRAIN_KWCOCO" \
        --out_dpath "$SHARDS_DPATH" \
        --bucket_attr source_category \
        --maxcount 4 \
        --maxsize_mb 1024 \
        --jpeg_quality 80 \
        --no-progress 2>&1 | tail -10
    echo "  shard tree:"
    find "$SHARDS_DPATH" -maxdepth 2 -type f -name '*.tar' | sort | sed 's/^/    /'
fi

# Build the source_to_target JSON for sweep. Demo data has classes
# named "star", "circle", etc.; we collapse them all to one "object"
# class so the sweep's category dimension is trivial.
CATEGORY_NAMES="object"
SRC_TO_TGT_JSON="$("$PYTHON_BIN" -c '
import json, kwcoco
ds = kwcoco.CocoDataset("'"$TRAIN_KWCOCO"'")
cats = [c["name"] for c in ds.dataset["categories"]]
print(json.dumps({c: "object" for c in cats}))
')"
echo "  source_to_target: $SRC_TO_TGT_JSON"

# ---------- 3. sweep (train + export + eval + bench)
echo
echo "=== 3. sweep (train + export + eval + bench) ==="
"$PYTHON_BIN" -m kwcoco_detector_kit sweep \
    --train_kwcoco "$TRAIN_KWCOCO" \
    --vali_kwcoco  "$VALI_KWCOCO" \
    --test_kwcoco  "$TEST_KWCOCO" \
    --kcd_root "$KCD_ROOT" \
    --trainer deimv2 \
    --variant deimv2_hgnetv2_n \
    --input_hw "$DEMO_INPUT_HW" \
    --train_policy fixed \
    --num_epochs "$DEMO_EPOCHS" \
    --batch_size "$DEMO_BATCH" \
    --val_batch_size "$DEMO_VAL_BATCH" \
    --train_wds_shards_dpath "$SHARDS_DPATH" \
    --train_wds_source_to_target "$SRC_TO_TGT_JSON" \
    --train_num_workers "$DEMO_TRAIN_WORKERS" \
    --val_num_workers "$DEMO_VAL_WORKERS" \
    --category_names "$CATEGORY_NAMES" \
    --lr "$DEMO_LR" \
    --backbone_lr "$DEMO_BACKBONE_LR" \
    --use_amp "$DEMO_AMP" \
    --scale_tier S \
    --num_gpus 1

# ---------- 4. eligibility manifest
echo
echo "=== 4. eligibility manifest ==="
"$PYTHON_BIN" -m kwcoco_detector_kit manifest \
    --auto \
    --kcd_root "$KCD_ROOT" \
    --out      "$KCD_ROOT/manifest.tsv"

# ---------- summary
echo
echo "=== demo done ==="
echo "  output:    $KCD_ROOT"
echo "  manifest:  $KCD_ROOT/manifest.tsv"
echo
echo "  recent log lines:"
find "$KCD_ROOT" -name 'log.txt' -exec tail -3 {} \; 2>/dev/null | sed 's/^/    /' || true
echo
echo "If you got here without an error, every WDS-path stage works."
