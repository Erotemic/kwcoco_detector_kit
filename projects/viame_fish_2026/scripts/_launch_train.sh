#!/usr/bin/env bash
# In-container entry point for a fish DEIMv2 training run.
#
# Invoked by the shared _sbatch_train.sh, which resolves it as
# "$KCD_REPO_ROOT/scripts/${KCD_LAUNCH_SCRIPT:-_launch_train.sh}". With
# KCD_REPO_ROOT pointing at projects/viame_fish_2026 this file is what runs.
#
# Deliberately much shorter than the sea-lion launcher, because this project
# skips two of its three data stages:
#
#   no tiling        box percentiles are p1 42x44, p50 150x109 on 1920x1200
#                    imagery, so whole frames resized to the model input keep
#                    even the smallest boxes well above threshold. Tiling would
#                    multiply data volume and epoch time to fix a problem this
#                    corpus does not have.
#   no scheme step   the model is single-class `fish`; there is no category
#                    collapse to apply at launch time because the corpus
#                    labels.txt already did it during prep.
#   no balancing     with one category there is nothing to balance.
#
# So this reduces to: verify the bundles, then hand them to the kit's sweep CLI.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

: "${KCD_RUN_NAME:?_launch_train.sh: KCD_RUN_NAME must be set}"
: "${KCD_VARIANT:?_launch_train.sh: KCD_VARIANT must be set}"
: "${KCD_NUM_GPUS:?_launch_train.sh: KCD_NUM_GPUS must be set}"

KCD_ROOT="${KCD_ROOT:-$KCD_RUNS_DPATH/$KCD_RUN_NAME}"
mkdir -p "$KCD_ROOT"

echo "=============================================================="
echo " fish DEIMv2 training: $KCD_RUN_NAME"
echo "=============================================================="
echo "  variant:   $KCD_VARIANT"
echo "  gpus:      $KCD_NUM_GPUS"
echo "  input_hw:  ${KCD_INPUT_HW:-?}"
echo "  epochs:    ${KCD_NUM_EPOCHS:-?}"
echo "  kcd_root:  $KCD_ROOT"
echo

kcd_require_path "train bundle" "$KCD_TRAIN_KWCOCO"
kcd_require_path "vali bundle"  "$KCD_VALI_KWCOCO"
kcd_require_path "test bundle"  "$KCD_TEST_KWCOCO"

# Init checkpoint. Resolved through the same helper the host-side pre-flight
# used, so the two can never disagree about which file a variant means.
INIT_FLAG=()
KCD_INIT_CHECKPOINT="$(kcd_resolve_init_checkpoint "$KCD_VARIANT")"
if [ -n "$KCD_INIT_CHECKPOINT" ]; then
    kcd_require_path "init checkpoint" "$KCD_INIT_CHECKPOINT"
    INIT_FLAG=(--init_checkpoint "$KCD_INIT_CHECKPOINT")
    echo "  init:      $KCD_INIT_CHECKPOINT"
else
    echo "  init:      <from scratch>"
fi

# Report what we are actually about to train on. Cheap, and it is the check
# that would have caught the baseline run's identical valid/test bundles.
"$PYTHON_BIN" - <<PYEOF
import json
for label, path in [('train', '$KCD_TRAIN_KWCOCO'),
                    ('vali',  '$KCD_VALI_KWCOCO'),
                    ('test',  '$KCD_TEST_KWCOCO')]:
    with open(path) as file:
        dset = json.load(file)
    names = {v['name'] for v in dset.get('videos', [])}
    print('  {:<6} sequences={:<5} images={:<9,} annotations={:<9,} categories={}'.format(
        label, len(names), len(dset['images']), len(dset['annotations']),
        [c['name'] for c in dset['categories']]))
    globals().setdefault('seqs', {})[label] = names

overlap = seqs['train'] & seqs['vali']
assert not overlap, 'train/vali share sequences: {}'.format(sorted(overlap)[:5])
for split in ('train', 'vali'):
    leak = seqs[split] & seqs['test']
    assert not leak, '{}/test share sequences: {}'.format(split, sorted(leak)[:5])
print('  split disjointness: OK')
PYEOF
echo

# Resolve KCD_RESUME_CKPT (default "auto"; see vf_resolve_resume_ckpt).
#
# The workdir is created by `sweep` and named after the candidate id, so on a
# first run there is nothing here and this correctly resolves to empty. On a
# re-run after a kill it finds the newest checkpoint and continues from it
# rather than silently restarting -- which is what happened when job 299 was
# launched into gen001's directory.
KCD_RESUME_CKPT_REQUESTED="${KCD_RESUME_CKPT-auto}"
_RESUME_WORKDIR=""
if [ -d "$KCD_ROOT/runs" ]; then
    _RESUME_WORKDIR="$(find "$KCD_ROOT/runs" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | head -n1)"
fi
KCD_RESUME_CKPT="$(vf_resolve_resume_ckpt "$_RESUME_WORKDIR")"
if [ -n "$KCD_RESUME_CKPT" ]; then
    kcd_require_path "resume checkpoint" "$KCD_RESUME_CKPT"
    echo "  resume:    $KCD_RESUME_CKPT"
    echo "             (KCD_RESUME_CKPT=${KCD_RESUME_CKPT_REQUESTED:-auto}; set it to"
    echo "              'fresh' to ignore existing checkpoints and start over)"
else
    echo "  resume:    <none> -- training from the init checkpoint"
fi
echo

TOTAL_BATCH=$(( KCD_NUM_GPUS * ${KCD_PER_GPU_BATCH:-4} ))
TOTAL_VAL_BATCH=$(( TOTAL_BATCH * ${KCD_VAL_BATCH_MULT:-1} ))

DIST_FLAG=(--num_gpus "$KCD_NUM_GPUS")
[ "$KCD_NUM_GPUS" -gt 1 ] && DIST_FLAG+=(--distributed true)

set -x
"$PYTHON_BIN" -m kwcoco_detector_kit sweep \
    --train_kwcoco "$KCD_TRAIN_KWCOCO" \
    --vali_kwcoco  "$KCD_VALI_KWCOCO" \
    --test_kwcoco  "$KCD_TEST_KWCOCO" \
    --kcd_root "$KCD_ROOT" \
    --trainer deimv2 \
    --variant "$KCD_VARIANT" \
    --input_hw "$KCD_INPUT_HW" \
    --train_policy "${KCD_TRAIN_POLICY:-fixed}" \
    --num_epochs "$KCD_NUM_EPOCHS" \
    --batch_size "$TOTAL_BATCH" \
    --val_batch_size "$TOTAL_VAL_BATCH" \
    --category_names "${KCD_CATEGORY_NAMES:-fish}" \
    --lr "$KCD_LR" \
    --backbone_lr "$KCD_BACKBONE_LR" \
    --use_amp "${KCD_USE_AMP:-true}" \
    --train_num_workers "${KCD_TRAIN_NUM_WORKERS:-8}" \
    --val_num_workers "${KCD_VAL_NUM_WORKERS:-4}" \
    ${KCD_DO_EXPORT:+--do_export "$KCD_DO_EXPORT"} \
    ${KCD_DO_EVAL:+--do_eval "$KCD_DO_EVAL"} \
    ${KCD_DO_BENCH:+--do_bench "$KCD_DO_BENCH"} \
    ${KCD_RESUME_CKPT:+--resume "$KCD_RESUME_CKPT"} \
    ${KCD_EVAL_DEVICE:+--eval_device "$KCD_EVAL_DEVICE"} \
    "${INIT_FLAG[@]}" \
    "${DIST_FLAG[@]}"
set +x

"$PYTHON_BIN" -m kwcoco_detector_kit manifest \
    --auto \
    --kcd_root "$KCD_ROOT" \
    --out      "$KCD_ROOT/manifest.tsv" \
    --out_json "$KCD_ROOT/manifest.json" \
    --allow_missing_desktop_bench true

echo
echo "=== done: $KCD_ROOT ==="
