#!/usr/bin/env bash
# Score RF-DETR and DEIMv2 against the same ground truth, one protocol, and
# print the comparison.
#
# The point of this script is that BOTH numbers come out of the same
# `kwcoco eval` invocation shape -- same true bundle, same IoU threshold, same
# AP implementation. Every fish number quoted before this was measured under
# whatever protocol its own trainer happened to use, which is how the RF-DETR
# baseline ended up reporting 0.7166 on 4,000 chips carved out of its own
# training sequences.
#
# Runs in the kit's docker image (kwcoco lives there, not on the host).
#
# Usage:
#   bash projects/viame_fish_2026/scripts/score_headtohead.sh
#
# Prerequisites:
#   1. DEIMv2 already scored (its predictions are cached from the eval run).
#   2. run_rfdetr_on_test.sh has produced computed_detections.csv.
#   3. convert_viame_dets_to_kwcoco.py has turned that into a kwcoco bundle.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

RUN_NAME="${KCD_RUN_NAME:-fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen001}"
CANDIDATE_ID="${KCD_CANDIDATE_ID:-deimv2_dinov3_x_1024x1024_fixed}"
EVAL_ROOT="$KCD_RUNS_DPATH/$RUN_NAME/eval/$CANDIDATE_ID"

# The single shared ground truth. Built once by the DEIMv2 eval; both models
# are scored against this exact file.
TRUE_ZIP="${VF_TRUE_ZIP:-$EVAL_ROOT/true_bbox_only.kwcoco.zip}"
DEIM_PRED="${VF_DEIM_PRED:-$EVAL_ROOT/pred_boxes.kwcoco.zip}"
RFDETR_PRED="${VF_RFDETR_PRED:-$VF_KCD_ROOT/rfdetr_test_inference/rfdetr_test_preds.kwcoco.json}"

OUT_DPATH="${VF_HEADTOHEAD_DPATH:-$VF_KCD_ROOT/headtohead}"
IOU="${VF_EVAL_IOU:-0.5}"
IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-aiq}"

kcd_require_path "shared ground truth" "$TRUE_ZIP"
kcd_require_path "DEIMv2 predictions" "$DEIM_PRED"
kcd_require_path "RF-DETR predictions" "$RFDETR_PRED"

mkdir -p "$OUT_DPATH/deimv2" "$OUT_DPATH/rfdetr"

echo "=============================================================="
echo " head-to-head on the held-out test split"
echo "=============================================================="
echo "  ground truth: $TRUE_ZIP"
echo "  DEIMv2 preds: $DEIM_PRED"
echo "  RF-DETR preds:$RFDETR_PRED"
echo "  iou_thresh:   $IOU"
echo

DOCKER_ARGS=(
    run --rm
    -v "$KCD_DATA_ROOT:$KCD_DATA_ROOT"
    -v "$VF_SSD_ROOT:$VF_SSD_ROOT"
    "$IMAGE"
)

score_one() {
    local label="$1" preds="$2" outdir="$3"
    echo "=== scoring $label ==="
    docker "${DOCKER_ARGS[@]}" python -m kwcoco eval \
        --true_dataset "$TRUE_ZIP" \
        --pred_dataset "$preds" \
        --out_dpath "$outdir" \
        --out_fpath "$outdir/detect_metrics.json" \
        --draw False \
        --iou_thresh "$IOU" \
        || echo "  kwcoco eval exited non-zero; checking for metrics anyway" >&2
    kcd_require_path "$label metrics" "$outdir/detect_metrics.json"
    echo
}

# DEIMv2's metrics already exist from its eval run; recompute here anyway so
# both sides are produced by an identical command rather than trusting that a
# previous invocation used the same settings.
score_one "DEIMv2"  "$DEIM_PRED"   "$OUT_DPATH/deimv2"
score_one "RF-DETR" "$RFDETR_PRED" "$OUT_DPATH/rfdetr"

echo "=============================================================="
python3 - "$OUT_DPATH/deimv2/detect_metrics.json" "$OUT_DPATH/rfdetr/detect_metrics.json" "$IOU" <<'PYEOF'
import json, pathlib, sys

def read(fpath):
    d = json.loads(pathlib.Path(fpath).read_text())
    key = next((k for k in d if k.startswith('area_range=all')), None)
    m = d[key]['nocls_measures']
    return {'ap': m.get('ap'), 'auc': m.get('auc'),
            'realpos': m.get('realpos_total'), 'nsupport': m.get('nsupport')}

deim, rfdetr, iou = read(sys.argv[1]), read(sys.argv[2]), sys.argv[3]
print(f' head-to-head: box AP @ IoU={iou}, single class `fish`,')
print(' held-out Test/ split, identical ground truth and protocol')
print('=' * 62)
print(f'  {"model":<12}{"AP":>10}{"AUC":>10}{"predictions":>16}')
print('  ' + '-' * 46)
for name, r in (('DEIMv2', deim), ('RF-DETR', rfdetr)):
    print(f'  {name:<12}{r["ap"]:>10.4f}{r["auc"]:>10.4f}{int(r["nsupport"]):>16,}')
print('  ' + '-' * 46)
if deim['ap'] is not None and rfdetr['ap'] is not None:
    delta = deim['ap'] - rfdetr['ap']
    winner = 'DEIMv2' if delta > 0 else 'RF-DETR'
    print(f'  delta: {abs(delta):.4f} AP in favour of {winner}')
if deim['realpos'] != rfdetr['realpos']:
    print(f'\n  WARNING: the two evals saw different numbers of true positives '
          f'({deim["realpos"]} vs {rfdetr["realpos"]}). They are NOT scoring the '
          f'same ground truth; the comparison is invalid.')
else:
    print(f'\n  both scored against {int(deim["realpos"]):,} ground-truth boxes.')
PYEOF
echo "=============================================================="
echo "  metrics: $OUT_DPATH/{deimv2,rfdetr}/detect_metrics.json"
