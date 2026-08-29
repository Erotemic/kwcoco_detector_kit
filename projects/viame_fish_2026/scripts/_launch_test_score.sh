#!/usr/bin/env bash
# In-container: score each run's vali-selected checkpoint on the TEST split
# under the same true-tiled protocol used for vali.
#
# Runs all four generations so the tile-trained models (gen006, gen007) and the
# whole-frame models (gen001, gen003) are measured the same way. The existing
# gen001/gen003 test numbers are WHOLE-IMAGE and predate true-tiled inference,
# so they cannot be compared against a true-tiled gen006 number -- rescoring
# them here is what makes the comparison mean anything.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
: "${KCD_TEST_KWCOCO:?_launch_test_score.sh: missing KCD_TEST_KWCOCO}"
: "${KCD_RUNS_DPATH:?_launch_test_score.sh: missing KCD_RUNS_DPATH}"

WINDOW="${KCD_TILED_EVAL_WINDOW:-$KCD_TILE_SIZE_ONDISK}"
OVERLAP="${KCD_TILED_EVAL_OVERLAP:-0.25}"
OUT_ROOT="${KCD_TEST_SCORE_OUT:-$KCD_TILE_DPATH/../test_score}"
# Written by score_epochs.py; supplies the per-run checkpoint choice.
VALI_SUMMARY="${KCD_VALI_SUMMARY:-$KCD_TILE_DPATH/../baseline_vali/summary_w${WINDOW}_o${OVERLAP}_bf16_s${KCD_VALI_SUMMARY_STRIDE:-8}.json}"

# Frozen and explicit: unset it would default to fp16 here while every vali
# number this is compared against was measured in bf16.
export KCD_AMP_DTYPE="${KCD_AMP_DTYPE:-bfloat16}"

echo "=============================================================="
echo " TEST scoring -- true-tiled ${WINDOW}px"
echo "=============================================================="
echo "  test:      $KCD_TEST_KWCOCO"
echo "  window:    ${WINDOW} px source, overlap ${OVERLAP}, keep_full"
echo "  amp:       ${KCD_AMP_DTYPE}"
echo "  selection: $VALI_SUMMARY"
echo "  runs:      ${KCD_TEST_SCORE_RUNS}"
echo "  out:       $OUT_ROOT"
echo

if [ ! -f "$VALI_SUMMARY" ]; then
    echo "ERROR: vali ranking summary not found: $VALI_SUMMARY" >&2
    echo "  Checkpoints are chosen on VALI, never here. Run:" >&2
    echo "    bash $SCRIPT_DIR/submit_baseline_vali.sh" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"
export KCD_TEST_SCORE_OUT="$OUT_ROOT"
export KCD_VALI_SUMMARY="$VALI_SUMMARY"
export KCD_TILED_EVAL_WINDOW="$WINDOW"
export KCD_TILED_EVAL_OVERLAP="$OVERLAP"
exec "$PYTHON_BIN" "$SCRIPT_DIR/score_test_once.py"
