#!/usr/bin/env bash
# Complete the record: rank gen006 on vali, then score all four generations on
# the test split under one protocol.
#
#   bash projects/viame_fish_2026/scripts/run_test_scores_end_to_end.sh
#
# Run from a tmux pane; both phases are foreground.
#
# ## Why two phases
#
# gen006 is the only run with staged epochs and no vali ranking -- it was left
# out of the gen007 scoring pass. Its `best_stg1.pth` is whatever DEIMv2's
# in-loop TILE-level validation preferred, and gen007 demonstrated those two
# criteria disagree: in-loop chose epoch 27, deployment geometry chose epoch 6.
# Picking a gen006 checkpoint by looking at test scores would make every number
# here meaningless, so phase 1 ranks it on vali and phase 2 just reports.
#
# gen001, gen003 and gen007 already have vali rankings; phase 1 only adds
# gen006 and merges into the existing summary.
#
# ## What phase 2 answers
#
# On vali, under true-tiled deployment geometry, the WHOLE-FRAME runs beat the
# TILE-trained ones:
#
#   gen003 (whole frame)  0.7689     gen007 (tiles)  0.7311  epoch 6
#   gen001 (whole frame)  0.7658
#
# Phase 2 puts the held-out split next to that. The existing gen001/gen003 test
# numbers are WHOLE-IMAGE and predate true-tiled inference, so they are
# rescored rather than quoted -- comparing across protocols is the mistake this
# project already made once with RF-DETR.
#
# ## Cost and controls
#
#   phase 1   14 gen006 checkpoints on stride-8 vali   ~2 h
#   phase 2   4 checkpoints on the FULL test split     ~2 h
#
#   KCD_SKIP_VALI=1   gen006 already ranked
#   KCD_SKIP_TEST=1   ranking only
#   KCD_YES=1         skip the abort window
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

GEN006=fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen006_final
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DPATH="${KCD_E2E_LOG_DPATH:-$KCD_SLURM_LOG_DPATH}/test_scores_$STAMP"
mkdir -p "$LOG_DPATH"

if docker info >/dev/null 2>&1; then
    export KCD_DOCKER_CMD="docker"
elif sudo -n true 2>/dev/null; then
    export KCD_DOCKER_CMD="sudo docker"
else
    echo "ERROR: docker needs elevation and passwordless sudo is unavailable." >&2
    exit 1
fi

phase() {
    local name="$1"; shift
    local log="$LOG_DPATH/$name.log"
    echo
    echo "=============================================================="
    echo " PHASE $name  ($(date -Is))"
    echo " log: $log"
    echo "=============================================================="
    set +e
    ( "$@" ) 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -ne 0 ]; then
        echo; echo "PHASE $name FAILED (exit $rc). Log: $log" >&2
        exit "$rc"
    fi
    echo "PHASE $name OK  ($(date -Is))"
}

WINDOW="${KCD_TILED_EVAL_WINDOW:-$KCD_TILE_SIZE_ONDISK}"
SUMMARY="$KCD_TILE_DPATH/../baseline_vali/summary_w${WINDOW}_o0.25_bf16_s8.json"

rank_gen006_on_vali() {
    # Only gen006. score_epochs.py merges into the existing summary, so the
    # gen001/gen003/gen007 rows already there are carried forward rather than
    # recomputed -- that pass took five hours.
    KCD_NO_SLURM=1 \
    KCD_EVAL_STRIDE=8 \
    KCD_RUN_NAME="fishtrack23_vali_1229_s8_gen006" \
    KCD_BASELINE_RUNS="$GEN006" \
        bash "$SCRIPT_DIR/submit_baseline_vali.sh"
}

score_all_on_test() {
    KCD_NO_SLURM=1 bash "$SCRIPT_DIR/submit_test_score.sh"
}

echo "=============================================================="
echo " test scores, end to end"
echo "=============================================================="
echo "  logs:    $LOG_DPATH"
echo "  summary: $SUMMARY"
echo "  phases:  vali=$([ "${KCD_SKIP_VALI:-0}" = 1 ] && echo skip || echo yes)" \
     "test=$([ "${KCD_SKIP_TEST:-0}" = 1 ] && echo skip || echo yes)"
echo "  expect:  ~4 h total"
echo

if [ "${KCD_YES:-0}" != "1" ]; then
    echo "Starting in 10s. Ctrl-C to abort."
    sleep 10
fi

if [ "${KCD_SKIP_VALI:-0}" != "1" ]; then
    phase 1-vali-gen006 rank_gen006_on_vali
fi

# Cheap guard: phase 2 refuses on a missing entry anyway, but failing here
# names the cause instead of surfacing it four checkpoints later.
if [ -f "$SUMMARY" ] && ! grep -q "$GEN006" "$SUMMARY"; then
    echo "ERROR: $GEN006 still has no row in $SUMMARY." >&2
    echo "  Re-run without KCD_SKIP_VALI=1." >&2
    exit 1
fi

if [ "${KCD_SKIP_TEST:-0}" != "1" ]; then
    phase 2-test score_all_on_test
fi

echo
echo "=============================================================="
echo " done  ($(date -Is))"
echo "=============================================================="
echo "  logs:    $LOG_DPATH"
echo "  vali:    $SUMMARY"
echo "  test:    $KCD_TILE_DPATH/../test_score/"
echo
echo "  Journal the test table next to the vali table. The point of the record"
echo "  is the gap between what tiling was predicted to do and what it did."
