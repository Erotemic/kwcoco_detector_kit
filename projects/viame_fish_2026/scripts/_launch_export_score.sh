#!/usr/bin/env bash
# In-container entry point for exporting and scoring an EXISTING checkpoint.
#
# Selected by setting KCD_LAUNCH_SCRIPT=_launch_export_score.sh, so this reuses
# the whole submit -> sbatch -> docker chain (GPU pinning by UUID, zombie
# container traps, leak detection) rather than re-implementing it.
#
# ## Why this exists separately from the sweep
#
# `kwcoco_detector_kit sweep` deliberately refuses to skip training unless the
# workdir carries a `.train_complete` marker, which is written only when
# training returns cleanly. A checkpoint left behind by a killed run therefore
# triggers a full retrain instead of an evaluation -- and that guard is right:
# evaluating a model whose training died for unknown reasons is how bad numbers
# get published.
#
# But gen001's training did not die for unknown reasons. It deadlocked in an
# NCCL all-reduce at epoch 13 (all four ranks stopped in the same frame; see
# docs/journals/2026-08-17_deim_gen001_deadlock_at_epoch13.md), while epoch 12's
# evaluation had completed normally and selected best_stg2.pth at vali AP
# 0.5440. The weights are sound; only the schedule was cut short.
#
# The honest way to act on that is to export and score the checkpoint through a
# path that never claims training finished -- NOT to fabricate the marker. This
# script touches no completion state, so `sweep` still sees an incomplete run
# and will still resume/retrain when asked. The two views stay consistent.
#
# ## What it does
#
#   1. eval  -- the model against the HELD-OUT test bundle
#   2. export -- ONNX + modelspec from the same checkpoint
#   3. bench  -- latency of the exported graph (best effort)
#
# Eval runs FIRST on purpose, mirroring the sweep's own ordering: eval loads the
# .pth directly, while ONNX export can fail on deploy-only bugs. The science
# must not be blocked by the deploy artifact.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

: "${KCD_RUN_NAME:?_launch_export_score.sh: KCD_RUN_NAME must be set}"

KCD_ROOT="${KCD_ROOT:-$KCD_RUNS_DPATH/$KCD_RUN_NAME}"
# The sweep lays out one workdir per candidate under $KCD_ROOT/runs/.
CANDIDATE_ID="${KCD_CANDIDATE_ID:-}"
if [ -z "$CANDIDATE_ID" ]; then
    mapfile -t _cands < <(find "$KCD_ROOT/runs" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null | sort)
    if [ "${#_cands[@]}" -ne 1 ]; then
        echo "ERROR: expected exactly one candidate under $KCD_ROOT/runs, found ${#_cands[@]}." >&2
        printf '  %s\n' "${_cands[@]:-<none>}" >&2
        echo "  Set KCD_CANDIDATE_ID to pick one." >&2
        exit 1
    fi
    CANDIDATE_ID="${_cands[0]}"
fi
WORKDIR="$KCD_ROOT/runs/$CANDIDATE_ID"

echo "=============================================================="
echo " export + score an existing checkpoint"
echo "=============================================================="
echo "  run:        $KCD_RUN_NAME"
echo "  candidate:  $CANDIDATE_ID"
echo "  workdir:    $WORKDIR"
echo "  test bundle:$KCD_TEST_KWCOCO"
echo

kcd_require_path "workdir" "$WORKDIR"
kcd_require_path "policy.json" "$WORKDIR/policy.json"
kcd_require_path "test bundle" "$KCD_TEST_KWCOCO"

# State the training situation plainly in the log rather than hiding it.
if [ -f "$WORKDIR/.train_complete" ]; then
    echo "  training:   COMPLETE (marker present)"
else
    echo "  training:   INCOMPLETE -- no .train_complete marker."
    echo "              Exporting a checkpoint from a run that did not finish."
    echo '              This script does not create that marker; sweep will'
    echo '              still treat the run as resumable.'
fi
echo

# Image capability check. The aiq image in use as of 2026-08-17 bakes kit
# e48ca1dab2a4 (2026-06-21), which predates the `export-onnx` and `bench`
# subcommands -- they appear in its --help text but were never registered. Eval
# still works there (run_kwcoco_eval is a module function and its signature is
# unchanged), so a stale image yields metrics but no deployable artifact.
# Say which situation we are in up front rather than discovering it three
# stages later.
HAVE_EXPORT_CLI=0
if "$PYTHON_BIN" -m kwcoco_detector_kit export-onnx --help >/dev/null 2>&1; then
    HAVE_EXPORT_CLI=1
    echo "  image:      export-onnx available"
else
    echo "  image:      STALE -- no export-onnx subcommand." >&2
    echo "              Eval will still run and produce metrics, but there will be" >&2
    echo "              no ONNX artifact. Rebuild the image for a deliverable:" >&2
    echo "                bash docker/opengroundingdino/build_aiq_cuda132_blackwell.sh" >&2
fi
echo

# ------------------------------------------------------------------ 1. eval
# Scores the HELD-OUT test bundle -- sequences neither this model nor the
# RF-DETR baseline has seen. Uses the sweep's own eval entry point so the
# protocol is identical to what a completed run would have produced.
echo "=== [1/3] eval on the held-out test split ==="
"$PYTHON_BIN" - <<PYEOF
import json, pathlib
# Importing the package registers the trainer plugins (deimv2 among them);
# the factory itself lives in _registry, not the package root.
import kwcoco_detector_kit.trainers  # noqa: F401
from kwcoco_detector_kit.trainers._registry import get_trainer
from kwcoco_detector_kit.eval.kwcoco_eval import run_kwcoco_eval

workdir = pathlib.Path("$WORKDIR")
policy = json.loads((workdir / "policy.json").read_text())
category_names = policy["category_names"]
print("  category_names from policy.json:", category_names)

trainer = get_trainer("deimv2")
out = run_kwcoco_eval(
    trainer=trainer,
    workdir=workdir,
    test_kwcoco="$KCD_TEST_KWCOCO",
    kcd_root=pathlib.Path("$KCD_ROOT"),
    candidate_id="$CANDIDATE_ID",
    category_names=category_names,
    force=bool(int("${KCD_FORCE_EVAL:-0}")),
    tiled_eval=${KCD_TILED_EVAL:-False},
    device="${KCD_EVAL_DEVICE:-cuda}",
    read_workers=int("${KCD_EVAL_READ_WORKERS:-4}"),
)
print("  wrote", out)
PYEOF
echo

# ---------------------------------------------------------------- 2. export
# Reads variant, input size and category_names from policy.json, and resolves
# the checkpoint via the trainer -- no completion state consulted.
echo "=== [2/3] ONNX export ==="
if [ "$HAVE_EXPORT_CLI" = "0" ]; then
    echo "  SKIPPED: this image has no export-onnx subcommand (see above)." >&2
else
"$PYTHON_BIN" -m kwcoco_detector_kit export-onnx "$WORKDIR" \
    ${KCD_FORCE_EXPORT:+--force} \
    ${KCD_EXPORT_SCORE_THRESH:+--score_thresh "$KCD_EXPORT_SCORE_THRESH"} \
    || echo "  WARNING: ONNX export failed; the eval metrics above still stand." >&2
fi
echo

# ----------------------------------------------------------------- 3. bench
echo "=== [3/3] bench (best effort; needs the .onnx) ==="
"$PYTHON_BIN" - <<PYEOF || echo "  WARNING: bench skipped/failed (non-fatal)." >&2
import pathlib
from kwcoco_detector_kit.eval.bench import run_onnx_bench
print("  wrote", run_onnx_bench(workdir=pathlib.Path("$WORKDIR")))
PYEOF

echo
echo "=============================================================="
echo " done"
echo "=============================================================="
echo "  metrics: $KCD_ROOT/eval/$CANDIDATE_ID/eval/detect_metrics.json"
ls -la "$WORKDIR"/*.onnx "$WORKDIR"/*modelspec* 2>/dev/null || true
