#!/usr/bin/env bash
# In-container: score the existing checkpoints on the FULL vali split under the
# frozen final-inference protocol, to establish the baseline `B` that gen006
# must beat.
#
# ## Why this exists
#
# Every comparison this project has made so far used either whole-image vali
# (which measures a tile-trained model at the wrong object scale) or the
# held-out test split (which is not ours to spend on decisions). `B` is the
# number gen006's success criterion is defined against, and it has to be fixed
# and recorded BEFORE gen006 runs, or the criterion is retrospective.
#
# ## The frozen protocol
#
#   true tiled, source window = $KCD_TILED_EVAL_WINDOW (1229 px)
#   overlap 0.25, keep_full, cross-window NMS 0.5
#   full 46-sequence vali split
#   each checkpoint under ITS OWN preprocessing contract
#
# That last point is automatic, not a flag: DEIMv2Predictor recovers mean/std
# from the generated train.yml of the run being scored. gen001 and gen003
# trained before DINO normalization was added, so their configs carry no
# Normalize op and they are scored unnormalized -- correct for them. gen006
# will carry one and be scored with it.
#
# This is a deliberate confound. It prevents attributing gen006's delta to any
# single change, but it is the right comparison: the question is whether the
# new recipe yields a better DEPLOYABLE detector, and preprocessing is part of
# the recipe. Feeding gen001/gen003 normalized inputs would not control the
# experiment, it would hand them a distribution they never trained on.
#
# ## Scope
#
# Baselines gen001 and gen003, plus the gen006 candidate at EVERY staged epoch.
# Scoring them together is the point: B and "which gen006 epoch is actually
# best" are the same measurement, and splitting them across two invocations is
# how two subtly different protocols get compared to each other.
#
# gen006's in-training "best epoch 4" ranked TILES at the model input size,
# which is not the geometry the detector is deployed at. It is a hypothesis to
# be checked here, not an answer to be carried forward.
#
# B is the best BASELINE row. A gen006 epoch is never part of its own baseline.
#
# gen005 is excluded: it aborted at epoch 3 and its surviving checkpoint is
# epoch 1, before its schedule developed. Add it to KCD_BASELINE_RUNS to score
# it as a diagnostic.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
: "${KCD_VALI_KWCOCO:?_launch_baseline_vali.sh: missing KCD_VALI_KWCOCO}"
: "${KCD_RUNS_DPATH:?_launch_baseline_vali.sh: missing KCD_RUNS_DPATH}"

WINDOW="${KCD_TILED_EVAL_WINDOW:-1229}"
OVERLAP="${KCD_TILED_EVAL_OVERLAP:-0.25}"
CANDIDATE_ID="${KCD_CANDIDATE_ID:-deimv2_dinov3_x_1024x1024_fixed}"
BASELINE_RUNS="fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen001 fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen003_bf16_fresh"
CANDIDATE_RUNS="fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen006_final"
RUNS="${KCD_BASELINE_RUNS:-$BASELINE_RUNS $CANDIDATE_RUNS}"
OUT_ROOT="${KCD_BASELINE_OUT:-$KCD_TILE_DPATH/../baseline_vali}"

# Evaluation precision is frozen and explicit. Left unset it would default to
# fp16 here while gen006 trains and reranks under bf16, so the baseline and the
# candidate would be measured at different precision.
export KCD_AMP_DTYPE="${KCD_AMP_DTYPE:-bfloat16}"

echo "=============================================================="
echo " B and the gen006 epoch curve -- full vali, true-tiled ${WINDOW}px"
echo "=============================================================="
echo "  vali:     $KCD_VALI_KWCOCO"
echo "  window:   ${WINDOW} px source, overlap ${OVERLAP}, keep_full"
echo "  runs:     $RUNS"
echo "  amp:      ${KCD_AMP_DTYPE}  (frozen; must match gen006's eval precision)"
echo "  stride:   ${KCD_EVAL_STRIDE:-1}  (>1 = stage-1 ranking only, NOT B)"
echo "  out:      $OUT_ROOT"
echo

mkdir -p "$OUT_ROOT"

# One driver, one protocol, every checkpoint. score_epochs.py scores a run at
# EVERY staged epoch when it has a staging/ dir and at its autoselected
# checkpoint otherwise, so gen001/gen003 contribute one row each and gen006
# contributes fourteen -- from a single pass under a single ruler.
export KCD_VALI_KWCOCO KCD_RUNS_DPATH KCD_CANDIDATE_ID KCD_BASELINE_RUNS
export KCD_TILED_EVAL_WINDOW KCD_TILED_EVAL_OVERLAP
KCD_CANDIDATE_ID="$CANDIDATE_ID" \
KCD_BASELINE_RUNS="$RUNS" \
KCD_BASELINE_OUT="$OUT_ROOT" \
KCD_TILED_EVAL_WINDOW="$WINDOW" \
KCD_TILED_EVAL_OVERLAP="$OVERLAP" \
KCD_EVAL_STRIDE="${KCD_EVAL_STRIDE:-1}" \
    exec "$PYTHON_BIN" "$SCRIPT_DIR/score_epochs.py"
