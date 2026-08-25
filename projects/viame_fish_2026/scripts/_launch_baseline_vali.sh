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
# gen001 and gen003 only. gen005 is excluded from B: it aborted at epoch 3 and
# its surviving checkpoint is epoch 1, before its schedule developed. It can be
# scored as a diagnostic by adding it to KCD_BASELINE_RUNS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
: "${KCD_VALI_KWCOCO:?_launch_baseline_vali.sh: missing KCD_VALI_KWCOCO}"
: "${KCD_RUNS_DPATH:?_launch_baseline_vali.sh: missing KCD_RUNS_DPATH}"

WINDOW="${KCD_TILED_EVAL_WINDOW:-1229}"
OVERLAP="${KCD_TILED_EVAL_OVERLAP:-0.25}"
CANDIDATE_ID="${KCD_CANDIDATE_ID:-deimv2_dinov3_x_1024x1024_fixed}"
RUNS="${KCD_BASELINE_RUNS:-fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen001 fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen003_bf16_fresh}"
OUT_ROOT="${KCD_BASELINE_OUT:-$KCD_TILE_DPATH/../baseline_vali}"

echo "=============================================================="
echo " baseline B -- full vali, true-tiled ${WINDOW}px"
echo "=============================================================="
echo "  vali:     $KCD_VALI_KWCOCO"
echo "  window:   ${WINDOW} px source, overlap ${OVERLAP}, keep_full"
echo "  runs:     $RUNS"
echo "  out:      $OUT_ROOT"
echo

mkdir -p "$OUT_ROOT"

for RUN in $RUNS; do
    WORKDIR="$KCD_RUNS_DPATH/$RUN/runs/$CANDIDATE_ID"
    if [ ! -d "$WORKDIR" ]; then
        echo "[$RUN] SKIP -- no workdir at $WORKDIR" >&2
        continue
    fi
    echo "[$RUN] scoring ..."
    KCD_BASELINE_RUN="$RUN" KCD_BASELINE_WORKDIR="$WORKDIR" \
    KCD_BASELINE_EVALROOT="$OUT_ROOT/$RUN" \
    "$PYTHON_BIN" - <<'PYEOF'
import json, os, pathlib
import kwcoco_detector_kit.trainers  # registers the plugins
from kwcoco_detector_kit.trainers._registry import get_trainer
from kwcoco_detector_kit.eval.kwcoco_eval import run_kwcoco_eval

run = os.environ["KCD_BASELINE_RUN"]
workdir = pathlib.Path(os.environ["KCD_BASELINE_WORKDIR"])
evalroot = pathlib.Path(os.environ["KCD_BASELINE_EVALROOT"])
evalroot.mkdir(parents=True, exist_ok=True)

policy = json.loads((workdir / "policy.json").read_text())
category_names = policy["category_names"]

# Record the contract this checkpoint is being scored under, so the journal
# entry can state it rather than assert it.
cfg_fpath = workdir / "generated_configs" / "train.yml"
contract = "unnormalized"
if cfg_fpath.exists():
    import yaml
    cfg = yaml.safe_load(cfg_fpath.read_text()) or {}
    try:
        ops = cfg["val_dataloader"]["dataset"]["transforms"]["ops"]
    except (KeyError, TypeError):
        ops = []
    for op in ops or []:
        if isinstance(op, dict) and op.get("type") == "Normalize":
            contract = f"normalized mean={op.get('mean')} std={op.get('std')}"
            break
print(f"  {run}: categories={category_names}  preprocessing={contract}")

window = int(os.environ.get("KCD_TILED_EVAL_WINDOW", "1229"))
out = run_kwcoco_eval(
    trainer=get_trainer("deimv2"),
    workdir=workdir,
    test_kwcoco=os.environ["KCD_VALI_KWCOCO"],
    kcd_root=evalroot,
    candidate_id=run,
    category_names=category_names,
    force=bool(int(os.environ.get("KCD_FORCE_EVAL", "0"))),
    tiled_eval=True,
    tiled_window=window,
    tiled_overlap=float(os.environ.get("KCD_TILED_EVAL_OVERLAP", "0.25")),
    tiled_keep_full=True,
    tiled_batch=int(os.environ.get("KCD_TILED_EVAL_BATCH", "64")),
    device=os.environ.get("KCD_EVAL_DEVICE", "cuda"),
    read_workers=int(os.environ.get("KCD_EVAL_READ_WORKERS", "4")),
)
(evalroot / "contract.txt").write_text(contract + "\n")
print(f"  {run}: wrote {out}")
PYEOF
    echo
done

echo "=============================================================="
echo " B summary"
echo "=============================================================="
"$PYTHON_BIN" - <<PYEOF
import json, pathlib
root = pathlib.Path("$OUT_ROOT")
rows = []
for d in sorted(root.iterdir()):
    m = list(d.rglob("detect_metrics.json")) if d.is_dir() else []
    if not m:
        continue
    data = json.loads(m[0].read_text())
    for key, blob in data.items():
        if not isinstance(blob, dict):
            continue
        ap = (blob.get("nocls_measures") or {}).get("ap")
        if ap is not None:
            contract = (d / "contract.txt")
            rows.append((d.name, float(ap), key,
                         contract.read_text().strip() if contract.exists() else "?"))
            break
for name, ap, key, contract in rows:
    print(f"  {name:52} AP {ap:.4f}   [{key}]  {contract}")
if rows:
    b = max(r[1] for r in rows)
    print()
    print(f"  B = {b:.4f}   (success: >= {b+0.01:.4f}, strong: >= {b+0.02:.4f})")
    print("  RECORD THIS IN THE JOURNAL BEFORE LAUNCHING gen006.")
PYEOF
