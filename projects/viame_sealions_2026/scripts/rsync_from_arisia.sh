#!/usr/bin/env bash
# Pull training results back from arisia for local analysis on namek.
# Mirror of rsync_to_arisia.sh in the other direction.
#
# By default fetches the diagnostic / analysis surface, NOT the full
# checkpoint tree (which can be tens of GB):
#   - nccl_traces/        per-rank flight-recorder dumps on hang/timeout
#   - slurm_logs/         sbatch stdout/stderr captures
#   - runs/*/log.txt      DEIMv2 per-epoch train+eval JSONL
#   - runs/*/eval/        coco_eval JSON / pth artifacts
#   - runs/*/summary/     tensorboard event files
#   - sweeps/*/index.tsv  per-sweep status table
#   - manifest.{tsv,json} eligibility manifest if present
#
# Override with KCD_FETCH_FULL=1 to also pull checkpoints, tile bundles,
# detector_prepared mscoco files (large — use sparingly).
#
# Usage:
#   bash scripts/rsync_from_arisia.sh                  # default
#   KCD_FETCH_FULL=1 bash scripts/rsync_from_arisia.sh # heavy
#   SRC=arisia:/other/path DEST=/tmp/foo bash scripts/rsync_from_arisia.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

# Default src/dest = the canonical runs/ tree under KCD_TRAINING_ROOT.
# Pulls every per-experiment workspace under runs/* and the shared
# tile_cache/. Pass SRC/DEST to point elsewhere (e.g. a different
# kcd_sealion-rooted tree or another host).
SRC="${SRC:-arisia:$KCD_TRAINING_ROOT/}"
DEST="${DEST:-$KCD_TRAINING_ROOT/}"

# Slurm logs live under $KCD_SLURM_LOG_DPATH on every host (on the
# data drive, NOT in the kit checkout). Same canonical path means we
# don't need a host-specific remote source path.
REMOTE_LOG_SRC="${REMOTE_LOG_SRC:-arisia:$KCD_SLURM_LOG_DPATH/}"
LOCAL_LOG_DEST="${LOCAL_LOG_DEST:-$KCD_SLURM_LOG_DPATH/}"

mkdir -p "$DEST" "$LOCAL_LOG_DEST"

if [ "${KCD_FETCH_FULL:-0}" = "1" ]; then
    echo "=== Full fetch (checkpoints + tiles + everything) ==="
    rsync -avh --info=progress2 \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        "$SRC" "$DEST"
else
    echo "=== Diagnostic fetch (logs + traces + eval only) ==="
    # Layout (since the 2026-05-22 refactor):
    #   $KCD_TRAINING_ROOT/runs/<run_name>/         per-experiment workspace
    #     ├── nccl_traces/, eval/, summary/, ...
    #     ├── sweeps/<ts>/                          per-sweep status table
    #     └── runs/<candidate>/                     DEIMv2 output dir
    rsync -avh --info=progress2 \
        --include '*/' \
        --include 'runs/*/nccl_traces/***' \
        --include 'runs/*/manifest.tsv' \
        --include 'runs/*/manifest.json' \
        --include 'runs/*/sweeps/*/index.tsv' \
        --include 'runs/*/sweeps/*/*.json' \
        --include 'runs/*/runs/*/log.txt' \
        --include 'runs/*/runs/*/NaN.pth' \
        --include 'runs/*/runs/*/eval/***' \
        --include 'runs/*/runs/*/summary/***' \
        --include 'runs/*/runs/*/generated_configs/***' \
        --include 'runs/*/runs/*/detector_prepared/*.json' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '*' \
        "$SRC" "$DEST"
fi

echo
echo "=== Slurm logs (canonical $KCD_SLURM_LOG_DPATH) ==="
rsync -avh --info=progress2 \
    --include '*.out' \
    --include '*.err' \
    --exclude '*' \
    "$REMOTE_LOG_SRC" "$LOCAL_LOG_DEST"

# Legacy: pre-2026-05-22 runs wrote slurm logs into the kit checkout
# instead of the data drive. Sweep those too so old runs aren't
# stranded. Skipped silently if the legacy path is empty.
LEGACY_LOG_SRC="${LEGACY_LOG_SRC:-arisia:/home/local/KHQ/jon.crall/code/kwcoco_detector_kit/projects/viame_sealions_2026/training_runs/slurm_logs/}"
echo
echo "=== Slurm logs (legacy kit-checkout path) ==="
rsync -avh --info=progress2 \
    --include '*.out' \
    --include '*.err' \
    --exclude '*' \
    "$LEGACY_LOG_SRC" "$LOCAL_LOG_DEST" || \
    echo "  (legacy path empty or unreachable — fine; only matters for pre-2026-05-22 jobs)"

echo
echo "Done."
echo "  Results under: $DEST"
echo "  Slurm logs:    $LOCAL_LOG_DEST"
if [ "${KCD_FETCH_FULL:-0}" != "1" ]; then
    echo "  (Set KCD_FETCH_FULL=1 to also grab checkpoints + tile bundles.)"
fi
