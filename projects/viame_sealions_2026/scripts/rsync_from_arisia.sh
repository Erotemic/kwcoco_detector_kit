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

# Default src = arisia's KCD_ROOT_PUP_VS_NONPUP. The default below
# matches the canonical arisia layout from paths.sh; pass SRC to point
# elsewhere (e.g. a different experiment's kcd_root or another host).
SRC="${SRC:-arisia:/data/users/jon.crall/kcd_sealion/pup_vs_nonpup/}"

# Local landing zone. Defaults to $KCD_ROOT_PUP_VS_NONPUP on this host
# so the same env vars resolve to the same paths everywhere.
DEST="${DEST:-$KCD_ROOT_PUP_VS_NONPUP/}"

# Also fetch the slurm logs that live next to the project tree on the
# remote (per submit_pup_vs_nonpup.sh: LOG_DPATH=$KCD_REPO_ROOT/training_runs/slurm_logs).
REMOTE_LOG_SRC="${REMOTE_LOG_SRC:-arisia:/home/local/KHQ/jon.crall/code/kwcoco_detector_kit/projects/viame_sealions_2026/training_runs/slurm_logs/}"
LOCAL_LOG_DEST="${LOCAL_LOG_DEST:-$KCD_REPO_ROOT/training_runs/slurm_logs/}"

mkdir -p "$DEST" "$LOCAL_LOG_DEST"

if [ "${KCD_FETCH_FULL:-0}" = "1" ]; then
    echo "=== Full fetch (checkpoints + tiles + everything) ==="
    rsync -avh --info=progress2 \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        "$SRC" "$DEST"
else
    echo "=== Diagnostic fetch (logs + traces + eval only) ==="
    rsync -avh --info=progress2 \
        --include '*/' \
        --include 'nccl_traces/***' \
        --include 'manifest.tsv' \
        --include 'manifest.json' \
        --include 'sweeps/*/index.tsv' \
        --include 'runs/*/log.txt' \
        --include 'runs/*/eval/***' \
        --include 'runs/*/summary/***' \
        --include 'runs/*/generated_configs/***' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '*' \
        "$SRC" "$DEST"
fi

echo
echo "=== Slurm logs ==="
rsync -avh --info=progress2 \
    --include '*.out' \
    --include '*.err' \
    --exclude '*' \
    "$REMOTE_LOG_SRC" "$LOCAL_LOG_DEST"

echo
echo "Done."
echo "  Results under: $DEST"
echo "  Slurm logs:    $LOCAL_LOG_DEST"
if [ "${KCD_FETCH_FULL:-0}" != "1" ]; then
    echo "  (Set KCD_FETCH_FULL=1 to also grab checkpoints + tile bundles.)"
fi
