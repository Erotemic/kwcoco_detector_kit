#!/usr/bin/env bash
# Pull training results back from the aiq-gpu Blackwell box for local
# analysis on namek. Mirror of rsync_from_arisia.sh, with two differences:
#
#   * aiq-gpu has NO slurm, so there are no sbatch *.out logs to fetch.
#     Instead we pull the tmux `tee` logs the standalone runs write to
#     $KCD_TRAINING_ROOT/*.log (e.g. aiq_pup_640.log, aiq_tiles_1280.log).
#   * The tile cache lives on a separate SSD symlinked into the tree; the
#     default diagnostic fetch skips it (tiles are large + reproducible).
#
# By default fetches the diagnostic / analysis surface, NOT the full
# checkpoint tree:
#   - runs/*/runs/*/log.txt      DEIMv2 per-epoch train+eval JSONL
#   - runs/*/runs/*/eval/        coco_eval JSON / metrics
#   - runs/*/runs/*/summary/     tensorboard event files
#   - runs/*/tiled_compare/      whole-image-vs-tiled rescore outputs
#   - runs/*/sweeps/*/index.tsv  per-sweep status table
#   - *.log                      tmux tee logs at the training root
#
# Override with KCD_FETCH_FULL=1 to also pull checkpoints + everything
# under runs/ (large — checkpoints are tens of GB). Even then the tile
# cache is excluded unless KCD_FETCH_TILES=1.
#
# Usage:
#   bash scripts/rsync_from_aiq.sh                    # default (diagnostic)
#   KCD_FETCH_FULL=1 bash scripts/rsync_from_aiq.sh   # + checkpoints
#   SRC=aiq-gpu:/other DEST=/tmp/foo bash scripts/rsync_from_aiq.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

# Remote host alias for aiq-gpu (override via AIQ_HOST or a full SRC).
AIQ_HOST="${AIQ_HOST:-aiq-gpu}"

# Default src/dest = the canonical runs/ tree under KCD_TRAINING_ROOT.
# Same canonical path on both hosts, so run names (…_4gpu_aiq_…) don't
# collide with the arisia runs already under namek's runs/.
SRC="${SRC:-$AIQ_HOST:$KCD_TRAINING_ROOT/}"
DEST="${DEST:-$KCD_TRAINING_ROOT/}"

mkdir -p "$DEST"

if [ "${KCD_FETCH_FULL:-0}" = "1" ]; then
    echo "=== Full fetch (checkpoints + everything under runs/) ==="
    # Exclude the tile cache unless explicitly requested — the 1280 build
    # alone is large and reproducible from the corpus.
    TILE_EXCLUDES=(--exclude 'tile_cache' --exclude 'ssd-data')
    [ "${KCD_FETCH_TILES:-0}" = "1" ] && TILE_EXCLUDES=()
    rsync -avh --info=progress2 \
        "${TILE_EXCLUDES[@]}" \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        "$SRC" "$DEST"
else
    echo "=== Diagnostic fetch (logs + eval + tiled_compare only) ==="
    rsync -avh --info=progress2 \
        --include '*/' \
        --include 'runs/*/manifest.tsv' \
        --include 'runs/*/manifest.json' \
        --include 'runs/*/sweeps/*/index.tsv' \
        --include 'runs/*/sweeps/*/*.json' \
        --include 'runs/*/tiled_compare/***' \
        --include 'runs/*/runs/*/log.txt' \
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
echo "=== tmux tee logs ($KCD_TRAINING_ROOT/*.log) ==="
# aiq-gpu standalone runs tee their stdout here instead of slurm *.out.
rsync -avh --info=progress2 \
    --include '*.log' \
    --exclude '*/' \
    --exclude '*' \
    "$AIQ_HOST:$KCD_TRAINING_ROOT/" "$KCD_TRAINING_ROOT/" || \
    echo "  (no *.log at the training root — fine)"

echo
echo "Done."
echo "  Results under: $DEST"
if [ "${KCD_FETCH_FULL:-0}" != "1" ]; then
    echo "  (Set KCD_FETCH_FULL=1 to also grab checkpoints; +KCD_FETCH_TILES=1 for tiles.)"
fi
