#!/usr/bin/env bash
# Internal boilerplate. Builds the scheme-agnostic universal tile cache.
# Runs inside the kit's docker image after _sbatch_train.sh sets up
# mounts. Dispatched via KCD_LAUNCH_SCRIPT=_launch_tiles.sh.
#
# Contract: KCD_TILE_* env vars must be set (defaults in paths.sh).
# KCD_UNIVERSAL_TRAIN_KWCOCO must point at the un-collapsed source
# bundle. KCD_TILE_CACHE_DPATH must exist. No GPU required.
#
# This script intentionally produces the SAME TILE_HASH path that
# _launch_train.sh resolves, so a training job submitted after this
# completes finds its cache immediately and exits the tile step.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"

: "${KCD_UNIVERSAL_TRAIN_KWCOCO:?_launch_tiles.sh: missing KCD_UNIVERSAL_TRAIN_KWCOCO}"
: "${KCD_TILE_SIZE:?_launch_tiles.sh: missing KCD_TILE_SIZE}"
: "${KCD_TILE_SOURCE_SCALES:?_launch_tiles.sh: missing KCD_TILE_SOURCE_SCALES}"
: "${KCD_TILE_STRIDE_FRAC:?_launch_tiles.sh: missing KCD_TILE_STRIDE_FRAC}"
: "${KCD_TILE_MIN_GT_AREA_FRAC:?_launch_tiles.sh: missing KCD_TILE_MIN_GT_AREA_FRAC}"
: "${KCD_TILE_MIN_KEEP_FRACTION:?_launch_tiles.sh: missing KCD_TILE_MIN_KEEP_FRACTION}"
: "${KCD_TILE_OVERSIZE_FACTOR:?_launch_tiles.sh: missing KCD_TILE_OVERSIZE_FACTOR}"
: "${KCD_TILE_KEEP_NEGATIVE:?_launch_tiles.sh: missing KCD_TILE_KEEP_NEGATIVE}"
: "${KCD_TILE_CATEGORY_NAMES:?_launch_tiles.sh: missing KCD_TILE_CATEGORY_NAMES}"
: "${KCD_TILE_MODE:?_launch_tiles.sh: missing KCD_TILE_MODE}"

kcd_require_path "universal train.kwcoco.zip" "$KCD_UNIVERSAL_TRAIN_KWCOCO" || exit 1

# Tile-cache key — IDENTICAL hash computation to _launch_train.sh so
# both produce/consume the same cache subdir. Any divergence between
# these two implementations silently breaks cache sharing.
WRITER_FINGERPRINT=$("$PYTHON_BIN" -c "
from kwcoco_detector_kit.data import tile
print('v{}:{}'.format(
    getattr(tile, '_TILE_WRITER_VERSION', 1),
    ','.join(sorted(tile._PASSTHROUGH_ANN_FIELDS)),
))
" 2>/dev/null || echo 'unknown')

TILE_PARAMS_BODY=$(printf '%s\n' \
    "tile_mode=$KCD_TILE_MODE" \
    "tile_size=$KCD_TILE_SIZE" \
    "source_scales=$KCD_TILE_SOURCE_SCALES" \
    "stride_frac=$KCD_TILE_STRIDE_FRAC" \
    "min_gt_area_frac=$KCD_TILE_MIN_GT_AREA_FRAC" \
    "min_keep_fraction=$KCD_TILE_MIN_KEEP_FRACTION" \
    "oversize_factor=$KCD_TILE_OVERSIZE_FACTOR" \
    "keep_negative=$KCD_TILE_KEEP_NEGATIVE" \
    "category_names=$KCD_TILE_CATEGORY_NAMES" \
    "writer_passthrough=$WRITER_FINGERPRINT")
TILE_HASH=$(printf '%s' "$TILE_PARAMS_BODY" | sha1sum | cut -c1-8)

TILE_DIR="$KCD_TILE_CACHE_DPATH/_universal/$TILE_HASH"
UNIVERSAL_TILES="$TILE_DIR/tiles.kwcoco.zip"
mkdir -p "$TILE_DIR"
printf '%s\n' "$TILE_PARAMS_BODY" > "$TILE_DIR/tile_params.txt"

echo "=== tile build (universal, scheme-agnostic) ==="
echo "  source:    $KCD_UNIVERSAL_TRAIN_KWCOCO"
echo "  cache:     $UNIVERSAL_TILES"
echo "  hash:      $TILE_HASH (see tile_params.txt)"
echo "  size:      $KCD_TILE_SIZE"
echo "  scales:    $KCD_TILE_SOURCE_SCALES"
echo "  stride:    $KCD_TILE_STRIDE_FRAC"
echo "  min_gt_area_frac:   $KCD_TILE_MIN_GT_AREA_FRAC"
echo "  min_keep_fraction:  $KCD_TILE_MIN_KEEP_FRACTION"
echo "  oversize_factor:    $KCD_TILE_OVERSIZE_FACTOR"
echo "  keep_negative:      $KCD_TILE_KEEP_NEGATIVE"
echo

# Disk guard — tile build for the v2 corpus is ~225 GB. Default
# threshold higher than train's 30 GB.
KCD_MIN_FREE_GB="${KCD_MIN_FREE_GB:-50}"
free_kb=$(df -k --output=avail "$KCD_TILE_CACHE_DPATH" 2>/dev/null | tail -n1 | tr -d ' ')
if [ -n "$free_kb" ]; then
    free_gb=$(( free_kb / 1024 / 1024 ))
    echo "  free disk: ${free_gb} GB at $KCD_TILE_CACHE_DPATH (need >= ${KCD_MIN_FREE_GB})"
    if [ "$free_gb" -lt "$KCD_MIN_FREE_GB" ]; then
        echo "ERROR: ${free_gb} GB free; need >= ${KCD_MIN_FREE_GB}" >&2
        exit 1
    fi
fi
echo

# Skip if already built (idempotent).
if [ -f "$UNIVERSAL_TILES" ] && [ "${KCD_FORCE_RETILE:-0}" != "1" ]; then
    sz=$(stat -c%s "$UNIVERSAL_TILES" 2>/dev/null || echo 0)
    if [ "$sz" -gt 102400 ]; then
        echo "Cache already valid at $UNIVERSAL_TILES (size $sz bytes)."
        echo "Set KCD_FORCE_RETILE=1 to rebuild."
        exit 0
    fi
fi

# Tile against the v2 norm bundle's 9 categories. coco_export filters
# annotations whose source category name isn't in --category_names
# (coco_export.py:92-93), so this list MUST match the source bundle's
# category vocabulary or all annotations get dropped silently.
# apply_scheme collapses these 9 to the scheme target downstream.
"$PYTHON_BIN" -m kwcoco_detector_kit tile \
    "$KCD_UNIVERSAL_TRAIN_KWCOCO" "$UNIVERSAL_TILES" \
    --mode "$KCD_TILE_MODE" \
    --tile_size "$KCD_TILE_SIZE" \
    --source_scales "$KCD_TILE_SOURCE_SCALES" \
    --stride_frac "$KCD_TILE_STRIDE_FRAC" \
    --min_gt_area_frac "$KCD_TILE_MIN_GT_AREA_FRAC" \
    --min_keep_fraction "$KCD_TILE_MIN_KEEP_FRACTION" \
    --oversize_factor "$KCD_TILE_OVERSIZE_FACTOR" \
    --keep_negative "$KCD_TILE_KEEP_NEGATIVE" \
    --category_names "$KCD_TILE_CATEGORY_NAMES"

echo
echo "Done. Tile cache: $UNIVERSAL_TILES"
echo "  $(du -sh "$TILE_DIR" 2>/dev/null | cut -f1) on disk."
