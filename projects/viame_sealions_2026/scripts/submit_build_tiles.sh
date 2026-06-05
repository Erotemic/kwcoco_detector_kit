#!/usr/bin/env bash
# Build the universal scheme-agnostic tile cache used by every
# subsequent training run. Runs as its own slurm + docker job — no
# GPU required.
#
# Why this is a separate job: tiling 6k+ source images at multiple
# scales takes 2-4 hours on the v2 corpus. Decoupling it from
# training means (a) you can fan out many training jobs against one
# cache build, (b) submit_train_* fails fast if the cache is missing
# instead of silently kicking off an interactive tile build that
# burns slurm walltime before any GPU work starts.
#
# Cache params come from paths.sh (canonical KCD_TILE_* defaults).
# Override here only if you're intentionally building a different
# cache variant.
#
# Submit:
#   bash projects/viame_sealions_2026/scripts/submit_build_tiles.sh
#
# Override walltime/resources from the shell if needed:
#   KCD_TIME_LIMIT=12:00:00 KCD_CPUS_PER_TASK=32 KCD_MEM=128G \
#       bash projects/viame_sealions_2026/scripts/submit_build_tiles.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

# Compute the same tile-hash that _launch_train.sh / _launch_tiles.sh
# will, and bake it into the run name. Different params → different
# job name and different log file, so you can build multiple cache
# variants without name collisions.
WRITER_FINGERPRINT=$(python3 -c "
from kwcoco_detector_kit.data import tile
print('v{}:{}'.format(
    getattr(tile, '_TILE_WRITER_VERSION', 1),
    ','.join(sorted(tile._PASSTHROUGH_ANN_FIELDS)),
))
" 2>/dev/null || echo 'unknown')
TILE_PARAMS_BODY=$(printf '%s\n' \
    "tile_size=$KCD_TILE_SIZE" \
    "source_scales=$KCD_TILE_SOURCE_SCALES" \
    "stride_frac=$KCD_TILE_STRIDE_FRAC" \
    "min_gt_area_frac=$KCD_TILE_MIN_GT_AREA_FRAC" \
    "min_keep_fraction=$KCD_TILE_MIN_KEEP_FRACTION" \
    "oversize_factor=$KCD_TILE_OVERSIZE_FACTOR" \
    "keep_negative=$KCD_TILE_KEEP_NEGATIVE" \
    "writer_passthrough=$WRITER_FINGERPRINT")
TILE_HASH=$(printf '%s' "$TILE_PARAMS_BODY" | sha1sum | cut -c1-8)

export KCD_RUN_NAME="build_tiles_${TILE_HASH}"

echo "=== submit_build_tiles ==="
echo "  source:      $KCD_UNIVERSAL_TRAIN_KWCOCO"
echo "  cache root:  $KCD_TILE_CACHE_DPATH"
echo "  tile params:"
printf '%s\n' "$TILE_PARAMS_BODY" | sed 's/^/    /'
echo "  TILE_HASH:   $TILE_HASH"
echo "  run name:    $KCD_RUN_NAME"
echo

exec bash "$SCRIPT_DIR/_submit_tiles.sh"
