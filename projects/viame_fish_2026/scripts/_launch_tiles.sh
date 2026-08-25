#!/usr/bin/env bash
# In-container entry point for building the native-resolution tile bundles.
#
# Selected with KCD_LAUNCH_SCRIPT=_launch_tiles.sh, so this reuses the same
# sbatch + docker + mount machinery as training. No GPU is needed, but the
# kit (kwcoco, kwimage) is, which is why this runs in the image rather than
# on the host like prep_all.sh.
#
# Deliberately simpler than the sea-lion equivalent: that project maintains a
# hashed "universal" tile cache shared across schemes, because it trains many
# class schemes off one tiling. Fish has a single scheme, so tiles land at a
# fixed path and the train script points at them. If the tile parameters
# change, change KCD_TILE_DPATH (or KCD_TILE_SIZE, which is in its default)
# so the old bundle is not silently reused.
#
# Tiles TRAIN and VALI. Not test -- see the note in paths.sh.
#
# EVERY PATH HERE MUST BE A KCD_* VARIABLE. This script runs inside the
# container, where $HOME is /root, and _submit_train.sh forwards only ^KCD_
# names -- so a VF_* path silently re-derives to /root/ssd-data/... That is
# exactly how slurm job 493 failed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"

: "${KCD_TILE_SIZE:?_launch_tiles.sh: missing KCD_TILE_SIZE}"
: "${KCD_TILE_SOURCE_SCALES:?_launch_tiles.sh: missing KCD_TILE_SOURCE_SCALES}"
: "${KCD_TILE_STRIDE_FRAC:?_launch_tiles.sh: missing KCD_TILE_STRIDE_FRAC}"
: "${KCD_TILE_DPATH:?_launch_tiles.sh: missing KCD_TILE_DPATH}"
: "${KCD_TILE_SRC_TRAIN:?_launch_tiles.sh: missing KCD_TILE_SRC_TRAIN}"
: "${KCD_TILE_SRC_VALI:?_launch_tiles.sh: missing KCD_TILE_SRC_VALI}"

# Sources are passed explicitly rather than read from KCD_TRAIN_KWCOCO, which
# the gen005 submit script repoints at the TILED bundle -- tiling that would
# be a silent no-op or a nested re-tile.
kcd_require_path "source train bundle" "$KCD_TILE_SRC_TRAIN" || exit 1
kcd_require_path "source vali bundle"  "$KCD_TILE_SRC_VALI"  || exit 1

echo "=============================================================="
echo " fish tiling: ${KCD_TILE_SIZE}px native windows"
echo "=============================================================="
echo "  mode:          $KCD_TILE_MODE"
echo "  tile_size:     $KCD_TILE_SIZE"
echo "  source_scales: $KCD_TILE_SOURCE_SCALES"
TILE_OVERLAP_PCT="$(awk -v s="$KCD_TILE_STRIDE_FRAC" 'BEGIN{printf "%.0f", (1-s)*100}')"
echo "  stride_frac:   $KCD_TILE_STRIDE_FRAC  (${TILE_OVERLAP_PCT}% overlap, $(awk -v s="$KCD_TILE_STRIDE_FRAC" -v t="$KCD_TILE_SIZE" 'BEGIN{printf "%.0f", s*t}') px stride)"
echo "  keep_negative: $KCD_TILE_KEEP_NEGATIVE"
echo "  out:           $KCD_TILE_DPATH"
echo

tile_split() {
    local name="$1" src="$2" dst="$3"
    if [ -f "$dst" ]; then
        echo "[$name] already tiled -> $dst (delete to rebuild)"
        return 0
    fi
    echo "[$name] tiling $src"
    mkdir -p "$(dirname "$dst")"
    "$PYTHON_BIN" -m kwcoco_detector_kit tile \
        "$src" "$dst" \
        --mode "$KCD_TILE_MODE" \
        --tile_size "$KCD_TILE_SIZE" \
        --source_scales "$KCD_TILE_SOURCE_SCALES" \
        --stride_frac "$KCD_TILE_STRIDE_FRAC" \
        --min_gt_area_frac "$KCD_TILE_MIN_GT_AREA_FRAC" \
        --min_keep_fraction "$KCD_TILE_MIN_KEEP_FRACTION" \
        --oversize_factor "$KCD_TILE_OVERSIZE_FACTOR" \
        --keep_negative "$KCD_TILE_KEEP_NEGATIVE" \
        --category_names "$KCD_TILE_CATEGORY_NAMES" \
        --output_ext .jpg \
        --jpeg_quality "$KCD_TILE_JPEG_QUALITY"
    echo "[$name] wrote $dst"
}

tile_split train "$KCD_TILE_SRC_TRAIN" "$KCD_TILE_TRAIN_KWCOCO"
tile_split vali  "$KCD_TILE_SRC_VALI"  "$KCD_TILE_VALI_KWCOCO"

echo
echo "=============================================================="
echo " done"
echo "=============================================================="
"$PYTHON_BIN" - <<'PYEOF'
import json, os
for name, key in (("train", "KCD_TILE_TRAIN_KWCOCO"), ("vali", "KCD_TILE_VALI_KWCOCO")):
    p = os.environ[key]
    d = json.load(open(p))
    n_img, n_ann = len(d["images"]), len(d["annotations"])
    empty = n_img - len({a["image_id"] for a in d["annotations"]})
    print(f"  {name}: {n_img:,} tiles, {n_ann:,} annotations, "
          f"{empty:,} empty ({100*empty/max(1,n_img):.1f}%)")
PYEOF
echo "  disk: $(du -sh "$KCD_TILE_DPATH" 2>/dev/null | cut -f1)"
