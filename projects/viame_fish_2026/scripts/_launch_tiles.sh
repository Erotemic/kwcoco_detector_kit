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
# change, change VF_TILE_DPATH (or KCD_TILE_SIZE, which is in its default)
# so the old bundle is not silently reused.
#
# Tiles TRAIN and VALI. Not test -- see the note in paths.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"

: "${KCD_TILE_SIZE:?_launch_tiles.sh: missing KCD_TILE_SIZE}"
: "${KCD_TILE_SOURCE_SCALES:?_launch_tiles.sh: missing KCD_TILE_SOURCE_SCALES}"
: "${KCD_TILE_STRIDE_FRAC:?_launch_tiles.sh: missing KCD_TILE_STRIDE_FRAC}"
: "${VF_TILE_DPATH:?_launch_tiles.sh: missing VF_TILE_DPATH}"

kcd_require_path "train bundle" "$VF_TRAIN_KWCOCO" || exit 1
kcd_require_path "vali bundle"  "$VF_VALI_KWCOCO"  || exit 1

echo "=============================================================="
echo " fish tiling: ${KCD_TILE_SIZE}px native windows"
echo "=============================================================="
echo "  mode:          $KCD_TILE_MODE"
echo "  tile_size:     $KCD_TILE_SIZE"
echo "  source_scales: $KCD_TILE_SOURCE_SCALES"
echo "  stride_frac:   $KCD_TILE_STRIDE_FRAC  (overlap $(python3 -c "print(f'{(1-float('$KCD_TILE_STRIDE_FRAC'))*100:.0f}%')"))"
echo "  keep_negative: $KCD_TILE_KEEP_NEGATIVE"
echo "  out:           $VF_TILE_DPATH"
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

tile_split train "$VF_TRAIN_KWCOCO" "$VF_TILE_TRAIN_KWCOCO"
tile_split vali  "$VF_VALI_KWCOCO"  "$VF_TILE_VALI_KWCOCO"

echo
echo "=============================================================="
echo " done"
echo "=============================================================="
"$PYTHON_BIN" - <<'PYEOF'
import json, os
for name, key in (("train", "VF_TILE_TRAIN_KWCOCO"), ("vali", "VF_TILE_VALI_KWCOCO")):
    p = os.environ[key]
    d = json.load(open(p))
    n_img, n_ann = len(d["images"]), len(d["annotations"])
    empty = n_img - len({a["image_id"] for a in d["annotations"]})
    print(f"  {name}: {n_img:,} tiles, {n_ann:,} annotations, "
          f"{empty:,} empty ({100*empty/max(1,n_img):.1f}%)")
PYEOF
echo "  disk: $(du -sh "$VF_TILE_DPATH" 2>/dev/null | cut -f1)"
