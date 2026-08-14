#!/usr/bin/env bash
# Measure the FishTrack23 corpus on the training host and package a small,
# transferable manifest describing it.
#
# The manifest is what a workstation needs to choose resolution, chip size,
# class balance, and splits without copying imagery. It contains counts,
# percentiles, and (optionally) the raw annotation CSVs -- never pixels.
#
# Usage, on the training host:
#
#   cd ~/code/kwcoco_detector_kit
#   bash projects/viame_fish_2026/scripts/collect_data_manifest.sh
#   bash projects/viame_fish_2026/scripts/collect_data_manifest.sh --with-annotations
#
# Then, from the workstation:
#
#   rsync -avhP aiq-gpu:/data/users/jon.crall/fish/inventory/ ./fish-inventory/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

WITH_ANNOTATIONS=0
for arg in "$@"; do
    case "$arg" in
        --with-annotations) WITH_ANNOTATIONS=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

export VF_INVENTORY_DPATH="${VF_INVENTORY_DPATH:-$VF_WORK_DPATH/inventory}"

if [ ! -d "$VF_INPUT_DPATH" ]; then
    echo "MISSING dir: VF_INPUT_DPATH = $VF_INPUT_DPATH"
    echo "Mirror the data first: bash $SCRIPT_DIR/setup_data.sh"
    exit 1
fi

mkdir -p "$VF_INVENTORY_DPATH"

echo "Inventorying $VF_INPUT_DPATH"
python3 "$SCRIPT_DIR/inventory_data.py" \
    --input "$VF_INPUT_DPATH" \
    --out-dir "$VF_INVENTORY_DPATH"

# A full-depth tree listing costs nothing to transfer and answers layout
# questions the JSON summary flattens away.
find "$VF_INPUT_DPATH" -maxdepth 3 | sort > "$VF_INVENTORY_DPATH/tree_depth3.txt"
du -sh "$VF_INPUT_DPATH" > "$VF_INVENTORY_DPATH/disk_usage.txt" 2>/dev/null || true
df -h "$VF_WORK_DPATH" "$VF_DATA_DPATH" > "$VF_INVENTORY_DPATH/disk_free.txt" 2>/dev/null || true
nvidia-smi -L > "$VF_INVENTORY_DPATH/gpus.txt" 2>/dev/null || true
nproc > "$VF_INVENTORY_DPATH/nproc.txt" 2>/dev/null || true

if [ "$WITH_ANNOTATIONS" = "1" ]; then
    ANNOT_TGZ="$VF_INVENTORY_DPATH/annotations.tar.gz"
    echo "Packaging annotation CSVs into $ANNOT_TGZ"
    ( cd "$VF_INPUT_DPATH" && find . -type f -name '*.csv' -print0 \
        | tar --null -T - -czf "$ANNOT_TGZ" ) || true
    du -sh "$ANNOT_TGZ"
fi

echo
echo "Manifest written to $VF_INVENTORY_DPATH"
du -sh "$VF_INVENTORY_DPATH"
ls -lh "$VF_INVENTORY_DPATH"
