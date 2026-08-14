#!/usr/bin/env bash
# Resume/synchronize the FishTrack23 training data from numenor.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

mkdir -p "$VF_DATA_DPATH"

rsync -avhP --info=progress2 \
    "$VF_DATA_SOURCE" \
    "$VF_DATA_DPATH/"

echo
echo "Data setup finished"
echo "VF_DATA_DPATH=$VF_DATA_DPATH"
echo "VF_INPUT_DPATH=$VF_INPUT_DPATH"
ls -ld "$VF_DATA_DPATH" "$VF_INPUT_DPATH"
