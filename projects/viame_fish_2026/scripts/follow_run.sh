#!/usr/bin/env bash
# Follow the newest attempt for a run definition.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

RUN_NAME="${1:-fishtrack23_rfdetr_l_seg720_4gpu_viame0227_gen001}"
LATEST_LINK="$VF_RUNS_DPATH/$RUN_NAME/latest"

if [ ! -e "$LATEST_LINK" ]; then
    echo "ERROR: no latest run exists at: $LATEST_LINK"
    exit 1
fi

RUN_DPATH="$(readlink -f "$LATEST_LINK")"
echo "Following $RUN_DPATH/train.log"
tail -f "$RUN_DPATH/train.log"
