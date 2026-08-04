#!/usr/bin/env bash
# Fast preflight for data, binary, config, disk, and visible GPUs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

FAILED=0

check_file() {
    if [ -f "$2" ]; then
        echo "OK file: $1 = $2"
    else
        echo "MISSING file: $1 = $2"
        FAILED=1
    fi
}

check_dir() {
    if [ -d "$2" ]; then
        echo "OK dir:  $1 = $2"
    else
        echo "MISSING dir: $1 = $2"
        FAILED=1
    fi
}

check_dir VF_WORK_DPATH "$VF_WORK_DPATH"
check_dir VF_DATA_DPATH "$VF_DATA_DPATH"
check_dir VF_INPUT_DPATH "$VF_INPUT_DPATH"
check_file VF_PROJECT_CONFIG "$VF_PROJECT_CONFIG"
check_file setup_viame.sh "$VF_CURRENT_VIAME_LINK/setup_viame.sh"
check_file installed_config "$VF_CURRENT_VIAME_LINK/configs/pipelines/$VF_CONFIG_NAME"

if [ "$FAILED" != "0" ]; then
    echo
echo "Preflight failed. Fix the missing paths above."
    exit 1
fi

source "$VF_CURRENT_VIAME_LINK/setup_viame.sh"

if command -v viame_train_detector >/dev/null 2>&1; then
    echo "OK command: $(command -v viame_train_detector)"
else
    echo "MISSING command: viame_train_detector"
    exit 1
fi

echo
echo "GPU inventory"
nvidia-smi -L

echo
echo "Work-disk capacity"
df -h "$VF_WORK_DPATH"

echo
echo "Input directory sample"
find "$VF_INPUT_DPATH" -maxdepth 2 -type f | head -n 20

echo
echo "Preflight passed"
