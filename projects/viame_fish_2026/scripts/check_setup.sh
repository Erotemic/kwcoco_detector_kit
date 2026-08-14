#!/usr/bin/env bash
# Fast preflight for data, binary, selected config, disk, and visible GPUs.

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
check_file setup_viame.sh "$VF_CURRENT_VIAME_LINK/setup_viame.sh"
check_file selected_config "$VF_CONFIG_FPATH"

if [ -n "$VF_EXPECTED_CONFIG_NAME" ] && [ "$VF_CONFIG_NAME" != "$VF_EXPECTED_CONFIG_NAME" ]; then
    echo "WRONG config: selected $VF_CONFIG_NAME"
    echo "Expected by this run: $VF_EXPECTED_CONFIG_NAME"
    echo "Select it with:"
    echo "  bash $SCRIPT_DIR/setup_config.sh $VF_EXPECTED_CONFIG_NAME"
    FAILED=1
fi

if [ "$FAILED" != "0" ]; then
    echo
    echo "Preflight failed. Fix the missing or mismatched paths above."
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
echo "Selected config"
echo "Name:   $VF_CONFIG_NAME"
echo "Path:   $VF_CONFIG_FPATH"
echo "Origin: ${VF_CONFIG_ORIGIN:-unknown}"
sha256sum "$VF_CONFIG_FPATH"
grep -nE 'rf_detr:(segmentation|resolution|max_epochs|val_subsample|ddp_timeout) =' "$VF_CONFIG_FPATH" || true

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
