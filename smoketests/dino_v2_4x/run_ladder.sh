#!/usr/bin/env bash
# Run the smoke ladder up to a selected stage.
set -euo pipefail

THIS_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_STAGE="${MAX_STAGE:-03}"

run_stage() {
    local script="$1"
    echo
    echo "############################################################"
    echo "# $script"
    echo "############################################################"
    bash "$THIS_DPATH/$script"
}

run_stage 00_mock_demo_dataload.sh

if [ "$MAX_STAGE" = "00" ]; then exit 0; fi
run_stage 01_ogdino_demo_1gpu.sh

if [ "$MAX_STAGE" = "01" ]; then exit 0; fi
run_stage 02_ogdino_demo_4gpu.sh

if [ "$MAX_STAGE" = "02" ]; then exit 0; fi
run_stage 03_ogdino_viame_subset_4gpu.sh

if [ "$MAX_STAGE" = "03" ]; then exit 0; fi
run_stage 04_ogdino_viame_full_4gpu.sh
