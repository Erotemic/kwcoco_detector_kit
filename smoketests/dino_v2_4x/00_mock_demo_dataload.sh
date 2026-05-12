#!/usr/bin/env bash
# CPU / tiny-model smoke: generate kwcoco demo data and run mock_tiny through
# the sweep path. This proves the container can import the kit, read kwcoco
# images, write run artifacts, and exercise the orchestration layer.
set -euo pipefail
source "$(dirname "$0")/common.sh"

ROOT="$KCD_SMOKE_ROOT/00_mock_demo_dataload"
DATA="$ROOT/data"
RUN_ROOT="$ROOT/run"

print_env_summary
make_demo_splits "$DATA"
run_mock_sweep "$RUN_ROOT" \
    "$DATA/train.kwcoco.zip" \
    "$DATA/vali.kwcoco.zip" \
    "$DATA/test.kwcoco.zip"

echo
echo "PASS: mock demo dataload smoke"
echo "RUN_ROOT=$RUN_ROOT"
