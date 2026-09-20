#!/usr/bin/env bash
# Explicit stable RF-DETR build profile for Ampere/Ada/Hopper workstations.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export KCD_RFDETR_CUDA_PROFILE=cu130
export AUTO_IMAGE_TAG="${IMAGE_TAG:-kwcoco-detector-kit:rfdetr-cu130}"
export TAG_VARIANT=0
exec "$HERE/build_auto.sh"
