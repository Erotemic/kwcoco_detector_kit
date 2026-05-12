#!/usr/bin/env bash
# Build an OpenGroundingDINO training image for arisia-style hosts:
# Docker 28 + NVIDIA driver 595.58.03 + CUDA toolkit/runtime 13.2.
#
# PyTorch CUDA 13.2 wheels are currently expected from the PyTorch nightly
# CUDA 13.2 index. If your cluster mirror provides stable cu132 wheels, set
# TORCH_PRE=0 and TORCH_INDEX_URL accordingly.
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-kwcoco-detector-kit:ogdino-cu132-arisia}"
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:13.2.0-devel-ubuntu24.04}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu132}"
TORCH_PRE="${TORCH_PRE:-1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

cd "$(dirname "$0")/../.."

if [ ! -f tpl/Open-GroundingDino/models/GroundingDINO/ops/setup.py ]; then
    echo "Open-GroundingDino submodule is missing; initializing it now."
    git submodule update --init tpl/Open-GroundingDino
fi
if [ ! -f tpl/Open-GroundingDino/models/GroundingDINO/ops/setup.py ]; then
    echo "Failed to initialize tpl/Open-GroundingDino; cannot build Docker image." >&2
    exit 1
fi

docker build \
    -f docker/opengroundingdino/Dockerfile \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
    --build-arg TORCH_INDEX_URL="$TORCH_INDEX_URL" \
    --build-arg TORCH_PRE="$TORCH_PRE" \
    -t "$IMAGE_TAG" \
    .

echo
echo "Built $IMAGE_TAG"
