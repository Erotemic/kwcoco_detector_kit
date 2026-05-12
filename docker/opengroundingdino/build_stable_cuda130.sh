#!/usr/bin/env bash
# Build the stable OpenGroundingDINO training image: CUDA toolkit 13.0 +
# PyTorch cu130. This image runs on arisia's 595.58.03 / CUDA-13.2-capable
# driver while avoiding PyTorch nightly wheels.
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-kwcoco-detector-kit:ogdino-cu130}"
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:13.0.1-devel-ubuntu24.04}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
TORCH_PRE="${TORCH_PRE:-0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

cd "$(dirname "$0")/../.."

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
