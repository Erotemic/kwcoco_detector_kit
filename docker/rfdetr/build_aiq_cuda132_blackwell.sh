#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-kwcoco-detector-kit:rfdetr-cu132-aiq}"
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:13.2.0-runtime-ubuntu24.04}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu132}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_PRE="${TORCH_PRE:-1}"
BUILD_ULIMIT_NOFILE="${BUILD_ULIMIT_NOFILE:-1048576:1048576}"
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

cd "$(dirname "$0")/../.."
if [ ! -f tpl/rf-detr/src/rfdetr/__init__.py ]; then
    git submodule update --init tpl/rf-detr
fi

git_sha() {
    local repo="$1" sha dirty
    sha="$(git -C "$repo" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
    dirty="$(git -C "$repo" status --porcelain 2>/dev/null || true)"
    [ -n "$dirty" ] && sha="${sha}-dirty"
    echo "$sha"
}

docker build \
    -f docker/rfdetr/Dockerfile \
    --ulimit nofile="$BUILD_ULIMIT_NOFILE" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
    --build-arg TORCH_INDEX_URL="$TORCH_INDEX_URL" \
    --build-arg TORCH_PRE="$TORCH_PRE" \
    --build-arg KCD_KIT_SHA="$(git_sha .)" \
    --build-arg KCD_RFDETR_SHA="$(git_sha tpl/rf-detr)" \
    --build-arg KCD_DOCKERFILE_SHA="$(sha256sum docker/rfdetr/Dockerfile | cut -c1-16)" \
    --build-arg KCD_BUILD_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    -t "$IMAGE_TAG" .

echo "Built $IMAGE_TAG"
echo "GPU smoke: docker run --rm --gpus all $IMAGE_TAG 'python -c \"import torch, rfdetr; print(torch.__version__, torch.cuda.get_device_name())\"'"
