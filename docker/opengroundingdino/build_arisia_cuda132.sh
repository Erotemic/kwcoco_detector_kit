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
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
# Raise the build containers' open-file limit so uv's parallel bytecode
# compilation doesn't fail with "Too many open files" on high-core hosts.
BUILD_ULIMIT_NOFILE="${BUILD_ULIMIT_NOFILE:-1048576:1048576}"
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

cd "$(dirname "$0")/../.."

ensure_submodule() {
    local path="$1"
    local sentinel="$2"
    if [ ! -f "$sentinel" ]; then
        echo "$path submodule is missing; initializing it now."
        git submodule update --init "$path"
    fi
    if [ ! -f "$sentinel" ]; then
        echo "Failed to initialize $path; cannot build Docker image." >&2
        exit 1
    fi
}

ensure_submodule tpl/Open-GroundingDino tpl/Open-GroundingDino/models/GroundingDINO/ops/setup.py
ensure_submodule tpl/DEIMv2 tpl/DEIMv2/train.py

source "$(dirname "$0")/_build_provenance.sh"
mapfile -t PROV_ARGS < <(kcd_provenance_build_args)

docker build \
    -f docker/opengroundingdino/Dockerfile \
    "${PROV_ARGS[@]}" \
    --ulimit nofile="$BUILD_ULIMIT_NOFILE" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
    --build-arg TORCH_INDEX_URL="$TORCH_INDEX_URL" \
    --build-arg TORCH_PRE="$TORCH_PRE" \
    --build-arg TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    -t "$IMAGE_TAG" \
    .

echo
echo "Built $IMAGE_TAG"
