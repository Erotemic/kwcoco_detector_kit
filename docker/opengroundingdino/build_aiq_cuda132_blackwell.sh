#!/usr/bin/env bash
# Build an OpenGroundingDINO / DEIMv2 training image for the aiq-gpu
# cluster: 4x NVIDIA RTX PRO 6000 Blackwell Max-Q + driver 595.58.03 +
# CUDA toolkit/runtime 13.2.
#
# This is the arisia cu132 profile with ONE critical change:
# TORCH_CUDA_ARCH_LIST=12.0. The RTX PRO 6000 Blackwell reports compute
# capability 12.0 (sm_120). The Dockerfile bakes TORCH_CUDA_ARCH_LIST
# into the env and uses it to JIT-compile the MultiScaleDeformableAttention
# CUDA ops at build time. Arisia's default (8.6, Ampere) would produce an
# image whose custom ops have no sm_120 kernel, failing at runtime with
# "no kernel image is available for execution on the device".
#
# CUDA 13.2 + driver 595.58.03 are identical to arisia, so the same
# nightly cu132 PyTorch wheels are used (those wheels ship sm_120 kernels
# for torch's own ops; this arch list governs the kit's compiled ops).
#
# Usage (from anywhere; the script cd's to the repo root):
#   bash docker/opengroundingdino/build_aiq_cuda132_blackwell.sh
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-kwcoco-detector-kit:ogdino-cu132-aiq}"
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:13.2.0-devel-ubuntu24.04}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu132}"
TORCH_PRE="${TORCH_PRE:-1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
# Blackwell sm_120. This is the ONLY value that differs from the arisia
# build. Include 8.6 too (e.g. "8.6;12.0") only if you intend to run the
# same image on Ampere hosts — it enlarges the build but adds portability.
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
# uv parallelizes bytecode compilation across all cores. On this 128-core
# EPYC box that opens far more files at once than the default nofile soft
# limit (1024), which fails the `uv pip install` step with "Too many open
# files (os error 24)". Raise the limit for the build containers.
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

echo "Image tag:            $IMAGE_TAG"
echo "Base image:           $BASE_IMAGE"
echo "Torch index:          $TORCH_INDEX_URL (pre=$TORCH_PRE)"
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST  (Blackwell sm_120)"
echo

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
echo
echo "Smoke-test the Blackwell kernels before training:"
echo "  docker run --rm --gpus all $IMAGE_TAG python3 -c \\"
echo "    'import torch; print(torch.__version__, torch.cuda.get_device_capability())'"
