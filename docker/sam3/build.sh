#!/usr/bin/env bash
# Build the SAM3 admission image for broadly compatible Ampere+ inference.
set -euo pipefail

cd "$(dirname "$0")/../.."

IMAGE_TAG="${KCD_SAM3_IMAGE:-kwcoco-detector-kit:sam3-cu126}"
AUTO_IMAGE_TAG="${KCD_SAM3_AUTO_IMAGE:-kwcoco-detector-kit:sam3-auto}"
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:12.6.3-runtime-ubuntu24.04}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
BUILD_ULIMIT_NOFILE="${BUILD_ULIMIT_NOFILE:-1048576:1048576}"
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

if [ ! -f tpl/sam3/sam3/model_builder.py ]; then
    echo "tpl/sam3 is missing; initializing the SAM3 submodule." >&2
    git submodule update --init --recursive tpl/sam3
fi
if [ ! -f tpl/sam3/sam3/model_builder.py ]; then
    echo "Failed to initialize tpl/sam3; cannot build SAM3 image." >&2
    exit 1
fi

git_sha() {
    local repo="$1" sha dirty
    sha="$(git -c safe.directory='*' -C "$repo" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
    dirty="$(git -c safe.directory='*' -C "$repo" status --porcelain 2>/dev/null || true)"
    [ -n "$dirty" ] && sha="${sha}-dirty"
    printf '%s\n' "$sha"
}

echo "SAM3 Docker profile: cu126"
echo "Base image: $BASE_IMAGE"
echo "Torch: $TORCH_VERSION / torchvision $TORCHVISION_VERSION"
echo "Torch index: $TORCH_INDEX_URL"
echo "Image tags: $AUTO_IMAGE_TAG, $IMAGE_TAG"

cmd=(
    docker build
    -f docker/sam3/Dockerfile
    --ulimit "nofile=$BUILD_ULIMIT_NOFILE"
    --build-arg "BASE_IMAGE=$BASE_IMAGE"
    --build-arg "PYTHON_VERSION=$PYTHON_VERSION"
    --build-arg "TORCH_INDEX_URL=$TORCH_INDEX_URL"
    --build-arg "TORCH_VERSION=$TORCH_VERSION"
    --build-arg "TORCHVISION_VERSION=$TORCHVISION_VERSION"
    --build-arg "KCD_KIT_SHA=$(git_sha .)"
    --build-arg "KCD_SAM3_SHA=$(git_sha tpl/sam3)"
    --build-arg "KCD_DOCKERFILE_SHA=$(sha256sum docker/sam3/Dockerfile | cut -c1-16)"
    --build-arg "KCD_BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    -t "$AUTO_IMAGE_TAG"
    -t "$IMAGE_TAG"
    .
)

if [ "${KCD_DOCKER_DRYRUN:-0}" = "1" ]; then
    printf 'DRY RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

"${cmd[@]}"
