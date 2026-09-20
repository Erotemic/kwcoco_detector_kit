#!/usr/bin/env bash
# Build the RF-DETR image using a host-appropriate CUDA/PyTorch profile.
#
# Policy:
#   * Ampere/Ada/Hopper-class hosts use stable PyTorch cu130 by default.
#   * Blackwell (compute capability >= 12.0) uses the existing cu132 nightly
#     profile when the host driver supports CUDA >= 13.2.
#
# Override with KCD_RFDETR_CUDA_PROFILE=cu130|cu132.
set -euo pipefail

cd "$(dirname "$0")/../.."

AUTO_IMAGE_TAG="${AUTO_IMAGE_TAG:-kwcoco-detector-kit:rfdetr-auto}"
TAG_VARIANT="${TAG_VARIANT:-1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
BUILD_ULIMIT_NOFILE="${BUILD_ULIMIT_NOFILE:-1048576:1048576}"
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

_version_ge() {
    local lhs rhs
    lhs="$(printf '%s\n' "$1" | awk -F. '{printf "%03d%03d%03d\n", $1, $2, $3}')"
    rhs="$(printf '%s\n' "$2" | awk -F. '{printf "%03d%03d%03d\n", $1, $2, $3}')"
    [ "$lhs" -ge "$rhs" ]
}

_cap_ge() {
    awk -v lhs="$1" -v rhs="$2" 'BEGIN { exit !(lhs + 0 >= rhs + 0) }'
}

_find_nvidia_smi() {
    local p
    for p in nvidia-smi /usr/bin/nvidia-smi /usr/local/cuda/bin/nvidia-smi \
             /usr/lib/nvidia-smi /opt/nvidia/bin/nvidia-smi; do
        if command -v "$p" >/dev/null 2>&1 || [ -x "$p" ]; then
            printf '%s\n' "$p"
            return 0
        fi
    done
    return 1
}

_detect_host_cuda() {
    if [ -n "${HOST_CUDA_VERSION:-}" ]; then
        printf '%s\n' "$HOST_CUDA_VERSION"
        return 0
    fi
    local smi ver
    smi="$(_find_nvidia_smi 2>/dev/null || true)"
    if [ -n "$smi" ]; then
        ver="$("$smi" 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -1)"
        if [ -n "$ver" ]; then
            printf '%s\n' "$ver"
            return 0
        fi
    fi
    return 1
}

_detect_max_compute_cap() {
    if [ -n "${HOST_COMPUTE_CAP:-}" ]; then
        printf '%s\n' "$HOST_COMPUTE_CAP"
        return 0
    fi
    local smi
    smi="$(_find_nvidia_smi 2>/dev/null || true)"
    [ -n "$smi" ] || return 1
    "$smi" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | awk 'NF { if ($1 + 0 > max) max = $1 + 0; seen=1 } END { if (seen) printf "%.1f\n", max }'
}

if [ ! -f tpl/rf-detr/src/rfdetr/__init__.py ]; then
    echo "tpl/rf-detr is missing; initializing the RF-DETR submodule."
    git submodule update --init tpl/rf-detr
fi
if [ ! -f tpl/rf-detr/src/rfdetr/__init__.py ]; then
    echo "Failed to initialize tpl/rf-detr; cannot build RF-DETR image." >&2
    exit 1
fi

profile="${KCD_RFDETR_CUDA_PROFILE:-${KCD_DOCKER_CUDA_PROFILE:-auto}}"
host_cuda="$(_detect_host_cuda || true)"
compute_cap="$(_detect_max_compute_cap || true)"

if [ "$profile" = "auto" ]; then
    if [ -z "$host_cuda" ]; then
        echo "Could not detect host CUDA from nvidia-smi." >&2
        echo "Set HOST_CUDA_VERSION or KCD_RFDETR_CUDA_PROFILE=cu130|cu132." >&2
        exit 1
    fi
    if [ -n "$compute_cap" ] && _cap_ge "$compute_cap" "12.0"; then
        if _version_ge "$host_cuda" "13.2"; then
            profile="cu132"
        else
            echo "Blackwell-class GPU detected (compute capability $compute_cap)," >&2
            echo "but the host reports CUDA $host_cuda; the known RF-DETR Blackwell" >&2
            echo "profile requires CUDA >= 13.2." >&2
            exit 1
        fi
    elif _version_ge "$host_cuda" "13.0"; then
        # Stable cu130 is intentional on Ampere/Ada/Hopper even when a newer
        # driver advertises CUDA 13.2. A newer driver can run the cu130 image.
        profile="cu130"
    else
        echo "Host reports CUDA $host_cuda; supported RF-DETR profiles require CUDA >= 13.0." >&2
        exit 1
    fi
fi

case "$profile" in
    cu130|cuda130|stable)
        profile="cu130"
        VARIANT_IMAGE_TAG="${VARIANT_IMAGE_TAG:-kwcoco-detector-kit:rfdetr-cu130}"
        BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:13.0.1-runtime-ubuntu24.04}"
        TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
        TORCH_PRE="${TORCH_PRE:-0}"
        ;;
    cu132|cuda132|blackwell|aiq)
        profile="cu132"
        VARIANT_IMAGE_TAG="${VARIANT_IMAGE_TAG:-kwcoco-detector-kit:rfdetr-cu132}"
        BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:13.2.0-runtime-ubuntu24.04}"
        TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu132}"
        TORCH_PRE="${TORCH_PRE:-1}"
        ;;
    *)
        echo "Unknown RF-DETR CUDA profile: $profile" >&2
        echo "Expected auto, cu130, or cu132." >&2
        exit 1
        ;;
esac

git_sha() {
    local repo="$1" sha dirty
    sha="$(git -C "$repo" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
    dirty="$(git -C "$repo" status --porcelain 2>/dev/null || true)"
    [ -n "$dirty" ] && sha="${sha}-dirty"
    printf '%s\n' "$sha"
}

echo "Host CUDA: ${host_cuda:-unknown}"
echo "Max GPU compute capability: ${compute_cap:-unknown}"
echo "Selected RF-DETR profile: $profile"
echo "Base image: $BASE_IMAGE"
echo "Torch index: $TORCH_INDEX_URL"
echo "Auto tag: $AUTO_IMAGE_TAG"
if [ "$TAG_VARIANT" = "1" ]; then
    echo "Variant tag: $VARIANT_IMAGE_TAG"
fi

tags=(-t "$AUTO_IMAGE_TAG")
if [ "$TAG_VARIANT" = "1" ]; then
    tags+=(-t "$VARIANT_IMAGE_TAG")
fi

cmd=(
    docker build
    -f docker/rfdetr/Dockerfile
    --ulimit "nofile=$BUILD_ULIMIT_NOFILE"
    --build-arg "BASE_IMAGE=$BASE_IMAGE"
    --build-arg "PYTHON_VERSION=$PYTHON_VERSION"
    --build-arg "TORCH_INDEX_URL=$TORCH_INDEX_URL"
    --build-arg "TORCH_PRE=$TORCH_PRE"
    --build-arg "KCD_KIT_SHA=$(git_sha .)"
    --build-arg "KCD_RFDETR_SHA=$(git_sha tpl/rf-detr)"
    --build-arg "KCD_DOCKERFILE_SHA=$(sha256sum docker/rfdetr/Dockerfile | cut -c1-16)"
    --build-arg "KCD_BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "${tags[@]}"
    .
)

if [ "${KCD_DOCKER_DRYRUN:-0}" = "1" ]; then
    printf 'DRY RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

"${cmd[@]}"

echo
echo "Built $AUTO_IMAGE_TAG using RF-DETR profile $profile"
if [ "$TAG_VARIANT" = "1" ]; then
    echo "Also tagged $VARIANT_IMAGE_TAG"
fi
