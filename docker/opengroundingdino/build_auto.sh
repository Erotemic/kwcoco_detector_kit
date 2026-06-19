#!/usr/bin/env bash
# Auto-select an OpenGroundingDINO Docker CUDA profile for the current host.
#
# The selected image is always tagged as:
#
#     kwcoco-detector-kit:ogdino-auto
#
# It is also tagged with the profile-specific tag unless TAG_VARIANT=0.
set -euo pipefail

cd "$(dirname "$0")/../.."

AUTO_IMAGE_TAG="${AUTO_IMAGE_TAG:-kwcoco-detector-kit:ogdino-auto}"
TAG_VARIANT="${TAG_VARIANT:-1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
# Raise the build containers' open-file limit. uv compiles bytecode in
# parallel across all cores; on high-core hosts that exhausts the default
# nofile soft limit (1024) and fails `uv pip install` with "Too many open
# files (os error 24)".
BUILD_ULIMIT_NOFILE="${BUILD_ULIMIT_NOFILE:-1048576:1048576}"
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

_version_ge() {
    # Return true when $1 >= $2 for simple dotted numeric versions.
    local lhs rhs
    lhs="$(printf '%s\n' "$1" | awk -F. '{printf "%03d%03d%03d\n", $1, $2, $3}')"
    rhs="$(printf '%s\n' "$2" | awk -F. '{printf "%03d%03d%03d\n", $1, $2, $3}')"
    [ "$lhs" -ge "$rhs" ]
}

_find_nvidia_smi() {
    # Try PATH first, then common system locations (aiq-gpu and other managed
    # clusters may not have nvidia-smi on the login-shell PATH).
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

    # Primary: nvidia-smi (most authoritative — reports driver CUDA level).
    local smi ver
    smi="$(_find_nvidia_smi 2>/dev/null || true)"
    if [ -n "$smi" ]; then
        ver="$("$smi" 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -1)"
        if [ -n "$ver" ]; then
            printf '%s\n' "$ver"
            return 0
        fi
    fi

    # Fallback 1: nvcc --version (toolkit, may lag the driver slightly).
    local nvcc_out
    nvcc_out="$(nvcc --version 2>/dev/null || true)"
    ver="$(printf '%s\n' "$nvcc_out" | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -1)"
    if [ -n "$ver" ]; then
        echo "NOTE: nvidia-smi not found on PATH; CUDA version detected from nvcc." >&2
        printf '%s\n' "$ver"
        return 0
    fi

    # Fallback 2: CUDA version.json (present in CUDA 11.4+ toolkit installs).
    local vf
    for vf in /usr/local/cuda/version.json /usr/local/cuda-*/version.json; do
        [ -f "$vf" ] || continue
        ver="$(python3 -c "
import json, sys
try:
    d = json.load(open('$vf'))
    print(d.get('cuda', d).get('version', ''))
except Exception:
    pass
" 2>/dev/null || true)"
        if [ -n "$ver" ]; then
            echo "NOTE: nvidia-smi not found on PATH; CUDA version detected from $vf." >&2
            printf '%s\n' "$ver"
            return 0
        fi
    done

    return 1
}

_detect_arch_list() {
    # Derive TORCH_CUDA_ARCH_LIST from the GPU's compute capability so the
    # MultiScaleDeformableAttention ops are compiled with a kernel that
    # matches the device. Hardcoding the wrong arch (e.g. 8.6 on Blackwell
    # sm_120) produces an image that fails at runtime with "no kernel image
    # is available for execution on the device". Override by exporting
    # TORCH_CUDA_ARCH_LIST yourself.
    local smi
    smi="$(_find_nvidia_smi 2>/dev/null || true)"
    if [ -z "$smi" ]; then
        return 1
    fi
    # compute_cap is reported like "8.6", "9.0", "12.0". Take the unique
    # set across all GPUs and join with ';' so a mixed box still works.
    "$smi" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | awk 'NF' | sort -u | paste -sd';' -
}

_ensure_one_submodule() {
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

_ensure_submodule() {
    _ensure_one_submodule tpl/Open-GroundingDino tpl/Open-GroundingDino/models/GroundingDINO/ops/setup.py
    _ensure_one_submodule tpl/DEIMv2 tpl/DEIMv2/train.py
}

profile="${KCD_DOCKER_CUDA_PROFILE:-${CUDA_PROFILE:-auto}}"
host_cuda="$(_detect_host_cuda || true)"

if [ "$profile" = "auto" ]; then
    if [ -z "$host_cuda" ]; then
        echo "Could not detect host CUDA (tried nvidia-smi, nvcc, version.json)." >&2
        echo "Override with one of:" >&2
        echo "  HOST_CUDA_VERSION=13.2                    # let auto pick the profile" >&2
        echo "  KCD_DOCKER_CUDA_PROFILE=aiq               # aiq-gpu: Blackwell sm_120 + cu132" >&2
        echo "  KCD_DOCKER_CUDA_PROFILE=cu132             # arisia: Ampere sm_86 + cu132" >&2
        echo "  KCD_DOCKER_CUDA_PROFILE=cu130             # stable cu130" >&2
        exit 1
    fi
    if _version_ge "$host_cuda" "13.2"; then
        profile="cu132"
    elif _version_ge "$host_cuda" "13.0"; then
        profile="cu130"
    else
        echo "Host reports CUDA $host_cuda, but the supported OGDino profiles require CUDA >= 13.0." >&2
        exit 1
    fi
fi

case "$profile" in
    aiq|blackwell)
        # aiq-gpu: 4x RTX PRO 6000 Blackwell (sm_120) + CUDA 13.2.
        # Same cu132 base as arisia; only the arch list and image tag differ.
        profile="cu132"
        VARIANT_IMAGE_TAG="${VARIANT_IMAGE_TAG:-kwcoco-detector-kit:ogdino-cu132-aiq}"
        BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:13.2.0-devel-ubuntu24.04}"
        TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu132}"
        TORCH_PRE="${TORCH_PRE:-1}"
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
        ;;
    cu132|cuda132|arisia)
        profile="cu132"
        VARIANT_IMAGE_TAG="${VARIANT_IMAGE_TAG:-kwcoco-detector-kit:ogdino-cu132-arisia}"
        BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:13.2.0-devel-ubuntu24.04}"
        TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu132}"
        TORCH_PRE="${TORCH_PRE:-1}"
        ;;
    cu130|cuda130|stable)
        profile="cu130"
        VARIANT_IMAGE_TAG="${VARIANT_IMAGE_TAG:-kwcoco-detector-kit:ogdino-cu130}"
        BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:13.0.1-devel-ubuntu24.04}"
        TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
        TORCH_PRE="${TORCH_PRE:-0}"
        ;;
    *)
        echo "Unknown KCD_DOCKER_CUDA_PROFILE=$profile; expected auto, aiq, cu130, or cu132." >&2
        exit 1
        ;;
esac

# Resolve the CUDA arch list. Prefer an explicit override; otherwise
# detect it from the GPU compute capability; fall back to Ampere 8.6
# (arisia) only if detection fails.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    TORCH_CUDA_ARCH_LIST="$(_detect_arch_list || true)"
    if [ -n "$TORCH_CUDA_ARCH_LIST" ]; then
        echo "Detected GPU compute capability -> TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
    else
        TORCH_CUDA_ARCH_LIST="8.6"
        echo "WARNING: could not detect GPU compute capability; defaulting" >&2
        echo "         TORCH_CUDA_ARCH_LIST=8.6 (Ampere). Override if wrong." >&2
    fi
fi

echo "Host CUDA: ${host_cuda:-unknown}"
echo "Selected profile: $profile"
echo "Base image: $BASE_IMAGE"
echo "Torch index: $TORCH_INDEX_URL"
echo "Arch list: $TORCH_CUDA_ARCH_LIST"
echo "Auto tag: $AUTO_IMAGE_TAG"
if [ "$TAG_VARIANT" = "1" ]; then
    echo "Variant tag: $VARIANT_IMAGE_TAG"
fi

source "$(dirname "$0")/_build_provenance.sh"
mapfile -t PROV_ARGS < <(kcd_provenance_build_args)

tags=(-t "$AUTO_IMAGE_TAG")
if [ "$TAG_VARIANT" = "1" ]; then
    tags+=(-t "$VARIANT_IMAGE_TAG")
fi

cmd=(
    docker build
    -f docker/opengroundingdino/Dockerfile
    --ulimit "nofile=$BUILD_ULIMIT_NOFILE"
    --build-arg "BASE_IMAGE=$BASE_IMAGE"
    --build-arg "PYTHON_VERSION=$PYTHON_VERSION"
    --build-arg "TORCH_INDEX_URL=$TORCH_INDEX_URL"
    --build-arg "TORCH_PRE=$TORCH_PRE"
    --build-arg "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
    "${PROV_ARGS[@]}"
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
echo "Built $AUTO_IMAGE_TAG using profile $profile"
if [ "$TAG_VARIANT" = "1" ]; then
    echo "Also tagged $VARIANT_IMAGE_TAG"
fi
