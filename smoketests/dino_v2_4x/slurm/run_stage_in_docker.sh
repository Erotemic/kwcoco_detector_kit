#!/usr/bin/env bash
# Slurm job payload: run one smoke stage inside the Docker image.
set -euo pipefail

STAGE_SCRIPT="${STAGE_SCRIPT:?set STAGE_SCRIPT relative to smoketests/dino_v2_4x}"
IMAGE_TAG="${IMAGE_TAG:-kwcoco-detector-kit:ogdino-cu132-arisia}"

HOST_KIT_DPATH="${HOST_KIT_DPATH:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONTAINER_KIT_DPATH="${CONTAINER_KIT_DPATH:-/workspace/kwcoco_detector_kit}"
if [ -z "${DATA_DPATH:-}" ]; then
    for cand in \
        /data/users/jon.crall/dvc-repos/viame_sealions_2026 \
        /media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026
    do
        if [ -d "$cand" ]; then
            DATA_DPATH="$cand"
            break
        fi
    done
    DATA_DPATH="${DATA_DPATH:-/data/users/jon.crall/dvc-repos/viame_sealions_2026}"
fi

if [ -z "${KCD_EXPT_DPATH:-}" ]; then
    if [ -d /data/users/jon.crall/dvc-repos ] || [ -d /data/users/jon.crall/dvc-repos/viame_sealions_2026_expt ]; then
        KCD_EXPT_DPATH=/data/users/jon.crall/dvc-repos/viame_sealions_2026_expt
    else
        KCD_EXPT_DPATH="${SCRATCH:-/tmp}/kcd_smoketests"
    fi
fi

KCD_SMOKE_ROOT_HOST="${KCD_SMOKE_ROOT_HOST:-$KCD_EXPT_DPATH/smoketests/dino_v2_4x}"
KCD_CACHE_ROOT_HOST="${KCD_CACHE_ROOT_HOST:-$KCD_EXPT_DPATH/cache/opengroundingdino}"

CONTAINER_KCD_SMOKE_ROOT="${CONTAINER_KCD_SMOKE_ROOT:-$KCD_SMOKE_ROOT_HOST}"
CONTAINER_KCD_CACHE_ROOT="${CONTAINER_KCD_CACHE_ROOT:-$KCD_CACHE_ROOT_HOST}"

NUM_GPUS_REQUESTED="${NUM_GPUS_REQUESTED:-0}"
DOCKER_SHM_SIZE="${DOCKER_SHM_SIZE:-32g}"
KCD_RUNTIME_PIP_DEPS="${KCD_RUNTIME_PIP_DEPS:-colorlog transformers>=4.35,<4.47}"

mkdir -p "$KCD_EXPT_DPATH" "$KCD_SMOKE_ROOT_HOST" "$KCD_CACHE_ROOT_HOST"

echo "=== Slurm context ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-}"
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NUM_GPUS_REQUESTED=$NUM_GPUS_REQUESTED"
echo "DATA_DPATH=$DATA_DPATH"
echo "KCD_EXPT_DPATH=$KCD_EXPT_DPATH"
echo "KCD_SMOKE_ROOT_HOST=$KCD_SMOKE_ROOT_HOST"
echo "KCD_CACHE_ROOT_HOST=$KCD_CACHE_ROOT_HOST"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
fi

docker_args=(
    run --rm
    --ipc=host
    --shm-size="$DOCKER_SHM_SIZE"
    -v "$HOST_KIT_DPATH:$CONTAINER_KIT_DPATH"
    -v "$KCD_SMOKE_ROOT_HOST:$CONTAINER_KCD_SMOKE_ROOT"
    -v "$KCD_CACHE_ROOT_HOST:$CONTAINER_KCD_CACHE_ROOT"
    -w "$CONTAINER_KIT_DPATH"
    -e KIT_DPATH="$CONTAINER_KIT_DPATH"
    -e STAGE_SCRIPT="$STAGE_SCRIPT"
    -e KCD_SMOKE_ROOT="$CONTAINER_KCD_SMOKE_ROOT"
    -e KCD_CACHE_ROOT="$CONTAINER_KCD_CACHE_ROOT"
    -e PYTHON_BIN=python
    -e NUM_EPOCHS="${NUM_EPOCHS:-}"
    -e INPUT_SIZE="${INPUT_SIZE:-}"
    -e BATCH_SIZE="${BATCH_SIZE:-}"
    -e VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-}"
    -e VIAME_SUBSET_TRAIN_IMAGES="${VIAME_SUBSET_TRAIN_IMAGES:-}"
    -e VIAME_SUBSET_VALI_IMAGES="${VIAME_SUBSET_VALI_IMAGES:-}"
    -e VIAME_SUBSET_TEST_IMAGES="${VIAME_SUBSET_TEST_IMAGES:-}"
    -e KCD_RUNTIME_PIP_DEPS="$KCD_RUNTIME_PIP_DEPS"
)

if [ -d "$DATA_DPATH" ]; then
    docker_args+=(
        -v "$DATA_DPATH:$DATA_DPATH:ro"
        -e DATA_DPATH="$DATA_DPATH"
    )
else
    echo "WARNING: DATA_DPATH does not exist on host: $DATA_DPATH" >&2
fi

if [ "${NUM_GPUS_REQUESTED}" != "0" ]; then
    if [ -n "${DOCKER_GPUS:-}" ]; then
        docker_args+=(--gpus "$DOCKER_GPUS")
    elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        # Docker's --gpus parser needs nested quotes for comma-separated
        # device lists. Without them, `device=0,1,2,3` can be interpreted as
        # both a Count and DeviceIDs request.
        docker_args+=(--gpus "\"device=$CUDA_VISIBLE_DEVICES\"")
    else
        docker_args+=(--gpus all)
    fi
fi

docker_args+=(
    "$IMAGE_TAG"
    bash -lc '
        set -euo pipefail
        if [ -n "${KCD_RUNTIME_PIP_DEPS:-}" ]; then
            python - <<PY
import importlib.util
import importlib.metadata
import subprocess
import sys

from packaging.requirements import Requirement

deps = [d for d in "${KCD_RUNTIME_PIP_DEPS}".split() if d]
missing = []
for dep in deps:
    req = Requirement(dep)
    module = req.name.replace("-", "_")
    if importlib.util.find_spec(module) is None:
        missing.append(dep)
        continue
    try:
        version = importlib.metadata.version(req.name)
    except importlib.metadata.PackageNotFoundError:
        missing.append(dep)
        continue
    if req.specifier and not req.specifier.contains(version, prereleases=True):
        missing.append(dep)
if missing:
    print("[kcd-runtime-patch] installing missing deps: " + " ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
else:
    print("[kcd-runtime-patch] runtime pip deps already present")
PY
        fi
        : "${STAGE_SCRIPT:?container missing STAGE_SCRIPT}"
        bash "$KIT_DPATH/smoketests/dino_v2_4x/$STAGE_SCRIPT"
    '
)

echo
echo "=== Docker command ==="
printf '+'
printf ' %q' docker "${docker_args[@]}"
echo

docker "${docker_args[@]}"
