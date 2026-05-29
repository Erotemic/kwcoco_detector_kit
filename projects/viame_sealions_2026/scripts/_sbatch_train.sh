#!/usr/bin/env bash
# Internal boilerplate. Sbatch script — runs on the compute node, sets
# up NCCL diagnostics + dev mounts, then invokes _launch_train.sh
# inside the kit's docker image.
#
# Called by `sbatch` from _submit_train.sh. Don't invoke directly.
# All KCD_* hyperparams must be forwarded via the submit script's
# --export=ALL,KCD_*=...
#
# Resource directives (--gres, --time, --cpus-per-task, --mem) are
# supplied on the sbatch CLI, so this script intentionally omits
# #SBATCH resource lines — the CLI is the single source of truth.
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail

# _submit_train.sh writes all KCD_* env to KCD_ENV_FPATH (a single
# file path that's safe to pass through sbatch --export, unlike the
# comma-separated KEY=VAL form which truncates values containing
# commas). Source it FIRST so the rest of this script sees the env.
if [ -n "${KCD_ENV_FPATH:-}" ] && [ -f "$KCD_ENV_FPATH" ]; then
    source "$KCD_ENV_FPATH"
fi

if [ -z "${KCD_REPO_ROOT:-}" ]; then
    KCD_REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
fi
source "$KCD_REPO_ROOT/scripts/paths.sh"

: "${KCD_RUN_NAME:?_sbatch_train.sh: missing KCD_RUN_NAME (sbatch --export forwarding failed?)}"
: "${KCD_NUM_GPUS:?_sbatch_train.sh: missing KCD_NUM_GPUS}"

KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-arisia}"
KCD_ROOT="$KCD_RUNS_DPATH/$KCD_RUN_NAME"
export KCD_ROOT

echo "=== Slurm context ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<manual>}"
echo "HOSTNAME=$(hostname)"
echo "RUN_NAME=$KCD_RUN_NAME"
echo "KCD_ROOT=$KCD_ROOT"
echo "IMAGE=$KCD_IMAGE"
echo "SCHEME=$KCD_SCHEME  VARIANT=$KCD_VARIANT  GPUS=$KCD_NUM_GPUS"
nvidia-smi -L || true
echo

# NCCL diagnostics. Flight recorder only by default (cheap; only fires
# on watchdog timeout). Set KCD_NCCL_DEBUG=verbose for live collective
# logging during active debugging; KCD_NCCL_DEBUG=0 to disable.
KCD_NCCL_DEBUG="${KCD_NCCL_DEBUG:-1}"
NCCL_DEBUG_FLAGS=()
if [ "$KCD_NCCL_DEBUG" != "0" ]; then
    NCCL_DEBUG_FLAGS=(
        -e TORCH_FR_BUFFER_SIZE=20000
        -e TORCH_NCCL_TRACE_BUFFER_SIZE=20000
        -e TORCH_NCCL_DUMP_ON_TIMEOUT=1
        -e TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$KCD_ROOT/nccl_traces/rank_"
        -e TORCH_NCCL_DESYNC_DEBUG=1
    )
fi
if [ "$KCD_NCCL_DEBUG" = "verbose" ]; then
    NCCL_DEBUG_FLAGS+=(-e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=COLL)
fi

# Dev overrides: mount host source over the image's baked copies so a
# kit-side fix can be tested without rebuilding the docker image
# (~10-15 min each rebuild).
DEV_MOUNT_FLAGS=()
if [ "${KCD_DEV_MOUNT_DEIMV2:-0}" = "1" ]; then
    if [ -d "$KCD_KIT_DPATH/tpl/DEIMv2" ]; then
        DEV_MOUNT_FLAGS+=(-v "$KCD_KIT_DPATH/tpl/DEIMv2:/opt/kwcoco_detector_kit/tpl/DEIMv2")
        echo "DEV: mounting host tpl/DEIMv2 over image's copy" >&2
    else
        echo "WARNING: KCD_DEV_MOUNT_DEIMV2=1 but $KCD_KIT_DPATH/tpl/DEIMv2 not found" >&2
    fi
fi
if [ "${KCD_DEV_MOUNT_KIT:-0}" = "1" ]; then
    # Image installs the kit via `pip install -e .` at /opt/kwcoco_detector_kit/.
    # Overlaying the Python package dir lets us iterate on kit code (sweep,
    # tile, configs, ...) without a rebuild. tpl/ submodules and the entry
    # points stay baked.
    if [ -d "$KCD_KIT_DPATH/kwcoco_detector_kit" ]; then
        DEV_MOUNT_FLAGS+=(-v "$KCD_KIT_DPATH/kwcoco_detector_kit:/opt/kwcoco_detector_kit/kwcoco_detector_kit")
        echo "DEV: mounting host kwcoco_detector_kit/ over image's copy" >&2
    else
        echo "WARNING: KCD_DEV_MOUNT_KIT=1 but $KCD_KIT_DPATH/kwcoco_detector_kit not found" >&2
    fi
fi

# Shared shm size scales with gpu count: DataLoader workers' ipc.
shm_gb=$(( 16 + 8 * KCD_NUM_GPUS ))

# Forward every KCD_* var into the docker container. Wildcard (vs an
# explicit list) mirrors _submit_train.sh's env-file write: adding a
# new KCD_* knob in the launcher never requires also remembering to
# update this list. The KCD_ namespace is reserved for our config so
# this won't sweep in unrelated state. `compgen -v` (vs `-e`) catches
# vars set without `export` too.
KCD_ENV_FLAGS=()
while IFS= read -r v; do
    val="${!v:-}"
    if [ -n "$val" ]; then
        KCD_ENV_FLAGS+=(-e "$v=$val")
    fi
done < <(compgen -v | grep -E '^KCD_' | sort -u)
echo "[_sbatch_train.sh] forwarding ${#KCD_ENV_FLAGS[@]} KCD_* values to docker run (${#KCD_ENV_FLAGS[@]} = 2 * n_vars)"

docker run --rm \
    --gpus all \
    --ipc=host \
    --shm-size="${shm_gb}g" \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
    -e SLURM_JOB_ID="${SLURM_JOB_ID:-manual}" \
    "${KCD_ENV_FLAGS[@]}" \
    "${NCCL_DEBUG_FLAGS[@]}" \
    -v "$KCD_DATA_ROOT:$KCD_DATA_ROOT" \
    -v "$KCD_DATA_DPATH:$KCD_DATA_DPATH" \
    -v "$KCD_REPO_ROOT:$KCD_REPO_ROOT" \
    "${DEV_MOUNT_FLAGS[@]}" \
    -w "$KCD_REPO_ROOT" \
    "$KCD_IMAGE" \
    bash "$KCD_REPO_ROOT/scripts/_launch_train.sh"

echo
echo "Done. Output under: $KCD_ROOT"
