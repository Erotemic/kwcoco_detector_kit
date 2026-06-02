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
    # Heartbeat watchdog: if a NCCL collective doesn't complete
    # within KCD_NCCL_HEARTBEAT_TIMEOUT_SEC (default 600s), torch
    # raises a real error instead of hanging silently. Job 2556
    # (2026-05-31 → 2026-06-01) hung 30+ hours in loss.backward()
    # because a stuck NCCL allreduce never timed out; this would
    # have killed it after 10 minutes with an actionable traceback.
    NCCL_DEBUG_FLAGS+=(
        -e "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${KCD_NCCL_HEARTBEAT_TIMEOUT_SEC:-600}"
        # Blocking wait makes collective failures synchronous —
        # the traceback then points at the actual stuck call site
        # instead of bubbling up later from a watcher thread.
        -e "TORCH_NCCL_BLOCKING_WAIT=${KCD_NCCL_BLOCKING_WAIT:-1}"
        -e "TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
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
if [ "${KCD_DEV_MOUNT_DATALOADER:-0}" = "1" ]; then
    # Mirror of KCD_DEV_MOUNT_DEIMV2 for the kwcoco_dataloader submodule.
    # The image installs the submodule via `pip install -e
    # /opt/kwcoco_detector_kit/tpl/kwcoco_dataloader`, so overlaying the
    # source dir lets reader/writer edits take effect without a rebuild.
    if [ -d "$KCD_KIT_DPATH/tpl/kwcoco_dataloader/kwcoco_dataloader" ]; then
        DEV_MOUNT_FLAGS+=(-v "$KCD_KIT_DPATH/tpl/kwcoco_dataloader/kwcoco_dataloader:/opt/kwcoco_detector_kit/tpl/kwcoco_dataloader/kwcoco_dataloader")
        echo "DEV: mounting host tpl/kwcoco_dataloader/kwcoco_dataloader over image's copy" >&2
    else
        echo "WARNING: KCD_DEV_MOUNT_DATALOADER=1 but $KCD_KIT_DPATH/tpl/kwcoco_dataloader/kwcoco_dataloader not found" >&2
    fi
fi

# Extra host->container bind mounts. Use case: data on a path outside
# the default $KCD_DATA_ROOT / $KCD_DATA_DPATH / $KCD_REPO_ROOT mounts,
# typically when the data path is a SYMLINK to a target the container
# can't see (e.g. /data/.../ssd-data -> /home/.../ssd-data with /home
# not bind-mounted). gen003 single_sealion 2026-06-01 hit this.
#
# Format: colon-separated list of host paths to mount at the same
# path inside the container. The kit's `mkdir -p` etc. then succeeds
# because the symlink resolves through to a real directory the
# container CAN see.
#
#   export KCD_EXTRA_MOUNTS="/home/.../ssd-data:/some/other/path"
EXTRA_MOUNT_FLAGS=()
if [ -n "${KCD_EXTRA_MOUNTS:-}" ]; then
    IFS=':' read -ra _extra <<< "$KCD_EXTRA_MOUNTS"
    for _p in "${_extra[@]}"; do
        if [ -n "$_p" ]; then
            EXTRA_MOUNT_FLAGS+=(-v "$_p:$_p")
            echo "EXTRA MOUNT: $_p" >&2
        fi
    done
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

# Respect slurm's per-job GPU allocation. CUDA_VISIBLE_DEVICES is set
# by slurm to the host-side indices of the GPUs assigned to this
# job (e.g. "0" for job A, "1" for job B). `--gpus all` would
# override slurm's allocation by exposing every physical GPU to
# every container; concurrent jobs then all default to "GPU 0"
# inside the container and collide on the same physical device
# (gen002 OOM 2026-05-30: jobs 2508 + 2537 both landed on UUID
# ebfc1af1 with 45 GB held by the other). `--gpus device=<idx>`
# pins docker to exactly the GPUs slurm reserved.
# Docker --gpus=device=<host-idx> exposes ONLY that physical GPU to the
# container, but inside the container that GPU is remapped to logical
# index 0 (the only visible device). So:
#   - host side: CUDA_VISIBLE_DEVICES=1 (slurm's pin to physical GPU 1)
#   - --gpus=device=1 → container sees one device, at logical idx 0
#   - container side: CUDA_VISIBLE_DEVICES MUST be "0" to match the
#     post-remap logical index, or "" to let torch see all visible
#     devices. Forwarding the host idx (1) makes torch look for logical
#     device 1 inside the container — which doesn't exist — and silently
#     ends up with zero usable GPUs, model stays on CPU, DDP errors
#     ("module parameters {device(type='cpu')}"). gen002 2544 hit this.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # Multi-GPU: docker's --gpus value parser splits on commas and
    # mis-interprets any purely-numeric segment as a Count, then
    # errors "cannot set both Count and DeviceIDs on device request"
    # when DeviceIDs are also set. Reproduced 2026-06-01 on arisia
    # twice (jobs 2572, 2574) with --gpus device=0,1 — both the
    # equals form and the space form hit this.
    #
    # Workaround: pass GPU UUIDs instead of indices. UUIDs always
    # contain hex letters, so no comma-separated segment can be
    # mistaken for a count. Get them from nvidia-smi keyed on
    # the slurm-pinned indices in CUDA_VISIBLE_DEVICES.
    GPU_UUIDS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader \
                  -i "$CUDA_VISIBLE_DEVICES" 2>/dev/null \
                | tr '\n' ',' | sed 's/,$//')
    if [ -n "$GPU_UUIDS" ]; then
        GPU_FLAG=(--gpus "device=${GPU_UUIDS}")
    else
        # nvidia-smi unavailable or refused our indices; fall back
        # to the index form. Single-GPU jobs (no comma) work fine
        # this way; multi-GPU will fail at docker — surface that
        # rather than silently mis-running.
        GPU_FLAG=(--gpus "device=${CUDA_VISIBLE_DEVICES}")
        echo "WARN: nvidia-smi UUID lookup failed; falling back to indices" >&2
    fi
    # Build container-side device list: 0,1,2,... up to the count slurm
    # gave us. Slurm sets CUDA_VISIBLE_DEVICES to a comma-separated host
    # index list ("0", "1,2", etc.); count its entries.
    n_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
    CONTAINER_CUDA_VISIBLE=$(seq -s, 0 $((n_gpus - 1)))
else
    GPU_FLAG=(--gpus all)
    CONTAINER_CUDA_VISIBLE=""
    echo "WARN: CUDA_VISIBLE_DEVICES unset; using --gpus all (collision risk)" >&2
fi
docker run --rm \
    "${GPU_FLAG[@]}" \
    --ipc=host \
    --shm-size="${shm_gb}g" \
    -e CUDA_VISIBLE_DEVICES="${CONTAINER_CUDA_VISIBLE}" \
    -e SLURM_JOB_ID="${SLURM_JOB_ID:-manual}" \
    "${KCD_ENV_FLAGS[@]}" \
    "${NCCL_DEBUG_FLAGS[@]}" \
    -v "$KCD_DATA_ROOT:$KCD_DATA_ROOT" \
    -v "$KCD_DATA_DPATH:$KCD_DATA_DPATH" \
    -v "$KCD_REPO_ROOT:$KCD_REPO_ROOT" \
    "${DEV_MOUNT_FLAGS[@]}" \
    "${EXTRA_MOUNT_FLAGS[@]}" \
    -w "$KCD_REPO_ROOT" \
    "$KCD_IMAGE" \
    bash "$KCD_REPO_ROOT/scripts/_launch_train.sh"

echo
echo "Done. Output under: $KCD_ROOT"
