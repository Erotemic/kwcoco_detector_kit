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
    # Multi-GPU on arisia: docker's --gpus value parser is broken
    # for comma-separated device lists in the version installed.
    # We tried:
    #   --gpus=device=0,1     → splits on ',' → reads "1" as Count
    #                            → "cannot set both Count and DeviceIDs"
    #                            (job 2572, 2026-06-01)
    #   --gpus device=0,1     → same bug (job 2574)
    #   --gpus device=UUID1,UUID2 → parser tries to read UUID2 as a
    #                            count → "value must be either 'all'
    #                            or an integer" (job after 1cb28cf)
    # All three forms route through the same broken value parser.
    #
    # Workaround: bypass --gpus entirely. The pre-19.03 nvidia-
    # docker2 path uses --runtime=nvidia plus NVIDIA_VISIBLE_DEVICES
    # env var — same effect, doesn't touch the buggy parser.
    # NVIDIA_VISIBLE_DEVICES is read by the runtime hook before
    # the container starts, exposes exactly the listed GPUs, and
    # accepts comma-separated UUIDs or indices.
    #
    # We use UUIDs (not indices) so the env var matches what the
    # slurm-context log echoes and so concurrent jobs can't have
    # subtle ordering ambiguity. nvidia-smi resolves the slurm-
    # pinned indices to UUIDs.
    GPU_UUIDS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader \
                  -i "$CUDA_VISIBLE_DEVICES" 2>/dev/null \
                | tr '\n' ',' | sed 's/,$//')
    if [ -z "$GPU_UUIDS" ]; then
        # Fall back to indices if nvidia-smi lookup fails.
        # NVIDIA_VISIBLE_DEVICES accepts both forms.
        GPU_UUIDS="$CUDA_VISIBLE_DEVICES"
        echo "WARN: nvidia-smi UUID lookup failed; using indices" >&2
    fi
    GPU_FLAG=(--runtime=nvidia \
              -e "NVIDIA_VISIBLE_DEVICES=${GPU_UUIDS}" \
              -e "NVIDIA_DRIVER_CAPABILITIES=compute,utility")
    # Build container-side device list: 0,1,2,... up to the count slurm
    # gave us. Slurm sets CUDA_VISIBLE_DEVICES to a comma-separated host
    # index list ("0", "1,2", etc.); count its entries.
    n_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
    CONTAINER_CUDA_VISIBLE=$(seq -s, 0 $((n_gpus - 1)))
else
    GPU_FLAG=(--runtime=nvidia \
              -e "NVIDIA_VISIBLE_DEVICES=all" \
              -e "NVIDIA_DRIVER_CAPABILITIES=compute,utility")
    CONTAINER_CUDA_VISIBLE=""
    echo "WARN: CUDA_VISIBLE_DEVICES unset; exposing all GPUs (collision risk)" >&2
fi
# ============================================================
# Zombie defense: deterministic container ID + cleanup traps
# ============================================================
# Without this, `docker run` losing its parent shell (slurm cancel,
# OOM kill, host reboot mid-job) can leave the container running
# with the GPUs reserved. The leaked container holds VRAM until
# manually killed, blocking subsequent jobs with "out of memory"
# errors on a GPU that's actually free per nvidia-smi -- because
# nvidia-smi sees the leaked container's processes still pinning
# the device.
#
# Defense layers:
#   1. Stable --name + --cidfile so cleanup has a target even if
#      the wrapper PID dies.
#   2. EXIT/INT/TERM/HUP traps that docker-stop the container.
#      EXIT covers normal exit + most errors; INT covers Ctrl-C;
#      TERM covers slurm scancel; HUP covers SSH disconnect.
#   3. Baseline GPU PID snapshot + post-run leak detection.
#      Diagnostic-only by default; KCD_KILL_GPU_LEAKS=1 escalates
#      to SIGKILL on same-user leaks.
#   4. Container labels for post-hoc audit (which slurm job /
#      kit run produced this surviving container).

# Stable container identity. docker --name only accepts
# [a-zA-Z0-9_.-], length <= 128. Replace anything else with '-'
# and truncate.
_kcd_sanitize_name() {
    local s="$1"
    echo -n "${s}" | tr -c 'a-zA-Z0-9_.-' '-' | cut -c1-128
}
KCD_CONTAINER_NAME="kcd-${SLURM_JOB_ID:-manual}-$(_kcd_sanitize_name "$KCD_RUN_NAME")"
KCD_CONTAINER_NAME="$(_kcd_sanitize_name "$KCD_CONTAINER_NAME")"
# --cidfile requires the file NOT to exist; mktemp -u gives a
# unique path without creating the file.
KCD_CIDFILE="$(mktemp -u --tmpdir kcd-cid.XXXXXX 2>/dev/null || echo "/tmp/kcd-cid.$$")"

# Baseline GPU PIDs (own UID only -- don't snoop on neighbors).
# Subtract these from the post-run set to detect leaks attributable
# to this job.
KCD_GPU_BASELINE_PIDS="$(nvidia-smi --query-compute-apps=pid \
    --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ' ')"
KCD_USER_UID="$(id -u)"

_kcd_gpu_leak_report() {
    local cur_pids new_pids my_leaks pid pid_uid cgr
    cur_pids="$(nvidia-smi --query-compute-apps=pid \
        --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ' ')"
    new_pids="$(comm -13 \
        <(echo "$KCD_GPU_BASELINE_PIDS" | tr ' ' '\n' | sort -u) \
        <(echo "$cur_pids" | tr ' ' '\n' | sort -u))"
    my_leaks=""
    for pid in $new_pids; do
        [ -z "$pid" ] && continue
        pid_uid="$(stat -c '%u' "/proc/$pid" 2>/dev/null || true)"
        [ "$pid_uid" = "$KCD_USER_UID" ] && my_leaks="$my_leaks $pid"
    done
    if [ -n "$my_leaks" ]; then
        echo "[_sbatch_train.sh] GPU LEAK: same-user processes still on GPU after cleanup:" >&2
        for pid in $my_leaks; do
            ps -o pid,ppid,pgid,sid,etime,user,cmd -p "$pid" 2>/dev/null >&2 || true
            cgr="$(head -1 "/proc/$pid/cgroup" 2>/dev/null || echo "<no cgroup>")"
            echo "  cgroup: $cgr" >&2
        done
        if [ "${KCD_KILL_GPU_LEAKS:-0}" = "1" ]; then
            echo "[_sbatch_train.sh] KCD_KILL_GPU_LEAKS=1; SIGKILL'ing leaks: $my_leaks" >&2
            # shellcheck disable=SC2086
            kill -9 $my_leaks 2>/dev/null || true
        else
            echo "[_sbatch_train.sh] (set KCD_KILL_GPU_LEAKS=1 to auto-SIGKILL these)" >&2
        fi
    fi
}

_kcd_cleanup() {
    local exit_code=$?
    # Don't let cleanup errors abort the trap path.
    set +e
    trap - EXIT INT TERM HUP

    # Stop+remove the container by name (in case --cidfile race)
    # and by cidfile. --rm on docker run means most paths leave
    # nothing to clean, but the docker daemon won't honor --rm if
    # the container is mid-stop or got orphaned; belt and suspenders.
    # Speed matters: slurm's default KillWait is 30s. If our cleanup
    # takes >30s, bash gets SIGKILL'd before docker stop returns,
    # leaving the container running orphaned (--rm only fires when
    # the container exits, not when the docker run client dies).
    # We saw this with 48h walltime hits — container at 2 days old
    # in docker ps with no slurm job, GPUs pinned.
    #
    # docker kill (SIGKILL the container) is the safe choice here.
    # The container is being terminated either way; giving it time
    # to flush state is nice but not worth losing the cleanup race.
    if [ -f "$KCD_CIDFILE" ]; then
        local cid
        cid="$(cat "$KCD_CIDFILE" 2>/dev/null)"
        if [ -n "$cid" ]; then
            docker kill "$cid" >/dev/null 2>&1
            docker rm -f "$cid" >/dev/null 2>&1
        fi
        rm -f "$KCD_CIDFILE"
    fi
    docker kill "$KCD_CONTAINER_NAME" >/dev/null 2>&1
    docker rm -f "$KCD_CONTAINER_NAME" >/dev/null 2>&1

    # Leak diagnostics on every exit path.
    _kcd_gpu_leak_report

    exit "$exit_code"
}
trap _kcd_cleanup EXIT INT TERM HUP

echo "=== Container guard ==="
echo "  name:    $KCD_CONTAINER_NAME"
echo "  cidfile: $KCD_CIDFILE"
echo "  traps:   EXIT INT TERM HUP -> docker stop + leak check"
echo "  KCD_KILL_GPU_LEAKS=${KCD_KILL_GPU_LEAKS:-0}  (1 = SIGKILL same-user GPU leaks on exit)"

docker run --rm \
    --name "$KCD_CONTAINER_NAME" \
    --cidfile "$KCD_CIDFILE" \
    --label "kcd.run_name=$KCD_RUN_NAME" \
    --label "kcd.slurm_job_id=${SLURM_JOB_ID:-manual}" \
    --label "kcd.user=$(whoami)" \
    --label "kcd.created_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
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
