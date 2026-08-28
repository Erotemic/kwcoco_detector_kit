#!/usr/bin/env bash
# No-slurm replacement for _sbatch_train.sh: run the kit's launch script
# directly via `docker run` on the current host, in the FOREGROUND so the
# user drives it in tmux and captures logs with `tee`.
#
# Reached when a submit_*.sh / _submit_tiles.sh is invoked with
# KCD_NO_SLURM=1 — those dispatchers `exec` here instead of sbatch. All
# KCD_* env is already exported by the wrapper, so it's forwarded into the
# container as-is (same `compgen -v | grep KCD_` snapshot the slurm path
# uses).
#
# Differences from the slurm path:
#   * --gpus all (proven on namek/aiq-gpu) instead of --runtime=nvidia.
#   * No slurm allocation / container-guard traps; tmux owns the process.
#   * CPU-only jobs (KCD_GRES=none, e.g. tile build) get no --gpus.
#
# GPU selection: defaults to all visible GPUs. Set KCD_NUM_GPUS to match
# the DDP world size your launch script expects.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Load the CALLING project's paths.sh, not this directory's. This script is
# shared the same way _sbatch_train.sh is (see viame_fish_2026's
# _submit_train.sh), and every path below -- the mounts, the workdir and the
# launch script -- already resolves through $KCD_REPO_ROOT. Sourcing this
# directory's paths.sh regardless would hand a fish run the sea-lion bundles.
# Mirrors _sbatch_train.sh:25-28. Falls back to SCRIPT_DIR so a sea-lion
# wrapper that does not set KCD_REPO_ROOT keeps working unchanged.
# shellcheck source=./paths.sh
if [ -n "${KCD_REPO_ROOT:-}" ] && [ -f "$KCD_REPO_ROOT/scripts/paths.sh" ]; then
    source "$KCD_REPO_ROOT/scripts/paths.sh"
else
    source "$SCRIPT_DIR/paths.sh"
fi

: "${KCD_RUN_NAME:?_run_standalone.sh: KCD_RUN_NAME must be set}"
# Default to the locally-built auto-profile image (build_auto.sh tags
# ogdino-auto with the CUDA profile matching THIS host's driver). Avoids
# the cu132 image's nvidia-container "cuda>=13.2" requirement failing on
# hosts with an older driver. Override per host (aiq submit sets cu132-aiq).
KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-auto}"
LAUNCH="${KCD_LAUNCH_SCRIPT:-_launch_train.sh}"

# Host-side pre-flight: a training launch needs the variant's pretrained
# init checkpoint. Check it HERE, before paying for a container start.
if [ "$LAUNCH" = "_launch_train.sh" ]; then
    kcd_require_init_checkpoint "${KCD_VARIANT:?_run_standalone.sh: missing KCD_VARIANT}" || exit 1
fi
SHM_GB=$(( 16 + 8 * ${KCD_NUM_GPUS:-1} ))

# GPU flag: omit for CPU-only jobs (tile build sets KCD_GRES=none).
GPU_FLAGS=()
if [ "${KCD_GRES:-}" != "none" ]; then
    GPU_FLAGS=(--gpus "${KCD_GPUS:-all}")
fi

# Forward every KCD_* value into the container (same snapshot rule as the
# slurm path so a new knob never needs to be whitelisted here).
ENV_FLAGS=()
while IFS= read -r v; do
    val="${!v:-}"
    [ -n "$val" ] && ENV_FLAGS+=(-e "$v=$val")
done < <(compgen -v | grep -E '^KCD_' | sort -u)

# Some hosts put the tile cache / training root on a separate disk that's
# symlinked into /data (aiq-gpu: .../kcd_sealion/ssd-data -> /home/.../ssd-data).
# The literal path is visible through the /data mount, but the symlink TARGET
# isn't mounted, so it dangles inside the container (mkdir -p fails). Mount
# each symlinked backing path at its real location so the link resolves.
EXTRA_MOUNTS=()
_kcd_mount_real() {
    local p real
    p="$1"
    [ -z "$p" ] && return 0
    real="$(readlink -f "$p" 2>/dev/null || true)"
    [ -z "$real" ] && return 0
    [ "$real" = "$p" ] && return 0          # not symlinked; already covered
    case "$real/" in
        "$KCD_DATA_ROOT"/*|"$KCD_DATA_DPATH"/*) return 0 ;;  # under a mounted root
    esac
    # de-dup
    local m
    for m in "${EXTRA_MOUNTS[@]:-}"; do
        [ "$m" = "$real:$real" ] && return 0
    done
    EXTRA_MOUNTS+=(-v "$real:$real")
    echo "  symlink mount: $p -> $real"
}
_kcd_mount_real "${KCD_TILE_CACHE_DPATH:-}"
_kcd_mount_real "${KCD_TRAINING_ROOT:-}"

# Dev-mount host code over the baked image copies (no rebuild needed).
DEV_MOUNTS=()
if [ "${KCD_DEV_MOUNT_KIT:-0}" = "1" ]; then
    DEV_MOUNTS+=(-v "$KCD_KIT_DPATH/kwcoco_detector_kit:/opt/kwcoco_detector_kit/kwcoco_detector_kit")
fi
if [ "${KCD_DEV_MOUNT_DEIMV2:-0}" = "1" ]; then
    DEV_MOUNTS+=(-v "$KCD_KIT_DPATH/tpl/DEIMv2:/opt/kwcoco_detector_kit/tpl/DEIMv2")
fi
if [ "${KCD_DEV_MOUNT_KWCOCO_DATALOADER:-0}" = "1" ]; then
    DEV_MOUNTS+=(-v "$KCD_KIT_DPATH/tpl/kwcoco_dataloader:/opt/kwcoco_detector_kit/tpl/kwcoco_dataloader")
fi

echo "=== standalone docker run (no slurm) ==="
echo "  run_name: $KCD_RUN_NAME"
echo "  image:    $KCD_IMAGE"
echo "  launch:   $LAUNCH"
echo "  gpus:     ${GPU_FLAGS[*]:-<none (CPU)>}   shm: ${SHM_GB}g"
echo "  env vars: ${#ENV_FLAGS[@]} forwarded (2 per KCD_* value)"
echo

# Hosts where the invoking user is not in the `docker` group need `sudo docker`.
# Elevating only this call keeps paths.sh, $HOME and the run's own environment
# resolving as the real user -- running the whole submit under sudo would
# resolve every VF_*/KCD_* default against root's home instead.
set -x
${KCD_DOCKER_CMD:-docker} run --rm \
    --label "kcd.run_name=$KCD_RUN_NAME" \
    "${GPU_FLAGS[@]}" \
    --ipc=host \
    --shm-size="${SHM_GB}g" \
    "${ENV_FLAGS[@]}" \
    -v "$KCD_DATA_ROOT:$KCD_DATA_ROOT" \
    -v "$KCD_DATA_DPATH:$KCD_DATA_DPATH" \
    -v "$KCD_REPO_ROOT:$KCD_REPO_ROOT" \
    "${EXTRA_MOUNTS[@]}" \
    "${DEV_MOUNTS[@]}" \
    -w "$KCD_REPO_ROOT" \
    "$KCD_IMAGE" \
    bash "$KCD_REPO_ROOT/scripts/$LAUNCH"
