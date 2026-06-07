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
# shellcheck source=./paths.sh
source "$SCRIPT_DIR/paths.sh"

: "${KCD_RUN_NAME:?_run_standalone.sh: KCD_RUN_NAME must be set}"
KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-arisia}"
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

# Dev-mount host code over the baked image copies (no rebuild needed).
DEV_MOUNTS=()
if [ "${KCD_DEV_MOUNT_KIT:-0}" = "1" ]; then
    DEV_MOUNTS+=(-v "$KCD_KIT_DPATH/kwcoco_detector_kit:/opt/kwcoco_detector_kit/kwcoco_detector_kit")
fi
if [ "${KCD_DEV_MOUNT_DEIMV2:-0}" = "1" ]; then
    DEV_MOUNTS+=(-v "$KCD_KIT_DPATH/tpl/DEIMv2:/opt/kwcoco_detector_kit/tpl/DEIMv2")
fi

echo "=== standalone docker run (no slurm) ==="
echo "  run_name: $KCD_RUN_NAME"
echo "  image:    $KCD_IMAGE"
echo "  launch:   $LAUNCH"
echo "  gpus:     ${GPU_FLAGS[*]:-<none (CPU)>}   shm: ${SHM_GB}g"
echo "  env vars: ${#ENV_FLAGS[@]} forwarded (2 per KCD_* value)"
echo

set -x
docker run --rm \
    --label "kcd.run_name=$KCD_RUN_NAME" \
    "${GPU_FLAGS[@]}" \
    --ipc=host \
    --shm-size="${SHM_GB}g" \
    "${ENV_FLAGS[@]}" \
    -v "$KCD_DATA_ROOT:$KCD_DATA_ROOT" \
    -v "$KCD_DATA_DPATH:$KCD_DATA_DPATH" \
    -v "$KCD_REPO_ROOT:$KCD_REPO_ROOT" \
    "${DEV_MOUNTS[@]}" \
    -w "$KCD_REPO_ROOT" \
    "$KCD_IMAGE" \
    bash "$KCD_REPO_ROOT/scripts/$LAUNCH"
