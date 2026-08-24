#!/usr/bin/env bash
# Build the native-resolution tile bundles for train and vali.
#
# One-off prep for gen005. Run this BEFORE submitting the tiled training run;
# _launch_tiles.sh skips a split whose output already exists, so a re-run after
# an interruption is cheap.
#
# Expect roughly 1.47 M train tiles (5.87 per frame at 25% overlap) and ~95 GB
# on the NVMe, extrapolated from the existing frames cache (34 GB / 251k frames
# = 0.061 KB per Kpx). Check `df -h` first if the disk has moved since.
#
# Tiling is CPU/IO-bound and touches no GPU, but the shared _submit_train.sh
# always passes --gres=gpu:$KCD_NUM_GPUS, and --gres=gpu:0 is not a request
# this scheduler is known to accept. So it asks for one GPU and leaves it
# idle. That is fine in the intended order -- tile first, then train -- but it
# does mean this is not free to run beside a 4-GPU job.
#
# Submit (from the kit root, on aiq-gpu):
#   bash projects/viame_fish_2026/scripts/submit_build_tiles.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

# Drive the tiling launcher instead of the training one, reusing the same
# sbatch + docker machinery.
export KCD_LAUNCH_SCRIPT=_launch_tiles.sh

# Resolve the tiling SOURCES on the host and hand them over under KCD_ names,
# since only ^KCD_ is forwarded into the container. Passing them explicitly
# also keeps this independent of KCD_TRAIN_KWCOCO, which gen005 repoints at
# the tiled bundle.
export KCD_TILE_SRC_TRAIN="${KCD_TILE_SRC_TRAIN:-$VF_TRAIN_KWCOCO}"
export KCD_TILE_SRC_VALI="${KCD_TILE_SRC_VALI:-$VF_VALI_KWCOCO}"
export KCD_RUN_NAME="${KCD_RUN_NAME:-fishtrack23_build_tiles_${KCD_TILE_SIZE}}"

# _submit_train.sh's contract still applies (it pre-flights bundles and the
# init checkpoint) even though nothing trains and no GPU is used.
export KCD_VARIANT=deimv2_dinov3_x
export KCD_CATEGORY_NAMES=fish
export KCD_NUM_EPOCHS=1
export KCD_INPUT_HW="${KCD_INPUT_HW:-[1024, 1024]}"
export KCD_LR=5e-4
export KCD_BACKBONE_LR=1e-5

# Many cores and plenty of RAM; the one GPU is requested only to satisfy
# --gres (see the note above) and goes unused.
export KCD_NUM_GPUS="${KCD_NUM_GPUS:-1}"
export KCD_CPUS_PER_TASK="${KCD_CPUS_PER_TASK:-32}"
export KCD_MEM="${KCD_MEM:-96G}"
export KCD_TIME_LIMIT="${KCD_TIME_LIMIT:-12:00:00}"

export KCD_NO_SLURM="${KCD_NO_SLURM:-0}"
export KCD_DOCKER_GPU_MODE="${KCD_DOCKER_GPU_MODE:-gpus}"
export KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-aiq}"

echo "fish tile build"
echo "  window:  ${KCD_TILE_SIZE}px native (RF-DETR's baseline uses 720)"
echo "  scales:  $KCD_TILE_SOURCE_SCALES"
echo "  stride:  $KCD_TILE_STRIDE_FRAC"
echo "  out:     $KCD_TILE_DPATH"
echo "  src:     $KCD_TILE_SRC_TRAIN"
echo

exec bash "$SCRIPT_DIR/_submit_train.sh"
