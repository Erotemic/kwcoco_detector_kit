#!/usr/bin/env bash
# Re-score a trained run WHOLE-IMAGE vs TILED on a non-slurm host (namek)
# via a direct `docker run`, and print the class-agnostic AP delta.
#
# Wraps rescore_tiled_compare.py. Dev-mounts the host kit + DEIMv2 over the
# image's baked copies so the new tiled-eval code (and the PIL-truncated-image
# patch) are live WITHOUT rebuilding the image.
#
# Usage (assuming cwd = ~/code/kwcoco_detector_kit):
#   # pup_vs_nonpup (NFS is dropped by the scheme -> no distractor needed):
#   bash projects/viame_sealions_2026/scripts/rescore_tiled_compare.sh \
#       pup_vs_nonpup_deimv2_dinov3_s_2gpu_arisia_gen005_v2splits \
#       pup,nonpup_sealion
#
#   # single_sealion:
#   bash projects/viame_sealions_2026/scripts/rescore_tiled_compare.sh \
#       single_sealion_deimv2_dinov3_s_2gpu_arisia_gen005_v2splits \
#       sealion
#
# Override the image with KCD_IMAGE=...; tiling knobs with KCD_TILED_EVAL_OVERLAP
# / KCD_TILED_EVAL_WINDOW; device with KCD_EVAL_DEVICE (default cuda).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./paths.sh
source "$SCRIPT_DIR/paths.sh"

RUN_NAME="${1:?usage: rescore_tiled_compare.sh <run_name> <category_names_csv> [distractors_csv]}"
CATEGORY_NAMES="${2:?missing category_names (e.g. pup,nonpup_sealion or sealion)}"
DISTRACTORS="${3:-}"

# Default to the locally-built auto-profile image (build_auto.sh tags it
# ogdino-auto with the CUDA profile matching THIS host's driver). The
# cu132-arisia tag requires a CUDA>=13.2 driver and fails on hosts with an
# older driver (e.g. namek: driver 580 / CUDA 13.0). Override per host.
KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-auto}"
KCD_EVAL_DEVICE="${KCD_EVAL_DEVICE:-cuda}"
KCD_ROOT="$KCD_RUNS_DPATH/$RUN_NAME"

if [ ! -d "$KCD_ROOT" ]; then
    echo "ERROR: run dir not found: $KCD_ROOT" >&2
    exit 1
fi

# Dev-mount host kit + DEIMv2 over the baked copies (no rebuild needed), and
# mount the kit checkout at its host path so the projects/ script (not baked
# into the image) is readable inside the container.
DEV_MOUNTS=(
    -v "$KCD_KIT_DPATH:$KCD_KIT_DPATH"
    -v "$KCD_KIT_DPATH/kwcoco_detector_kit:/opt/kwcoco_detector_kit/kwcoco_detector_kit"
    -v "$KCD_KIT_DPATH/tpl/DEIMv2:/opt/kwcoco_detector_kit/tpl/DEIMv2"
)

GPU_FLAGS=()
[ "$KCD_EVAL_DEVICE" = "cuda" ] && GPU_FLAGS=(--gpus all)

echo "run:        $RUN_NAME"
echo "image:      $KCD_IMAGE"
echo "device:     $KCD_EVAL_DEVICE"
echo "categories: $CATEGORY_NAMES   distractors: ${DISTRACTORS:-<none>}"
echo

set -x
docker run --rm "${GPU_FLAGS[@]}" \
    -e PYTHONUNBUFFERED=1 \
    -v "$KCD_DATA_ROOT:$KCD_DATA_ROOT" \
    -v "$KCD_DATA_DPATH:$KCD_DATA_DPATH" \
    "${DEV_MOUNTS[@]}" \
    -w /opt/kwcoco_detector_kit \
    "$KCD_IMAGE" \
    python3 -u "$KCD_KIT_DPATH/projects/viame_sealions_2026/scripts/rescore_tiled_compare.py" \
        --kcd_root "$KCD_ROOT" \
        --category_names "$CATEGORY_NAMES" \
        ${DISTRACTORS:+--distractor_classes "$DISTRACTORS"} \
        --device "$KCD_EVAL_DEVICE" \
        --overlap "${KCD_TILED_EVAL_OVERLAP:-0.25}" \
        --max_dets "${KCD_TILED_EVAL_MAX_DETS:-1000}" \
        ${KCD_FORCE_WHOLEIMAGE:+--force_wholeimage} \
        ${KCD_TILED_EVAL_WINDOW:+--window "$KCD_TILED_EVAL_WINDOW"}
