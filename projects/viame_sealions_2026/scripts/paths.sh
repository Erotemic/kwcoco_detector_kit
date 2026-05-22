#!/usr/bin/env bash
# Canonical paths for viame_sealions_2026 scripts. Every script that needs
# a fixed location sources this file so paths are defined once.
#
# Override conventions:
#   - Every variable uses `${VAR:-default}` so individual values can be
#     overridden from the calling shell:  KCD_ROOT_BASE=/tmp/foo bash ...
#   - The host-specific defaults below assume arisia's filesystem layout
#     (/data/users/jon.crall/...). On a host with different layout, set
#     KCD_DATA_ROOT and KCD_DATA_DPATH once in your shell rc and the
#     rest derives from those.
#
# Usage:
#
#     SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#     source "$SCRIPT_DIR/paths.sh"

# -- Roots ---------------------------------------------------------------

# Project tree (this file's grandparent dir). Since the 2026-05 reorg
# the project lives inside the kit at `projects/viame_sealions_2026/`.
# Scripts/docs/tests are versioned here; the actual sea-lion data
# (training_ready_v1, unpacked/, ...) lives at $KCD_DATA_DPATH below.
KCD_REPO_ROOT="${KCD_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Parent of all sea-lion experiment data. On arisia this is the shared
# user-data volume; on namek override in your shell rc.
KCD_DATA_ROOT="${KCD_DATA_ROOT:-/data/users/jon.crall}"

# Sea-lion data directory (holds training_ready_v1/, unpacked/, etc.).
# Default matches arisia's dvc-repos location. On namek override to the
# raid mount, e.g.:
#   export KCD_DATA_DPATH=/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026
KCD_DATA_DPATH="${KCD_DATA_DPATH:-$KCD_DATA_ROOT/dvc-repos/viame_sealions_2026}"

# Where trained-model workspaces (per-experiment kcd_root) live. The
# kit's `--kcd_root` writes train/eval/manifest artifacts here.
KCD_TRAINING_ROOT="${KCD_TRAINING_ROOT:-$KCD_DATA_ROOT/kcd_sealion}"

# Where downloaded pretrained checkpoints live, regardless of source
# (HuggingFace, Drive, etc.).
KCD_PRETRAINED_ROOT="${KCD_PRETRAINED_ROOT:-$KCD_DATA_ROOT/pretrained_models}"

# kwcoco_detector_kit checkout on the host (we shell out to its slurm
# follow utility, run the docker image built from it, etc.). On arisia
# this is $HOME/code/kwcoco_detector_kit; on namek it's the same path
# under joncrall's home.
KCD_KIT_DPATH="${KCD_KIT_DPATH:-$HOME/code/kwcoco_detector_kit}"

# -- Dataset paths -------------------------------------------------------

KCD_TRAINING_READY_DIR="${KCD_TRAINING_READY_DIR:-$KCD_DATA_DPATH/training_ready_v1}"
KCD_SCHEMES_DIR="${KCD_SCHEMES_DIR:-$KCD_TRAINING_READY_DIR/by_scheme}"

# -- Pretrained checkpoints ----------------------------------------------

# DEIMv2-DINOv3-S COCO-finetuned (50.9 AP on COCO). Foundation backbone
# (DINOv3, self-supervised on ~1.7B images) + full DEIM head trained on
# COCO. The strongest publicly-available init for `deimv2_dinov3_s`.
KCD_DEIMV2_DINOV3_S_COCO_DIR="${KCD_DEIMV2_DINOV3_S_COCO_DIR:-$KCD_PRETRAINED_ROOT/deimv2_dinov3_s_coco}"
KCD_DEIMV2_DINOV3_S_COCO_PTH="${KCD_DEIMV2_DINOV3_S_COCO_PTH:-$KCD_DEIMV2_DINOV3_S_COCO_DIR/deimv2_dinov3_s_coco.pth}"

# -- Per-experiment workspaces -------------------------------------------

KCD_ROOT_PUP_VS_NONPUP="${KCD_ROOT_PUP_VS_NONPUP:-$KCD_TRAINING_ROOT/pup_vs_nonpup}"

# Sanity helper: callers can use `kcd_require_path foo /some/path` to fail
# fast with a useful message when a required file/dir is missing.
kcd_require_path() {
    local label="$1"
    local path="$2"
    if [ ! -e "$path" ]; then
        echo "ERROR: $label not found at: $path" >&2
        echo "  Override by setting the matching KCD_* variable, or run the" >&2
        echo "  fetch/build script that produces it." >&2
        return 1
    fi
}
