#!/usr/bin/env bash
# Canonical paths for viame_sealions_2026 scripts. Every script that needs
# a fixed location sources this file so paths are defined once.
#
# # Canonical layout (every host)
#
# Every host that runs this project — arisia, namek, future workstations
# — must expose the canonical data root at:
#
#     /data/users/jon.crall/
#
# On hosts where the physical storage lives elsewhere, this is a symlink
# (e.g. namek symlinks /data/users/jon.crall -> /media/joncrall/raid/...).
# Scripts hard-code the canonical path; the per-host symlink is the
# compatibility shim. Run `scripts/check_paths.sh` to verify your host
# is set up correctly.
#
# # Future migration (planned, not yet active)
#
# The shared data store (kwcoco bundles, raw imagery) is planned to
# move out from under jon.crall to a shared location at:
#
#     /data/Public/VIAME/
#
# Personal work directories (kcd_sealion training workspaces,
# pretrained_models, ...) will stay under /data/users/jon.crall/.
# When the move happens, override KCD_DATA_DPATH in your shell rc to
# point at the new shared location; KCD_TRAINING_ROOT etc. stay put.
#
# # Override conventions
#
# Every variable uses `${VAR:-default}` so individual values can be
# overridden from the calling shell or shell rc:
#
#     export KCD_DATA_DPATH=/data/Public/VIAME/viame_sealions_2026
#
# # Usage
#
#     SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#     source "$SCRIPT_DIR/paths.sh"

# -- Roots ---------------------------------------------------------------

# Project tree (this file's grandparent dir). Since the 2026-05 reorg
# the project lives inside the kit at `projects/viame_sealions_2026/`.
# Scripts/docs/tests are versioned here; the actual sea-lion data
# (training_ready_v1, unpacked/, ...) lives at $KCD_DATA_DPATH below.
KCD_REPO_ROOT="${KCD_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Canonical user data root — same on every host (via symlink where
# needed). Run scripts/check_paths.sh to verify your host satisfies
# this contract.
KCD_DATA_ROOT="${KCD_DATA_ROOT:-/data/users/jon.crall}"

# Sea-lion data directory (holds training_ready_v1/, unpacked/, etc.).
# Currently under jon.crall; will move to /data/Public/VIAME/ in the
# future without affecting the work-dir variables below.
KCD_DATA_DPATH="${KCD_DATA_DPATH:-$KCD_DATA_ROOT/dvc-repos/viame_sealions_2026}"

# Where trained-model workspaces (per-experiment kcd_root) live. The
# kit's `--kcd_root` writes train/eval/manifest artifacts here. Stays
# under jon.crall even after the public-data-store migration.
KCD_TRAINING_ROOT="${KCD_TRAINING_ROOT:-$KCD_DATA_ROOT/kcd_sealion}"

# Where downloaded pretrained checkpoints live, regardless of source
# (HuggingFace, Drive, etc.). Stays under jon.crall after the migration.
KCD_PRETRAINED_ROOT="${KCD_PRETRAINED_ROOT:-$KCD_DATA_ROOT/pretrained_models}"

# Where slurm stdout/stderr land. Lives on the data drive (NOT inside
# the kit checkout, which is on the SSD on workstations) so log volume
# doesn't fill workstation root filesystems.
KCD_SLURM_LOG_DPATH="${KCD_SLURM_LOG_DPATH:-$KCD_TRAINING_ROOT/slurm_logs}"

# kwcoco_detector_kit checkout on the host. Used by submit_*.sh to find
# follow_job.py outside the docker container. Default is $HOME-relative
# because the kit checkout is per-user; it doesn't live on $KCD_DATA_ROOT.
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
