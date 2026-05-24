#!/usr/bin/env bash
# Canonical paths for viame_sealions_2026 scripts. Every script that needs
# a fixed location sources this file so paths are defined once.
#
# # Canonical layout (every host)
#
# Two roots, with different read/write contracts:
#
#   /data/Public/VIAME/        — shared data store, READ-ONLY.
#                                Holds the official viame_sealions_2026
#                                tree (kwcoco bundles, raw imagery).
#                                Same path on every host (per-host
#                                symlink to the actual storage).
#
#   /data/users/jon.crall/     — per-user work area, READ-WRITE.
#                                Holds training workspaces
#                                (kcd_sealion/), downloaded pretrained
#                                checkpoints, and slurm logs. Same
#                                canonical path on every host.
#
# Run `scripts/check_paths.sh` to verify your host is set up correctly.
#
# # Override conventions
#
# Every variable uses `${VAR:-default}` so individual values can be
# overridden from the calling shell or shell rc, e.g.:
#
#     export KCD_DATA_DPATH=/some/other/copy/of/viame_sealions_2026
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

# Per-user work area — same canonical path on every host (via symlink
# where needed). Read-write: holds kcd_sealion/, pretrained_models/,
# slurm_logs/.
KCD_DATA_ROOT="${KCD_DATA_ROOT:-/data/users/jon.crall}"

# Shared sea-lion data store — READ-ONLY for this project. Holds the
# official training_ready_v1/, unpacked imagery, scheme bundles. Lives
# under /data/Public/VIAME/ since 2026-05-22; the prior location at
# $KCD_DATA_ROOT/dvc-repos/viame_sealions_2026 may still exist as a
# legacy symlink on some hosts but should not be relied on.
KCD_DATA_DPATH="${KCD_DATA_DPATH:-/data/Public/VIAME/viame_sealions_2026}"

# Where trained-model workspaces (per-experiment kcd_root) live. The
# kit's `--kcd_root` writes train/eval/manifest artifacts here. Stays
# under jon.crall even after the public-data-store migration.
KCD_TRAINING_ROOT="${KCD_TRAINING_ROOT:-$KCD_DATA_ROOT/kcd_sealion}"

# Where downloaded pretrained checkpoints live, regardless of source
# (HuggingFace, Drive, etc.). Stays under jon.crall after the migration.
KCD_PRETRAINED_ROOT="${KCD_PRETRAINED_ROOT:-$KCD_DATA_ROOT/pretrained_models}"

# Where slurm stdout/stderr land. Lives on the data drive (NOT inside
# the kit checkout, which is on the SSD on workstations) so log volume
# doesn't fill workstation root filesystems. Placed under
# $KCD_DATA_ROOT (user-owned) rather than $KCD_TRAINING_ROOT (which
# may be root-owned because docker writes into it) so the submit-side
# `mkdir -p` succeeds without elevated permissions.
KCD_SLURM_LOG_DPATH="${KCD_SLURM_LOG_DPATH:-$KCD_DATA_ROOT/slurm_logs}"

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

# DEIMv2-HGNetv2-N COCO (43.0 AP, 3.6M params, native 320x320). Mobile-
# class HGNetv2 B0 backbone + DEIM head, COCO-pretrained. The fastest
# tier that's still useful as a real baseline (atto/femto/pico are
# floor models). Native input is 320; doesn't support dynamic input.
KCD_DEIMV2_HGNETV2_N_COCO_DIR="${KCD_DEIMV2_HGNETV2_N_COCO_DIR:-$KCD_PRETRAINED_ROOT/deimv2_hgnetv2_n_coco}"
KCD_DEIMV2_HGNETV2_N_COCO_PTH="${KCD_DEIMV2_HGNETV2_N_COCO_PTH:-$KCD_DEIMV2_HGNETV2_N_COCO_DIR/deimv2_hgnetv2_n_coco.pth}"

# -- Per-experiment workspaces -------------------------------------------
#
# Convention: descriptive submit_train_*.sh scripts derive their
# experiment-specific KCD_ROOT from KCD_RUN_NAME (the script's basename
# minus the submit_train_ prefix and .sh suffix). Example:
#     submit_train_pup_vs_nonpup_deimv2_dinov3_s_4gpu_v1.sh
#         -> KCD_RUN_NAME=pup_vs_nonpup_deimv2_dinov3_s_4gpu_v1
#         -> KCD_ROOT=$KCD_TRAINING_ROOT/runs/pup_vs_nonpup_deimv2_dinov3_s_4gpu_v1
#
# Tile bundles are shared per-scheme (independent of variant), so each
# scheme tiles once and every variant reuses the result.
KCD_RUNS_DPATH="${KCD_RUNS_DPATH:-$KCD_TRAINING_ROOT/runs}"
KCD_TILE_CACHE_DPATH="${KCD_TILE_CACHE_DPATH:-$KCD_TRAINING_ROOT/tile_cache}"

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
