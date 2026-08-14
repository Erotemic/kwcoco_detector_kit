#!/usr/bin/env bash
# Canonical paths for the VIAME fish detector project.
# Override any VF_* variable in the calling shell when a host differs.

export VF_PROJECT_DPATH="${VF_PROJECT_DPATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export VF_WORK_DPATH="${VF_WORK_DPATH:-/data/users/${USER}/fish}"
export VF_DOWNLOAD_DPATH="${VF_DOWNLOAD_DPATH:-$VF_WORK_DPATH/downloads}"
export VF_SOFTWARE_DPATH="${VF_SOFTWARE_DPATH:-$VF_WORK_DPATH/software}"
export VF_RUNS_DPATH="${VF_RUNS_DPATH:-$VF_WORK_DPATH/runs}"
export VF_LOGS_DPATH="${VF_LOGS_DPATH:-$VF_WORK_DPATH/logs}"

export VF_DATA_SOURCE="${VF_DATA_SOURCE:-numenor:/data/Public/NOAA/FishTrack23-Latest/}"
export VF_DATA_DPATH="${VF_DATA_DPATH:-$HOME/ssd-data/FishTrack23-Latest}"
export VF_INPUT_DPATH="${VF_INPUT_DPATH:-$VF_DATA_DPATH/Train}"

export VF_VIAME_VERSION="${VF_VIAME_VERSION:-0.22.7-rc2}"
export VF_VIAME_ARCHIVE="${VF_VIAME_ARCHIVE:-$VF_DOWNLOAD_DPATH/VIAME-v${VF_VIAME_VERSION}-Linux-64Bit.tar.gz}"
export VF_CURRENT_VIAME_LINK="${VF_CURRENT_VIAME_LINK:-$VF_WORK_DPATH/viame-current}"

# setup_config.sh records the selected config name here. The state stores only
# the basename so the selection follows viame-current after a binary upgrade.
export VF_CONFIG_SELECTION_FPATH="${VF_CONFIG_SELECTION_FPATH:-$VF_WORK_DPATH/selected_config.env}"

if [ -z "${VF_CONFIG_NAME+x}" ] && [ -f "$VF_CONFIG_SELECTION_FPATH" ]; then
    source "$VF_CONFIG_SELECTION_FPATH"
fi

export VF_CONFIG_NAME="${VF_CONFIG_NAME:-train_detector_rf_detr_l_seg_720.conf}"
export VF_CONFIG_DPATH="${VF_CONFIG_DPATH:-$VF_CURRENT_VIAME_LINK/configs/pipelines}"
export VF_CONFIG_FPATH="${VF_CONFIG_FPATH:-$VF_CONFIG_DPATH/$VF_CONFIG_NAME}"

# ========================================================================
# DEIMv2 / kwcoco_detector_kit path (gen001 onwards)
# ========================================================================
#
# Everything above this line belongs to the VIAME-native RF-DETR runbook.
# Everything below drives the kit pipeline, which is a separate stack with
# its own storage contract.
#
# ## Storage rule: generated training data goes on the NVMe, not /data
#
# aiq-gpu has two very different filesystems (measured 2026-08-14, from
# the host's own inventory/disk_free.txt):
#
#   /dev/md0         37T   20T avail   mounted at /data   -- RAID array
#   /dev/nvme0n1p2  1.8T  506G avail   mounted at /       -- NVMe SSD
#
# The RF-DETR run put its 771 GB of extracted PNG frames and its 383 GB
# chip cache under /data/users/$USER/fish/, i.e. on md0, and then read
# 650k small files from it in random order every epoch. Whether or not
# that was the binding constraint on those ~3.2 h epochs, it is a
# needless one. Every path below that a training job READS lands on the
# NVMe instead.
#
# The budget works because we extract only ANNOTATED frames. Of the
# ~4.4M frames in the 420 videos, 250,753 carry annotations (~6%). As
# JPEG q95 that is roughly 75 GB, which fits the 506 GB free with room
# to spare -- where extracting every frame as PNG would not fit at all.
export VF_SSD_ROOT="${VF_SSD_ROOT:-$HOME/ssd-data}"

# Root for everything the kit pipeline generates. On the NVMe by design.
export VF_KCD_ROOT="${VF_KCD_ROOT:-$VF_SSD_ROOT/fish_kcd}"

# Extracted video frames, one directory per sequence. Named to match
# VIAME's own convention (frame%06d, 1-based) so a frame index in a
# VIAME CSV means the same thing here as it does in a VIAME run.
export VF_FRAMES_DPATH="${VF_FRAMES_DPATH:-$VF_KCD_ROOT/frames}"

# The kwcoco bundle: split .kwcoco.zip files plus their assets.
export VF_BUNDLE_DPATH="${VF_BUNDLE_DPATH:-$VF_KCD_ROOT/bundle}"

# Held-out test source. The FishTrack23 release ships its own Test/
# directory (54 videos + 18 image dirs = 72 sequences). The RF-DETR run
# trained with `-i $VF_DATA_DPATH/Train` and nothing else, so it provably
# never saw any of this. That makes Test/ an honest held-out set for BOTH
# models and resolves the contamination problem the 2026-08-14 orientation
# journal recorded as unresolvable. Do not train on it, ever.
export VF_TEST_INPUT_DPATH="${VF_TEST_INPUT_DPATH:-$VF_DATA_DPATH/Test}"

# VIAME class-folding file shipped with the corpus. One line: the output
# class `fish` followed by 321 aliases. Reusing it is what makes our
# single-class model label-compatible with the RF-DETR model, which was
# trained through this exact file (its rf_detr_mgpu_params.json records
# class_names: ["fish"]).
export VF_LABELS_FPATH="${VF_LABELS_FPATH:-$VF_INPUT_DPATH/labels.txt}"

# Split bundles consumed by the kit's `sweep` CLI.
export VF_TRAIN_KWCOCO="${VF_TRAIN_KWCOCO:-$VF_BUNDLE_DPATH/train.kwcoco.zip}"
export VF_VALI_KWCOCO="${VF_VALI_KWCOCO:-$VF_BUNDLE_DPATH/vali.kwcoco.zip}"
export VF_TEST_KWCOCO="${VF_TEST_KWCOCO:-$VF_BUNDLE_DPATH/test.kwcoco.zip}"

# Fraction of Train/ sequences held out as validation. Whole sequences
# only -- a frame-level split puts adjacent frames of one fish track on
# both sides of the boundary, which is how the RF-DETR run ended up
# selecting its checkpoint on near-duplicates of its own training data.
export VF_VALI_FRACTION="${VF_VALI_FRACTION:-0.12}"
export VF_SPLIT_SEED="${VF_SPLIT_SEED:-0}"

# Temporal subsampling of annotated frames, applied when BUILDING the
# kwcoco splits (not at extraction time, so changing it never re-runs
# ffmpeg). Stride 1 = every annotated frame, matching what RF-DETR
# consumed. Tracks average ~39 annotated frames, so stride 2-3 still
# leaves 13-20 well-separated samples per track at a fraction of the
# epoch cost. Kept as a knob rather than a decision baked into the data.
export VF_FRAME_STRIDE="${VF_FRAME_STRIDE:-1}"

# -- kit variables -------------------------------------------------------
#
# The kit's own KCD_* namespace. _submit_train.sh snapshots every KCD_*
# variable into the job env, so anything the container needs must be
# named KCD_*, not VF_*.
export KCD_DATA_ROOT="${KCD_DATA_ROOT:-/data/users/${USER}}"
export KCD_KIT_DPATH="${KCD_KIT_DPATH:-$HOME/code/kwcoco_detector_kit}"
export KCD_REPO_ROOT="${KCD_REPO_ROOT:-$VF_PROJECT_DPATH}"

# Run workspaces (checkpoints, eval, manifests) on the NVMe: the trainer
# writes checkpoints every epoch and reads them back for rescoring.
export KCD_TRAINING_ROOT="${KCD_TRAINING_ROOT:-$VF_KCD_ROOT/runs_root}"
export KCD_RUNS_DPATH="${KCD_RUNS_DPATH:-$KCD_TRAINING_ROOT/runs}"

# Pretrained DEIMv2 checkpoints are shared with the sea-lion project and
# are read once at startup, so they can stay on /data.
export KCD_PRETRAINED_ROOT="${KCD_PRETRAINED_ROOT:-$KCD_DATA_ROOT/pretrained_models}"
export KCD_SLURM_LOG_DPATH="${KCD_SLURM_LOG_DPATH:-$KCD_DATA_ROOT/slurm_logs}"

# No tiling stage for fish. The sea-lion project tiles because its
# targets are a few dozen pixels on 10k-wide aerials; here the box size
# percentiles (measured over all 665,228 boxes) are:
#
#     p1 42x44   p5 59x54   p50 150x109   p95 359x230   (pixels)
#
# on 1920x1200 imagery. Even the 1st percentile box survives a whole-frame
# resize to 1024 with ~22 px of width left. Tiling would multiply the data
# volume and the epoch cost to solve a problem this corpus does not have.
# (It also means RF-DETR's small_box_area=75 / small_action=remove deleted
# essentially nothing -- contrary to what the orientation journal assumed.)
export KCD_USE_WEBDATASET="${KCD_USE_WEBDATASET:-0}"

kcd_require_path() {
    local label="$1"
    local path="$2"
    if [ ! -e "$path" ]; then
        echo "ERROR: $label not found at: $path" >&2
        echo "  Override the matching VF_*/KCD_* variable, or run the" >&2
        echo "  prep step that produces it (scripts/prep_all.sh)." >&2
        return 1
    fi
}

# -- Pretrained DEIMv2 checkpoints ---------------------------------------
#
# Shared with the sea-lion project; both live under $KCD_PRETRAINED_ROOT.
# Only S and X are currently on aiq-gpu (checked 2026-08-14); fetch others
# with the sea-lion project's fetch_pretrained.sh, which writes to the same
# directory.
export KCD_DEIMV2_DINOV3_S_COCO_PTH="${KCD_DEIMV2_DINOV3_S_COCO_PTH:-$KCD_PRETRAINED_ROOT/deimv2_dinov3_s_coco/deimv2_dinov3_s_coco.pth}"
export KCD_DEIMV2_DINOV3_M_COCO_PTH="${KCD_DEIMV2_DINOV3_M_COCO_PTH:-$KCD_PRETRAINED_ROOT/deimv2_dinov3_m_coco/deimv2_dinov3_m_coco.pth}"
export KCD_DEIMV2_DINOV3_L_COCO_PTH="${KCD_DEIMV2_DINOV3_L_COCO_PTH:-$KCD_PRETRAINED_ROOT/deimv2_dinov3_l_coco/deimv2_dinov3_l_coco.pth}"
export KCD_DEIMV2_DINOV3_X_COCO_PTH="${KCD_DEIMV2_DINOV3_X_COCO_PTH:-$KCD_PRETRAINED_ROOT/deimv2_dinov3_x_coco/deimv2_dinov3_x_coco.pth}"
export KCD_DEIMV2_HGNETV2_N_COCO_PTH="${KCD_DEIMV2_HGNETV2_N_COCO_PTH:-$KCD_PRETRAINED_ROOT/deimv2_hgnetv2_n_coco/deimv2_hgnetv2_n_coco.pth}"

# Same contract as the sea-lion project's helper of this name, so
# _launch_train.sh behaves identically across the two projects.
kcd_resolve_init_checkpoint() {
    local variant="$1"
    [ "${KCD_TRAIN_FROM_SCRATCH:-0}" = "1" ] && return 0
    if [ -n "${KCD_INIT_CHECKPOINT:-}" ]; then
        printf '%s\n' "$KCD_INIT_CHECKPOINT"
        return 0
    fi
    case "$variant" in
        deimv2_dinov3_s)  printf '%s\n' "$KCD_DEIMV2_DINOV3_S_COCO_PTH" ;;
        deimv2_dinov3_m)  printf '%s\n' "$KCD_DEIMV2_DINOV3_M_COCO_PTH" ;;
        deimv2_dinov3_l)  printf '%s\n' "$KCD_DEIMV2_DINOV3_L_COCO_PTH" ;;
        deimv2_dinov3_x)  printf '%s\n' "$KCD_DEIMV2_DINOV3_X_COCO_PTH" ;;
        deimv2_hgnetv2_n) printf '%s\n' "$KCD_DEIMV2_HGNETV2_N_COCO_PTH" ;;
        *) ;;
    esac
}

kcd_require_init_checkpoint() {
    local variant="$1"
    [ "${KCD_TRAIN_FROM_SCRATCH:-0}" = "1" ] && return 0
    local ckpt
    ckpt="$(kcd_resolve_init_checkpoint "$variant")"
    [ -z "$ckpt" ] && return 0
    if [ ! -e "$ckpt" ]; then
        echo "ERROR: $variant pretrained checkpoint not found at: $ckpt" >&2
        echo "  Fetch it with the sea-lion project's helper (same destination):" >&2
        echo "    bash projects/viame_sealions_2026/scripts/fetch_pretrained.sh $variant" >&2
        echo "  or set KCD_TRAIN_FROM_SCRATCH=1 to skip pretrained init." >&2
        return 1
    fi
}

# Host-side pre-flight for a fish training run. Deliberately much smaller
# than the sea-lion equivalent: there is no tile cache to validate because
# this project does not tile (see the box-size note above).
kcd_require_train_inputs() {
    local rc=0
    kcd_require_path "train bundle" "${KCD_TRAIN_KWCOCO:-}" || rc=1
    kcd_require_path "vali bundle"  "${KCD_VALI_KWCOCO:-}"  || rc=1
    kcd_require_path "test bundle"  "${KCD_TEST_KWCOCO:-}"  || rc=1
    if [ "$rc" -ne 0 ]; then
        echo "  Build them first:  bash $VF_PROJECT_DPATH/scripts/prep_all.sh" >&2
    fi
    return "$rc"
}

# The kit CLI reads these; they mirror the VF_* split paths so the
# container (which only receives KCD_* vars) can find the bundles.
export KCD_TRAIN_KWCOCO="${KCD_TRAIN_KWCOCO:-$VF_BUNDLE_DPATH/train.kwcoco.json}"
export KCD_VALI_KWCOCO="${KCD_VALI_KWCOCO:-$VF_BUNDLE_DPATH/vali.kwcoco.json}"
export KCD_TEST_KWCOCO="${KCD_TEST_KWCOCO:-$VF_BUNDLE_DPATH/test.kwcoco.json}"

# The corpus and everything generated from it live on the NVMe, which is
# outside $KCD_DATA_ROOT, so the shared _sbatch_train.sh would not mount
# it. One bind mount at an identical path covers frames, bundle and run
# workspaces.
export KCD_EXTRA_MOUNTS="${KCD_EXTRA_MOUNTS:-$VF_SSD_ROOT}"
