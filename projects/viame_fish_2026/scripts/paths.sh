#!/usr/bin/env bash
# Canonical paths for the VIAME fish detector project.
# Override any VF_* variable in the calling shell when a host differs.

export VF_PROJECT_DPATH="${VF_PROJECT_DPATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Account name used to build the default work paths.
#
# `$USER` is NOT set inside the training container -- docker does not inherit
# it -- and paths.sh is sourced there by _launch_train.sh under `set -u`, so a
# bare ${USER} aborts the job after the container has already started (job 292).
# The sea-lion project sidesteps this by hardcoding the name; resolve it
# instead, falling back through LOGNAME and `id -un` before that hardcoded
# last resort.
#
# In-container this value is mostly moot: _submit_train.sh forwards every KCD_*
# variable from the host, so the paths that actually matter arrive already
# resolved and the ${VAR:-default} branches below are never evaluated. It only
# has to not explode.
export VF_USER="${VF_USER:-${USER:-${LOGNAME:-$(id -un 2>/dev/null || echo jon.crall)}}}"

export VF_WORK_DPATH="${VF_WORK_DPATH:-/data/users/${VF_USER}/fish}"
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
#
# ## Do NOT move the source corpus to /data
#
# It looks like cold data that is read once during extraction. Half of it
# is not. FishTrack23 splits into two kinds of sequence with very
# different access patterns (measured 2026-08-14):
#
#   420 videos (.mp4)          10 GB   decoded ONCE during extraction
#   102 image dirs (36k PNGs)  38 GB   read EVERY EPOCH during training
#
# The image directories are already frames on disk, so nothing extracts
# them -- the kwcoco bundles point `file_name` straight at the original
# PNGs. They are 12% of the training images and they are hot. Moving the
# corpus to the md0 RAID array would put them back on the exact slow
# path this layout exists to avoid.
#
# Splitting the corpus (videos to /data, image dirs on the NVMe) would
# reclaim only 10 GB, against 506 GB free and a ~80 GB total need
# (~75 GB of extracted frames + ~5 GB of run workspace, the latter
# measured from the sea-lion dinov3_x runs). Not worth fragmenting the
# corpus across two filesystems and breaking the VIAME runbook's single
# `-i` input directory. Leave it whole, on the NVMe.
export VF_SSD_ROOT="${VF_SSD_ROOT:-$HOME/ssd-data}"

# -- ffmpeg / ffprobe ----------------------------------------------------
#
# Frame extraction needs BOTH: ffmpeg to decode, and ffprobe to recover the
# frame rate for the ~20 video CSVs that carry no `fps:` metadata comment.
# `apt install ffmpeg` provides both.
#
# Resolved from PATH. Deliberately NOT from a VIAME install, even though VIAME
# bundles static builds of both: the DEIMv2 path is the non-VIAME stack, and
# making it reach into a VIAME tree for a system tool would couple the two in
# the wrong direction. Nothing in this pipeline should require VIAME unless it
# is actually driving VIAME.
#
# Override either variable to point at a specific binary (a static build, a
# non-standard prefix) without touching PATH.
export VF_FFMPEG="${VF_FFMPEG:-ffmpeg}"
export VF_FFPROBE="${VF_FFPROBE:-ffprobe}"

# Root for everything the kit pipeline generates. On the NVMe by design.
export VF_KCD_ROOT="${VF_KCD_ROOT:-$VF_SSD_ROOT/fish_kcd}"

# Extracted video frames, one directory per sequence. Named to match
# VIAME's own convention (frame%06d, 1-based) so a frame index in a
# VIAME CSV means the same thing here as it does in a VIAME run.
export VF_FRAMES_DPATH="${VF_FRAMES_DPATH:-$VF_KCD_ROOT/frames}"

# The kwcoco bundle: split .kwcoco.json files (plain JSON -- the converters
# are stdlib-only and cannot write kwcoco's zipped form).
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
# .json, not .zip: prep_all.sh writes plain JSON because the converters are
# stdlib-only and cannot produce kwcoco's zipped form. These originally said
# .zip and nothing noticed, because the training path reads the KCD_* aliases
# below and those were right. The head-to-head scripts read the VF_* names and
# broke immediately. Defined once here; KCD_* now derives from these rather
# than repeating the literals.
export VF_TRAIN_KWCOCO="${VF_TRAIN_KWCOCO:-$VF_BUNDLE_DPATH/train.kwcoco.json}"
export VF_VALI_KWCOCO="${VF_VALI_KWCOCO:-$VF_BUNDLE_DPATH/vali.kwcoco.json}"
export VF_TEST_KWCOCO="${VF_TEST_KWCOCO:-$VF_BUNDLE_DPATH/test.kwcoco.json}"

# ========================================================================
# HOLDOUT DISCIPLINE -- read before quoting a test number
# ========================================================================
#
# VF_TEST_KWCOCO is the FINAL holdout. It is not a second validation set, and
# it must not inform any design decision: not the schedule, not the dtype, not
# the input resolution, not which checkpoint to keep. All of that comes from
# vali, which exists for exactly this purpose and is sequence-disjoint from
# train for exactly this reason.
#
# This rule is written down because it was broken. Between 2026-08-23 and
# 2026-08-25 the test scores were used to argue at least four decisions: that
# gen003 was fine despite lower vali, that gen003 was not update-starved, that
# bf16 cost nothing, and -- most seriously -- that the project was
# resolution-limited rather than schedule-limited, which is what motivated the
# whole tiling effort. Every one of those conclusions happens to survive on
# vali evidence alone, and they have been re-grounded on it, but the process
# was wrong and the holdout is correspondingly less clean than its two recorded
# scores suggest.
#
# Scope of the contamination, for whoever reports the final number: test has
# been SCORED twice (gen001 job 299, gen003 job 491) and CONSULTED for
# decisions roughly four times. Two model evaluations against 33,434 images and
# 84,694 annotations is mild leakage, not a ruined split -- but the final
# figure should be reported with that history attached rather than as a clean
# generalization estimate.
#
# Going forward: decide on vali, score test once, on the single model you have
# already chosen.


# ========================================================================
# Native-resolution tiling (gen005 onwards)
# ========================================================================
#
# ## Why
#
# Whole-frame training resizes 1920x1200 to the model's 1024x1024, which is
# an x-scale of 0.533 and a y-scale of 0.853 -- it discards nearly half the
# horizontal resolution AND distorts aspect by 1.6x. 97% of the corpus is
# 1920-wide, so this is the common case, not an edge case. The RF-DETR
# baseline never pays it: its VIAME config uses adaptive chipping
# (chip_width 720, chip_step 480, chip_adaptive_thresh 1.6 Mpx), so every
# image above 1.6 Mpx is cut into 720px windows at NATIVE scale.
#
# gen003's own metrics show the cost: AP_small 0.192 against AP_large 0.653.
# The sea-lion project hit the same wall and measured the fix -- windowed
# native-resolution eval lifted pup AP 0.123 -> 0.838
# (viame_sealions_2026/docs/journals/2026-06-06_gen005_small_object_floor.md).
#
# ## The window is 1024, deliberately larger than RF-DETR's 720
#
# A 1024 window is 2.0x the area of a 720 window, so FEWER of them cover a
# frame: 5.87 tiles/frame at 25% overlap versus RF-DETR's 7.90. More context
# per window, full native resolution, and 26% fewer crops than the baseline.
#
# And because the window equals the model input, each tile is fed 1:1 -- no
# resize, no aspect distortion, anywhere in the pipeline.
#
# ## Parameter notes
#
# source_scales is 1.0 ONLY, unlike the sea-lion recipe's 1.0,0.5. Their pups
# vanish at scale; fish are p50 150x109 and p1 42x44, all detectable at
# native. A single scale also keeps this a clean resolution experiment rather
# than two changes at once. DEIMv2's FPN supplies internal multi-scale
# features regardless.
#
# stride_frac 0.75 gives a 256 px seam, comfortably wider than the p50 fish
# (150 px), so a typical fish appears whole in at least one window.
#
# oversize_factor 1.2 leaves the 968x728 images (2.5% of the corpus) untiled,
# since they are already below the window size.
export KCD_TILE_SIZE="${KCD_TILE_SIZE:-1024}"
export KCD_TILE_SOURCE_SCALES="${KCD_TILE_SOURCE_SCALES:-1.0}"
export KCD_TILE_STRIDE_FRAC="${KCD_TILE_STRIDE_FRAC:-0.75}"
export KCD_TILE_MIN_GT_AREA_FRAC="${KCD_TILE_MIN_GT_AREA_FRAC:-0.0005}"
export KCD_TILE_MIN_KEEP_FRACTION="${KCD_TILE_MIN_KEEP_FRACTION:-0.20}"
export KCD_TILE_OVERSIZE_FACTOR="${KCD_TILE_OVERSIZE_FACTOR:-1.2}"
export KCD_TILE_KEEP_NEGATIVE="${KCD_TILE_KEEP_NEGATIVE:-true}"
export KCD_TILE_CATEGORY_NAMES="${KCD_TILE_CATEGORY_NAMES:-fish}"
export KCD_TILE_MODE="${KCD_TILE_MODE:-multiscale}"
export KCD_TILE_JPEG_QUALITY="${KCD_TILE_JPEG_QUALITY:-90}"

# Where the tiled bundles land. Train and vali are tiled; TEST IS NOT.
# The held-out test number must stay comparable to the scores already on
# record, both measured over whole test images, so test keeps its full frames
# and tiled EVAL slides windows over them at inference time.
# Vali is tiled because DEIMv2's per-epoch eval drives checkpoint selection
# and has to measure the same thing training optimises.
export VF_TILE_DPATH="${VF_TILE_DPATH:-$VF_KCD_ROOT/tiles_${KCD_TILE_SIZE}}"
export VF_TILE_TRAIN_KWCOCO="${VF_TILE_TRAIN_KWCOCO:-$VF_TILE_DPATH/train/tiles.kwcoco.json}"
export VF_TILE_VALI_KWCOCO="${VF_TILE_VALI_KWCOCO:-$VF_TILE_DPATH/vali/tiles.kwcoco.json}"

# Bridge the tile paths into KCD_* names. This is NOT cosmetic: _submit_train.sh
# forwards only variables matching ^KCD_ into the container
# (_submit_train.sh:72), so a VF_* path is re-derived in-container from
# $HOME -- which is /root there, not the user's home. That is the same hazard
# the VF_USER comment at the top of this file describes for $USER, and it is
# what made the first tiling job (slurm 493) look for
# /root/ssd-data/fish_kcd/bundle/train.kwcoco.json.
#
# Anything the container must see has to be resolved on the host and exported
# under a KCD_ name. The ${VAR:-default} form means the forwarded value wins
# in-container and the host default applies on the host.
export KCD_TILE_DPATH="${KCD_TILE_DPATH:-$VF_TILE_DPATH}"
export KCD_TILE_TRAIN_KWCOCO="${KCD_TILE_TRAIN_KWCOCO:-$VF_TILE_TRAIN_KWCOCO}"
export KCD_TILE_VALI_KWCOCO="${KCD_TILE_VALI_KWCOCO:-$VF_TILE_VALI_KWCOCO}"

# The ACTUAL on-disk tile size. tile.py cuts
# `round(tile_size * oversize_factor)` px windows and derives its stride from
# that enlarged value, so with tile_size 1024 and oversize 1.2 the emitted
# tiles are 1229x1229, not 1024x1024. The oversize knob exists for a
# trainer-side load-time crop that upstream calls "future" and has not
# implemented, so those tiles are simply resized down to the model input.
#
# Tiled EVAL must slide a window of this size, not of the model input, or the
# model is measured at a different object scale than it trained at (1229->1024
# is 0.833; a 1024 window would be 1.0).
#
# Read from the tile bundle's own metadata when it exists, so a re-tile with
# different parameters cannot leave this stale. Falls back to the arithmetic.
kcd_ondisk_tile_size() {
    local meta="$KCD_TILE_TRAIN_KWCOCO"
    if [ -s "$meta" ]; then
        python3 -c "
import json,sys
d=json.load(open('$meta'))
im=(d.get('images') or [{}])[0]
w=im.get('width')
print(int(w) if w else '')
" 2>/dev/null && return 0
    fi
    return 0
}
export KCD_TILE_SIZE_ONDISK="${KCD_TILE_SIZE_ONDISK:-$(kcd_ondisk_tile_size)}"
: "${KCD_TILE_SIZE_ONDISK:=$(python3 -c "print(int(round($KCD_TILE_SIZE*$KCD_TILE_OVERSIZE_FACTOR)))" 2>/dev/null || echo "$KCD_TILE_SIZE")}"
export KCD_TILE_SIZE_ONDISK

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
export KCD_DATA_ROOT="${KCD_DATA_ROOT:-/data/users/${VF_USER}}"
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

# This project has no "scheme". The sea-lion project uses schemes to collapse
# its 9 source categories into per-run class sets (pup_vs_nonpup,
# lifestage_6cls, ...); here the corpus's own labels.txt already folds every
# species to `fish` during prep, so there is nothing left to select at launch.
#
# The value still has to exist: the shared _sbatch_train.sh runs under `set -u`
# and echoes $KCD_SCHEME in its job banner, so leaving it unset aborts the job
# before the container starts. Naming it after what the model actually predicts
# keeps the banner and the slurm logs honest.
export KCD_SCHEME="${KCD_SCHEME:-single_fish}"

# -- Resume resolution ---------------------------------------------------
#
# KCD_RESUME_CKPT accepts:
#
#   auto (default)   resume from the most RECENT checkpoint in the workdir,
#                    or start fresh if there are none
#   <path>           resume from exactly that file
#   null|none|fresh  start fresh, even if checkpoints exist
#   noop|no|false|0
#
# "Most recent", not "best", is deliberate. Resuming after a kill wants the
# latest training state; the best checkpoint may be several epochs stale, and
# resuming from it silently discards the epochs since. Best-checkpoint
# selection is a separate concern that happens at eval time.
#
# NOTE on how much this can currently recover: with train_policy=fixed,
# DEIMv2 only writes last.pth and its periodic checkpoints while
# `epoch < collate_fn.stop_epoch` (det_solver.py:111), and the kit sets that
# stop_epoch to 1 for fixed policy -- so after epoch 0 the ONLY files that
# update are best_stg1/best_stg2.pth, on eval improvement. Until that gate is
# addressed, "most recent" in practice means "most recently improved". This
# function is still correct; it just cannot conjure checkpoints nobody wrote.
vf_resolve_resume_ckpt() {
    local workdir="$1"
    local requested="${KCD_RESUME_CKPT-auto}"

    case "$(printf '%s' "$requested" | tr '[:upper:]' '[:lower:]')" in
        ''|null|none|fresh|noop|no|false|0)
            return 0 ;;                      # print nothing -> fresh run
        auto)
            ;;                               # fall through to discovery
        *)
            printf '%s\n' "$requested"       # explicit path, used verbatim
            return 0 ;;
    esac

    [ -d "$workdir" ] || return 0            # nothing trained here yet

    # Newest by mtime across every shape of checkpoint this stack writes:
    # DEIMv2's last.pth / checkpointNNNN.pth / best_stg*.pth, plus the kit's
    # per-epoch selection-journal staging (staging/epoch_NNNN.pth) when
    # kcd_journal_dir is enabled.
    local newest
    newest="$(find "$workdir" -maxdepth 2 -type f \
        \( -name 'last.pth' -o -name 'checkpoint*.pth' \
           -o -name 'best_stg*.pth' -o -name 'epoch_*.pth' \) \
        -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n1 | cut -d' ' -f2-)"
    [ -n "$newest" ] && printf '%s\n' "$newest"
    return 0
}

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
export KCD_TRAIN_KWCOCO="${KCD_TRAIN_KWCOCO:-$VF_TRAIN_KWCOCO}"
export KCD_VALI_KWCOCO="${KCD_VALI_KWCOCO:-$VF_VALI_KWCOCO}"
export KCD_TEST_KWCOCO="${KCD_TEST_KWCOCO:-$VF_TEST_KWCOCO}"

# The UNTILED train bundle, kept addressable even when KCD_TRAIN_KWCOCO has
# been pointed at the tiled one. Sequence-aware sampling needs both: the tiler
# stamps tile_source_gid on every tile but does NOT copy video_id, so the only
# place sequence identity exists is the source. Without this, balancing can
# group tiles by source FRAME (which the measurement shows is already almost
# uniform -- gini 0.013) and would do essentially nothing.
export KCD_TILE_SOURCE_KWCOCO="${KCD_TILE_SOURCE_KWCOCO:-$VF_TRAIN_KWCOCO}"

# The corpus and everything generated from it live on the NVMe, which is
# outside $KCD_DATA_ROOT, so it has to be bind-mounted into the container
# explicitly. The shared _sbatch_train.sh already mounts $KCD_DATA_DPATH at an
# identical path, so pointing that at the NVMe root covers the corpus, the
# extracted frames, the kwcoco bundles and the run workspaces in one mount.
#
# For the sea-lion project this variable means "the shared read-only data
# store" (/data/Public/VIAME/viame_sealions_2026). This project has no such
# store, so it takes the role of "where this project's data lives".
#
# Do NOT also list it in KCD_EXTRA_MOUNTS: that would pass the same
# source:target pair to `docker run -v` twice, which fails with a duplicate
# mount point error.
export KCD_DATA_DPATH="${KCD_DATA_DPATH:-$VF_SSD_ROOT}"
