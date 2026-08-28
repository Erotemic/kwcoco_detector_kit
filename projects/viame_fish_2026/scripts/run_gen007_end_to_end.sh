#!/usr/bin/env bash
# gen007, end to end: preflight -> build image -> train -> score on vali.
#
#   bash projects/viame_fish_2026/scripts/run_gen007_end_to_end.sh
#
# Run it from a tmux pane. Every phase is FOREGROUND (this host has no slurm),
# so the pane owns the job; detaching is what backgrounds it.
#
# ## Phases
#
#   0  preflight   data, checkpoint, GPUs, disk, docker. Seconds, and it fails
#                  here rather than 40 minutes into a build.
#   1  build       the image, WITH its pytest gate. The image is the
#                  reproducibility unit: it bakes the DEIMv2 fork whose solver
#                  honours KCD_BALANCE_REPLACEMENT and the kwcoco_dataloader
#                  fork that works on kwcoco >= 0.9. Verified afterwards
#                  against local HEAD, because a stale image is the one failure
#                  that looks like success.
#   2  train       ~13 h. 34 epochs x 96,000 unique tiles / batch 32.
#   3  score       the staged epochs + gen001/gen003 on VALI at stride 8.
#                  Ranking only -- it does not touch the test split, and a
#                  stride-8 number is never a reportable AP.
#
# ## Skipping and resuming
#
# Each phase is independently skippable, so a failure in one does not force
# redoing the others:
#
#   KCD_SKIP_BUILD=1   image already built and verified
#   KCD_SKIP_TRAIN=1   re-score an existing run
#   KCD_SKIP_SCORE=1   train only
#   KCD_YES=1          skip the abort window
#
# Training is NOT auto-resumed: gen007 pins KCD_RESUME_CKPT=fresh, so re-running
# phase 2 restarts from the COCO checkpoint. That is deliberate for a recipe
# experiment -- resuming into a half-finished schedule would silently blend two
# configurations. To resume deliberately, run the submit script directly with
# KCD_RESUME_CKPT=auto.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

RUN_SCRIPT="$SCRIPT_DIR/submit_train_fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen007_seqbalance.sh"
RUN_NAME="$(basename "$RUN_SCRIPT" .sh)"; RUN_NAME="${RUN_NAME#submit_train_}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DPATH="${KCD_E2E_LOG_DPATH:-$KCD_SLURM_LOG_DPATH}/gen007_e2e_$STAMP"
mkdir -p "$LOG_DPATH"

# Elevate ONLY the docker calls. Running this whole script under sudo would
# resolve every VF_*/KCD_* default against root's home instead of the user's.
if docker info >/dev/null 2>&1; then
    DOCKER_CMD="docker"; SUDO=""
elif sudo -n true 2>/dev/null; then
    DOCKER_CMD="sudo docker"; SUDO="sudo"
else
    echo "ERROR: docker needs elevation and passwordless sudo is unavailable." >&2
    echo "  Add yourself to the 'docker' group, or run with sudo cached." >&2
    exit 1
fi
export KCD_DOCKER_CMD="$DOCKER_CMD"

# Run one phase, tee'd to its own log, propagating the PHASE's exit status
# rather than tee's (which is why pipefail alone is not enough here).
phase() {
    local name="$1"; shift
    local log="$LOG_DPATH/$name.log"
    echo
    echo "=============================================================="
    echo " PHASE $name  ($(date -Is))"
    echo " log: $log"
    echo "=============================================================="
    set +e
    ( "$@" ) 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -ne 0 ]; then
        echo
        echo "PHASE $name FAILED (exit $rc). Log: $log" >&2
        echo "  Re-run this script with KCD_SKIP_* set for the phases that" >&2
        echo "  already succeeded." >&2
        exit "$rc"
    fi
    echo "PHASE $name OK  ($(date -Is))"
}

# ---------------------------------------------------------------- phase 0
preflight() {
    local rc=0
    kcd_require_path "tiled train bundle" "$KCD_TILE_TRAIN_KWCOCO" || rc=1
    kcd_require_path "tiled vali bundle"  "$KCD_TILE_VALI_KWCOCO"  || rc=1
    kcd_require_path "untiled train (sequence identity)" "$KCD_TILE_SOURCE_KWCOCO" || rc=1
    kcd_require_init_checkpoint deimv2_dinov3_x || rc=1
    [ "$rc" -ne 0 ] && return "$rc"

    # The two bundles must be DIFFERENT files. Equal means KCD_TRAIN_KWCOCO was
    # never pointed at the tile cache, and every tile would be its own sequence.
    if [ "$KCD_TILE_SOURCE_KWCOCO" = "$KCD_TILE_TRAIN_KWCOCO" ]; then
        echo "ERROR: tiled and untiled train bundles are the same file." >&2
        return 1
    fi

    if ! [[ "${KCD_TILE_SIZE_ONDISK:-}" =~ ^[0-9]+$ ]] || [ "${KCD_TILE_SIZE_ONDISK}" -lt 64 ]; then
        echo "ERROR: KCD_TILE_SIZE_ONDISK did not resolve ('${KCD_TILE_SIZE_ONDISK:-}')." >&2
        echo "  It comes from the tile cache metadata; the eval window needs it." >&2
        return 1
    fi
    echo "  tile size on disk: ${KCD_TILE_SIZE_ONDISK}px"

    # Distinguish "image not built yet" from "docker cannot reach the GPUs".
    # Both make the probe fail, and only the second is fatal -- but treating
    # them alike would let a launch discover a broken GPU runtime 40 minutes
    # into the build, which is exactly what this phase exists to prevent.
    local tag="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-aiq}"
    if ! $DOCKER_CMD image inspect "$tag" >/dev/null 2>&1; then
        echo "  NOTE: image $tag not built yet; GPU check deferred to phase 1."
    else
        local probe ngpu
        probe="$($DOCKER_CMD run --rm --gpus all "$tag" \
                  nvidia-smi --query-gpu=index --format=csv,noheader 2>&1)" || true
        ngpu="$(printf '%s\n' "$probe" | grep -cE '^[0-9]+$' || true)"
        if [ "${ngpu:-0}" -lt 1 ]; then
            echo "ERROR: docker cannot reach any GPU." >&2
            printf '  %s\n' "$probe" | head -3 >&2
            echo "  gen007 needs 4. Check the NVIDIA container runtime" >&2
            echo "  (nvidia-container-toolkit) and that /dev/nvidia* exist." >&2
            echo "  Set KCD_E2E_ALLOW_NO_GPU=1 only to rehearse phases 0-1." >&2
            [ "${KCD_E2E_ALLOW_NO_GPU:-0}" = "1" ] || return 1
            echo "  KCD_E2E_ALLOW_NO_GPU=1 set; continuing without GPUs." >&2
        else
            echo "  GPUs visible to docker: $ngpu"
            if [ "$ngpu" -lt 4 ]; then
                echo "ERROR: gen007 expects 4 GPUs, found $ngpu." >&2
                return 1
            fi
        fi
    fi

    local free_gb
    free_gb="$(df -BG --output=avail "$KCD_TRAINING_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')"
    echo "  free on training root: ${free_gb:-?} GB"
    # 34 staged epochs at ~800 MB plus the build's layers.
    if [ -n "$free_gb" ] && [ "$free_gb" -lt 80 ]; then
        echo "ERROR: need ~80 GB for 34 staged checkpoints; have ${free_gb} GB." >&2
        return 1
    fi

    if [ -n "$(git -C "$KCD_KIT_DPATH" status --porcelain 2>/dev/null)" ]; then
        echo "  NOTE: working tree is dirty; the image will be stamped -dirty."
    fi
    echo "  kit HEAD:    $(git -C "$KCD_KIT_DPATH" rev-parse --short HEAD 2>/dev/null || echo '?')"
    echo "  DEIMv2 HEAD: $(git -C "$KCD_KIT_DPATH/tpl/DEIMv2" rev-parse --short HEAD 2>/dev/null || echo '?')"
    return 0
}

# ---------------------------------------------------------------- phase 1
build_image() {
    $SUDO bash "$KCD_KIT_DPATH/docker/opengroundingdino/build_aiq_cuda132_blackwell.sh"
}

verify_image() {
    # A stale image is the failure that looks like success: the run would train
    # with the with-replacement sampler while every log line claimed otherwise.
    local tag="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-aiq}"
    local want_deimv2 want_dl got_deimv2 got_dl
    want_deimv2="$(git -C "$KCD_KIT_DPATH/tpl/DEIMv2" rev-parse HEAD | cut -c1-12)"
    want_dl="$(git -C "$KCD_KIT_DPATH/tpl/kwcoco_dataloader" rev-parse HEAD | cut -c1-12)"
    got_deimv2="$($DOCKER_CMD inspect "$tag" --format '{{index .Config.Labels "kcd.deimv2_sha"}}')"
    got_dl="$($DOCKER_CMD inspect "$tag" --format '{{index .Config.Labels "kcd.kwcoco_dataloader_sha"}}')"
    echo "  deimv2            image=$got_deimv2  local=$want_deimv2"
    echo "  kwcoco_dataloader image=$got_dl      local=$want_dl"
    [ "$got_deimv2" = "$want_deimv2" ] || { echo "ERROR: image DEIMv2 sha != local HEAD." >&2; return 1; }
    [ "$got_dl" = "$want_dl" ] || { echo "ERROR: image kwcoco_dataloader sha != local HEAD." >&2; return 1; }
    # The one line that makes KCD_BALANCE_REPLACEMENT=False mean anything.
    $DOCKER_CMD run --rm "$tag" \
        grep -q "replacement=bool(_kcd_cfg.get('kcd_sample_replacement'" \
        /opt/kwcoco_detector_kit/tpl/DEIMv2/engine/solver/_solver.py \
        || { echo "ERROR: baked solver does not forward kcd_sample_replacement." >&2; return 1; }
    echo "  baked solver forwards kcd_sample_replacement: yes"
}

# ---------------------------------------------------------------- phase 2/3
train_gen007() { bash "$RUN_SCRIPT"; }

score_vali() {
    # VALI ONLY, stride 8. Ranking, not a reportable AP, and never the test split.
    KCD_NO_SLURM=1 \
    KCD_EVAL_STRIDE="${KCD_E2E_SCORE_STRIDE:-8}" \
    KCD_BASELINE_RUNS="fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen001 fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen003_bf16_fresh $RUN_NAME" \
        bash "$SCRIPT_DIR/submit_baseline_vali.sh"
}

# ---------------------------------------------------------------- drive
echo "=============================================================="
echo " gen007 end to end"
echo "=============================================================="
echo "  run:    $RUN_NAME"
echo "  logs:   $LOG_DPATH"
echo "  docker: $DOCKER_CMD"
echo "  phases: build=$([ "${KCD_SKIP_BUILD:-0}" = 1 ] && echo skip || echo yes)" \
     "train=$([ "${KCD_SKIP_TRAIN:-0}" = 1 ] && echo skip || echo yes)" \
     "score=$([ "${KCD_SKIP_SCORE:-0}" = 1 ] && echo skip || echo yes)"
echo "  expect: ~13 h of GPU for training alone"
echo

phase 0-preflight preflight

if [ "${KCD_YES:-0}" != "1" ]; then
    echo
    echo "Starting in 10s. Ctrl-C to abort."
    sleep 10
fi

if [ "${KCD_SKIP_BUILD:-0}" != "1" ]; then
    phase 1-build build_image
fi
phase 1-verify verify_image          # always: cheap, and catches a stale image

if [ "${KCD_SKIP_TRAIN:-0}" != "1" ]; then
    phase 2-train train_gen007
fi

if [ "${KCD_SKIP_SCORE:-0}" != "1" ]; then
    phase 3-score score_vali
fi

echo
echo "=============================================================="
echo " done  ($(date -Is))"
echo "=============================================================="
echo "  logs:     $LOG_DPATH"
echo "  run dir:  $KCD_RUNS_DPATH/$RUN_NAME"
echo
echo "  Phase 3 ranked checkpoints on a stride-8 VALI subsample. That is a"
echo "  RANKING, not a reportable AP. Re-score the top 2-3 at stride 1:"
echo "    KCD_NO_SLURM=1 KCD_EVAL_STRIDE=1 \\"
echo "      KCD_BASELINE_RUNS=\"<baselines> $RUN_NAME\" \\"
echo "      bash $SCRIPT_DIR/submit_baseline_vali.sh"
echo
echo "  The TEST split stays untouched until exactly one checkpoint is chosen."
