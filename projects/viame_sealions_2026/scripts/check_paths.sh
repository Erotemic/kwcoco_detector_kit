#!/usr/bin/env bash
# Verify the host satisfies the viame_sealions_2026 path contract.
#
# Contract (see scripts/paths.sh):
#   - /data/users/jon.crall/ exists (real dir or symlink that resolves).
#   - /data/Public/VIAME/ exists (shared read-only data store root).
#   - $KCD_DATA_DPATH exists and is readable (the project's data dir).
#     Read-only by design — writability is NOT required (and a
#     write-check would actually be misleading).
#   - $KCD_TRAINING_ROOT exists or can be created (where runs write).
#   - $KCD_PRETRAINED_ROOT exists or can be created.
#   - $KCD_KIT_DPATH looks like a kwcoco_detector_kit checkout.
#   - $KCD_KIT_DPATH/smoketests/dino_v2_4x/slurm/follow_job.py exists.
#
# Exit code: 0 if every required check passes, non-zero otherwise.
# Warnings (yellow) are non-fatal.
#
# Usage (assuming cwd = ~/code/kwcoco_detector_kit):
#   bash projects/viame_sealions_2026/scripts/check_paths.sh
set -u  # -e off: we report all failures, not just the first

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

# tty-aware coloring
if [ -t 1 ]; then
    RED=$'\033[31m'; YEL=$'\033[33m'; GRN=$'\033[32m'; DIM=$'\033[2m'; OFF=$'\033[0m'
else
    RED=''; YEL=''; GRN=''; DIM=''; OFF=''
fi

n_fail=0
n_warn=0

pass() { printf '  %s✓%s %s\n' "$GRN" "$OFF" "$1"; }
fail() { printf '  %s✗%s %s\n' "$RED" "$OFF" "$1"; n_fail=$((n_fail+1)); }
warn() { printf '  %s!%s %s\n' "$YEL" "$OFF" "$1"; n_warn=$((n_warn+1)); }

resolve() {
    # readlink -f resolves symlinks (and reports errors for broken ones).
    # On macOS readlink -f is non-portable, but arisia/namek are Linux.
    readlink -f "$1" 2>/dev/null
}

echo "$(hostname)  $(date -Is)"
echo
echo "== Canonical user data root =="
if [ -d "$KCD_DATA_ROOT" ]; then
    real="$(resolve "$KCD_DATA_ROOT")"
    if [ -L "$KCD_DATA_ROOT" ]; then
        pass "$KCD_DATA_ROOT -> $real (symlink)"
    else
        pass "$KCD_DATA_ROOT (real dir)"
    fi
    if ! [ -w "$KCD_DATA_ROOT" ]; then
        warn "  not writable by $(whoami) — work dirs below may fail to create"
    fi
else
    fail "$KCD_DATA_ROOT missing"
    echo "    Required canonical path. On hosts where storage lives elsewhere,"
    echo "    create a symlink:"
    echo "      sudo mkdir -p /data/users && sudo ln -s <storage> /data/users/jon.crall"
fi

echo
echo "== Shared data store (read-only: $KCD_DATA_DPATH) =="
if [ -d "$KCD_DATA_DPATH" ]; then
    real="$(resolve "$KCD_DATA_DPATH")"
    if [ -L "$KCD_DATA_DPATH" ]; then
        pass "$KCD_DATA_DPATH -> $real (symlink)"
    else
        pass "$KCD_DATA_DPATH (real dir)"
    fi
    if [ -d "$KCD_TRAINING_READY_DIR" ]; then
        pass "$KCD_TRAINING_READY_DIR present (per-scheme kwcoco bundles)"
    else
        warn "$KCD_TRAINING_READY_DIR missing under $KCD_DATA_DPATH"
        echo "      The official data store should already have this. If you're"
        echo "      on a host that doesn't mirror /data/Public/VIAME/, override"
        echo "      KCD_DATA_DPATH to point at your local copy."
    fi
    # Read-only by design — no writability check here.
else
    fail "$KCD_DATA_DPATH missing"
    echo "    This is the shared read-only data store (/data/Public/VIAME/...)"
    echo "    On hosts that don't mount it directly, create a symlink to your"
    echo "    local copy, or override KCD_DATA_DPATH."
fi

echo
echo "== Work directories =="
# Note on writability: these dirs are routinely created/written by
# docker (running as root inside the container) on the training host.
# When that happens, the host-side ls shows them as root-owned and not
# writable by the submit user. Training still works because the next
# docker run also runs as root. So we treat "exists but not writable"
# as a warning, not a failure — and only fail when the dir is missing
# AND the parent isn't writable either (no way to create it at all).
for label_var in \
    "training-workspaces:$KCD_TRAINING_ROOT" \
    "pretrained-models:$KCD_PRETRAINED_ROOT" \
    "slurm-logs:$KCD_SLURM_LOG_DPATH"
do
    label="${label_var%%:*}"
    p="${label_var#*:}"
    if [ -d "$p" ]; then
        if [ -w "$p" ]; then
            pass "$label: $p (writable)"
        else
            owner="$(stat -c '%U:%G' "$p" 2>/dev/null || echo '?')"
            warn "$label: $p exists but is not writable by $(whoami) — owner=$owner"
            echo "      (probably created by a prior docker run; safe to ignore"
            echo "       unless host-side mkdir/write is also needed)"
        fi
    else
        parent="$(dirname "$p")"
        if [ -d "$parent" ] && [ -w "$parent" ]; then
            pass "$label: $p (will be created in writable parent $parent)"
        else
            fail "$label: $p missing and parent $parent not writable"
        fi
    fi
done

echo
echo "== Disk headroom on data root =="
if free_kb=$(df -k --output=avail "$KCD_DATA_ROOT" 2>/dev/null | tail -n1 | tr -d ' '); then
    if [ -n "$free_kb" ]; then
        free_gb=$(( free_kb / 1024 / 1024 ))
        msg="$free_gb GB free at $KCD_DATA_ROOT"
        if [ "$free_gb" -lt 30 ]; then
            fail "$msg (need >= 30 GB for a training run)"
        elif [ "$free_gb" -lt 100 ]; then
            warn "$msg (under 100 GB; consider cleaning up before launching)"
        else
            pass "$msg"
        fi
    fi
fi

echo
echo "== kwcoco_detector_kit checkout =="
if [ -d "$KCD_KIT_DPATH" ]; then
    if [ -f "$KCD_KIT_DPATH/pyproject.toml" ] && [ -d "$KCD_KIT_DPATH/kwcoco_detector_kit" ]; then
        pass "$KCD_KIT_DPATH looks like a kit checkout"
    else
        fail "$KCD_KIT_DPATH exists but doesn't look like a kit checkout (no pyproject.toml + kwcoco_detector_kit/)"
    fi
    FOLLOW_SCRIPT="$KCD_KIT_DPATH/smoketests/dino_v2_4x/slurm/follow_job.py"
    if [ -f "$FOLLOW_SCRIPT" ]; then
        pass "follow_job.py found at $FOLLOW_SCRIPT"
    else
        fail "follow_job.py missing at $FOLLOW_SCRIPT"
    fi
else
    fail "$KCD_KIT_DPATH missing — git clone the kit there or set KCD_KIT_DPATH"
fi

echo
echo "== Optional tools =="
if command -v docker >/dev/null 2>&1; then
    pass "docker on PATH ($(docker --version 2>/dev/null | head -n1))"
else
    warn "docker not on PATH (only needed on the training host)"
fi
if command -v sbatch >/dev/null 2>&1; then
    pass "sbatch on PATH (slurm submission host)"
else
    warn "sbatch not on PATH (this host can't submit slurm jobs)"
fi
if command -v hf >/dev/null 2>&1; then
    pass "hf CLI on PATH (for fetch_pretrained.sh)"
elif command -v huggingface-cli >/dev/null 2>&1; then
    warn "only deprecated huggingface-cli found; install: pip install --user --upgrade 'huggingface_hub>=0.27'"
else
    warn "neither 'hf' nor 'huggingface-cli' on PATH (needed for the host-side fetch_pretrained.sh; runs inside docker on arisia)"
fi

echo
echo "== Summary =="
echo "  failures: $n_fail"
echo "  warnings: $n_warn"
if [ "$n_fail" = "0" ]; then
    echo "  $GRN""path contract satisfied$OFF"
    exit 0
else
    echo "  $RED""path contract NOT satisfied — fix the failures above before running training$OFF"
    exit 1
fi
