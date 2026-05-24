#!/usr/bin/env bash
# Reset permissions on the project's work dirs to world-readable +
# world-writable. Solves the "root-owned by docker, can't write from
# host" problem without needing sudo or messing with UID/GID mapping.
#
# Targets:
#   $KCD_TRAINING_ROOT   (e.g. /data/users/jon.crall/kcd_sealion/)
#   $KCD_PRETRAINED_ROOT (e.g. /data/users/jon.crall/pretrained_models/)
#   $KCD_SLURM_LOG_DPATH (e.g. /data/users/jon.crall/slurm_logs/)
#
# This data isn't sensitive for the sea-lion project — public NOAA
# imagery + derived artifacts. If that ever changes, narrow the chmod
# to a specific group instead of o+rw.
#
# Usage (assuming cwd = ~/code/kwcoco_detector_kit):
#   bash projects/viame_sealions_2026/scripts/fixup_perms.sh
#   bash projects/viame_sealions_2026/scripts/fixup_perms.sh /custom/path  # extra paths
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-arisia}"

# Default targets — the work dirs that docker has written into. Skip
# any that don't exist yet. Extra targets can be passed positionally.
TARGETS=()
for p in "$KCD_TRAINING_ROOT" "$KCD_PRETRAINED_ROOT" "$KCD_SLURM_LOG_DPATH" "$@"; do
    if [ -e "$p" ]; then
        TARGETS+=("$p")
    fi
done

if [ "${#TARGETS[@]}" -eq 0 ]; then
    echo "Nothing to fix — none of the target paths exist." >&2
    exit 0
fi

echo "Will chmod -R a+rwX (world-readable + writable) on:"
for p in "${TARGETS[@]}"; do echo "  $p"; done
echo

# We need a mount that covers all targets. They all live under
# $KCD_DATA_ROOT, so mount that one path.
docker run --rm \
    -v "$KCD_DATA_ROOT:$KCD_DATA_ROOT" \
    "$KCD_IMAGE" \
    bash -c "chmod -R a+rwX ${TARGETS[*]@Q}"

echo
echo "Done. Verify with:"
for p in "${TARGETS[@]}"; do
    echo "  ls -ld $p"
done
