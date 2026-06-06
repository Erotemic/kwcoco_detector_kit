#!/usr/bin/env bash
# Sync the canonical sealion corpus from this host to arisia.
#
# Historical note (pre-2026-06-05): this script used to sync the kit
# project subtree (scripts/, docs/, tests/, README.md, AGENT.md)
# between machines. That was always redundant with `git push && git
# pull`. Replaced 2026-06-05 with the data-sync logic that used to
# live at /data/Public/VIAME/viame_sealions_2026/legacy-tools/scripts/
# rsync_to_arisia.sh. Code sync = git. This script = data.
#
# Syncs the corpus tree at $KCD_DATA_DPATH (default
# /data/Public/VIAME/viame_sealions_2026) to its corresponding
# location on arisia. By policy the source/ subtree (immutable Girder
# mirror + email payload) is namek-only and excluded — arisia
# consumes unpacked/.
#
# Usage:
#   bash projects/viame_sealions_2026/scripts/rsync_to_arisia.sh           # default
#   bash projects/viame_sealions_2026/scripts/rsync_to_arisia.sh -n        # dry run
#   bash projects/viame_sealions_2026/scripts/rsync_to_arisia.sh /alt/src  # override src
#   bash projects/viame_sealions_2026/scripts/rsync_to_arisia.sh "" host:/alt/dest
set -euo pipefail

# Parse the optional -n / --dry-run flag before positional args, so the
# existing positional ordering (src then dest) still works.
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        -n|--dry-run)
            DRY_RUN=1
            shift
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "ERROR: unknown flag $1" >&2
            exit 2
            ;;
        *)
            break
            ;;
    esac
done

# Source paths.sh so $KCD_DATA_DPATH resolves the same way as every
# other script in this project.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./paths.sh
source "$SCRIPT_DIR/paths.sh"

# A bare positional arg overrides the data root; pass "" to keep the
# default while overriding only DEST.
SRC_DIR="${1:-$KCD_DATA_DPATH}"
[ -z "$SRC_DIR" ] && SRC_DIR="$KCD_DATA_DPATH"
DEST="${2:-arisia:/data/Public/VIAME/viame_sealions_2026/}"

# Sanity: the src dir should look like the corpus tree, not an
# arbitrary directory.
for marker in README.md notes.md unpacked/all_norm.kwcoco.zip; do
    if [ ! -e "$SRC_DIR/$marker" ]; then
        echo "ERROR: $SRC_DIR doesn't look like viame_sealions_2026" >&2
        echo "       (missing $marker)" >&2
        echo "       Pass the corpus path as the first argument if syncing" >&2
        echo "       from a non-standard location." >&2
        exit 1
    fi
done

echo "src:  $SRC_DIR"
echo "dest: $DEST"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "mode: DRY RUN (no files transferred)"
fi
echo

RSYNC_ARGS=(-avh --info=progress2)
if [ "$DRY_RUN" -eq 1 ]; then
    RSYNC_ARGS+=(--dry-run --itemize-changes)
fi

rsync "${RSYNC_ARGS[@]}" \
    --include '/README.md' \
    --include '/notes.md' \
    --include '/docs/***' \
    --include '/unpacked/***' \
    --include '/legacy-tools/***' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '*' \
    "${SRC_DIR%/}/" \
    "${DEST}"
