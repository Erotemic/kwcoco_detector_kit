#!/usr/bin/env bash
# Sync the project subtree (scripts/, docs/, tests/) to its
# corresponding location in the kit checkout on arisia. The canonical
# transport is `git push && git pull on arisia` — this script exists
# only to push uncommitted in-flight work between machines.
#
# Since the 2026-05 reorg, the project lives at
# `kwcoco_detector_kit/projects/viame_sealions_2026/` in both the local
# and remote kit checkouts. Data artifacts (training_ready_v1/,
# unpacked/, ...) are NOT synced — they live outside the kit and are
# transferred separately, if at all.
#
# Usage:
#   bash scripts/rsync_to_arisia.sh                          # default src/dest
#   bash scripts/rsync_to_arisia.sh /path/to/other/project   # override src
#   bash scripts/rsync_to_arisia.sh "" host:/other/dest      # override dest only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# A bare positional arg uses repo root; pass "" to keep the default
# while overriding only DEST.
SRC_DIR="${1:-$REPO_ROOT}"
[ -z "$SRC_DIR" ] && SRC_DIR="$REPO_ROOT"
DEST="${2:-arisia:/home/joncrall/code/kwcoco_detector_kit/projects/viame_sealions_2026/}"

# Sanity: the src dir should look like this project subtree, not an
# arbitrary tree. Check for marker files that always exist.
for marker in scripts/paths.sh docs/class_schemes.yaml; do
    if [ ! -f "$SRC_DIR/$marker" ]; then
        echo "ERROR: $SRC_DIR doesn't look like viame_sealions_2026" >&2
        echo "       (missing $marker)" >&2
        echo "       Pass the project path as the first argument if syncing" >&2
        echo "       from a non-standard location." >&2
        exit 1
    fi
done

echo "src:  $SRC_DIR"
echo "dest: $DEST"
echo "NOTE: prefer 'git push && ssh arisia \"cd $KCD_KIT_DPATH && git pull\"'"
echo "      when the work is already committed."
echo

rsync -avh --info=progress2 \
    --include '/scripts/***' \
    --include '/docs/***' \
    --include '/tests/***' \
    --include '/README.md' \
    --include '/AGENT.md' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '*' \
    "${SRC_DIR%/}/" \
    "${DEST}"
