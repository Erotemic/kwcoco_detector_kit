#!/usr/bin/env bash
# Configure this clone's `git push` to also handle submodules.
#
# Sets `push.recurseSubmodules = on-demand` in the LOCAL repo config
# (so `git push` from the kit root checks each configured submodule;
# any submodule whose currently-checked-out commit isn't yet on its
# remote gets pushed first).
#
# Re-runnable / idempotent. Safe: writes only to .git/config of the
# current clone — no global or system-wide changes.
#
# Optional: also configure the LFS-style "fetch + clone everything"
# behavior so `git fetch` updates submodules too, and `git status`
# shows the submodule pointer changes.
#
# Run once per fresh clone:
#   bash scripts/setup_git.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d .git ] && [ ! -f .git ]; then
    echo "ERROR: $REPO_ROOT does not look like a git working tree." >&2
    exit 1
fi

# 1. `git push` from the parent auto-pushes submodules whose pinned
#    commits aren't on their remote yet. The push fails BEFORE
#    touching the parent's remote if any submodule push fails -- the
#    parent never references a commit a clone wouldn't be able to
#    fetch.
git config --local push.recurseSubmodules on-demand
echo "  push.recurseSubmodules = on-demand"

# 2. `git fetch` from the parent also fetches submodules.
git config --local fetch.recurseSubmodules on-demand
echo "  fetch.recurseSubmodules = on-demand"

# 3. `git status` / `git diff` summarises submodule pointer changes
#    (e.g. "1 file changed, 1 insertion(+), 1 deletion(-)" with the
#    old/new commit displayed) rather than the per-file diff of the
#    submodule's working tree.
git config --local status.submoduleSummary 1
echo "  status.submoduleSummary = 1"
git config --local diff.submodule log
echo "  diff.submodule = log"

# 4. `git submodule update` defaults to merging on top of local
#    submodule branches when possible (vs. always detached HEAD).
#    Helpful if you sometimes edit submodule code locally.
git config --local submodule.recurse true
echo "  submodule.recurse = true"

echo
echo "Local clone configured. \`git push\` from this clone will now:"
echo "  - check each configured submodule"
echo "  - push any submodule whose pinned commit isn't on its remote yet"
echo "  - abort BEFORE pushing the parent if any submodule push fails"
echo
echo "To inspect what would happen without pushing:"
echo "  git push --dry-run"
echo
echo "To revert: git config --local --unset push.recurseSubmodules"
