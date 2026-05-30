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

# 1. `git push` from the parent auto-pushes each submodule's
#    current-branch commits to that branch's UPSTREAM. We configure
#    each submodule's local branch + upstream below (see step 6) so
#    on-demand pushes go to the right branch on the fork.
git config --local push.recurseSubmodules on-demand
echo "  push.recurseSubmodules = on-demand"

# 2. `git fetch` from the parent also fetches submodules.
git config --local fetch.recurseSubmodules on-demand
echo "  fetch.recurseSubmodules = on-demand"

# 2b. `git pull` from the parent also advances submodules to the
#     parent's new pointer. Without this, the parent gets the new
#     pinned commit but the submodule's working tree stays at the
#     old one, and dev-mounts (KCD_DEV_MOUNT_DEIMV2=1) pick up stale
#     code from the host's submodule dir.
git config --local pull.recurseSubmodules on-demand
echo "  pull.recurseSubmodules = on-demand"

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

# 5. Drop any pre-push hook a previous version of this script may
#    have installed (we moved the branch-aware push into step 6
#    rather than carrying a hook).
HOOK_FPATH="$(git rev-parse --git-dir)/hooks/pre-push"
if [ -f "$HOOK_FPATH" ] && grep -q "push_submodules.sh" "$HOOK_FPATH" 2>/dev/null; then
    rm -f "$HOOK_FPATH"
    echo "  removed stale pre-push hook"
fi

# 6. For every submodule that declares `branch = X` in .gitmodules,
#    ensure the submodule has a LOCAL branch X tracking origin/X.
#    Without this, the submodule sits on detached HEAD after
#    `git submodule update --init` and `git push --recurse-submodules`
#    has no idea which branch to push to (falls back to refs/heads/main
#    which is wrong for branches like ddp-loss-key-alignment or
#    dev/0.1.3).
echo
echo "Configuring submodule tracking branches:"
git config -f .gitmodules --get-regexp '^submodule\..*\.path$' | while read -r key path; do
    name=$(echo "$key" | sed -e 's/^submodule\.//' -e 's/\.path$//')
    cfg_branch=$(git config -f .gitmodules --get "submodule.${name}.branch" 2>/dev/null || true)
    if [ -z "$cfg_branch" ]; then
        echo "  $path: no branch in .gitmodules; leaving as-is"
        continue
    fi
    if [ ! -d "$path/.git" ] && [ ! -f "$path/.git" ]; then
        echo "  $path: not initialized (run 'git submodule update --init $path')"
        continue
    fi

    (
        cd "$path"
        git fetch --quiet origin "$cfg_branch" 2>/dev/null || true

        # CRITICAL: never move HEAD away from the kit's pinned commit.
        # That commit is what `git submodule update --init` just
        # checked out (detached HEAD). If we `git checkout <branch>`
        # naively, we'd lose the kit's pin and land on whatever the
        # local branch tip happens to be — which can be a wildly
        # different commit from a sibling line of work on the fork.
        #
        # `git checkout -B <branch> HEAD` is atomic: reset local
        # branch to current HEAD AND check it out. The kit-pinned
        # commit stays the working-tree commit AND becomes the tip
        # of the named branch. The branch's upstream is then set so
        # push goes to the right place on the fork.
        pinned=$(git rev-parse HEAD)
        git checkout --quiet -B "$cfg_branch" "$pinned"
        git branch --set-upstream-to="origin/$cfg_branch" "$cfg_branch" 2>/dev/null || true
        echo "  $path: on '$cfg_branch' at $(git rev-parse --short HEAD), tracking origin/$cfg_branch"
    )
done

echo
echo "Local clone configured. \`git push\` from this clone will now:"
echo "  - recursively push each submodule's current-branch commits to its"
echo "    upstream (origin/<branch>), per push.recurseSubmodules=on-demand"
echo "  - abort the parent push BEFORE touching its remote if any submodule"
echo "    push fails"
echo
echo "To inspect what would happen without pushing:"
echo "  git push --dry-run"
echo
echo "To revert:"
echo "  git config --local --unset push.recurseSubmodules"
