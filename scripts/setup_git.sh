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

# 1. `git push` from the parent verifies (via the built-in `check`
#    mode) that every referenced submodule commit is already on its
#    remote, BUT does not attempt to push the submodules itself —
#    git's built-in `on-demand` mode pushes to the submodule's
#    UPSTREAM (typically `main`), ignoring the `branch = ...` hint
#    in .gitmodules. The actual submodule push is done by our
#    `scripts/push_submodules.sh` via the pre-push hook installed
#    below, which DOES respect the `branch` field.
git config --local push.recurseSubmodules check
echo "  push.recurseSubmodules = check"

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

# 5. Install a pre-push hook that calls scripts/push_submodules.sh
#    BEFORE the parent push runs. Hook reads .gitmodules per-submodule
#    `branch` and pushes HEAD:refs/heads/<branch>, which handles the
#    common detached-HEAD-after-submodule-update case.
HOOK_FPATH="$(git rev-parse --git-dir)/hooks/pre-push"
mkdir -p "$(dirname "$HOOK_FPATH")"
cat > "$HOOK_FPATH" <<'HOOK'
#!/usr/bin/env bash
# Pre-push hook installed by scripts/setup_git.sh.
#
# Pushes every submodule (per scripts/push_submodules.sh: respects
# .gitmodules `branch =` hints) before the parent push runs. The
# parent push only proceeds if every submodule push succeeds.
set -e
PUSH_SUBMODULES_SH="$(git rev-parse --show-toplevel)/scripts/push_submodules.sh"
if [ -x "$PUSH_SUBMODULES_SH" ]; then
    echo "[pre-push] Pushing submodules first..."
    bash "$PUSH_SUBMODULES_SH"
fi
exit 0
HOOK
chmod +x "$HOOK_FPATH"
echo "  pre-push hook -> $HOOK_FPATH"

echo
echo "Local clone configured. \`git push\` from this clone will now:"
echo "  1. (pre-push hook) push each submodule via scripts/push_submodules.sh,"
echo "     respecting the \`branch = ...\` field in .gitmodules"
echo "  2. (built-in check) verify every submodule commit referenced by the"
echo "     parent is now on its remote — aborts the parent push if not"
echo
echo "To inspect what would happen without pushing:"
echo "  bash scripts/push_submodules.sh --dry-run"
echo "  git push --dry-run"
echo
echo "To revert:"
echo "  git config --local --unset push.recurseSubmodules"
echo "  rm $HOOK_FPATH"
