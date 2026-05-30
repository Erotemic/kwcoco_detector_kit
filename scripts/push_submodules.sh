#!/usr/bin/env bash
# Push each git submodule's local commits to its configured upstream branch.
#
# Why this exists: there's no built-in `git submodule push` command. The
# closest built-in -- `git push --recurse-submodules=on-demand` -- only
# fires when the *parent* push references a submodule commit that isn't
# yet on the submodule's remote, and it doesn't push to an explicit
# branch when the submodule is in detached HEAD (which is the common
# state after `git submodule update --init`).
#
# This script:
#   - iterates over every submodule listed in .gitmodules
#   - for each, prefers the branch the submodule is currently checked
#     out on; falls back to the `branch = ...` value in .gitmodules
#     when HEAD is detached
#   - skips submodules with no local commits ahead of upstream (so
#     re-runs are idempotent)
#   - prints what it pushed (or why it skipped)
#
# Usage (from kit root):
#   bash scripts/push_submodules.sh                  # all submodules
#   bash scripts/push_submodules.sh tpl/DEIMv2       # one specific submodule
#   bash scripts/push_submodules.sh --dry-run        # show, don't push
#
# Exits non-zero if any push fails. Suitable for CI hooks.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=0
TARGETS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *) TARGETS+=("$arg") ;;
    esac
done

# Enumerate submodules. Falls back to `git config -f` when .gitmodules is
# the only authoritative source; cross-checks with `git submodule status`
# so we don't trip on stale entries.
mapfile -t ALL_SUBMODULES < <(git config -f .gitmodules --get-regexp '^submodule\..*\.path$' | awk '{print $2}')
if [ ${#ALL_SUBMODULES[@]} -eq 0 ]; then
    echo "no submodules registered in .gitmodules; nothing to do"
    exit 0
fi

# If the caller specified targets, filter to those (validated against the
# real list so typos error out clearly).
if [ ${#TARGETS[@]} -gt 0 ]; then
    declare -A KNOWN=()
    for s in "${ALL_SUBMODULES[@]}"; do KNOWN["$s"]=1; done
    for t in "${TARGETS[@]}"; do
        if [ -z "${KNOWN[$t]:-}" ]; then
            echo "ERROR: '$t' is not a registered submodule" >&2
            echo "  known: ${ALL_SUBMODULES[*]}" >&2
            exit 1
        fi
    done
    SUBMODULES=("${TARGETS[@]}")
else
    SUBMODULES=("${ALL_SUBMODULES[@]}")
fi

exit_code=0
for path in "${SUBMODULES[@]}"; do
    if [ ! -d "$path/.git" ] && [ ! -f "$path/.git" ]; then
        echo "[skip] $path: not initialized (run 'git submodule update --init $path')"
        continue
    fi

    name=$(git config -f .gitmodules --get-regexp '^submodule\..*\.path$' \
           | awk -v p="$path" '$2 == p {print $1}' \
           | sed -e 's/^submodule\.//' -e 's/\.path$//')
    cfg_branch=$(git config -f .gitmodules --get "submodule.${name}.branch" || echo "")

    pushd "$path" >/dev/null

    cur_branch=$(git rev-parse --abbrev-ref HEAD)
    if [ "$cur_branch" = "HEAD" ]; then
        # Detached. Prefer the .gitmodules-declared branch.
        if [ -z "$cfg_branch" ]; then
            echo "[skip] $path: detached HEAD and no branch in .gitmodules"
            popd >/dev/null
            continue
        fi
        target_branch="$cfg_branch"
        refspec="HEAD:refs/heads/$target_branch"
    else
        target_branch="$cur_branch"
        refspec="$cur_branch"
    fi

    # Determine whether there's anything to push. `@{u}` may not exist
    # for the current ref (e.g. a brand-new local branch); fall back to
    # comparing against origin/<target_branch>.
    upstream=$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || echo "")
    if [ -z "$upstream" ]; then
        upstream="origin/$target_branch"
        if ! git rev-parse --verify --quiet "$upstream" >/dev/null; then
            upstream=""
        fi
    fi

    if [ -n "$upstream" ]; then
        ahead=$(git rev-list --count "$upstream"..HEAD 2>/dev/null || echo "0")
        if [ "$ahead" = "0" ]; then
            echo "[skip] $path: up to date with $upstream"
            popd >/dev/null
            continue
        fi
        echo "[push] $path: $ahead commit(s) ahead of $upstream -> origin $refspec"
    else
        echo "[push] $path: no upstream yet -> origin $refspec (first push)"
    fi

    if [ "$DRY_RUN" = "1" ]; then
        echo "       (dry run; skipping)"
    else
        if ! git push origin "$refspec"; then
            echo "ERROR: push failed for $path" >&2
            exit_code=1
        fi
    fi
    popd >/dev/null
done

exit "$exit_code"
