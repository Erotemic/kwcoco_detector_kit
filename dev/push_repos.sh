#!/usr/bin/env bash
# Push submodule commits and then the parent repository.
#
# This is intentionally conservative:
# - Refuses dirty repos unless ALLOW_DIRTY=1.
# - Pushes each detached submodule HEAD to its current origin HEAD branch.
# - Pushes the parent repo last, after submodule commits are available remotely.
# - DRYRUN=1 prints the commands without executing them.
set -euo pipefail

ROOT_DPATH="${ROOT_DPATH:-$(git rev-parse --show-toplevel)}"
REMOTE="${REMOTE:-origin}"
DRYRUN="${DRYRUN:-0}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
PUSH_TAGS="${PUSH_TAGS:-0}"

run() {
    if [ "$DRYRUN" = "1" ]; then
        printf '+'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

require_clean() {
    local repo_dpath="$1"
    local label="$2"
    if [ "$ALLOW_DIRTY" = "1" ]; then
        return
    fi
    local dirty
    dirty="$(git -C "$repo_dpath" status --porcelain)"
    if [ -n "$dirty" ]; then
        echo "Refusing to push dirty repo: $label ($repo_dpath)" >&2
        echo "$dirty" >&2
        echo "Set ALLOW_DIRTY=1 to bypass this check." >&2
        exit 1
    fi
}

remote_head_branch() {
    local repo_dpath="$1"
    local remote="$2"
    local ref
    ref="$(git -C "$repo_dpath" symbolic-ref -q --short "refs/remotes/$remote/HEAD" || true)"
    if [ -n "$ref" ]; then
        echo "${ref#"$remote/"}"
        return
    fi
    # Fallback for older / unusual remotes.
    ref="$(git -C "$repo_dpath" remote show "$remote" 2>/dev/null | sed -n 's/.*HEAD branch: //p' | head -n 1)"
    if [ -n "$ref" ]; then
        echo "$ref"
        return
    fi
    echo "main"
}

push_repo() {
    local repo_dpath="$1"
    local label="$2"
    local remote="$3"
    local branch="$4"
    local sha
    sha="$(git -C "$repo_dpath" rev-parse --short HEAD)"
    echo
    echo "=== Push $label ==="
    echo "repo=$repo_dpath"
    echo "remote=$remote"
    echo "branch=$branch"
    echo "head=$sha"
    require_clean "$repo_dpath" "$label"
    run git -C "$repo_dpath" push "$remote" "HEAD:$branch"
    if [ "$PUSH_TAGS" = "1" ]; then
        run git -C "$repo_dpath" push "$remote" --tags
    fi
}

cd "$ROOT_DPATH"

echo "Root: $ROOT_DPATH"
echo "Remote: $REMOTE"
if [ "$DRYRUN" = "1" ]; then
    echo "Mode: dry run"
fi

git submodule update --init

git submodule foreach --quiet '
    set -e
    remote="${REMOTE:-origin}"
    if ! git remote get-url "$remote" >/dev/null 2>&1; then
        echo "Skipping $name: no remote named $remote"
        exit 0
    fi
    dirty="$(git status --porcelain)"
    if [ -n "$dirty" ] && [ "${ALLOW_DIRTY:-0}" != "1" ]; then
        echo "Refusing to push dirty submodule: $name ($sm_path)" >&2
        echo "$dirty" >&2
        exit 1
    fi
    # Push detached HEAD to the branch tracked by origin/HEAD.
    branch="$(git symbolic-ref -q --short "refs/remotes/$remote/HEAD" || true)"
    branch="${branch#"$remote/"}"
    if [ -z "$branch" ]; then
        branch="$(git remote show "$remote" 2>/dev/null | sed -n "s/.*HEAD branch: //p" | head -n 1)"
    fi
    if [ -z "$branch" ]; then
        branch=main
    fi
    echo
    echo "=== Push submodule $name ==="
    echo "path=$sm_path"
    echo "remote=$remote"
    echo "branch=$branch"
    echo "head=$(git rev-parse --short HEAD)"
    if [ "${DRYRUN:-0}" = "1" ]; then
        echo "+ git push $remote HEAD:$branch"
    else
        git push "$remote" "HEAD:$branch"
    fi
    if [ "${PUSH_TAGS:-0}" = "1" ]; then
        if [ "${DRYRUN:-0}" = "1" ]; then
            echo "+ git push $remote --tags"
        else
            git push "$remote" --tags
        fi
    fi
'

parent_branch="$(git branch --show-current)"
if [ -z "$parent_branch" ]; then
    parent_branch="$(remote_head_branch "$ROOT_DPATH" "$REMOTE")"
fi
push_repo "$ROOT_DPATH" "parent" "$REMOTE" "$parent_branch"

echo
echo "Push complete."
