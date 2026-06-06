#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

echo "Repo: $ROOT"

if [[ ! -f .gitmodules ]]; then
    echo "No .gitmodules found"
    exit 1
fi

echo
echo "Reading submodules from .gitmodules"
mapfile -t URL_LINES < <(git config -f .gitmodules --get-regexp '^submodule\..*\.url$')
mapfile -t PATH_LINES < <(git config -f .gitmodules --get-regexp '^submodule\..*\.path$')

declare -a SUBMODULE_URLS=()
declare -a SUBMODULE_PATHS=()

for line in "${URL_LINES[@]}"; do
    url="${line#* }"
    SUBMODULE_URLS+=("$url")
done

for line in "${PATH_LINES[@]}"; do
    path="${line#* }"
    SUBMODULE_PATHS+=("$path")
done

echo
echo "Removing global url.*.insteadOf rewrites that match these submodule URLs"
while read -r key prefix; do
    [[ -z "${key:-}" ]] && continue

    for url in "${SUBMODULE_URLS[@]}"; do
        if [[ "$url" == "$prefix"* ]]; then
            echo "  unset: $key = $prefix"
            git config --global --unset-all "$key" || true
        fi
    done
done < <(git config --global --get-regexp '^url\..*\.insteadof$' || true)

echo
echo "Remaining matching GitLab/GitHub rewrites:"
git config --show-origin --get-regexp 'url\..*\.insteadof' || true

echo
echo "Syncing submodule URLs from .gitmodules"
git submodule sync --recursive

echo
echo "Deinitializing submodules"
git submodule deinit -f --all || true

echo
echo "Removing stale submodule working trees and metadata"
for path in "${SUBMODULE_PATHS[@]}"; do
    echo "  rm -rf $path"
    rm -rf "$path"

    echo "  rm -rf .git/modules/$path"
    rm -rf ".git/modules/$path"
done

echo
echo "Reinitializing submodules from recorded commits"
git submodule update --init --recursive --jobs 8

echo
echo "Setting each submodule origin URL to .gitmodules URL"
for line in "${URL_LINES[@]}"; do
    key="${line%% *}"
    url="${line#* }"

    name="${key#submodule.}"
    name="${name%.url}"
    path="$(git config -f .gitmodules --get "submodule.$name.path")"

    if [[ -d "$path" ]]; then
        echo "  $path -> $url"
        git -C "$path" remote set-url origin "$url"
    fi
done

echo
echo "Final submodule status:"
git submodule status --recursive

echo
echo "Final submodule remotes:"
for path in "${SUBMODULE_PATHS[@]}"; do
    echo
    echo "[$path]"
    git -C "$path" remote -v || true
done
