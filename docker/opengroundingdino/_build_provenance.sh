#!/usr/bin/env bash
# Emit the provenance --build-arg flags for the kit docker image, one token
# per line (so callers can `mapfile` them into a docker build arg array).
#
# Bakes, into /etc/kcd_provenance.json + image LABELs (see Dockerfile):
#   * git HEAD of the kit and each submodule (DEIMv2, Open-GroundingDino,
#     kwcoco_dataloader) — "<sha>-dirty" if the working tree is modified
#   * a short hash of the Dockerfile itself
#   * the build time (UTC)
#
# Why: the image content-hash alone can't answer "how do I reproduce this?"
# if the image was never published. Baking the repo SHAs lets future
# detective work proceed (and corroborates claims) from the running
# container or `docker inspect`, even with no registry copy.
#
# Usage (from a build_*.sh that has already cd'd to the repo root):
#   source "$(dirname "$0")/_build_provenance.sh"
#   mapfile -t PROV_ARGS < <(kcd_provenance_build_args)
#   docker build "${PROV_ARGS[@]}" ...

kcd_provenance_build_args() {
    _kcd_sha() {
        local dir="$1" sha dirty
        sha="$(git -C "$dir" rev-parse --short=12 HEAD 2>/dev/null)" || { echo unknown; return; }
        dirty="$(git -C "$dir" status --porcelain 2>/dev/null)"
        [ -n "$dirty" ] && sha="${sha}-dirty"
        echo "$sha"
    }
    local kit deimv2 ogdino dl dfsha bt
    kit="$(_kcd_sha .)"
    deimv2="$(_kcd_sha tpl/DEIMv2)"
    ogdino="$(_kcd_sha tpl/Open-GroundingDino)"
    dl="$(_kcd_sha tpl/kwcoco_dataloader)"
    dfsha="$(sha256sum docker/opengroundingdino/Dockerfile 2>/dev/null | cut -c1-16)"
    [ -z "$dfsha" ] && dfsha=unknown
    # Date.now is fine here (host shell at build time, not a workflow).
    bt="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)"
    printf '%s\n' \
        "--build-arg" "KCD_KIT_SHA=$kit" \
        "--build-arg" "KCD_DEIMV2_SHA=$deimv2" \
        "--build-arg" "KCD_OGDINO_SHA=$ogdino" \
        "--build-arg" "KCD_DATALOADER_SHA=$dl" \
        "--build-arg" "KCD_DOCKERFILE_SHA=$dfsha" \
        "--build-arg" "KCD_BUILD_TIME=$bt"
}
