#!/usr/bin/env bash
# Make the prebuilt OpenGroundingDINO CUDA extension available when users
# bind-mount an editable checkout over /workspace/kwcoco_detector_kit.
set -euo pipefail

BAKED_REPO="${KCD_BAKED_REPO:-/opt/kwcoco_detector_kit}"
MOUNTED_REPO="${KCD_MOUNTED_REPO:-/workspace/kwcoco_detector_kit}"

copy_extension_if_needed() {
    local src_repo="$1"
    local dst_repo="$2"
    local src_ops="$src_repo/tpl/Open-GroundingDino/models/GroundingDINO/ops"
    local dst_ops="$dst_repo/tpl/Open-GroundingDino/models/GroundingDINO/ops"

    [ -d "$dst_ops" ] || return 0
    if compgen -G "$dst_ops/MultiScaleDeformableAttention*.so" >/dev/null; then
        return 0
    fi
    if compgen -G "$src_ops/MultiScaleDeformableAttention*.so" >/dev/null; then
        cp "$src_ops"/MultiScaleDeformableAttention*.so "$dst_ops"/
    fi
    if [ -d "$dst_repo/tpl/Open-GroundingDino" ] && \
       ! compgen -G "$dst_repo/tpl/Open-GroundingDino/MultiScaleDeformableAttention*.so" >/dev/null && \
       compgen -G "$src_repo/tpl/Open-GroundingDino/MultiScaleDeformableAttention*.so" >/dev/null; then
        cp "$src_repo"/tpl/Open-GroundingDino/MultiScaleDeformableAttention*.so \
           "$dst_repo"/tpl/Open-GroundingDino/
    fi
}

if [ -d "$MOUNTED_REPO" ] && [ "$MOUNTED_REPO" != "$BAKED_REPO" ]; then
    copy_extension_if_needed "$BAKED_REPO" "$MOUNTED_REPO"
fi

exec "$@"
