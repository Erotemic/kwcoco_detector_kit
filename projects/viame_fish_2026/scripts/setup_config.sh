#!/usr/bin/env bash
# Copy the project-owned config snapshot into the active VIAME installation.
# The include paths in a VIAME training config are relative to
# VIAME_INSTALL/configs/pipelines, so training should use this installed copy.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

CONFIG_SOURCE="${1:-$VF_PROJECT_CONFIG}"

if [ ! -f "$CONFIG_SOURCE" ]; then
    echo "ERROR: config does not exist: $CONFIG_SOURCE"
    exit 1
fi

if [ ! -f "$VF_CURRENT_VIAME_LINK/setup_viame.sh" ]; then
    echo "ERROR: active VIAME install is missing: $VF_CURRENT_VIAME_LINK"
    echo "Run setup_binaries.sh first."
    exit 1
fi

mkdir -p "$VF_CURRENT_VIAME_LINK/configs/pipelines"
cp -v "$CONFIG_SOURCE" "$VF_CURRENT_VIAME_LINK/configs/pipelines/$VF_CONFIG_NAME"

echo
sha256sum "$CONFIG_SOURCE"
echo "Installed config: $VF_CURRENT_VIAME_LINK/configs/pipelines/$VF_CONFIG_NAME"
