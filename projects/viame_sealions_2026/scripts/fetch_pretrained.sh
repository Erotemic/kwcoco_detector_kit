#!/usr/bin/env bash
# Fetch the DEIMv2-DINOv3-S COCO-pretrained checkpoint to the canonical
# path defined in scripts/paths.sh.
#
# After this runs successfully:
#   - $KCD_DEIMV2_DINOV3_S_COCO_DIR contains the full HF snapshot
#   - $KCD_DEIMV2_DINOV3_S_COCO_PTH is a real file (the .pth checkpoint)
# Other scripts (launch_pup_vs_nonpup_arisia.sh, etc.) read those vars
# directly — no need to look up filenames manually.
#
# Idempotent: if the canonical .pth already exists, exits successfully
# without re-downloading.
#
# This needs `huggingface-cli` on $PATH. It's installed inside the kit's
# docker image (transformers extra). To run on the host instead:
#     pip install --user huggingface_hub
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

REPO_ID="${KCD_HF_REPO:-Intellindust/DEIMv2_DINOv3_S_COCO}"
DEST_DIR="$KCD_DEIMV2_DINOV3_S_COCO_DIR"
CANONICAL_PTH="$KCD_DEIMV2_DINOV3_S_COCO_PTH"

if [ -f "$CANONICAL_PTH" ] && [ -z "${KCD_FORCE_REFETCH:-}" ]; then
    echo "Already on disk: $CANONICAL_PTH"
    echo "  (set KCD_FORCE_REFETCH=1 to redownload)"
    exit 0
fi

mkdir -p "$DEST_DIR"
echo "Downloading $REPO_ID -> $DEST_DIR"
huggingface-cli download "$REPO_ID" --local-dir "$DEST_DIR" >/dev/null

# The HF repo ships model.safetensors, not a .pth. DEIMv2's
# load_tuning_state expects torch.load() returning a dict with a 'model'
# key. Convert once at fetch time so launch can just point at a .pth.
SAFETENSORS_FPATH="$DEST_DIR/model.safetensors"
PTH_FPATH="$(find "$DEST_DIR" -maxdepth 2 -name '*.pth' -not -name "$(basename "$CANONICAL_PTH")" 2>/dev/null | head -n1)"

if [ -n "$PTH_FPATH" ]; then
    if [ "$PTH_FPATH" != "$CANONICAL_PTH" ]; then
        ln -sf "$(basename "$PTH_FPATH")" "$CANONICAL_PTH"
    fi
elif [ -f "$SAFETENSORS_FPATH" ]; then
    echo
    echo "Converting safetensors -> DEIMv2 .pth ..."
    python3 "$SCRIPT_DIR/safetensors_to_deimv2_pth.py" \
        --src "$SAFETENSORS_FPATH" \
        --dst "$CANONICAL_PTH"
else
    echo "ERROR: neither .pth nor model.safetensors found in $DEST_DIR" >&2
    echo "Contents:" >&2
    ls -la "$DEST_DIR" >&2
    exit 1
fi

echo
echo "OK: $CANONICAL_PTH"
ls -lh "$CANONICAL_PTH"
