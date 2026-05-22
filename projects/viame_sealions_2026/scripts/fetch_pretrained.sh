#!/usr/bin/env bash
# Fetch a DEIMv2 COCO-pretrained checkpoint to the canonical path
# defined in scripts/paths.sh. Knows about the variants we use:
#
#   deimv2_dinov3_s    -> Intellindust/DEIMv2_DINOv3_S_COCO    (50.9 AP, foundation backbone)
#   deimv2_hgnetv2_n   -> Intellindust/DEIMv2_HGNetv2_N_COCO   (43.0 AP, mobile baseline)
#
# Add a new variant by extending the dispatch block below.
#
# After this runs successfully:
#   - $KCD_<VARIANT>_COCO_DIR contains the full HF snapshot
#   - $KCD_<VARIANT>_COCO_PTH is a real file (the .pth checkpoint)
# Other scripts read those vars directly — no need to look up filenames.
#
# Idempotent: if the canonical .pth already exists, exits successfully
# without re-downloading.
#
# Needs `huggingface-cli` on $PATH. Installed inside the kit's docker
# image; to run on the host instead: pip install --user huggingface_hub.
#
# Usage (assuming cwd = ~/code/kwcoco_detector_kit):
#   bash projects/viame_sealions_2026/scripts/fetch_pretrained.sh                       # default: deimv2_dinov3_s
#   bash projects/viame_sealions_2026/scripts/fetch_pretrained.sh deimv2_hgnetv2_n      # baseline
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

VARIANT="${1:-deimv2_dinov3_s}"

case "$VARIANT" in
    deimv2_dinov3_s)
        REPO_ID="${KCD_HF_REPO:-Intellindust/DEIMv2_DINOv3_S_COCO}"
        DEST_DIR="$KCD_DEIMV2_DINOV3_S_COCO_DIR"
        CANONICAL_PTH="$KCD_DEIMV2_DINOV3_S_COCO_PTH"
        ;;
    deimv2_hgnetv2_n)
        REPO_ID="${KCD_HF_REPO:-Intellindust/DEIMv2_HGNetv2_N_COCO}"
        DEST_DIR="$KCD_DEIMV2_HGNETV2_N_COCO_DIR"
        CANONICAL_PTH="$KCD_DEIMV2_HGNETV2_N_COCO_PTH"
        ;;
    *)
        echo "ERROR: unknown variant: $VARIANT" >&2
        echo "Known: deimv2_dinov3_s, deimv2_hgnetv2_n" >&2
        echo "Add a case branch in $0 to support more." >&2
        exit 1
        ;;
esac

if [ -f "$CANONICAL_PTH" ] && [ -z "${KCD_FORCE_REFETCH:-}" ]; then
    echo "Already on disk: $CANONICAL_PTH"
    echo "  (set KCD_FORCE_REFETCH=1 to redownload)"
    exit 0
fi

mkdir -p "$DEST_DIR"
echo "Downloading $REPO_ID -> $DEST_DIR"
huggingface-cli download "$REPO_ID" --local-dir "$DEST_DIR" >/dev/null

# DEIMv2's load_tuning_state expects torch.load() returning a dict
# with a 'model' key. HF repos ship either a .pth (already correct
# shape) or a model.safetensors that needs converting.
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
