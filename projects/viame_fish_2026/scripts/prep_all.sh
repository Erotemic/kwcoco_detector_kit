#!/usr/bin/env bash
# Full data prep for the DEIMv2 fish run, start to finish. RUN THIS ON THE HOST
# (aiq-gpu): it needs ffmpeg, the corpus, and enough cores to be worth doing.
#
#   1. extract the annotated frames of Train/ and Test/ to JPEG on the NVMe
#   2. convert each to kwcoco, folding species -> `fish` via the corpus labels.txt
#   3. split Train/ into sequence-disjoint train/vali
#
# Every step is idempotent: re-running skips sequences already extracted and
# rewrites the (cheap) kwcoco files. Safe to resume after an interruption.
#
# Usage:
#   bash projects/viame_fish_2026/scripts/prep_all.sh
#   VF_FRAME_STRIDE=3 bash .../prep_all.sh     # subsample when building splits
#   bash .../prep_all.sh --limit 5             # smoke test on 5 videos per input
#
# Long job: run it under tmux with tee, per project convention.
#   tmux new -s fishprep
#   bash projects/viame_fish_2026/scripts/prep_all.sh 2>&1 | tee ~/fish_prep.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

LIMIT_ARGS=()
if [ "${1:-}" = "--limit" ]; then
    LIMIT_ARGS=(--limit "${2:?--limit needs a count}")
    echo "SMOKE TEST: only ${2} videos per input directory"
fi

JOBS="${VF_PREP_JOBS:-$(( $(nproc) / 4 ))}"
[ "$JOBS" -lt 1 ] && JOBS=1

echo "=============================================================="
echo " fish DEIMv2 data prep"
echo "=============================================================="
echo "  corpus (train):  $VF_INPUT_DPATH"
echo "  corpus (test):   $VF_TEST_INPUT_DPATH"
echo "  labels:          $VF_LABELS_FPATH"
echo "  frames ->        $VF_FRAMES_DPATH"
echo "  bundle ->        $VF_BUNDLE_DPATH"
echo "  ffmpeg jobs:     $JOBS"
echo "  frame stride:    $VF_FRAME_STRIDE"
echo

kcd_require_path "train corpus" "$VF_INPUT_DPATH"
kcd_require_path "test corpus"  "$VF_TEST_INPUT_DPATH"
kcd_require_path "labels.txt"   "$VF_LABELS_FPATH"

# Both are needed: ffprobe recovers the frame rate for the ~20 videos whose
# CSV omits an fps comment. Checked together so a missing ffprobe surfaces now
# rather than partway through extraction.
for tool in "$VF_FFMPEG" "$VF_FFPROBE"; do
    if ! command -v "$tool" >/dev/null 2>&1 && [ ! -x "$tool" ]; then
        echo "ERROR: $tool not found." >&2
        echo "  sudo apt install ffmpeg    # provides both ffmpeg and ffprobe" >&2
        echo "  or set VF_FFMPEG / VF_FFPROBE to specific binaries." >&2
        exit 1
    fi
done

# The frames land on the NVMe, which is the smaller of the two filesystems.
# Annotated-only extraction is ~75 GB at q95; refuse to start if that clearly
# will not fit rather than filling the root filesystem and wedging the host.
avail_gb=$(df -BG --output=avail "$VF_SSD_ROOT" | tail -1 | tr -dc '0-9')
echo "  NVMe free:       ${avail_gb} GB"
if [ "$avail_gb" -lt 120 ] && [ ${#LIMIT_ARGS[@]} -eq 0 ]; then
    echo "ERROR: need ~120 GB free on $VF_SSD_ROOT for frame extraction; have ${avail_gb} GB." >&2
    echo "  Free space, or point VF_FRAMES_DPATH somewhere with room (but keep it" >&2
    echo "  off /data -- that is the md0 RAID array this design exists to avoid)." >&2
    exit 1
fi
echo

# ---------------------------------------------------------------- 1. extract
for split in train test; do
    case "$split" in
        train) input="$VF_INPUT_DPATH" ;;
        test)  input="$VF_TEST_INPUT_DPATH" ;;
    esac
    echo "=== [1/3] extracting frames: $split ==="
    python3 "$SCRIPT_DIR/extract_frames.py" \
        --input "$input" \
        --out-dir "$VF_FRAMES_DPATH/$split" \
        --jobs "$JOBS" \
        --ffmpeg "$VF_FFMPEG" \
        --ffprobe "$VF_FFPROBE" \
        "${LIMIT_ARGS[@]}"
    echo
done

# ---------------------------------------------------------------- 2. convert
mkdir -p "$VF_BUNDLE_DPATH"

echo "=== [2/3] converting Train/ to kwcoco ==="
python3 "$SCRIPT_DIR/convert_viame_to_kwcoco.py" \
    --input "$VF_INPUT_DPATH" \
    --frames "$VF_FRAMES_DPATH/train" \
    --labels "$VF_LABELS_FPATH" \
    --out "$VF_BUNDLE_DPATH/train_all.kwcoco.json"
echo

echo "=== [2/3] converting Test/ to kwcoco (held-out; never trained on) ==="
# Same labels.txt as Train, so the test set carries the identical class
# definition. RF-DETR never saw any of these sequences.
python3 "$SCRIPT_DIR/convert_viame_to_kwcoco.py" \
    --input "$VF_TEST_INPUT_DPATH" \
    --frames "$VF_FRAMES_DPATH/test" \
    --labels "$VF_LABELS_FPATH" \
    --out "$VF_BUNDLE_DPATH/test.kwcoco.json"
echo

# ----------------------------------------------------------------- 3. splits
echo "=== [3/3] building sequence-disjoint train/vali splits ==="
python3 "$SCRIPT_DIR/build_splits.py" \
    --in-kwcoco "$VF_BUNDLE_DPATH/train_all.kwcoco.json" \
    --out-train "$VF_BUNDLE_DPATH/train.kwcoco.json" \
    --out-vali  "$VF_BUNDLE_DPATH/vali.kwcoco.json" \
    --vali-fraction "$VF_VALI_FRACTION" \
    --seed "$VF_SPLIT_SEED" \
    --stride "$VF_FRAME_STRIDE"

echo
echo "=============================================================="
echo " prep complete"
echo "=============================================================="
du -sh "$VF_FRAMES_DPATH" 2>/dev/null || true
ls -la "$VF_BUNDLE_DPATH"
echo
echo "Next: submit training with"
echo "  bash $SCRIPT_DIR/submit_train_fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen001.sh"
