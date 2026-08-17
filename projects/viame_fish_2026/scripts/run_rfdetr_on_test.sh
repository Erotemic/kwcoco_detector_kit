#!/usr/bin/env bash
# Run the trained RF-DETR detector over the held-out Test/ split, so it can be
# scored against DEIMv2 under one protocol.
#
# RUNS ON THE HOST under VIAME (not in the kit's docker image) -- this is the
# VIAME-native stack, and the model is a VIAME detector package.
#
# ## What makes the comparison fair
#
# The image list is generated FROM `test.kwcoco.json`, the same bundle DEIMv2
# was scored on, rather than from a directory walk. Both models therefore see
# byte-identical inputs: the same extracted frames, in the same set. A
# directory walk would drift the moment either side re-extracted anything.
#
# Two hazards this guards against:
#
#   * The downsampler. VIAME's common_default_input_with_downsampler.pipe sets
#     target_frame_rate 5 against a reader whose frame_time is 1 (i.e. 1 Hz),
#     so it should pass everything through -- but "should" is not good enough
#     when a silent drop would mean the two models scored different image sets.
#     The rate is overridden AND the output coverage is verified afterwards.
#   * Score truncation. The pipeline's class_probablity_filter threshold is
#     0.0 with keep_all_classes true, which is what AP needs. Do not raise it
#     here: a threshold applied before scoring truncates the ranking and
#     lowers AP for reasons that have nothing to do with the model.
#
# Usage:
#   bash projects/viame_fish_2026/scripts/run_rfdetr_on_test.sh
#
# Long job (33,434 images through a windowed segmentation detector). Use tmux.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

DETECTOR_ZIP="${VF_RFDETR_ZIP:-$VF_RUNS_DPATH/fishtrack23_rfdetr_l_seg720_4gpu_viame0227_gen001/attempt_20260804_195055/fish_detector.zip}"
OUT_DPATH="${VF_RFDETR_INFER_DPATH:-$VF_KCD_ROOT/rfdetr_test_inference}"
TEST_KWCOCO="${VF_TEST_KWCOCO:-$VF_BUNDLE_DPATH/test.kwcoco.json}"

kcd_require_path "RF-DETR detector zip" "$DETECTOR_ZIP"
kcd_require_path "test bundle" "$TEST_KWCOCO"
kcd_require_path "VIAME install" "$VF_CURRENT_VIAME_LINK"

PROJECT_DIR="$OUT_DPATH/project"
LIST_FPATH="$OUT_DPATH/input_list.txt"
CSV_FPATH="$OUT_DPATH/computed_detections.csv"

echo "=============================================================="
echo " RF-DETR inference over the held-out test split"
echo "=============================================================="
echo "  detector: $DETECTOR_ZIP"
echo "  test set: $TEST_KWCOCO"
echo "  output:   $OUT_DPATH"
echo

mkdir -p "$PROJECT_DIR/category_models"

# 1. Unpack the detector where detector_project_folder.pipe expects it:
#    it includes $VIAME_PROJECT_DIR/category_models/detector.pipe, and that
#    pipe's `relativepath weight` resolves trained_detector.pth beside it.
echo "=== unpacking detector ==="
unzip -o -q "$DETECTOR_ZIP" -d "$PROJECT_DIR/category_models"
ls -la "$PROJECT_DIR/category_models"
echo

# 2. Image list straight from the kwcoco bundle DEIMv2 was scored on.
echo "=== building image list from the test bundle ==="
python3 - "$TEST_KWCOCO" "$LIST_FPATH" <<'PYEOF'
import json, sys, pathlib
bundle, out = sys.argv[1], sys.argv[2]
dset = json.loads(pathlib.Path(bundle).read_text())
paths = [im['file_name'] for im in dset['images']]
missing = [p for p in paths[:2000] if not pathlib.Path(p).exists()]
if missing:
    raise SystemExit('ERROR: {} of the first 2000 image paths do not exist, '
                     'e.g. {}'.format(len(missing), missing[:3]))
pathlib.Path(out).write_text('\n'.join(paths) + '\n')
print('  wrote {} image paths -> {}'.format(len(paths), out))
PYEOF
N_INPUT=$(wc -l < "$LIST_FPATH")
echo

# 3. Run the pipeline.
echo "=== running VIAME pipeline over $N_INPUT images ==="
export VIAME_PROJECT_DIR="$PROJECT_DIR"

# VIAME's setup script is not written for `set -u`. It does:
#     export KWIVER_PLUGIN_PATH=$this_dir/...:$KWIVER_PLUGIN_PATH
# i.e. appends to a dozen PATH-like variables assuming unset reads as empty,
# which aborts immediately under `set -u`:
#     setup_viame.sh: line 11: KWIVER_PLUGIN_PATH: unbound variable
#
# This project's older VIAME runbook (_launch_viame_train.sh) never hit it
# because that script does not enable the strict flags at all. Relaxing them
# just around the source is better than dropping them for the whole script:
# everything else here -- the coverage verification especially -- wants them.
set +u
set +e
# shellcheck disable=SC1091
source "$VF_CURRENT_VIAME_LINK/setup_viame.sh"
set -e
set -u

command -v kwiver >/dev/null || {
    echo "ERROR: 'kwiver' not on PATH after sourcing setup_viame.sh." >&2
    echo "  VIAME_INSTALL=${VIAME_INSTALL:-<unset>}" >&2
    exit 1
}
echo "  kwiver: $(command -v kwiver)"
cd "$OUT_DPATH"

set -x
kwiver runner "$VF_CURRENT_VIAME_LINK/configs/pipelines/detector_project_folder.pipe" \
    -s input:video_filename="$LIST_FPATH" \
    -s downsampler:target_frame_rate=1000 \
    -s detector_writer:file_name="$CSV_FPATH"
set +x
echo

# 4. Verify coverage. This is the check that actually protects the comparison:
#    if the downsampler (or anything else) dropped frames, the two models did
#    not see the same data and the head-to-head is meaningless.
echo "=== verifying coverage ==="
kcd_require_path "detections csv" "$CSV_FPATH"
python3 - "$CSV_FPATH" "$N_INPUT" <<'PYEOF'
import csv, sys
csv_fpath, n_input = sys.argv[1], int(sys.argv[2])
seen = set()
rows = 0
with open(csv_fpath, errors='replace') as f:
    for row in csv.reader(f):
        if not row or row[0].lstrip().startswith('#'):
            continue
        if len(row) >= 9:
            rows += 1
            seen.add(row[1].strip())
print('  detections:        {:,}'.format(rows))
print('  distinct images:   {:,} of {:,} input'.format(len(seen), n_input))
if len(seen) < n_input:
    print('\n  NOTE: {:,} input images produced no detection rows.'.format(n_input - len(seen)))
    print('  That is legitimate if the detector genuinely found nothing there,')
    print('  but a large fraction usually means frames were dropped. The VIAME')
    print('  CSV only lists images WITH detections, so this cannot distinguish')
    print('  the two cases on its own -- compare against DEIMv2 per-image')
    print('  coverage before trusting a head-to-head.')
PYEOF

echo
echo "=============================================================="
echo " done: $CSV_FPATH"
echo "=============================================================="
echo "Next: convert and score against the same ground truth."
echo "  python3 $SCRIPT_DIR/convert_viame_dets_to_kwcoco.py \\"
echo "      --csv $CSV_FPATH --like $TEST_KWCOCO \\"
echo "      --out $OUT_DPATH/rfdetr_test_preds.kwcoco.json --category fish"
