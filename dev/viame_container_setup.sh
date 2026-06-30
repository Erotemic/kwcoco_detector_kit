#!/usr/bin/env bash
#
# Wire the kwcoco_detector_kit ONNX detector into a *built* VIAME container.
#
# Run this INSIDE the running viame-gpu-local container, AFTER sourcing
# setup_viame.sh so that `python`/`pip` resolve to VIAME's internal Python.
#
__doc__='
Why this script exists
----------------------
The VIAME plugin (plugins/pytorch/kwcoco_detector_kit_detector.py) is baked
into the image by the normal CMake build, but at runtime it imports

    from kwcoco_detector_kit.predictors.onnx import OnnxPredictor
    import onnxruntime

Neither the kit nor onnxruntime are part of VIAME, so we install them into
VIAME internal Python here. We deliberately install the kit with --no-deps
so it does NOT pull torch>=2.5 / torchvision over the torch VIAME builds from
source. The ONNX inference path needs only onnxruntime + numpy + kwimage,
which is a torch-free import chain.

Prerequisites
-------------
  * A built viame-gpu-local container (see docker/viame_gpu_local.docker in
    the VIAME repo).
  * The kit checkout mounted into the container. Add a second --volume when
    you launch the container, e.g.

      docker run --gpus=all --shm-size=8g \
          --volume "$HOME/code/VIAME:/host-viame" \
          --volume "$HOME/code/kwcoco_detector_kit:/host-kwcoco-detector-kit" \
          -it viame:viame-gpu-local bash

Usage (inside the container)
----------------------------
    source /opt/noaa/viame/setup_viame.sh
    bash /host-kwcoco-detector-kit/dev/viame_container_setup.sh

Override variables in your shell, never by editing this script:
    KCD_KIT_DIR        kit checkout inside the container  (default: auto-detect)
    KCD_ORT_PACKAGE    onnxruntime distribution           (default: onnxruntime-gpu)
    KCD_SYMLINK        1 = symlink plugin/template/predictor from the host into
                       site-packages for live iteration without a rebuild (default: 0)
'
set -eu

# --- resolve the kit checkout ------------------------------------------------
KCD_KIT_DIR="${KCD_KIT_DIR:-}"
if [ -z "$KCD_KIT_DIR" ]; then
    for cand in /host-kwcoco-detector-kit "$HOME/code/kwcoco_detector_kit" /viame/../kwcoco_detector_kit; do
        if [ -f "$cand/pyproject.toml" ]; then
            KCD_KIT_DIR="$cand"
            break
        fi
    done
fi
if [ -z "$KCD_KIT_DIR" ] || [ ! -f "$KCD_KIT_DIR/pyproject.toml" ]; then
    echo "ERROR: could not find the kit checkout. Mount it and/or set KCD_KIT_DIR." >&2
    echo "       e.g. --volume \"\$HOME/code/kwcoco_detector_kit:/host-kwcoco-detector-kit\"" >&2
    exit 1
fi

KCD_ORT_PACKAGE="${KCD_ORT_PACKAGE:-onnxruntime-gpu}"
KCD_SYMLINK="${KCD_SYMLINK:-0}"

PY="$(command -v python3 || command -v python)"
echo "[kcd-viame-setup] kit checkout : $KCD_KIT_DIR"
echo "[kcd-viame-setup] python       : $PY"
case "$PY" in
    /opt/noaa/viame/*) : ;;
    *) echo "[kcd-viame-setup] WARNING: python is not under /opt/noaa/viame — did you 'source setup_viame.sh'?" >&2 ;;
esac

# --- install runtime deps (torch-free) ---------------------------------------
# CRITICAL: never touch opencv here. fletch builds VIAME's own python cv2 and
# VIAME ships a fake opencv_python-4.9.0.80.dist-info so pip believes opencv is
# already installed (see VIAME cmake/custom_install_fletch.cmake). Installing a
# pip opencv-python* OVERWRITES fletch's cv2 wrapper with a version-mismatched
# one -> `import cv2` crashes (gapi circular import / missing __version__),
# which breaks loading of the ENTIRE viame.pytorch plugin package. And
# `pip uninstall opencv*` is just as bad: it deletes files shared with fletch's
# cv2. So we neither install nor uninstall opencv.
#
# VIAME's python already ships kwimage / kwcoco / scriptconfig / kwutil, so the
# only things actually missing are onnxruntime and the kit itself. We name the
# kw* packages so a bare VIAME without them still gets them; when already
# present pip is a no-op and pulls no opencv.
echo "[kcd-viame-setup] installing onnxruntime + kw* stack (no opencv) ..."
"$PY" -m pip install --no-input \
    "$KCD_ORT_PACKAGE" \
    kwimage kwcoco kwutil scriptconfig

# Install the kit WITHOUT deps so torch/torchvision (and opencv) are never pulled.
echo "[kcd-viame-setup] installing kwcoco_detector_kit (--no-deps, editable) ..."
"$PY" -m pip install --no-input --no-deps -e "$KCD_KIT_DIR"

# --- optional: live-iteration symlinks ---------------------------------------
# The plugin .py is already baked into the image by the build; symlinking only
# matters when you want host edits to take effect without rebuilding.
if [ "$KCD_SYMLINK" = "1" ]; then
    SITE_PT="/opt/noaa/viame/lib/python3.10/site-packages/viame/arrows/pytorch"
    if [ -d "$SITE_PT" ]; then
        echo "[kcd-viame-setup] symlinking plugin into $SITE_PT"
        ln -sf /host-viame/plugins/pytorch/kwcoco_detector_kit_detector.py \
               "$SITE_PT/kwcoco_detector_kit_detector.py"
    else
        echo "[kcd-viame-setup] WARNING: $SITE_PT not found; skipping symlink" >&2
    fi
fi

# --- verify ------------------------------------------------------------------
echo "[kcd-viame-setup] verifying torch-free ONNX import chain ..."
"$PY" - <<'PYEOF'
import onnxruntime as ort
from kwcoco_detector_kit.predictors.onnx import OnnxPredictor  # noqa: F401
print("  onnxruntime providers :", ort.get_available_providers())
print("  OnnxPredictor import  : OK")
try:
    from kwiver.vital.algo import ImageObjectDetector  # noqa: F401
    import sys, pathlib
    sys.path.insert(0, "/host-viame/plugins/pytorch")
    from kwcoco_detector_kit_detector import KwcocoDetectorKitDetector  # noqa: F401
    print("  VIAME plugin import   : OK")
except Exception as ex:
    print(f"  VIAME plugin import   : SKIP ({type(ex).__name__}: {ex})")
PYEOF

echo "[kcd-viame-setup] done."
