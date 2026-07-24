#!/usr/bin/env bash
#
# Install the runtime deps the vendored kwcoco_detector_kit ONNX detector needs
# inside a *built* VIAME container.
#
# Run this INSIDE the running viame-gpu-local container, AFTER sourcing
# setup_viame.sh so that `python`/`pip` resolve to VIAME's internal Python.
#
__doc__='
Why this script exists
----------------------
The VIAME plugin (plugins/pytorch/kwcoco_detector_kit_detector.py) and the
OnnxPredictor it runs are BOTH baked into the image by the normal CMake build.
The predictor is VENDORED into VIAME (plugins/pytorch/
kwcoco_detector_kit_onnx_predictor.py), so VIAME inference does NOT depend on
kwcoco_detector_kit being installed -- you do not need the kit repo here at all.

What is still missing from a stock VIAME python are two small, torch-free
packages the plugin imports at runtime:

    import onnxruntime                         # the inference engine
    import kwconf                              # the plugin config (KwcocoDetectorKitConfig)

(kwimage / numpy, which the predictor also uses, already ship with VIAME.)
This script installs those two. No kit checkout, no editable install, no symlinks.

Prerequisites
-------------
  * A built viame-gpu-local container whose VIAME was built from a tree that
    includes the vendored predictor (resync with the kit''s
    dev/vendor_onnx_to_viame.py before building if needed).

Usage (inside the container)
----------------------------
    source /opt/noaa/viame/setup_viame.sh
    bash viame_container_setup.sh          # (copy this script in, or mount it)

Override variables in your shell, never by editing this script:
    KCD_ORT_PACKAGE    onnxruntime distribution           (default: onnxruntime-gpu)
'
set -eu

KCD_ORT_PACKAGE="${KCD_ORT_PACKAGE:-onnxruntime-gpu}"

PY="$(command -v python3 || command -v python)"
echo "[kcd-viame-setup] python : $PY"
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
# We name kwimage too so a bare VIAME without it still gets it; when already
# present pip is a no-op and pulls no opencv.
echo "[kcd-viame-setup] installing onnxruntime + kwconf (+ kwimage) (no opencv) ..."
"$PY" -m pip install --no-input \
    "$KCD_ORT_PACKAGE" \
    kwconf kwimage

# --- verify ------------------------------------------------------------------
echo "[kcd-viame-setup] verifying the vendored torch-free ONNX import chain ..."
"$PY" - <<'PYEOF'
import onnxruntime as ort
print("  onnxruntime providers :", ort.get_available_providers())
try:
    from viame.pytorch.kwcoco_detector_kit_onnx_predictor import (  # noqa: F401
        OnnxPredictor, __vendored_provenance__,
    )
    print("  vendored OnnxPredictor : OK "
          f"(kit_git_sha={__vendored_provenance__.get('kit_git_sha', '?')[:12]})")
except Exception as ex:
    print(f"  vendored OnnxPredictor : FAIL ({type(ex).__name__}: {ex})")
    print("    -> rebuild VIAME after vendoring, or check the build installed "
          "kwcoco_detector_kit_onnx_predictor into viame.pytorch")
try:
    from kwiver.vital.algo import ImageObjectDetector  # noqa: F401
    from viame.pytorch.kwcoco_detector_kit_detector import (  # noqa: F401
        KwcocoDetectorKitDetector,
    )
    print("  VIAME plugin import   : OK")
except Exception as ex:
    print(f"  VIAME plugin import   : SKIP ({type(ex).__name__}: {ex})")
PYEOF

echo "[kcd-viame-setup] done."
