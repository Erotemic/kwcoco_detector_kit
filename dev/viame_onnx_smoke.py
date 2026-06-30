#!/usr/bin/env python3
"""
Smoke-test for the OnnxPredictor and (optionally) the VIAME plugin.

Run on namek (or any machine with the ONNX export visible):

    python dev/viame_onnx_smoke.py \
        --package /data/users/jon.crall/kcd_sealion/workdirs/\
pup_vs_nonpup_deimv2_dinov3_x_2gpu_aiq_gen006_1280/export \
        --image /path/to/test_image.jpg

If --image is omitted, a synthetic 1280x1280 RGB image is used.
The --viame flag additionally tests the kwiver plugin (requires VIAME Python
path to be active).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np


def run_onnx_predictor(package: Path, image_np: np.ndarray) -> None:
    from kwcoco_detector_kit.predictors.onnx import OnnxPredictor

    print(f"\n[OnnxPredictor] loading from {package}")
    t0 = time.perf_counter()
    pred = OnnxPredictor(package, device="cpu", score_thresh=0.01)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  eval_spatial_size : {pred.eval_spatial_size}")
    print(f"  category_names    : {pred.category_names}")
    print(f"  load time         : {load_ms:.0f} ms")

    print(f"\n[OnnxPredictor] running on {image_np.shape} image ...")
    t0 = time.perf_counter()
    dets = pred.predict_image_kwimage(image_np)
    infer_ms = (time.perf_counter() - t0) * 1000
    print(f"  inference time    : {infer_ms:.0f} ms")
    print(f"  raw detections    : {len(dets)}")

    if len(dets) > 0:
        # Show top-5 by score
        import kwimage
        order = np.argsort(dets.scores)[::-1][:5]
        top = dets.take(order)
        for i, (box, score, cidx) in enumerate(
            zip(top.boxes.to_ltrb().data, top.scores, top.class_idxs)
        ):
            ci = int(cidx)
            names = pred.category_names or []
            # An undecodable index (e.g. a stale ['widget'] modelspec on a
            # multi-class model) must not abort the smoke test — fall back to
            # the raw integer and flag it.
            if 0 <= ci < len(names):
                name = names[ci]
            else:
                name = f"<cidx={ci} undecodable>"
            x0, y0, x1, y1 = box
            print(f"  [{i}] {name:20s}  score={score:.3f}  "
                  f"box=[{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}]")
    else:
        print("  (no detections above score_thresh=0.01 — model/image mismatch?)")

    # Check labels.txt sidecar
    package_p = Path(package)
    if package_p.is_dir():
        root = package_p
    else:
        root = package_p.parent
    labels_files = list(root.rglob("*.labels.txt"))
    if labels_files:
        print(f"\n[labels.txt] {labels_files[0]}:")
        print("  " + labels_files[0].read_text().strip().replace("\n", "\n  "))
    else:
        print("\n[labels.txt] NOT FOUND — re-export with current kit to generate it")


def run_viame_plugin(package: Path, image_np: np.ndarray) -> None:
    try:
        from kwiver.vital.algo import ImageObjectDetector  # noqa: F401
    except ImportError:
        print("\n[VIAME plugin] kwiver not importable — skipping VIAME test")
        print("  (source a VIAME setup_paths.sh and rerun with --viame)")
        return

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "VIAME" / "plugins" / "pytorch"))
    try:
        from kwcoco_detector_kit_detector import KwcocoDetectorKitDetector
    except ImportError as ex:
        print(f"\n[VIAME plugin] cannot import kwcoco_detector_kit_detector: {ex}")
        return

    from kwiver.vital.types import Image, ImageContainer

    print(f"\n[VIAME plugin] loading KwcocoDetectorKitDetector from {package}")
    det = KwcocoDetectorKitDetector()
    det.set_configuration({"package": str(package), "device": "cpu", "score_thresh": "0.01"})

    image_container = ImageContainer(Image(image_np))
    t0 = time.perf_counter()
    result = det.detect(image_container)
    infer_ms = (time.perf_counter() - t0) * 1000
    print(f"  VIAME detections  : {len(result)} in {infer_ms:.0f} ms")
    print("  [VIAME plugin] PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", required=True,
                    help="Path to exported ONNX package directory or .zip")
    ap.add_argument("--image", default=None,
                    help="Test image path (defaults to synthetic 1280x1280 RGB)")
    ap.add_argument("--viame", action="store_true",
                    help="Also test via the VIAME kwiver plugin (needs kwiver in path)")
    args = ap.parse_args()

    package = Path(args.package).expanduser()
    if not package.exists():
        print(f"ERROR: package path does not exist: {package}", file=sys.stderr)
        sys.exit(1)

    if args.image:
        import kwimage
        image_np = kwimage.imread(args.image, space="rgb")
    else:
        rng = np.random.RandomState(42)
        image_np = (rng.rand(1280, 1280, 3) * 255).astype(np.uint8)
        print("[smoke] using synthetic 1280×1280 RGB image (no real objects expected)")

    run_onnx_predictor(package, image_np)

    if args.viame:
        run_viame_plugin(package, image_np)

    print("\n[smoke] done")


if __name__ == "__main__":
    main()
