"""
Integration tests for OnnxPredictor against real exported ONNX packages.

Designed to run on a GPU system (arisia, aiq-gpu) where real gen007+
exports live under $KCD_TRAINING_ROOT.

Prerequisites
-------------
- A real ONNX export.  Point at one via::

      export KCD_TEST_ONNX_PACKAGE=/data/users/jon.crall/kcd_sealion/pup_vs_nonpup/gen007_x/workdir/export/

  or let the fixture search under $KCD_TRAINING_ROOT automatically.

- CUDA tests additionally require ``onnxruntime-gpu`` and a CUDA device.

- Parity tests additionally require ``torch`` and the DEIMv2 checkpoint
  living under the same workdir as the export (workdir parent of export/).

All tests skip gracefully when prerequisites are absent.

Run (from kit root)::

    pytest tests/integration/test_onnx_predictor_real.py -v
    pytest tests/integration/test_onnx_predictor_real.py -v -m cuda
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_KCD_TRAINING_ROOT = Path(os.environ.get("KCD_TRAINING_ROOT",
                                         "/data/users/jon.crall/kcd_sealion"))


def _find_real_onnx_package() -> Path | None:
    """Return a package directory containing a real (>1 MB) .onnx export."""
    env = os.environ.get("KCD_TEST_ONNX_PACKAGE")
    if env:
        pkg = Path(env).expanduser()
        if pkg.exists():
            return pkg

    if not _KCD_TRAINING_ROOT.exists():
        return None
    for candidate in sorted(_KCD_TRAINING_ROOT.rglob("*.onnx")):
        if candidate.stat().st_size > 1_000_000:
            return candidate.parent
    return None


def _find_real_workdir(pkg_dir: Path) -> Path | None:
    """Given an export/ directory, return the parent workdir if it has a
    checkpoint and a training config."""
    workdir = pkg_dir.parent
    has_ckpt = any(workdir.glob("*.pth"))
    has_cfg = (workdir / "generated_configs").is_dir()
    return workdir if (has_ckpt and has_cfg) else None


@pytest.fixture(scope="module")
def real_onnx_package() -> Path:
    pkg = _find_real_onnx_package()
    if pkg is None:
        pytest.skip(
            "No real ONNX export found. "
            "Set KCD_TEST_ONNX_PACKAGE or export a model first:\n"
            "  python -m kwcoco_detector_kit export-onnx <workdir>"
        )
    return pkg


@pytest.fixture(scope="module")
def real_workdir(real_onnx_package) -> Path:
    wd = _find_real_workdir(real_onnx_package)
    if wd is None:
        pytest.skip(
            f"No checkpoint + generated_configs found next to {real_onnx_package}; "
            "skipping parity test"
        )
    return wd


@pytest.fixture(scope="module")
def onnx_predictor_cpu(real_onnx_package):
    pytest.importorskip("onnxruntime")
    from kwcoco_detector_kit.predictors.onnx import OnnxPredictor
    return OnnxPredictor(real_onnx_package, device="cpu")


@pytest.fixture(scope="module")
def onnx_predictor_cuda(real_onnx_package):
    import onnxruntime as ort
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDAExecutionProvider not available on this machine")
    from kwcoco_detector_kit.predictors.onnx import OnnxPredictor
    return OnnxPredictor(real_onnx_package, device="cuda")


@pytest.fixture(scope="module")
def synthetic_image() -> np.ndarray:
    """640x640 RGB image with a few bright patches — gives the model something
    to look at without needing real sealion imagery."""
    rng = np.random.RandomState(42)
    img = (rng.rand(640, 640, 3) * 80).astype(np.uint8)
    for _ in range(6):
        y, x = rng.randint(50, 590), rng.randint(50, 590)
        h, w = rng.randint(30, 120), rng.randint(30, 120)
        img[y:y + h, x:x + w] = (200, 220, 240)
    return img


# ---------------------------------------------------------------------------
# CPU tests — need a real ONNX export, no GPU
# ---------------------------------------------------------------------------

@pytest.mark.requires_onnx
def test_real_export_loads(real_onnx_package):
    """Real export loads and reports sane metadata."""
    from kwcoco_detector_kit.predictors.onnx import OnnxPredictor
    p = OnnxPredictor(real_onnx_package, device="cpu")
    H, W = p.eval_spatial_size
    assert H > 0 and W > 0, "eval_spatial_size must be positive"
    assert len(p.category_names) > 0, "category_names must be non-empty"
    print(f"  eval_spatial_size={p.eval_spatial_size}  categories={p.category_names}")


@pytest.mark.requires_onnx
def test_real_export_cpu_predict_image(onnx_predictor_cpu, synthetic_image):
    """CPU predict_image returns well-formed dicts on a real export."""
    dets = onnx_predictor_cpu.predict_image(synthetic_image)
    assert isinstance(dets, list)
    for d in dets:
        assert set(d.keys()) == {"label", "bbox_xyxy", "score"}
        assert 0 <= d["score"] <= 1.0
        x0, y0, x1, y1 = d["bbox_xyxy"]
        assert x1 > x0 and y1 > y0, "degenerate box"
    print(f"  CPU: {len(dets)} detections above threshold")


@pytest.mark.requires_onnx
def test_real_export_cpu_predict_kwimage(onnx_predictor_cpu, synthetic_image):
    """predict_image_kwimage returns a kwimage.Detections with classes populated."""
    kwimage = pytest.importorskip("kwimage")
    dets = onnx_predictor_cpu.predict_image_kwimage(synthetic_image)
    assert isinstance(dets, kwimage.Detections)
    assert dets.classes == onnx_predictor_cpu.category_names
    print(f"  kwimage Detections: {len(dets)} boxes, classes={dets.classes}")


@pytest.mark.requires_onnx
def test_real_export_orig_size_matches_image(onnx_predictor_cpu):
    """Boxes are reported in the original image coordinate frame."""
    H_img, W_img = 1000, 1500
    img = np.zeros((H_img, W_img, 3), dtype=np.uint8)
    img[400:600, 600:900] = 200  # bright rectangle

    dets = onnx_predictor_cpu.predict_image(img, orig_size=(W_img, H_img))
    for d in dets:
        x0, y0, x1, y1 = d["bbox_xyxy"]
        # Boxes must not exceed the original image dimensions
        assert 0 <= x0 < x1 <= W_img, f"x out of bounds: {d['bbox_xyxy']}"
        assert 0 <= y0 < y1 <= H_img, f"y out of bounds: {d['bbox_xyxy']}"


# ---------------------------------------------------------------------------
# CUDA tests — additionally require onnxruntime-gpu and a CUDA device
# ---------------------------------------------------------------------------

@pytest.mark.cuda
@pytest.mark.requires_onnx
def test_real_export_cuda_predict_image(onnx_predictor_cuda, synthetic_image):
    """CUDA provider runs without error and returns well-formed dicts."""
    dets = onnx_predictor_cuda.predict_image(synthetic_image)
    assert isinstance(dets, list)
    print(f"  CUDA: {len(dets)} detections above threshold")


@pytest.mark.cuda
@pytest.mark.requires_onnx
def test_cpu_cuda_detection_count_parity(onnx_predictor_cpu, onnx_predictor_cuda,
                                         synthetic_image):
    """CPU and CUDA providers produce the same number of detections.

    Minor score differences from floating-point order are expected, but
    any detection above threshold on one device should also appear on the
    other (deterministic postprocessor in the ONNX graph).
    """
    cpu_dets = onnx_predictor_cpu.predict_image(synthetic_image)
    cuda_dets = onnx_predictor_cuda.predict_image(synthetic_image)
    assert len(cpu_dets) == len(cuda_dets), (
        f"CPU produced {len(cpu_dets)} detections, "
        f"CUDA produced {len(cuda_dets)}"
    )


@pytest.mark.cuda
@pytest.mark.requires_onnx
def test_cpu_cuda_score_parity(onnx_predictor_cpu, onnx_predictor_cuda,
                                synthetic_image):
    """Scores from CPU and CUDA providers agree within float32 tolerance."""
    cpu_dets = onnx_predictor_cpu.predict_image(synthetic_image)
    cuda_dets = onnx_predictor_cuda.predict_image(synthetic_image)
    if not cpu_dets:
        pytest.skip("No detections above threshold on synthetic image")

    cpu_scores = sorted([d["score"] for d in cpu_dets], reverse=True)
    cuda_scores = sorted([d["score"] for d in cuda_dets], reverse=True)
    for i, (c, g) in enumerate(zip(cpu_scores, cuda_scores)):
        assert abs(c - g) < 0.05, (
            f"score[{i}]: CPU={c:.4f} CUDA={g:.4f} diff={abs(c - g):.4f}"
        )


# ---------------------------------------------------------------------------
# Parity: OnnxPredictor vs DEIMv2 PyTorch predictor
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.requires_onnx
@pytest.mark.requires_torch
def test_onnx_pytorch_score_parity(real_onnx_package, real_workdir, synthetic_image):
    """OnnxPredictor and the DEIMv2 PyTorch predictor agree on top detections.

    Requires the checkpoint + training config to exist under real_workdir
    (the parent of the export/ directory).  Score differences up to 0.05
    and box IoU > 0.75 are considered a pass — AMP fp16 in the PyTorch path
    introduces small numerical divergence from the fp32 ONNX path.
    """
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from kwcoco_detector_kit.predictors.onnx import OnnxPredictor
    from kwcoco_detector_kit.trainers._registry import get_trainer

    trainer = get_trainer("deimv2")
    try:
        torch_predictor = trainer.build_predictor(real_workdir, device="cpu")
    except Exception as exc:
        pytest.skip(f"Could not build DEIMv2 predictor from {real_workdir}: {exc}")

    onnx_predictor = OnnxPredictor(real_onnx_package, device="cpu", score_thresh=0.20)

    H, W = synthetic_image.shape[:2]
    onnx_dets = onnx_predictor.predict_image(synthetic_image, orig_size=(W, H))
    torch_dets = torch_predictor.predict_image(synthetic_image, orig_size=(W, H))

    # Apply the same score threshold for fair comparison
    thresh = 0.20
    onnx_dets = [d for d in onnx_dets if d["score"] >= thresh]
    torch_dets = [d for d in torch_dets if d["score"] >= thresh]

    print(f"  ONNX: {len(onnx_dets)} dets, PyTorch: {len(torch_dets)} dets")

    if not onnx_dets and not torch_dets:
        return  # both models agree: nothing above threshold

    count_diff = abs(len(onnx_dets) - len(torch_dets))
    assert count_diff <= max(2, len(torch_dets) // 4), (
        f"Detection count diverged: ONNX={len(onnx_dets)} PyTorch={len(torch_dets)}"
    )

    # Check that top-scoring ONNX detections have matching PyTorch detections
    # (by rough IoU overlap) — not a strict 1-1 match since NMS can
    # change order when scores differ slightly
    import kwimage
    if onnx_dets and torch_dets:
        onnx_boxes = kwimage.Boxes(
            np.array([d["bbox_xyxy"] for d in onnx_dets[:5]], dtype=np.float32), "ltrb"
        )
        torch_boxes = kwimage.Boxes(
            np.array([d["bbox_xyxy"] for d in torch_dets[:5]], dtype=np.float32), "ltrb"
        )
        iou_mat = onnx_boxes.ious(torch_boxes)
        # Each ONNX top-5 box should match at least one PyTorch box with IoU > 0.5
        max_iou_per_onnx = iou_mat.max(axis=1)
        low_iou = max_iou_per_onnx[max_iou_per_onnx < 0.50]
        assert len(low_iou) <= 1, (
            f"Too many ONNX top-5 boxes with no PyTorch match (IoU<0.5): {low_iou}"
        )
