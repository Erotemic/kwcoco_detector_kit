"""
Tests for predictors.onnx.OnnxPredictor.

All tests that need onnxruntime are marked requires_onnx. The
no-torch isolation test runs via subprocess so it cannot be
contaminated by prior torch imports within the session.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_onnx(fpath: Path, *, H: int = 64, W: int = 64, K: int = 2,
                    score: float = 0.9) -> None:
    """Build a tiny constant-output ONNX that mirrors the DEIMv2 deploy format.

    Inputs: images (1,3,H,W) float32, orig_target_sizes (1,2) int64
    Outputs: labels (1,K) int64, boxes (1,K,4) float32, scores (1,K) float32
    """
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper, TensorProto

    labels_val = np.zeros((1, K), dtype=np.int64)
    boxes_val = np.array(
        [[[10.0, 10.0, 50.0, 50.0]] * K], dtype=np.float32
    )
    scores_val = np.full((1, K), fill_value=score, dtype=np.float32)

    nodes = [
        helper.make_node("Constant", [], ["labels"],
                         value=numpy_helper.from_array(labels_val)),
        helper.make_node("Constant", [], ["boxes"],
                         value=numpy_helper.from_array(boxes_val)),
        helper.make_node("Constant", [], ["scores"],
                         value=numpy_helper.from_array(scores_val)),
    ]
    graph = helper.make_graph(
        nodes, "mock_det",
        inputs=[
            helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, H, W]),
            helper.make_tensor_value_info("orig_target_sizes", TensorProto.INT64, [1, 2]),
        ],
        outputs=[
            helper.make_tensor_value_info("labels", TensorProto.INT64, [1, K]),
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, K, 4]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, K]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save(model, str(fpath))


def _make_mock_package(pkg_dir: Path, *, H: int = 64, W: int = 64,
                       category_names=("pup", "nonpup_sealion"),
                       score_thresh: float = 0.50) -> Path:
    """Write a mock ONNX + modelspec.json into pkg_dir; return onnx_fpath."""
    pkg_dir.mkdir(parents=True, exist_ok=True)
    onnx_fpath = pkg_dir / f"mock_h{H}_w{W}.onnx"
    _make_mock_onnx(onnx_fpath, H=H, W=W)

    spec = {
        "modelId": "mock-h64w64",
        "input": {"shape_hw": [H, W], "channels": 3, "dtype": "float32", "layout": "NCHW"},
        "preprocess": {"scale": 1.0 / 255.0,
                       "normalize_mean": [0.0, 0.0, 0.0],
                       "normalize_std": [1.0, 1.0, 1.0]},
        "postprocess": {"score_thresh": score_thresh, "nms_iou_thresh": 0.50, "topk": 100},
        "meta": {"variant": "mock", "category_names": list(category_names)},
    }
    (pkg_dir / f"mock_h{H}_w{W}.modelspec.json").write_text(json.dumps(spec, indent=2))
    return onnx_fpath


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_onnx
def test_onnx_predictor_loads(tmp_path):
    """OnnxPredictor loads from a directory package and exposes correct properties."""
    pytest.importorskip("onnxruntime")
    from kwcoco_detector_kit.predictors.onnx import OnnxPredictor

    _make_mock_package(tmp_path, H=64, W=64, category_names=["pup", "nonpup_sealion"])
    p = OnnxPredictor(tmp_path, score_thresh=0.50)

    assert p.eval_spatial_size == (64, 64)
    assert p.category_names == ["pup", "nonpup_sealion"]


@pytest.mark.requires_onnx
def test_onnx_predictor_predict_image_returns_dicts(tmp_path):
    """predict_image returns correctly shaped detection dicts."""
    pytest.importorskip("onnxruntime")
    from kwcoco_detector_kit.predictors.onnx import OnnxPredictor

    _make_mock_package(tmp_path, H=64, W=64, score_thresh=0.50)
    p = OnnxPredictor(tmp_path, score_thresh=0.50)

    img = np.zeros((128, 128, 3), dtype=np.uint8)
    dets = p.predict_image(img)

    assert isinstance(dets, list)
    for d in dets:
        assert set(d.keys()) == {"label", "bbox_xyxy", "score"}
        assert isinstance(d["label"], int)
        assert isinstance(d["score"], float)
        assert len(d["bbox_xyxy"]) == 4


@pytest.mark.requires_onnx
def test_onnx_predictor_score_thresh_filters(tmp_path):
    """Detections below score_thresh are dropped."""
    pytest.importorskip("onnxruntime")
    from kwcoco_detector_kit.predictors.onnx import OnnxPredictor

    # Mock returns score=0.9; threshold above that → no detections
    _make_mock_package(tmp_path, H=64, W=64, score_thresh=0.50)
    p_pass = OnnxPredictor(tmp_path, score_thresh=0.50)
    p_fail = OnnxPredictor(tmp_path, score_thresh=0.95)

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    assert len(p_pass.predict_image(img)) > 0
    assert len(p_fail.predict_image(img)) == 0


@pytest.mark.requires_onnx
def test_onnx_predictor_predict_image_kwimage(tmp_path):
    """predict_image_kwimage returns a kwimage.Detections with correct attributes."""
    pytest.importorskip("onnxruntime")
    kwimage = pytest.importorskip("kwimage")
    from kwcoco_detector_kit.predictors.onnx import OnnxPredictor

    _make_mock_package(tmp_path, H=64, W=64, category_names=["pup", "nonpup_sealion"],
                       score_thresh=0.50)
    p = OnnxPredictor(tmp_path, score_thresh=0.50)

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    dets = p.predict_image_kwimage(img)

    assert isinstance(dets, kwimage.Detections)
    assert isinstance(dets.boxes, kwimage.Boxes)
    assert dets.boxes.format == "ltrb"
    assert dets.scores is not None
    assert dets.class_idxs is not None
    assert dets.classes == ["pup", "nonpup_sealion"]


@pytest.mark.requires_onnx
def test_onnx_predictor_orig_size_inference(tmp_path):
    """orig_size defaults correctly to image dimensions."""
    pytest.importorskip("onnxruntime")
    from kwcoco_detector_kit.predictors.onnx import OnnxPredictor

    _make_mock_package(tmp_path, H=64, W=64, score_thresh=0.50)
    p = OnnxPredictor(tmp_path, score_thresh=0.50)

    # 200x300 image with no explicit orig_size — should not crash
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    dets = p.predict_image(img)
    assert isinstance(dets, list)


@pytest.mark.requires_onnx
def test_onnx_predictor_handles_grayscale(tmp_path):
    """2-D grayscale array is accepted and expanded to 3-channel internally."""
    pytest.importorskip("onnxruntime")
    from kwcoco_detector_kit.predictors.onnx import OnnxPredictor

    _make_mock_package(tmp_path, H=64, W=64, score_thresh=0.50)
    p = OnnxPredictor(tmp_path, score_thresh=0.50)

    gray = np.zeros((64, 64), dtype=np.uint8)
    dets = p.predict_image(gray)
    assert isinstance(dets, list)


@pytest.mark.requires_onnx
def test_onnx_predictor_zip_package(tmp_path):
    """OnnxPredictor loads from a .zip archive containing the ONNX + sidecar."""
    pytest.importorskip("onnxruntime")
    import zipfile
    from kwcoco_detector_kit.predictors.onnx import OnnxPredictor

    pkg_dir = tmp_path / "pkg"
    _make_mock_package(pkg_dir, H=64, W=64, score_thresh=0.50)

    zip_path = tmp_path / "package.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in pkg_dir.iterdir():
            zf.write(f, f.name)

    p = OnnxPredictor(zip_path, score_thresh=0.50)
    assert p.eval_spatial_size == (64, 64)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    assert isinstance(p.predict_image(img), list)


@pytest.mark.requires_onnx
def test_onnx_predictor_satisfies_protocol(tmp_path):
    """OnnxPredictor is recognised as a DetectorPredictor at runtime."""
    pytest.importorskip("onnxruntime")
    from kwcoco_detector_kit.predictors._interface import DetectorPredictor
    from kwcoco_detector_kit.predictors.onnx import OnnxPredictor

    _make_mock_package(tmp_path, H=64, W=64, score_thresh=0.50)
    p = OnnxPredictor(tmp_path, score_thresh=0.50)
    assert isinstance(p, DetectorPredictor)


@pytest.mark.requires_onnx
def test_onnx_predictor_no_torch(tmp_path):
    """OnnxPredictor must not import torch — verified in a clean subprocess."""
    pytest.importorskip("onnxruntime")

    pkg_dir = tmp_path / "pkg"
    _make_mock_package(pkg_dir, H=64, W=64, score_thresh=0.50)

    script = dedent(f"""
        import sys
        # Guarantee torch is not pre-loaded.
        assert 'torch' not in sys.modules, "torch pre-loaded before test"

        from kwcoco_detector_kit.predictors.onnx import OnnxPredictor
        import numpy as np

        p = OnnxPredictor({str(pkg_dir)!r}, score_thresh=0.5)
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        p.predict_image(img)
        p.predict_image_kwimage(img)

        torch_mods = [m for m in sys.modules if m == 'torch' or m.startswith('torch.')]
        assert not torch_mods, f"torch was imported: {{torch_mods}}"
        print("PASS")
    """)

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"no-torch subprocess failed (rc={proc.returncode}):\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "PASS" in proc.stdout
