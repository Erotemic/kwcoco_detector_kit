"""Tests for export.modelspec — the .modelspec.json sidecar."""
from __future__ import annotations

import json
from pathlib import Path


def test_write_modelspec_round_trip(tmp_path):
    from kwcoco_detector_kit.export.modelspec import write_modelspec

    onnx_fpath = tmp_path / "foo.onnx"
    onnx_fpath.write_bytes(b"x")
    out = write_modelspec(
        onnx_fpath, input_hw=(320, 416), input_channels=3,
        variant="deimv2_hgnetv2_n", category_name="widget",
        candidate_kind="real",
        postprocess_score_thresh=0.25, postprocess_nms_iou_thresh=0.45,
    )
    payload = json.loads(out.read_text())
    assert payload["input"]["shape_hw"] == [320, 416]
    assert payload["postprocess"]["score_thresh"] == 0.25
    assert payload["postprocess"]["nms_iou_thresh"] == 0.45
    assert payload["meta"]["variant"] == "deimv2_hgnetv2_n"
    assert payload["meta"]["category_name"] == "widget"


def test_write_modelspec_model_id_default(tmp_path):
    from kwcoco_detector_kit.export.modelspec import write_modelspec

    onnx_fpath = tmp_path / "deimv2_h320_w320.onnx"
    onnx_fpath.write_bytes(b"x")
    out = write_modelspec(
        onnx_fpath, input_hw=(320, 320), variant="deimv2_hgnetv2_n",
    )
    payload = json.loads(out.read_text())
    # default model_id = "<variant>-h<H>w<W>"
    assert payload["modelId"] == "deimv2_hgnetv2_n-h320w320"
