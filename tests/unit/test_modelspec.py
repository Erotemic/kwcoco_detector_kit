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
        variant="deimv2_hgnetv2_n", category_names=["widget"],
        candidate_kind="real",
        postprocess_score_thresh=0.25, postprocess_nms_iou_thresh=0.45,
    )
    payload = json.loads(out.read_text())
    assert payload["input"]["shape_hw"] == [320, 416]
    assert payload["postprocess"]["score_thresh"] == 0.25
    assert payload["postprocess"]["nms_iou_thresh"] == 0.45
    assert payload["meta"]["variant"] == "deimv2_hgnetv2_n"
    assert payload["meta"]["category_names"] == ["widget"]


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


def test_write_modelspec_provenance_and_imputed(tmp_path):
    """provenance + imputed blocks and category_names_source are recorded,
    and has_imputed_metadata flags downstream distrust."""
    from kwcoco_detector_kit.export.modelspec import write_modelspec

    onnx_fpath = tmp_path / "m.onnx"
    onnx_fpath.write_bytes(b"x")
    out = write_modelspec(
        onnx_fpath, input_hw=(640, 640), variant="deimv2_dinov3_x",
        category_names=["pup", "nonpup_sealion"],
        category_names_source="imputed:class_schemes.yaml:pup_vs_nonpup",
        source_checkpoint={"name": "best_stg2.pth", "sha256": "abc", "size_bytes": 7},
        provenance={"kit_sha": "deadbeef", "kit_dirty": False},
        imputed={"category_names": "derived from scheme target_order"},
    )
    payload = json.loads(out.read_text())
    assert payload["meta"]["category_names_source"] == "imputed:class_schemes.yaml:pup_vs_nonpup"
    assert payload["meta"]["has_imputed_metadata"] is True
    assert payload["provenance"]["kit_sha"] == "deadbeef"
    assert payload["provenance"]["source_checkpoint"]["sha256"] == "abc"
    assert "category_names" in payload["imputed"]
    # labels.txt sidecar written, one class per line, 0-indexed
    labels = onnx_fpath.with_suffix(".labels.txt").read_text().split()
    assert labels == ["pup", "nonpup_sealion"]


def test_write_modelspec_clean_has_no_imputed_block(tmp_path):
    """A clean, fully-specified export carries no 'imputed' block and flags
    has_imputed_metadata=False."""
    from kwcoco_detector_kit.export.modelspec import write_modelspec

    onnx_fpath = tmp_path / "m.onnx"
    onnx_fpath.write_bytes(b"x")
    out = write_modelspec(
        onnx_fpath, input_hw=(640, 640), variant="deimv2_dinov3_x",
        category_names=["pup", "nonpup_sealion"],
        category_names_source="policy.json",
    )
    payload = json.loads(out.read_text())
    assert payload["meta"]["has_imputed_metadata"] is False
    assert "imputed" not in payload
