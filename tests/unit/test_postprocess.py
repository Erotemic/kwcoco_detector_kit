"""Unit tests for kwcoco_detector_kit.data.postprocess."""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# apply_box_filters
# ---------------------------------------------------------------------------

@pytest.mark.requires_torch
def test_apply_box_filters_score_threshold():
    from kwcoco_detector_kit.data.postprocess import apply_box_filters

    records = [
        {"bbox_xyxy": [0, 0, 10, 10], "score": 0.9},
        {"bbox_xyxy": [5, 5, 15, 15], "score": 0.1},
    ]
    kept = apply_box_filters(records, score_thresh=0.5, nms_thresh=0.0)
    assert len(kept) == 1
    assert kept[0]["score"] == 0.9


@pytest.mark.requires_torch
def test_apply_box_filters_nms():
    from kwcoco_detector_kit.data.postprocess import apply_box_filters

    # Two heavily overlapping boxes with different scores.
    records = [
        {"bbox_xyxy": [0, 0, 100, 100], "score": 0.9},
        {"bbox_xyxy": [5, 5, 95, 95], "score": 0.8},
    ]
    kept = apply_box_filters(records, score_thresh=0.5, nms_thresh=0.5)
    assert len(kept) == 1


@pytest.mark.requires_torch
def test_apply_box_filters_empty():
    from kwcoco_detector_kit.data.postprocess import apply_box_filters

    assert apply_box_filters([], score_thresh=0.5, nms_thresh=0.5) == []


# ---------------------------------------------------------------------------
# detector_records_to_bbox_anns
# ---------------------------------------------------------------------------

@pytest.mark.requires_torch
def test_detector_records_to_bbox_anns_basic():
    from kwcoco_detector_kit.data.postprocess import detector_records_to_bbox_anns

    records = [{"bbox_xyxy": [10.0, 20.0, 50.0, 60.0], "score": 0.8, "label": 0}]
    post_cfg = {"score_thresh": 0.5, "nms_thresh": 0.0}
    anns = detector_records_to_bbox_anns(records, post_cfg, label_mapping={0: "widget"})
    assert len(anns) == 1
    ann = anns[0]
    assert ann["category_name"] == "widget"
    assert ann["bbox"] == [10.0, 20.0, 40.0, 40.0]  # xywh
    assert ann["score"] == 0.8


@pytest.mark.requires_torch
def test_detector_records_to_bbox_anns_no_label_mapping():
    from kwcoco_detector_kit.data.postprocess import detector_records_to_bbox_anns

    records = [{"bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "score": 0.9, "label": 3}]
    post_cfg = {"score_thresh": 0.5, "nms_thresh": 0.0}
    anns = detector_records_to_bbox_anns(records, post_cfg)
    assert anns[0]["category_name"] == "3"


# ---------------------------------------------------------------------------
# add_prediction_annotations
# ---------------------------------------------------------------------------

@pytest.mark.requires_torch
def test_add_prediction_annotations_creates_categories():
    import kwcoco
    from kwcoco_detector_kit.data.postprocess import add_prediction_annotations

    pred = kwcoco.CocoDataset()
    gid = pred.add_image(file_name="img.jpg")
    anns = [
        {"category_name": "widget", "bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.9},
        {"category_name": "widget", "bbox": [20.0, 20.0, 5.0, 5.0], "score": 0.7},
    ]
    add_prediction_annotations(pred, gid, anns, backend_name="mock")
    assert pred.n_annots == 2
    assert pred.n_cats == 1
    assert list(pred.cats.values())[0]["name"] == "widget"


@pytest.mark.requires_torch
def test_add_prediction_annotations_multi_category():
    import kwcoco
    from kwcoco_detector_kit.data.postprocess import add_prediction_annotations

    pred = kwcoco.CocoDataset()
    gid = pred.add_image(file_name="img.jpg")
    anns = [
        {"category_name": "cat", "bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.9},
        {"category_name": "dog", "bbox": [20.0, 20.0, 5.0, 5.0], "score": 0.7},
    ]
    add_prediction_annotations(pred, gid, anns, backend_name="mock")
    assert pred.n_cats == 2


@pytest.mark.requires_torch
def test_add_prediction_annotations_sets_backend_tag():
    import kwcoco
    from kwcoco_detector_kit.data.postprocess import add_prediction_annotations

    pred = kwcoco.CocoDataset()
    gid = pred.add_image(file_name="img.jpg")
    add_prediction_annotations(pred, gid, [
        {"category_name": "x", "bbox": [0.0, 0.0, 5.0, 5.0], "score": 0.8},
    ], backend_name="deimv2")
    ann = list(pred.anns.values())[0]
    assert ann["foundation_backend"] == "deimv2"
    assert ann["role"] == "prediction"


# ---------------------------------------------------------------------------
# predict_kwcoco image-directory coercion
# ---------------------------------------------------------------------------

@pytest.mark.requires_torch
def test_coerce_src_from_image_dir(tmp_path):
    """_coerce_src_kwcoco should build a kwcoco dataset from a plain image dir."""
    import kwimage
    from kwcoco_detector_kit.predict import _coerce_src_kwcoco

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(3):
        kwimage.imwrite(str(img_dir / f"frame_{i:02d}.jpg"),
                        np.zeros((32, 32, 3), dtype=np.uint8))

    dset = _coerce_src_kwcoco(img_dir)
    assert dset.n_images == 3
