"""Tests for data.coco_export — kwcoco -> MSCOCO json."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_export_mscoco_single_class_round_trip(synthetic_kwcoco, tmp_path):
    from kwcoco_detector_kit.data.coco_export import export_mscoco

    dst = tmp_path / "out.mscoco.json"
    export_mscoco(
        synthetic_kwcoco, dst, category_names=["widget"],
        include_segmentations=False,
    )
    assert dst.exists()
    payload = json.loads(dst.read_text())
    assert payload["categories"] == [
        {"id": 0, "name": "widget", "supercategory": "widget"},
    ]
    assert len(payload["images"]) > 0
    for img in payload["images"]:
        assert Path(img["file_name"]).is_absolute()
    for ann in payload["annotations"]:
        assert ann["category_id"] == 0


def test_export_mscoco_drops_unrequested_categories(synthetic_kwcoco_factory, tmp_path):
    src = synthetic_kwcoco_factory(
        "mixed", num_images=4, boxes_per_image=2,
        category_names=("widget", "gizmo"),
    )
    from kwcoco_detector_kit.data.coco_export import export_mscoco
    dst = tmp_path / "only_widget.mscoco.json"
    export_mscoco(src, dst, category_names=["widget"], include_segmentations=False)
    payload = json.loads(dst.read_text())
    assert [c["name"] for c in payload["categories"]] == ["widget"]
    assert all(a["category_id"] == 0 for a in payload["annotations"])
    assert payload["annotations"], "should keep at least the widget annotations"


def test_export_mscoco_multi_class_assigns_ids_in_order(synthetic_kwcoco_factory, tmp_path):
    src = synthetic_kwcoco_factory(
        "two_class", num_images=4, boxes_per_image=2,
        category_names=("widget", "gizmo"),
    )
    from kwcoco_detector_kit.data.coco_export import export_mscoco
    dst = tmp_path / "both.mscoco.json"
    # Reverse order: gizmo should get id=0, widget should get id=1.
    export_mscoco(src, dst, category_names=["gizmo", "widget"], include_segmentations=False)
    payload = json.loads(dst.read_text())
    assert payload["categories"] == [
        {"id": 0, "name": "gizmo", "supercategory": "gizmo"},
        {"id": 1, "name": "widget", "supercategory": "widget"},
    ]
    cat_ids_seen = {a["category_id"] for a in payload["annotations"]}
    assert cat_ids_seen == {0, 1}, "both categories should have annotations"


def test_export_mscoco_category_id_start_offset(synthetic_kwcoco_factory, tmp_path):
    src = synthetic_kwcoco_factory(
        "offset", num_images=2, category_names=("widget", "gizmo"),
    )
    from kwcoco_detector_kit.data.coco_export import export_mscoco
    dst = tmp_path / "offset.mscoco.json"
    export_mscoco(
        src, dst, category_names=["widget", "gizmo"],
        include_segmentations=False, category_id_start=1,
    )
    payload = json.loads(dst.read_text())
    assert [c["id"] for c in payload["categories"]] == [1, 2]
    assert {a["category_id"] for a in payload["annotations"]} <= {1, 2}


def test_export_mscoco_rejects_bare_string(synthetic_kwcoco, tmp_path):
    from kwcoco_detector_kit.data.coco_export import export_mscoco
    with pytest.raises(TypeError):
        export_mscoco(
            synthetic_kwcoco, tmp_path / "x.json",
            category_names="widget",  # type: ignore[arg-type]
        )


def test_export_mscoco_rejects_empty_list(synthetic_kwcoco, tmp_path):
    from kwcoco_detector_kit.data.coco_export import export_mscoco
    with pytest.raises(ValueError):
        export_mscoco(synthetic_kwcoco, tmp_path / "x.json", category_names=[])


def test_export_training_splits(synthetic_kwcoco_factory, tmp_path):
    from kwcoco_detector_kit.data.coco_export import export_training_splits

    train = synthetic_kwcoco_factory("train", num_images=4)
    vali = synthetic_kwcoco_factory("vali", num_images=2)
    out = tmp_path / "splits"
    paths = export_training_splits(
        train, vali, out, test_kwcoco=None, category_names=["widget"],
    )
    assert paths["train"].exists()
    assert paths["vali"].exists()
    assert "test" not in paths


def test_export_training_splits_multi_class(synthetic_kwcoco_factory, tmp_path):
    from kwcoco_detector_kit.data.coco_export import export_training_splits

    train = synthetic_kwcoco_factory(
        "train2", num_images=4, boxes_per_image=2,
        category_names=("widget", "gizmo"),
    )
    vali = synthetic_kwcoco_factory(
        "vali2", num_images=2, boxes_per_image=2,
        category_names=("widget", "gizmo"),
    )
    out = tmp_path / "splits2"
    paths = export_training_splits(
        train, vali, out, test_kwcoco=None,
        category_names=["widget", "gizmo"],
    )
    train_payload = json.loads(paths["train"].read_text())
    vali_payload = json.loads(paths["vali"].read_text())
    for payload in (train_payload, vali_payload):
        assert [c["name"] for c in payload["categories"]] == ["widget", "gizmo"]
        assert {a["category_id"] for a in payload["annotations"]} == {0, 1}
