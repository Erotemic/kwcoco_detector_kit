"""Tests for data.coco_export — kwcoco -> MSCOCO json."""
from __future__ import annotations

import json
from pathlib import Path


def test_export_mscoco_round_trip(synthetic_kwcoco, tmp_path):
    from kwcoco_detector_kit.data.coco_export import export_mscoco

    dst = tmp_path / "out.mscoco.json"
    export_mscoco(
        synthetic_kwcoco, dst, category_name="widget",
        include_segmentations=False, category_id=0,
    )
    assert dst.exists()
    payload = json.loads(dst.read_text())
    assert payload["categories"][0]["name"] == "widget"
    assert len(payload["images"]) > 0
    # File paths in the export are absolute (consumer-friendly).
    for img in payload["images"]:
        assert Path(img["file_name"]).is_absolute()


def test_export_mscoco_drops_other_categories(synthetic_kwcoco_factory, tmp_path):
    # Build a 2-category bundle, then export only one category.
    src = synthetic_kwcoco_factory("mixed", num_images=3, boxes_per_image=1)
    from kwcoco_detector_kit.data.coco_export import export_mscoco
    dst = tmp_path / "only_widget.mscoco.json"
    export_mscoco(src, dst, category_name="widget", include_segmentations=False)
    payload = json.loads(dst.read_text())
    cat_names = {c["name"] for c in payload["categories"]}
    assert cat_names == {"widget"}


def test_export_training_splits(synthetic_kwcoco_factory, tmp_path):
    from kwcoco_detector_kit.data.coco_export import export_training_splits

    train = synthetic_kwcoco_factory("train", num_images=4)
    vali = synthetic_kwcoco_factory("vali", num_images=2)
    out = tmp_path / "splits"
    paths = export_training_splits(
        train, vali, out, test_kwcoco=None, category_name="widget",
    )
    assert paths["train"].exists()
    assert paths["vali"].exists()
    assert "test" not in paths  # test_kwcoco was None
