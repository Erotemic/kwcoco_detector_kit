"""Unit tests for kwcoco_detector_kit.data.sam2_export."""
from __future__ import annotations

import json

import numpy as np
import pytest


@pytest.fixture
def kwcoco_with_polygons(tmp_path):
    """Build a tiny kwcoco dataset with polygon segmentation annotations."""
    import kwcoco
    import kwimage

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    img_path = bundle / "img_0000.jpg"
    kwimage.imwrite(str(img_path), np.zeros((64, 64, 3), dtype=np.uint8))

    dset = kwcoco.CocoDataset()
    dset.fpath = str(bundle / "data.kwcoco.json")
    cid = dset.add_category("widget")
    gid = dset.add_image(file_name=str(img_path.resolve()), width=64, height=64)
    # COCO flat polygon: [x1,y1,x2,y2,...]
    dset.add_annotation(
        image_id=gid,
        category_id=cid,
        bbox=[10.0, 10.0, 20.0, 20.0],
        segmentation=[[10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]],
        area=400.0,
    )
    dset.dump()
    return dset.fpath


@pytest.mark.requires_torch
def test_export_sam2_split_creates_expected_files(kwcoco_with_polygons, tmp_path):
    pytest.importorskip("pycocotools", reason="pycocotools required")
    from kwcoco_detector_kit.data.sam2_export import export_sam2_split

    out = tmp_path / "sam2_split"
    result = export_sam2_split(
        kwcoco_with_polygons, out, split_name="train", category_names=["widget"]
    )

    assert result["image_dpath"].is_dir()
    assert result["gt_dpath"].is_dir()
    assert result["file_list_fpath"].exists()
    assert result["metadata_fpath"].exists()

    # At least one image + annotation file
    image_files = list(result["image_dpath"].iterdir())
    gt_files = list(result["gt_dpath"].iterdir())
    assert len(image_files) == 1
    assert len(gt_files) == 1
    assert image_files[0].suffix == ".jpg"

    # Annotation file has expected structure
    ann_data = json.loads(gt_files[0].read_text())
    assert "annotations" in ann_data
    assert len(ann_data["annotations"]) == 1
    ann = ann_data["annotations"][0]
    assert "segmentation" in ann
    assert ann["segmentation"]["counts"]  # RLE counts present
    assert ann["category_name"] == "widget"


@pytest.mark.requires_torch
def test_export_sam2_split_skips_images_without_polygons(tmp_path):
    """Images that have no segmentation annotations should not appear in the split."""
    pytest.importorskip("pycocotools", reason="pycocotools required")
    import kwcoco
    import kwimage
    from kwcoco_detector_kit.data.sam2_export import export_sam2_split

    bundle = tmp_path / "no_seg_bundle"
    bundle.mkdir()
    img_path = bundle / "frame.jpg"
    kwimage.imwrite(str(img_path), np.zeros((32, 32, 3), dtype=np.uint8))

    dset = kwcoco.CocoDataset()
    dset.fpath = str(bundle / "data.kwcoco.json")
    cid = dset.add_category("widget")
    gid = dset.add_image(file_name=str(img_path.resolve()), width=32, height=32)
    # bbox-only annotation — no segmentation field
    dset.add_annotation(image_id=gid, category_id=cid, bbox=[0.0, 0.0, 10.0, 10.0])
    dset.dump()

    out = tmp_path / "sam2_no_seg"
    result = export_sam2_split(str(dset.fpath), out, split_name="train")

    assert list(result["image_dpath"].iterdir()) == []
    meta = json.loads(result["metadata_fpath"].read_text())
    assert meta["num_images"] == 0


@pytest.mark.requires_torch
def test_export_sam2_training_splits(kwcoco_with_polygons, tmp_path):
    pytest.importorskip("pycocotools", reason="pycocotools required")
    from kwcoco_detector_kit.data.sam2_export import export_sam2_training_splits

    out = tmp_path / "sam2"
    exports = export_sam2_training_splits(
        kwcoco_with_polygons, kwcoco_with_polygons, out
    )
    assert "train" in exports
    assert "vali" in exports
    assert (out / "train").is_dir()
    assert (out / "vali").is_dir()


@pytest.mark.requires_torch
def test_sam2_trainer_variants():
    from kwcoco_detector_kit.trainers.sam2 import SAM2Trainer

    for variant in SAM2Trainer.VARIANTS:
        t = SAM2Trainer(variant=variant)
        assert t.variant == variant


@pytest.mark.requires_torch
def test_sam2_trainer_unknown_variant():
    from kwcoco_detector_kit.trainers.sam2 import SAM2Trainer
    import pytest

    with pytest.raises(ValueError, match="Unknown SAM2 variant"):
        SAM2Trainer(variant="nonexistent_variant")
