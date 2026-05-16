"""Unit tests for kwcoco_detector_kit.export.labelme."""
from __future__ import annotations

import numpy as np
import pytest


def _make_pred_kwcoco_with_polygons(tmp_path):
    """Build a tiny kwcoco dataset with polygon segmentation annotations."""
    import kwcoco
    import kwimage

    bundle = tmp_path / "pred_bundle"
    bundle.mkdir()
    img_path = bundle / "frame_0000.jpg"
    kwimage.imwrite(str(img_path), np.zeros((64, 64, 3), dtype=np.uint8))

    dset = kwcoco.CocoDataset()
    dset.fpath = str(bundle / "pred.kwcoco.json")
    cid = dset.add_category("widget")
    gid = dset.add_image(
        file_name=str(img_path.resolve()),
        width=64, height=64, name="frame_0000",
    )
    # Polygon annotation in COCO format: flat list of [x,y,...] points per polygon
    dset.add_annotation(
        image_id=gid,
        category_id=cid,
        bbox=[10.0, 10.0, 20.0, 20.0],
        segmentation=[[10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]],
        score=0.9,
        role="prediction",
    )
    dset.dump()
    return dset.fpath


@pytest.mark.requires_torch
def test_export_to_labelme_writes_sidecar(tmp_path):
    from kwcoco_detector_kit.export.labelme import export_to_labelme

    pred_fpath = _make_pred_kwcoco_with_polygons(tmp_path)
    written = export_to_labelme(pred_fpath, score_thresh=0.0)
    assert len(written) == 1
    assert written[0].suffix == ".json"
    assert written[0].exists()


@pytest.mark.requires_torch
def test_export_to_labelme_score_filter(tmp_path):
    from kwcoco_detector_kit.export.labelme import export_to_labelme

    pred_fpath = _make_pred_kwcoco_with_polygons(tmp_path)
    # score_thresh above the annotation's score → nothing written
    written = export_to_labelme(pred_fpath, score_thresh=0.99)
    assert written == []


@pytest.mark.requires_torch
def test_export_to_labelme_only_missing(tmp_path):
    from kwcoco_detector_kit.export.labelme import export_to_labelme

    pred_fpath = _make_pred_kwcoco_with_polygons(tmp_path)
    written1 = export_to_labelme(pred_fpath, only_missing=True)
    assert len(written1) == 1
    # Second call with only_missing=True should skip the already-written sidecar
    written2 = export_to_labelme(pred_fpath, only_missing=True)
    assert written2 == []
    # But only_missing=False should overwrite
    written3 = export_to_labelme(pred_fpath, only_missing=False)
    assert len(written3) == 1
