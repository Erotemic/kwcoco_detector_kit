from __future__ import annotations

import json

import numpy as np
import pytest


@pytest.mark.requires_torch
def test_prediction_review_mask_iomin_dedupe(tmp_path):
    import kwcoco
    import kwimage

    from kwcoco_detector_kit.data.truth_review import PredictionReviewConfig, build_prediction_review

    image_fpath = tmp_path / "image.png"
    kwimage.imwrite(image_fpath, np.zeros((128, 128, 3), dtype=np.uint8))

    truth = kwcoco.CocoDataset()
    truth.fpath = tmp_path / "truth.kwcoco.zip"
    gid = truth.add_image(file_name=str(image_fpath), width=128, height=128)
    truth.add_category(name="poop")
    truth.dump()

    pred = kwcoco.CocoDataset()
    pred.fpath = tmp_path / "pred.kwcoco.zip"
    pred.add_image(id=gid, file_name=str(image_fpath), width=128, height=128)
    cid = pred.add_category(name="poop")
    outer = kwimage.Polygon(
        exterior=np.array([[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]])
    ).to_coco(style="new")
    inner = kwimage.Polygon(
        exterior=np.array([[10, 10], [50, 10], [50, 50], [10, 50], [10, 10]])
    ).to_coco(style="new")
    pred.add_annotation(
        image_id=gid, category_id=cid, bbox=[0, 0, 100, 100],
        segmentation=outer, score=0.8,
    )
    pred.add_annotation(
        image_id=gid, category_id=cid, bbox=[10, 10, 40, 40],
        segmentation=inner, score=0.9,
    )
    pred.dump()

    dst = tmp_path / "review"
    cfg = PredictionReviewConfig.cli(argv=False, data={
        "true": str(truth.fpath),
        "pred": str(pred.fpath),
        "dst_dpath": str(dst),
        "target_categories": "poop",
        "mask_iomin_thresh": 0.85,
        "top_n": 100,
    })
    build_prediction_review(cfg)
    payload = json.loads((dst / "review_queue.json").read_text())
    assert len(payload["items"]) == 1
    stats = payload["prediction_postprocess"]["mask_iomin_dedupe"]
    assert stats["total_predictions"] == 2
    assert stats["input"] == 2
    assert stats["suppressed"] == 1
    assert stats["kept"] == 1
