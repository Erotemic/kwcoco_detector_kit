"""Detection eval should tolerate non-detection kwcoco annotations."""
from __future__ import annotations


def test_filter_bbox_only_kwcoco_drops_caption_only_annotations(synthetic_kwcoco, tmp_path):
    import kwcoco

    from kwcoco_detector_kit.eval.kwcoco_eval import filter_bbox_only_kwcoco

    src = kwcoco.CocoDataset.coerce(str(synthetic_kwcoco))
    gid = next(iter(src.images()))
    cid = src.add_category(name="caption_meta")
    src.add_annotation(image_id=gid, category_id=cid, caption="not a detection")
    src.add_annotation(image_id=gid, category_id=cid, bbox=None)
    src.add_annotation(image_id=gid, category_id=cid, bbox=[1, 2, 3])
    src_fpath = tmp_path / "with_caption_only.kwcoco.zip"
    src._update_fpath(str(src_fpath))
    src.dump()

    dst_fpath, kept, dropped = filter_bbox_only_kwcoco(
        src_fpath, tmp_path / "bbox_only.kwcoco.zip")

    assert kept == 4
    assert dropped == 3
    filtered = kwcoco.CocoDataset.coerce(str(dst_fpath))
    assert filtered.n_annots == 4
    for ann in filtered.annots().objs:
        bbox = ann.get("bbox")
        assert isinstance(bbox, list)
        assert len(bbox) == 4
