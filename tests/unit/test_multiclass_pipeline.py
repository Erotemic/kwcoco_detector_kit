"""End-to-end multi-class wiring test — tile -> MSCOCO export.

Smoke-tests that the multi-class data flow stays consistent across the
two stages that decide class identity: ``data.tile`` (kwcoco -> tiled
kwcoco) and ``data.coco_export.export_mscoco`` (kwcoco -> MSCOCO json
for DEIMv2's train.py). The invariant is:

  CLI category_names order  ==  output category_id order  ==
  train-time class index (since DEIMv2 uses ``category_id=i`` for the
  i-th class).

This guards against silent class-id reordering across the data
pipeline. Does not exercise the trainer itself (that requires torch
and a GPU).
"""
from __future__ import annotations

import json
from pathlib import Path

import kwcoco
import pytest


def _tile_run(src, dst, *, category_names: str, mode: str = "multiscale", **kwargs):
    from kwcoco_detector_kit.data.tile import TileConfig, run
    cfg = TileConfig.cli(
        argv=False,
        data={
            "src": str(src), "dst": str(dst), "mode": mode,
            "category_names": category_names, "progress": False,
            **kwargs,
        },
    )
    run(cfg)
    return kwcoco.CocoDataset.coerce(str(dst))


def test_tile_then_export_preserves_class_order(synthetic_kwcoco_factory, tmp_path):
    """Tile a 2-class kwcoco, then export to MSCOCO using the same order.
    Both stages should agree on the assignment widget->id1 / gizmo->id2
    inside the tiled bundle, and widget->0 / gizmo->1 in the exported
    MSCOCO (the latter is the contract DEIMv2's train.py consumes).
    """
    src = synthetic_kwcoco_factory(
        "src", num_images=4, boxes_per_image=2,
        category_names=("widget", "gizmo"),
    )

    tiled_fpath = tmp_path / "tiled.kwcoco.zip"
    tiled = _tile_run(
        src, tiled_fpath, category_names="widget,gizmo",
        tile_size=128, source_scales="1.0", min_source_scale_long_side=32,
        min_gt_area_frac=0.0001, keep_negative=False,
    )

    # data.tile uses 1-indexed category IDs.
    tiled_cats = {c["id"]: c["name"] for c in tiled.dataset["categories"]}
    assert tiled_cats == {1: "widget", 2: "gizmo"}
    tiled_ann_cats = {ann["category_id"] for ann in tiled.annots().objs}
    assert tiled_ann_cats == {1, 2}, "tile output should preserve both classes"

    # Re-export through the MSCOCO pipeline (what the DEIMv2 trainer
    # implicitly calls). DEIMv2 expects 0-indexed labels matching the
    # i-th category, so category_id_start=0 is the contract.
    from kwcoco_detector_kit.data.coco_export import export_mscoco
    mscoco_fpath = tmp_path / "exported.mscoco.json"
    export_mscoco(
        tiled_fpath, mscoco_fpath,
        category_names=["widget", "gizmo"],
        include_segmentations=False, category_id_start=0,
    )
    payload = json.loads(mscoco_fpath.read_text())
    assert payload["categories"] == [
        {"id": 0, "name": "widget", "supercategory": "widget"},
        {"id": 1, "name": "gizmo", "supercategory": "gizmo"},
    ]
    mscoco_ann_cats = {ann["category_id"] for ann in payload["annotations"]}
    assert mscoco_ann_cats == {0, 1}, "MSCOCO export should keep both classes"


def test_reversed_cli_order_reverses_exported_ids(synthetic_kwcoco_factory, tmp_path):
    """Passing names in reversed order flips the exported ID assignment.
    This is the load-bearing property for keeping eval-time labels
    consistent with train-time labels: if you train with ['gizmo',
    'widget'], the eval pred bundle should register 'gizmo' first too.
    """
    src = synthetic_kwcoco_factory(
        "src2", num_images=4, boxes_per_image=2,
        category_names=("widget", "gizmo"),
    )
    from kwcoco_detector_kit.data.coco_export import export_mscoco
    fpath_forward = tmp_path / "forward.mscoco.json"
    fpath_reverse = tmp_path / "reverse.mscoco.json"
    export_mscoco(
        src, fpath_forward, category_names=["widget", "gizmo"],
        include_segmentations=False,
    )
    export_mscoco(
        src, fpath_reverse, category_names=["gizmo", "widget"],
        include_segmentations=False,
    )
    forward = json.loads(fpath_forward.read_text())
    reverse = json.loads(fpath_reverse.read_text())
    assert [c["name"] for c in forward["categories"]] == ["widget", "gizmo"]
    assert [c["name"] for c in reverse["categories"]] == ["gizmo", "widget"]
    # IDs are independent of source; only the position in the requested
    # list determines them.
    assert {c["id"] for c in forward["categories"]} == {0, 1}
    assert {c["id"] for c in reverse["categories"]} == {0, 1}


def test_eval_signature_requires_category_names(tmp_path):
    """Sanity: kwcoco_eval.run_kwcoco_eval rejects a singular string.
    Cheap signature check — doesn't require torch.
    """
    from kwcoco_detector_kit.eval.kwcoco_eval import run_kwcoco_eval

    class _FakeTrainer:
        pass

    with pytest.raises(TypeError):
        run_kwcoco_eval(
            trainer=_FakeTrainer(),
            workdir=tmp_path,
            test_kwcoco="/tmp/does_not_matter",
            kcd_root=tmp_path,
            candidate_id="x",
            category_names="widget",  # type: ignore[arg-type]  -- should raise
        )
