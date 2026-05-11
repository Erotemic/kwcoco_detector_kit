"""
Export a kwcoco bundle to MSCOCO json for the upstream trainers
(DEIMv2's ``train.py``, OpenGroundingDINO's ``coco2odvg.py``, MaskDINO,
etc.) that don't speak kwcoco directly.

The exported json has:

- ``categories`` — one entry, the requested ``category_name`` (others dropped).
- ``images`` — one row per source image, with absolute ``file_name``.
- ``annotations`` — one row per kept annotation, bbox in MSCOCO xywh.
  Optionally carries ``segmentation`` polygons. Bbox falls back to the
  segmentation's enclosing box when missing (mirrors v9's
  ``ensure_true_bboxes``).

Lifted from the prior project's ``coco_adapter._build_coco_export``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def export_mscoco(
    src,
    dst,
    *,
    category_name: str = "widget",
    include_segmentations: bool = True,
    category_id: int = 0,
):
    """Write a single MSCOCO json from a kwcoco bundle. Returns the dst path."""
    import kwcoco
    import kwimage

    src_dset = kwcoco.CocoDataset.coerce(src)
    export: dict = {
        "info": {},
        "images": [],
        "annotations": [],
        "categories": [{
            "id": category_id,
            "name": category_name,
            "supercategory": category_name,
        }],
    }

    kept_gids = set()
    for img in src_dset.images().objs:
        img = img.copy()
        try:
            file_name = src_dset.get_image_fpath(img["id"])
        except Exception:
            bundle_dpath = src_dset.bundle_dpath or "."
            file_name = Path(bundle_dpath) / img["file_name"]
        img["file_name"] = str(Path(file_name).resolve())
        export["images"].append({
            "id": img["id"],
            "file_name": img["file_name"],
            "width": img.get("width"),
            "height": img.get("height"),
        })
        kept_gids.add(img["id"])

    ann_id = 1
    for ann in src_dset.annots().objs:
        gid = ann["image_id"]
        if gid not in kept_gids:
            continue
        src_cid = ann.get("category_id")
        if src_cid is None:
            continue
        cat = src_dset.cats.get(src_cid)
        if cat is None or cat["name"] != category_name:
            continue
        new_ann = {
            "id": ann_id,
            "image_id": gid,
            "category_id": category_id,
            "iscrowd": int(ann.get("iscrowd", 0)),
            "bbox": ann.get("bbox"),
            "area": float(ann.get("area", 0.0)),
        }
        # Fall back to segmentation bbox when bbox is missing (mirror of v9 ensure_true_bboxes).
        if new_ann["bbox"] is None and ann.get("segmentation") is not None:
            seg = kwimage.Segmentation.coerce(ann["segmentation"]).to_multi_polygon()
            new_ann["bbox"] = list(seg.box().to_coco())
            new_ann["area"] = float(seg.area)
        elif new_ann["bbox"] is not None and not new_ann["area"]:
            new_ann["area"] = float(new_ann["bbox"][2] * new_ann["bbox"][3])
        if include_segmentations and ann.get("segmentation") is not None:
            new_ann["segmentation"] = ann["segmentation"]
            if not new_ann["area"]:
                seg = kwimage.Segmentation.coerce(ann["segmentation"]).to_multi_polygon()
                new_ann["area"] = float(seg.area)
        export["annotations"].append(new_ann)
        ann_id += 1

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(export))
    return dst


def export_training_splits(
    train_kwcoco,
    vali_kwcoco,
    output_dpath,
    *,
    test_kwcoco: Optional[str] = None,
    category_name: str = "widget",
    include_segmentations: bool = True,
    category_id: int = 0,
):
    """Export train+vali (+optional test) MSCOCO jsons next to each other."""
    output_dpath = Path(output_dpath)
    output_dpath.mkdir(parents=True, exist_ok=True)
    exports = {
        "train": export_mscoco(
            train_kwcoco,
            output_dpath / "train.mscoco.json",
            category_name=category_name,
            include_segmentations=include_segmentations,
            category_id=category_id,
        ),
        "vali": export_mscoco(
            vali_kwcoco,
            output_dpath / "vali.mscoco.json",
            category_name=category_name,
            include_segmentations=include_segmentations,
            category_id=category_id,
        ),
    }
    if test_kwcoco is not None:
        exports["test"] = export_mscoco(
            test_kwcoco,
            output_dpath / "test.mscoco.json",
            category_name=category_name,
            include_segmentations=include_segmentations,
            category_id=category_id,
        )
    return exports
