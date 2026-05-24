"""
Export a kwcoco bundle to MSCOCO json for the upstream trainers
(DEIMv2's ``train.py``, OpenGroundingDINO's ``coco2odvg.py``, MaskDINO,
etc.) that don't speak kwcoco directly.

The exported json has:

- ``categories`` — one entry per requested name (in caller-supplied order),
  with ``id`` starting at ``category_id_start`` (default 0) and incrementing.
  Annotations whose source category name is not in the requested list are
  dropped.
- ``images`` — one row per source image, with absolute ``file_name``.
- ``annotations`` — one row per kept annotation, bbox in MSCOCO xywh.
  Optionally carries ``segmentation`` polygons. Bbox falls back to the
  segmentation's enclosing box when missing (mirrors v9's
  ``ensure_true_bboxes``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence


def export_mscoco(
    src,
    dst,
    *,
    category_names: Sequence[str],
    include_segmentations: bool = True,
    category_id_start: int = 0,
):
    """Write a single MSCOCO json from a kwcoco bundle. Returns the dst path.

    Args:
        category_names: ordered sequence of source category names to keep.
            The order determines the MSCOCO ``category_id`` assigned
            (``category_id_start + i``), so it must match the order the
            trainer expects (it is used to interpret model class indices at
            eval time).
    """
    import kwcoco
    import kwimage

    if isinstance(category_names, str):
        raise TypeError(
            "category_names must be a sequence of names, not a single string"
        )
    category_names = list(category_names)
    if not category_names:
        raise ValueError("category_names must contain at least one name")

    name_to_id = {name: category_id_start + i for i, name in enumerate(category_names)}

    src_dset = kwcoco.CocoDataset.coerce(src)
    export: dict = {
        "info": {},
        "images": [],
        "annotations": [],
        "categories": [
            {"id": name_to_id[name], "name": name, "supercategory": name}
            for name in category_names
        ],
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
        if cat is None or cat["name"] not in name_to_id:
            continue
        new_ann = {
            "id": ann_id,
            "image_id": gid,
            "category_id": name_to_id[cat["name"]],
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
    category_names: Sequence[str],
    include_segmentations: bool = True,
    category_id_start: int = 0,
):
    """Export train+vali (+optional test) MSCOCO jsons next to each other."""
    output_dpath = Path(output_dpath)
    output_dpath.mkdir(parents=True, exist_ok=True)
    exports = {
        "train": export_mscoco(
            train_kwcoco,
            output_dpath / "train.mscoco.json",
            category_names=category_names,
            include_segmentations=include_segmentations,
            category_id_start=category_id_start,
        ),
        "vali": export_mscoco(
            vali_kwcoco,
            output_dpath / "vali.mscoco.json",
            category_names=category_names,
            include_segmentations=include_segmentations,
            category_id_start=category_id_start,
        ),
    }
    if test_kwcoco is not None:
        exports["test"] = export_mscoco(
            test_kwcoco,
            output_dpath / "test.mscoco.json",
            category_names=category_names,
            include_segmentations=include_segmentations,
            category_id_start=category_id_start,
        )
    return exports
