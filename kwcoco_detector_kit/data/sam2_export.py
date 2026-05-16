"""Export kwcoco datasets to the SAM2 fine-tuning data format.

SAM2 fine-tuning expects:
    <split>/
        images/         sa_<gid:08d>.jpg   (symlink or copy of source image)
        annotations/    sa_<gid:08d>.json  ({"annotations": [{"segmentation": <RLE>, ...}]})
        <split>.txt     newline-separated stem names that have annotations

``pycocotools`` (``pip install pycocotools``) is required at export time for
RLE mask encoding.  Import is lazy so it is not a hard dependency at import time.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


def _ensure_pycocotools():
    try:
        from pycocotools import mask as _  # noqa: F401
    except Exception as ex:
        raise ImportError(
            "pycocotools is required for SAM2 training data export.  "
            "Install it with: pip install pycocotools"
        ) from ex


def _collect_category_names(src_dset, category_names=None):
    if category_names is not None:
        return set(category_names)
    ann_cids = {ann.get("category_id") for ann in src_dset.annots().objs}
    names = set()
    for cid in sorted(c for c in ann_cids if c is not None):
        cat = src_dset.cats.get(cid)
        if cat is not None:
            names.add(cat["name"])
    return names


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        dst.write_bytes(src.read_bytes())


def _segmentation_to_rle(segmentation, dims):
    """Encode a kwcoco segmentation field as a COCO RLE dict."""
    import kwimage
    import numpy as np
    from pycocotools import mask as mask_utils

    h, w = int(dims[0]), int(dims[1])
    mask = kwimage.Segmentation.coerce(segmentation, dims=(h, w)).to_mask(dims=(h, w))
    data = np.asfortranarray(mask.data.astype("uint8"))
    rle = mask_utils.encode(data[:, :, None])[0]
    counts = rle.get("counts")
    if isinstance(counts, bytes):
        rle["counts"] = counts.decode("ascii")
    return rle


def export_sam2_split(
    src_kwcoco,
    split_dpath: str | Path,
    split_name: str,
    *,
    category_names=None,
) -> dict:
    """Export one kwcoco dataset split into SAM2 fine-tuning layout.

    Args:
        src_kwcoco: Path to (or instance of) a kwcoco dataset.
        split_dpath: Root directory for this split.
        split_name: ``"train"`` or ``"vali"`` — used for the file list name.
        category_names: Optional iterable of category names to include;
            defaults to all annotated categories.

    Returns:
        Dict with keys ``image_dpath``, ``gt_dpath``, ``file_list_fpath``,
        ``metadata_fpath``.
    """
    import kwcoco

    _ensure_pycocotools()

    src_dset = kwcoco.CocoDataset.coerce(src_kwcoco)
    usable_cats = _collect_category_names(src_dset, category_names)

    split_dpath = Path(split_dpath)
    image_dpath = split_dpath / "images"
    gt_dpath = split_dpath / "annotations"
    image_dpath.mkdir(parents=True, exist_ok=True)
    gt_dpath.mkdir(parents=True, exist_ok=True)

    file_list: list[str] = []
    records: list[dict] = []

    for img in src_dset.images().objs:
        gid = img["id"]
        anns = src_dset.annots(gid=gid).objs
        exported_anns = []

        height = img.get("height")
        width = img.get("width")
        if height is None or width is None:
            import kwimage as _kwimage
            shape = _kwimage.load_image_shape(src_dset.get_image_fpath(gid))
            height, width = shape[0], shape[1]

        for ann in anns:
            cid = ann.get("category_id")
            if cid is None:
                continue
            cat = src_dset.cats.get(cid)
            if cat is None or cat["name"] not in usable_cats:
                continue
            seg = ann.get("segmentation")
            if seg is None:
                continue
            rle = _segmentation_to_rle(seg, (height, width))
            area = float(ann.get("area") or 0.0)
            if not area:
                bbox = ann.get("bbox")
                if bbox is not None:
                    area = float(bbox[2] * bbox[3])
            exported_anns.append({
                "segmentation": rle,
                "area": area,
                "category_name": cat["name"],
                "source_annotation_id": ann.get("id"),
            })

        if not exported_anns:
            continue

        stem = f"sa_{int(gid):08d}"
        src_img_fpath = Path(src_dset.get_image_fpath(gid)).resolve()
        dst_img_fpath = image_dpath / f"{stem}.jpg"
        _safe_link_or_copy(src_img_fpath, dst_img_fpath)
        (gt_dpath / f"{stem}.json").write_text(json.dumps({"annotations": exported_anns}))
        file_list.append(stem)
        records.append({
            "gid": gid,
            "stem": stem,
            "source_image_fpath": str(src_img_fpath),
            "num_annotations": len(exported_anns),
        })

    file_list_fpath = split_dpath / f"{split_name}.txt"
    file_list_fpath.write_text("".join(f"{stem}\n" for stem in file_list))

    metadata_fpath = split_dpath / "metadata.json"
    metadata_fpath.write_text(json.dumps({
        "split": split_name,
        "num_images": len(file_list),
        "category_names": sorted(usable_cats),
        "records": records,
    }, indent=2))

    return {
        "image_dpath": image_dpath,
        "gt_dpath": gt_dpath,
        "file_list_fpath": file_list_fpath,
        "metadata_fpath": metadata_fpath,
    }


def export_sam2_training_splits(
    train_kwcoco,
    vali_kwcoco,
    output_dpath: str | Path,
    *,
    category_names=None,
) -> dict:
    """Export train + vali kwcoco splits into SAM2 fine-tuning layout.

    Args:
        train_kwcoco: Training kwcoco path or dataset.
        vali_kwcoco: Validation kwcoco path or dataset.
        output_dpath: Root output directory.  Two subdirs will be created:
            ``output_dpath/train/`` and ``output_dpath/vali/``.
        category_names: Optional category name filter (default = all annotated).

    Returns:
        Dict with ``"train"`` and ``"vali"`` keys, each mapping to the
        dict returned by :func:`export_sam2_split`.
    """
    output_dpath = Path(output_dpath)
    output_dpath.mkdir(parents=True, exist_ok=True)
    return {
        "train": export_sam2_split(
            train_kwcoco,
            output_dpath / "train",
            split_name="train",
            category_names=category_names,
        ),
        "vali": export_sam2_split(
            vali_kwcoco,
            output_dpath / "vali",
            split_name="vali",
            category_names=category_names,
        ),
    }
