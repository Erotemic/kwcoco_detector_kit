"""Inference-time postprocessing for detector and segmenter outputs.

Adapts raw predictor records (dicts with ``bbox_xyxy``/``mask``/``score``/``label``)
into kwcoco-ready annotation dicts and writes them into a kwcoco dataset.

The three main consumers are:
- ``detector_records_to_bbox_anns`` — box-only detections (no segmenter)
- ``detector_records_to_anns`` — detector boxes → segmenter masks → polygons
- ``mask_records_to_anns`` — backends that return masks directly (e.g. MaskDINO)

All three return a list of annotation dicts. Pass the list to
``add_prediction_annotations`` to write them into a kwcoco dataset.
"""
from __future__ import annotations

import numpy as np


def _resolve_category_name(label, label_mapping):
    """Map an integer/string label to a category name string.

    Falls back to ``str(label)`` when ``label_mapping`` is None or the key
    is absent, so callers always get a non-None string.
    """
    if label_mapping is None:
        return str(label)
    for key in [label, str(label)]:
        if key in label_mapping:
            return label_mapping[key]
    return str(label)


def apply_box_filters(records, score_thresh, nms_thresh):
    """Score-threshold then NMS over detector records.

    Args:
        records: Iterable of dicts with ``bbox_xyxy`` (x1,y1,x2,y2) and ``score``.
        score_thresh: Minimum score to keep (inclusive).
        nms_thresh: IoU threshold for non-max suppression (0 or None = skip NMS).

    Returns:
        Filtered list of records (same dicts, subset of input).
    """
    import kwimage

    filtered = [r for r in records if float(r.get("score", 0.0)) >= score_thresh]
    if not filtered:
        return []
    boxes = kwimage.Boxes(
        np.array([r["bbox_xyxy"] for r in filtered], dtype=float), "ltrb"
    )
    scores = np.array([float(r["score"]) for r in filtered], dtype=float)
    dets = kwimage.Detections(boxes=boxes, scores=scores, classes=["object"])
    dets.data["record_idxs"] = np.arange(len(filtered))
    if nms_thresh is not None and float(nms_thresh) > 0:
        dets = dets.non_max_supress(thresh=float(nms_thresh))
    keep = dets.data["record_idxs"].tolist()
    return [filtered[i] for i in keep]


def detector_records_to_bbox_anns(records, post_cfg, label_mapping=None):
    """Convert filtered detector records to bbox-only kwcoco annotation dicts.

    Args:
        records: Iterable of dicts with ``bbox_xyxy``, ``score``, ``label``.
        post_cfg: Dict with ``score_thresh`` and optionally ``nms_thresh``.
        label_mapping: Optional dict mapping label index (int) → category name (str).
            Unmapped labels fall back to ``str(label)``.

    Returns:
        List of annotation dicts with ``category_name``, ``bbox`` (COCO xywh), ``score``.
    """
    kept = apply_box_filters(
        records,
        score_thresh=post_cfg["score_thresh"],
        nms_thresh=post_cfg.get("nms_thresh", 0.0),
    )
    anns = []
    for r in kept:
        x1, y1, x2, y2 = map(float, r["bbox_xyxy"])
        anns.append({
            "category_name": _resolve_category_name(r.get("label", 0), label_mapping),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(r["score"]),
        })
    return anns


def detector_records_to_anns(image, records, segmenter, post_cfg, label_mapping=None):
    """Chain detector boxes through a segmenter to produce polygon annotations.

    Pipeline: filtered detector boxes → (optional) crop padding expand →
    segmenter mask prompt → polygon conversion/filtering → annotation dicts.

    Args:
        image: HWC uint8 numpy image (used for image-shape clamping and segmenter input).
        records: Iterable of dicts with ``bbox_xyxy``, ``score``, ``label``.
        segmenter: Object implementing ``predict_masks_for_boxes(image, boxes) → list[dict]``
            where each returned dict has a ``mask`` key (2-D bool array).
        post_cfg: Dict with ``score_thresh``, ``nms_thresh``, ``crop_padding``
            (pixels to expand box before passing to segmenter),
            ``polygon_simplify`` (tolerance, 0 = off), ``min_component_area``,
            ``keep_largest_component``.
        label_mapping: Optional dict mapping label index → category name.

    Returns:
        List of annotation dicts with ``category_name``, ``bbox`` (COCO xywh),
        ``segmentation``, ``score``, and diagnostic ``detector_bbox`` /
        ``prompt_bbox`` / ``foundation_prompt_source`` fields.
    """
    from kwcoco_detector_kit.util.polygon_utils import (
        expand_box_xyxy,
        mask_to_multi_polygon,
        segmentation_to_coco,
    )

    kept = apply_box_filters(
        records,
        score_thresh=post_cfg["score_thresh"],
        nms_thresh=post_cfg.get("nms_thresh", 0.0),
    )
    if not kept:
        return []

    crop_padding = float(post_cfg.get("crop_padding", 0))
    polygon_simplify = float(post_cfg.get("polygon_simplify", 0.0))
    min_component_area = float(post_cfg.get("min_component_area", 0.0))
    keep_largest_component = bool(post_cfg.get("keep_largest_component", True))

    padded_boxes = [expand_box_xyxy(r["bbox_xyxy"], crop_padding, image.shape) for r in kept]
    mask_infos = segmenter.predict_masks_for_boxes(image, padded_boxes)

    anns = []
    for record, prompt_box, mask_info in zip(kept, padded_boxes, mask_infos):
        mpoly = mask_to_multi_polygon(
            mask_info["mask"],
            polygon_simplify=polygon_simplify,
            min_component_area=min_component_area,
            keep_largest_component=keep_largest_component,
        )
        if not len(mpoly.data):
            continue
        x1, y1, x2, y2 = map(float, record["bbox_xyxy"])
        px1, py1, px2, py2 = map(float, prompt_box)
        anns.append({
            "category_name": _resolve_category_name(record.get("label", 0), label_mapping),
            "bbox": list(mpoly.box().to_coco()),
            "segmentation": segmentation_to_coco(mpoly),
            "score": float(record["score"]),
            "foundation_prompt_source": "detector_box",
            "detector_bbox": [x1, y1, x2 - x1, y2 - y1],
            "prompt_bbox": [px1, py1, px2 - px1, py2 - py1],
        })
    return anns


def mask_records_to_anns(mask_records, post_cfg, label_mapping=None):
    """Convert mask-producing backend records to kwcoco annotation dicts.

    For backends like MaskDINO that emit masks directly without a separate
    segmenter stage.

    Args:
        mask_records: Iterable of dicts with ``mask`` (2-D bool/uint8 array),
            ``bbox_xyxy``, ``score``, ``label``.
        post_cfg: Dict with ``score_thresh``, ``nms_thresh``, ``polygon_simplify``,
            ``min_component_area``, ``keep_largest_component``.
        label_mapping: Optional dict mapping label index → category name.

    Returns:
        List of annotation dicts with ``category_name``, ``bbox`` (COCO xywh),
        ``segmentation``, ``score``.
    """
    import kwimage
    from kwcoco_detector_kit.util.polygon_utils import mask_to_multi_polygon, segmentation_to_coco

    score_thresh = post_cfg["score_thresh"]
    nms_thresh = post_cfg.get("nms_thresh", 0.0)
    polygon_simplify = float(post_cfg.get("polygon_simplify", 0.0))
    min_component_area = float(post_cfg.get("min_component_area", 0.0))
    keep_largest_component = bool(post_cfg.get("keep_largest_component", True))

    filtered = [r for r in mask_records if float(r.get("score", 0.0)) >= score_thresh]
    if not filtered:
        return []

    boxes = kwimage.Boxes(
        np.array([r["bbox_xyxy"] for r in filtered], dtype=float), "ltrb"
    )
    scores = np.array([float(r["score"]) for r in filtered], dtype=float)
    dets = kwimage.Detections(boxes=boxes, scores=scores, classes=["object"])
    dets.data["record_idxs"] = np.arange(len(filtered))
    if nms_thresh is not None and float(nms_thresh) > 0:
        dets = dets.non_max_supress(thresh=float(nms_thresh))
    kept = [filtered[i] for i in dets.data["record_idxs"].tolist()]

    anns = []
    for record in kept:
        mpoly = mask_to_multi_polygon(
            record["mask"],
            polygon_simplify=polygon_simplify,
            min_component_area=min_component_area,
            keep_largest_component=keep_largest_component,
        )
        if not len(mpoly.data):
            continue
        anns.append({
            "category_name": _resolve_category_name(record.get("label", 0), label_mapping),
            "bbox": list(mpoly.box().to_coco()),
            "segmentation": segmentation_to_coco(mpoly),
            "score": float(record["score"]),
        })
    return anns


def add_prediction_annotations(pred_dset, image_id, anns, backend_name):
    """Write annotation dicts into a kwcoco dataset.

    Each dict must have a ``category_name`` key; all other keys are passed
    through as kwcoco annotation fields. Categories are created on demand via
    ``pred_dset.ensure_category`` so callers need not pre-register them.

    Args:
        pred_dset: Writable ``kwcoco.CocoDataset``.
        image_id: Image ID to attach annotations to.
        anns: Annotation dicts produced by one of the ``*_to_anns`` helpers.
        backend_name: Tag stored as ``foundation_backend`` on every annotation.
    """
    for ann in anns:
        ann = ann.copy()
        category_name = ann.pop("category_name")
        ann["image_id"] = image_id
        ann["category_id"] = pred_dset.ensure_category(category_name)
        ann["role"] = "prediction"
        ann["foundation_backend"] = str(backend_name)
        pred_dset.add_annotation(**ann)
