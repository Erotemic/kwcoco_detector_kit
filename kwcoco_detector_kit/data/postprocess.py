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


def _preferred_cpu_nms_impl():
    """Choose the fastest available CPU NMS implementation.

    ``kwimage`` historically exposes the optional ``kwimage_ext`` CPU backend
    under the compatibility name ``cython_cpu``.  The Rust-first
    ``kwimage_ext`` wheels keep that shim, but deliberately do not implement
    GPU NMS.  In particular, we must not use kwimage's ``impl="auto"`` here:
    versions that only probe whether the GPU shim imports can select a
    ``gpu_nms`` compatibility stub that raises ``NotImplementedError``.

    Prediction postprocessing is already operating on NumPy boxes/scores, so
    keeping NMS on the CPU also avoids an unnecessary CPU->GPU->CPU transfer.
    """
    import kwimage

    available = set(kwimage.available_nms_impls())
    for impl in ("cython_cpu", "cpu", "numpy"):
        if impl in available:
            return impl
    # ``numpy`` is kwimage's built-in reference implementation and should be
    # available even when optional acceleration packages are absent.
    return "numpy"


def _cpu_non_max_supress(dets, thresh):
    """Run NMS without allowing kwimage to auto-select a GPU backend."""
    return dets.non_max_supress(
        thresh=float(thresh),
        impl=_preferred_cpu_nms_impl(),
    )


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


def _box_to_native_xyxy(box, prediction_space=None):
    if prediction_space is None:
        return list(map(float, box))
    return prediction_space.box_to_native_xyxy(box)


def _mpoly_to_native(mpoly, prediction_space=None):
    if prediction_space is None:
        return mpoly
    return prediction_space.warp_multipolygon_to_native(mpoly)


def _annotation_category_identity(ann):
    """Return a stable category identity for prediction-like annotation dicts."""
    if "category_name" in ann:
        return ("name", ann.get("category_name"))
    if "category_id" in ann:
        return ("id", ann.get("category_id"))
    if "label" in ann:
        return ("label", ann.get("label"))
    return ("uncategorized", None)


def _annotation_ltrb(ann):
    bbox = ann.get("bbox")
    if bbox is not None:
        x, y, w, h = map(float, bbox)
        return np.array([x, y, x + w, y + h], dtype=float)
    bbox_xyxy = ann.get("bbox_xyxy")
    if bbox_xyxy is not None:
        return np.asarray(bbox_xyxy, dtype=float)
    return None


def suppress_mask_iomin_duplicates(anns, thresh, *, dims=None, return_stats=False):
    """Suppress same-class mask duplicates using intersection-over-minimum area.

    Ordinary IoU NMS can retain an almost completely contained prediction when
    its area is much smaller than the enclosing prediction.  This pass measures

        IoMin(A, B) = area(A intersect B) / min(area(A), area(B))

    and greedily keeps the higher-score annotation when IoMin reaches ``thresh``.
    Only annotations with segmentations participate; box-only annotations are
    never suppressed by this mask-specific rule.  Different categories never
    suppress each other.

    Args:
        anns: Prediction-like annotation dictionaries.  Category identity is
            read from ``category_name``, then ``category_id``, then ``label``.
        thresh: Threshold in (0, 1].  ``None`` or values <= 0 disable the pass.
        dims: Optional ``(height, width)`` used when coercing encoded masks.
        return_stats: If true, return ``(kept, stats)``.

    Returns:
        A subset of the original dictionaries, preserving original order.  When
        ``return_stats`` is true, also returns a small suppression summary.
    """
    anns = list(anns)
    threshold = 0.0 if thresh is None else float(thresh)
    if threshold <= 0 or len(anns) < 2:
        stats = {
            "threshold": threshold,
            "input": len(anns),
            "suppressed": 0,
            "kept": len(anns),
        }
        return (anns, stats) if return_stats else anns
    if not (0.0 < threshold <= 1.0):
        raise ValueError(f"mask IoMin threshold must be in (0, 1], got {threshold}")

    import kwimage

    # Greedy confidence ordering determines the winner; output ordering remains
    # unchanged so enabling this filter only removes records.
    order = sorted(
        range(len(anns)),
        key=lambda idx: (-float(anns[idx].get("score", 0.0)), idx),
    )
    geom_cache = {}
    bbox_cache = {}
    area_cache = {}

    def _coerce_geom(idx):
        if idx in geom_cache:
            return geom_cache[idx]
        seg = anns[idx].get("segmentation")
        if seg is None:
            geom_cache[idx] = None
            area_cache[idx] = 0.0
            return None
        try:
            coerced = kwimage.Segmentation.coerce(seg, dims=dims)
            geom = coerced.to_multi_polygon().to_shapely(fix=True)
            area = float(geom.area)
            if geom.is_empty or area <= 0:
                geom = None
                area = 0.0
        except Exception:
            # A malformed/unsupported segmentation should not cause a valid
            # prediction to disappear.  Leave it untouched instead.
            geom = None
            area = 0.0
        geom_cache[idx] = geom
        area_cache[idx] = area
        return geom

    def _bbox(idx):
        if idx not in bbox_cache:
            box = _annotation_ltrb(anns[idx])
            if box is None:
                geom = _coerce_geom(idx)
                if geom is not None:
                    minx, miny, maxx, maxy = geom.bounds
                    box = np.array([minx, miny, maxx, maxy], dtype=float)
            bbox_cache[idx] = box
        return bbox_cache[idx]

    kept_score_order = []
    suppressed = set()
    for idx in order:
        geom = _coerce_geom(idx)
        if geom is None:
            kept_score_order.append(idx)
            continue
        category = _annotation_category_identity(anns[idx])
        box = _bbox(idx)
        duplicate = False
        for prev_idx in kept_score_order:
            if _annotation_category_identity(anns[prev_idx]) != category:
                continue
            prev_geom = _coerce_geom(prev_idx)
            if prev_geom is None:
                continue
            prev_box = _bbox(prev_idx)
            if box is not None and prev_box is not None:
                if (
                    min(box[2], prev_box[2]) <= max(box[0], prev_box[0])
                    or min(box[3], prev_box[3]) <= max(box[1], prev_box[1])
                ):
                    continue
            min_area = min(area_cache[idx], area_cache[prev_idx])
            if min_area <= 0:
                continue
            inter_area = float(geom.intersection(prev_geom).area)
            if inter_area / min_area >= threshold:
                duplicate = True
                suppressed.add(idx)
                break
        if not duplicate:
            kept_score_order.append(idx)

    kept = [ann for idx, ann in enumerate(anns) if idx not in suppressed]
    stats = {
        "threshold": threshold,
        "input": len(anns),
        "suppressed": len(suppressed),
        "kept": len(kept),
    }
    return (kept, stats) if return_stats else kept


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
        dets = _cpu_non_max_supress(dets, nms_thresh)
    keep = dets.data["record_idxs"].tolist()
    return [filtered[i] for i in keep]


def detector_records_to_bbox_anns(
    records, post_cfg, label_mapping=None, prediction_space=None
):
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
        x1, y1, x2, y2 = _box_to_native_xyxy(
            r["bbox_xyxy"], prediction_space
        )
        anns.append({
            "category_name": _resolve_category_name(r.get("label", 0), label_mapping),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(r["score"]),
        })
    return anns


def detector_records_to_anns(
    image, records, segmenter, post_cfg, label_mapping=None, prediction_space=None
):
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
        mpoly = _mpoly_to_native(mpoly, prediction_space)
        x1, y1, x2, y2 = _box_to_native_xyxy(
            record["bbox_xyxy"], prediction_space
        )
        px1, py1, px2, py2 = _box_to_native_xyxy(
            prompt_box, prediction_space
        )
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


def mask_records_to_anns(
    mask_records, post_cfg, label_mapping=None, prediction_space=None
):
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
        dets = _cpu_non_max_supress(dets, nms_thresh)
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
        mpoly = _mpoly_to_native(mpoly, prediction_space)
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
