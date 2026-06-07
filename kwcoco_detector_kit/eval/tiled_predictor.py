"""
Windowed (tiled) inference wrapper for any ``DetectorPredictor``.

Why this exists
---------------
Detectors in this kit train on fixed-size tiles cut from full-resolution
imagery, but the default eval path (``deimv2.DEIMv2Predictor.predict_image``)
resizes each WHOLE image down to the model input size and runs a single
forward pass. For corpora where objects are small relative to the source
image — e.g. sea-lion pups (~46px) in multi-thousand-pixel aerials — that
whole-image resize shrinks a 46px object to a handful of pixels and the
detector cannot localize it. The symptom is COCO ``AP-small`` pinned near
zero while ``AP-large`` is healthy (see the gen005 forensic journal).

``TiledPredictor`` closes the train/eval resolution gap WITHOUT retraining:
it slides a native-resolution window (default = the model's
``eval_spatial_size``, i.e. the training tile size) across the full image,
runs the wrapped predictor on each window crop, translates each detection
back into full-image coordinates, and merges the per-window detections with
per-class non-maximum suppression. A 46px object stays 46px inside a 640
window, exactly as it appeared at training time.

It implements the same ``DetectorPredictor`` protocol as the thing it wraps,
so the eval and hard-negative-mining paths can use it transparently.

``keep_full`` (default True) additionally runs one whole-image pass and
folds those detections into the NMS merge, so large objects — which the
whole-image path already handles well — are never lost when they straddle
window seams or exceed a single window.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from kwcoco_detector_kit.data.tile import _grid_positions


def _per_class_nms(detections: List[dict], iou_thresh: float) -> List[dict]:
    """Greedy IoU NMS applied independently within each class label."""
    if not detections:
        return detections
    by_label: dict = {}
    for det in detections:
        by_label.setdefault(int(det["label"]), []).append(det)

    kept: List[dict] = []
    for label, dets in by_label.items():
        boxes = np.array([d["bbox_xyxy"] for d in dets], dtype=np.float64)
        scores = np.array([float(d["score"]) for d in dets], dtype=np.float64)
        for idx in _nms_indices(boxes, scores, iou_thresh):
            kept.append(dets[idx])
    return kept


def _nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """Indices surviving greedy NMS. ``boxes`` is Nx4 xyxy."""
    if boxes.shape[0] == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clip(min=0) * (yy2 - yy1).clip(min=0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thresh]
    return keep


class TiledPredictor:
    """Wrap a ``DetectorPredictor`` to run windowed inference + NMS merge.

    Args:
        base: the wrapped predictor (must expose ``eval_spatial_size`` and
            ``predict_image``).
        window: (H, W) window size in source pixels. Defaults to the base
            model's ``eval_spatial_size`` so each window is processed 1:1
            (no resize) — the whole point. Override only if you deliberately
            want a different inference scale.
        overlap: fractional overlap between adjacent windows in [0, 0.9].
            0.25 gives a quarter-window seam so objects on a boundary appear
            whole in at least one window.
        nms_thresh: IoU threshold for the cross-window NMS merge.
        keep_full: also run one whole-image pass and merge it in (protects
            large-object recall). Set False for a pure tiled pass.
    """

    def __init__(
        self,
        base,
        *,
        window: Optional[Sequence[int]] = None,
        overlap: float = 0.25,
        nms_thresh: float = 0.5,
        keep_full: bool = True,
        batch_size: int = 16,
    ):
        self._base = base
        if window is None:
            window = base.eval_spatial_size
        self._window: Tuple[int, int] = (int(window[0]), int(window[1]))
        self._overlap = float(max(0.0, min(overlap, 0.9)))
        self._nms_thresh = float(nms_thresh)
        self._keep_full = bool(keep_full)
        # Run windows through the base in batches when it supports a batched
        # forward (DEIMv2 does). One GPU call per `batch_size` windows turns a
        # ~64k-sequential-pass test set from hours into minutes. Falls back to
        # per-window predict_image otherwise.
        self._batch_size = max(1, int(batch_size))
        self._can_batch = callable(getattr(base, "predict_batch", None))

    @property
    def eval_spatial_size(self) -> Tuple[int, int]:
        return self._base.eval_spatial_size

    def predict_image(self, image_np, orig_size) -> List[dict]:
        H, W = int(image_np.shape[0]), int(image_np.shape[1])
        win_h, win_w = self._window

        # Image already fits in one window — no tiling benefit, defer.
        if H <= win_h and W <= win_w:
            return list(self._base.predict_image(image_np, orig_size))

        stride_h = max(1, int(round(win_h * (1.0 - self._overlap))))
        stride_w = max(1, int(round(win_w * (1.0 - self._overlap))))
        ys = _grid_positions(H, win_h, stride_h)
        xs = _grid_positions(W, win_w, stride_w)

        # Collect every window crop + its top-left offset, then score them
        # (batched when possible). Keeping crops as views avoids copies.
        crops = []
        offsets = []
        for y0 in ys:
            for x0 in xs:
                crops.append(image_np[y0:y0 + win_h, x0:x0 + win_w])
                offsets.append((x0, y0))

        merged: List[dict] = []
        for (x0, y0), dets in zip(offsets, self._score_crops(crops)):
            for det in dets:
                x1, y1, x2, y2 = det["bbox_xyxy"]
                merged.append({
                    "label": int(det["label"]),
                    "score": float(det["score"]),
                    "bbox_xyxy": [x1 + x0, y1 + y0, x2 + x0, y2 + y0],
                })

        if self._keep_full:
            for det in self._base.predict_image(image_np, orig_size):
                merged.append({
                    "label": int(det["label"]),
                    "score": float(det["score"]),
                    "bbox_xyxy": [float(v) for v in det["bbox_xyxy"]],
                })

        return _per_class_nms(merged, self._nms_thresh)

    def _score_crops(self, crops):
        """Yield a detection list per crop, batching base.predict_batch calls."""
        if not self._can_batch:
            for crop in crops:
                ch, cw = int(crop.shape[0]), int(crop.shape[1])
                yield self._base.predict_image(crop, (cw, ch))
            return
        for i in range(0, len(crops), self._batch_size):
            chunk = crops[i:i + self._batch_size]
            sizes = [(int(c.shape[1]), int(c.shape[0])) for c in chunk]
            for dets in self._base.predict_batch(chunk, sizes):
                yield dets
