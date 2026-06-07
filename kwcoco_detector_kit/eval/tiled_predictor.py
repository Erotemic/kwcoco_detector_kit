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

import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from kwcoco_detector_kit.data.tile import _grid_positions
from kwcoco_detector_kit._lineprofile import profile


@profile
def _per_class_nms(detections: List[dict], iou_thresh: float) -> List[dict]:
    """Greedy IoU NMS applied independently within each class label.

    Uses torchvision.ops.batched_nms (vectorized C++/CUDA, same greedy
    algorithm) when available — critical for tiled eval, where merging
    ~16k detections/image with a pure-Python O(n^2) loop dominated runtime
    (53 of 75 min on the gen005 pup test set). Falls back to the numpy
    implementation if torchvision isn't importable.
    """
    if not detections:
        return detections
    try:
        import torch
        from torchvision.ops import batched_nms
        # Build arrays via numpy first — torch.tensor(list-of-lists) is much
        # slower than np.array + from_numpy for ~16k rows (the conversion,
        # not the NMS, was otherwise the cost).
        boxes = np.array([d["bbox_xyxy"] for d in detections], dtype=np.float32)
        scores = np.array([d["score"] for d in detections], dtype=np.float32)
        idxs = np.array([d["label"] for d in detections], dtype=np.int64)
        keep = batched_nms(
            torch.from_numpy(boxes), torch.from_numpy(scores),
            torch.from_numpy(idxs), float(iou_thresh)).tolist()
        return [detections[i] for i in keep]
    except Exception:
        return _per_class_nms_numpy(detections, iou_thresh)


def _per_class_nms_numpy(detections: List[dict], iou_thresh: float) -> List[dict]:
    """Pure-numpy greedy per-class NMS (fallback when torchvision is absent)."""
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


def _to_detections(obj):
    """Coerce a predictor result (kwimage.Detections OR list-of-dicts) to a
    kwimage.Detections. New columnar predict_batch returns Detections; the
    per-window predict_image fallback and keep_full still return dicts."""
    import kwimage
    if isinstance(obj, kwimage.Detections):
        return obj
    if not obj:
        return kwimage.Detections(
            boxes=kwimage.Boxes(np.zeros((0, 4), np.float32), "ltrb"),
            scores=np.zeros(0, np.float32), class_idxs=np.zeros(0, np.int64))
    boxes = np.array([d["bbox_xyxy"] for d in obj], dtype=np.float32)
    scores = np.array([d["score"] for d in obj], dtype=np.float32)
    labels = np.array([d["label"] for d in obj], dtype=np.int64)
    return kwimage.Detections(boxes=kwimage.Boxes(boxes, "ltrb"),
                              scores=scores, class_idxs=labels)


@profile
def _nms_detections(det, iou_thresh: float):
    """Per-class NMS on a kwimage.Detections via torchvision.ops.batched_nms.

    Operates on the columnar arrays the Detections already holds — no
    list-of-dicts -> np.array conversion (that conversion, not the NMS, was
    the bespoke overhead). Returns the reduced Detections.
    """
    n = len(det)
    if n <= 1:
        return det
    boxes = np.ascontiguousarray(det.boxes.to_ltrb().data, dtype=np.float32)
    scores = np.ascontiguousarray(det.scores, dtype=np.float32)
    idxs = np.ascontiguousarray(det.class_idxs, dtype=np.int64)
    try:
        import torch
        from torchvision.ops import batched_nms
        keep = batched_nms(torch.from_numpy(boxes), torch.from_numpy(scores),
                           torch.from_numpy(idxs), float(iou_thresh)).cpu().numpy()
    except Exception:
        # numpy per-class greedy fallback
        keep_list = []
        for c in np.unique(idxs):
            sel = np.where(idxs == c)[0]
            local = _nms_indices(boxes[sel], scores[sel], iou_thresh)
            keep_list.extend(sel[local].tolist())
        keep = np.array(sorted(keep_list), dtype=np.int64)
    return det.take(keep)


def _detections_to_dicts(det) -> List[dict]:
    """Final boundary conversion (over the small capped set only)."""
    boxes = det.boxes.to_ltrb().data
    scores = det.scores
    labels = det.class_idxs
    out: List[dict] = []
    for i in range(len(det)):
        x1, y1, x2, y2 = boxes[i]
        out.append({
            "label": int(labels[i]),
            "score": float(scores[i]),
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
        })
    return out


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
        max_dets: Optional[int] = None,
        pre_nms_score_thresh: float = 0.0,
        pre_nms_topk: Optional[int] = None,
        per_window_nms: bool = True,
    ):
        self._base = base
        if window is None:
            window = base.eval_spatial_size
        self._window: Tuple[int, int] = (int(window[0]), int(window[1]))
        self._overlap = float(max(0.0, min(overlap, 0.9)))
        self._nms_thresh = float(nms_thresh)
        self._keep_full = bool(keep_full)
        # Per-window reduction BEFORE the global cross-window merge — this is
        # what keeps the merge cheap. Each window emits ~300 raw detections;
        # most are low-score junk. Applied in order per window:
        #   1. score floor  (lossless at the eval's score_thresh)
        #   2. per-window NMS (dedupe within the window — cheap, ~300 boxes;
        #      the global pass then only resolves cross-window overlaps)
        #   3. top-K by score (cap per window)
        # so the global NMS sees a fraction of the ~16k/image it used to.
        self._pre_nms_score_thresh = float(pre_nms_score_thresh)
        self._pre_nms_topk = int(pre_nms_topk) if pre_nms_topk else None
        self._per_window_nms = bool(per_window_nms)
        # Cap detections per image (top-K by score, after the cross-window
        # NMS merge). Tiled eval over many windows emits ~thousands of mostly
        # low-score detections/image (~16k for the sea-lion test set), which
        # bloats the pred bundle dump AND the downstream kwcoco-eval AP pass
        # (both O(n_detections)). A generous cap kills the junk tail with
        # negligible AP@0.5 impact. None = no cap (kit-agnostic default).
        self._max_dets = int(max_dets) if max_dets else None
        # Run windows through the base in batches when it supports a batched
        # forward (DEIMv2 does). One GPU call per `batch_size` windows turns a
        # ~64k-sequential-pass test set from hours into minutes. Falls back to
        # per-window predict_image otherwise.
        self._batch_size = max(1, int(batch_size))
        self._can_batch = callable(getattr(base, "predict_batch", None))
        # Accumulated wall-time (seconds) for the eval timing breakdown:
        # window inference (base forward) vs the cross-window NMS merge.
        self.t_infer = 0.0
        self.t_nms = 0.0

    @property
    def eval_spatial_size(self) -> Tuple[int, int]:
        return self._base.eval_spatial_size

    @profile
    def predict_image(self, image_np, orig_size) -> List[dict]:
        # Serial path: GPU inference then CPU merge+NMS. The eval loop can
        # instead call _infer_windows (GPU) and _merge_and_nms (CPU) on
        # separate stages to overlap them across images (see kwcoco_eval).
        raw = self._infer_windows(image_np, orig_size)
        if raw.get("deferred") is not None:
            return raw["deferred"]
        return self._merge_and_nms(raw)

    @profile
    def _infer_windows(self, image_np, orig_size) -> dict:
        """GPU phase: run every window (+ optional whole-image) through the
        base model. Returns the RAW per-window detections (crop coords) plus
        their offsets — NO per-window reduction, merge, or NMS (those are the
        CPU phase, _merge_and_nms). Keeping this phase pure-GPU lets the eval
        loop overlap it with the CPU NMS of the previous image.
        """
        H, W = int(image_np.shape[0]), int(image_np.shape[1])
        win_h, win_w = self._window

        # Image already fits in one window — no tiling benefit, defer.
        if H <= win_h and W <= win_w:
            return {"deferred": list(self._base.predict_image(image_np, orig_size))}

        stride_h = max(1, int(round(win_h * (1.0 - self._overlap))))
        stride_w = max(1, int(round(win_w * (1.0 - self._overlap))))
        ys = _grid_positions(H, win_h, stride_h)
        xs = _grid_positions(W, win_w, stride_w)

        crops, offsets = [], []
        for y0 in ys:
            for x0 in xs:
                crops.append(image_np[y0:y0 + win_h, x0:x0 + win_w])
                offsets.append((x0, y0))

        window_dets = list(self._score_crops(crops))  # GPU
        full_dets = None
        if self._keep_full:
            _t = time.perf_counter()
            full_dets = self._base.predict_image(image_np, orig_size)  # GPU
            self.t_infer += time.perf_counter() - _t
        return {"deferred": None, "offsets": offsets,
                "window_dets": window_dets, "full_dets": full_dets}

    @profile
    def _merge_and_nms(self, raw: dict) -> List[dict]:
        """CPU phase, fully columnar (kwimage.Detections): per-window reduce ->
        vectorized box offset -> concat -> global NMS -> cap, building dicts
        only for the small final set. The heavy ops are array math + the
        GIL-releasing batched_nms, so this runs truly parallel to the next
        image's _infer_windows when the eval loop pipelines it.
        """
        import kwimage
        _t = time.perf_counter()
        parts = []
        for (x0, y0), det in zip(raw["offsets"], raw["window_dets"]):
            det = self._reduce_window(_to_detections(det))
            if len(det):
                parts.append(det.translate((x0, y0)))  # vectorized
        full = raw.get("full_dets")
        if full is not None:
            full = _to_detections(full)
            if len(full):
                parts.append(full)
        if not parts:
            return []
        merged = parts[0] if len(parts) == 1 else kwimage.Detections.concatenate(parts)
        merged = _nms_detections(merged, self._nms_thresh)
        if self._max_dets is not None and len(merged) > self._max_dets:
            order = np.argsort(merged.scores)[::-1][:self._max_dets]
            merged = merged.take(np.ascontiguousarray(order))
        out = _detections_to_dicts(merged)
        self.t_nms += time.perf_counter() - _t
        return out

    @profile
    def _reduce_window(self, det):
        """Score-floor -> per-window NMS -> top-K, on one window's Detections."""
        if self._pre_nms_score_thresh > 0.0:
            det = det.compress(det.scores >= self._pre_nms_score_thresh)
        if self._per_window_nms and len(det) > 1:
            det = _nms_detections(det, self._nms_thresh)
        if self._pre_nms_topk is not None and len(det) > self._pre_nms_topk:
            order = np.argsort(det.scores)[::-1][:self._pre_nms_topk]
            det = det.take(np.ascontiguousarray(order))
        return det

    @profile
    def _score_crops(self, crops):
        """Yield a kwimage.Detections per crop, batching predict_batch calls."""
        if not self._can_batch:
            for crop in crops:
                ch, cw = int(crop.shape[0]), int(crop.shape[1])
                _t = time.perf_counter()
                dets = self._base.predict_image(crop, (cw, ch))
                self.t_infer += time.perf_counter() - _t
                yield _to_detections(dets)
            return
        for i in range(0, len(crops), self._batch_size):
            chunk = crops[i:i + self._batch_size]
            sizes = [(int(c.shape[1]), int(c.shape[0])) for c in chunk]
            _t = time.perf_counter()
            results = self._base.predict_batch(chunk, sizes)
            self.t_infer += time.perf_counter() - _t
            for det in results:
                yield det
