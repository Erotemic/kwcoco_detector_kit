"""ONNX Runtime adapter for RF-DETR raw detection/segmentation exports."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from kwcoco_detector_kit.predictors.onnx import _cuda_device_id, _providers_for_device


class RFDETROnnxPredictor:
    """Decode RF-DETR's raw ``dets``/``labels``/``masks`` ONNX contract.

    RF-DETR exports raw normalized ``cxcywh`` boxes, sigmoid class logits, and
    native mask logits. This adapter mirrors the KDK PyTorch path: stable top-k
    over the exported query/class grid, discard the final unused/background
    slot for contiguous custom datasets, scale boxes to the requested source
    size, and bilinearly resize mask logits before the zero-logit threshold.
    """

    def __init__(
        self,
        onnx_fpath: str | Path,
        *,
        device: str = "cpu",
        score_thresh: float = 0.0,
        category_names=None,
        providers=None,
    ):
        import onnxruntime as ort

        self.onnx_fpath = Path(onnx_fpath)
        self._score_thresh = float(score_thresh)
        self._category_names = list(category_names or [])
        if providers is None:
            providers = _providers_for_device(str(device))
            device_id = _cuda_device_id(str(device))
            if device_id is not None and "CUDAExecutionProvider" in providers:
                providers = [
                    ("CUDAExecutionProvider", {"device_id": device_id}),
                    "CPUExecutionProvider",
                ]
        self._session = ort.InferenceSession(str(self.onnx_fpath), providers=providers)
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        shape = inp.shape
        self._eval_h = int(shape[-2])
        self._eval_w = int(shape[-1])
        outputs = self._session.get_outputs()
        self._output_names = [out.name for out in outputs]
        self._box_idx = self._find_output("dets", fallback=0)
        self._logit_idx = self._find_output("labels", fallback=1)
        self._mask_idx = self._find_output("masks", fallback=None)

    def _find_output(self, token, fallback=None):
        for idx, name in enumerate(self._output_names):
            if token in name.lower():
                return idx
        return fallback

    @property
    def eval_spatial_size(self):
        return self._eval_h, self._eval_w

    @property
    def category_names(self):
        return self._category_names

    @property
    def supports_masks(self):
        return self._mask_idx is not None

    def _preprocess_one(self, image_np):
        from rfdetr.export._resize import _bilinear_resize_half_pixel

        arr = np.asarray(image_np)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        elif arr.shape[2] == 4:
            arr = arr[..., :3]
        chw = arr.astype(np.float32).transpose(2, 0, 1) / 255.0
        chw = _bilinear_resize_half_pixel(chw, self._eval_h, self._eval_w)
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        return (chw - mean) / std

    def _decode_one(self, boxes_cwh, logits, masks, orig_size):
        from rfdetr.export._resize import _bilinear_resize_half_pixel
        from rfdetr.export._topk import _select_topk_multiclass

        logits = np.asarray(logits)
        one = np.asarray(1, dtype=logits.dtype)
        scores_all = one / (one + np.exp(-logits.clip(-88, 88)))
        # KDK-trained RF-DETR follows the standard num_classes + background
        # layout; the final exported logit is background.
        # Upstream RF-DETR high-level PyTorch prediction performs top-k over
        # the complete exported grid before KDK removes the final custom-data
        # sentinel. Preserve that ordering here so parity is meaningful at the
        # actual detector-record boundary (including low score thresholds).
        scores, labels, query_idx = _select_topk_multiclass(
            scores_all,
            self._score_thresh,
            num_select=int(np.asarray(boxes_cwh).shape[0]),
        )
        if scores_all.shape[1] == len(self._category_names) + 1:
            keep = labels != (scores_all.shape[1] - 1)
            scores = scores[keep]
            labels = labels[keep]
            query_idx = query_idx[keep]
        elif self._category_names and scores_all.shape[1] != len(self._category_names):
            raise ValueError(
                "RF-DETR ONNX class layout does not match packaged category order: "
                f"logit_slots={scores_all.shape[1]} categories={len(self._category_names)}"
            )
        selected = np.asarray(boxes_cwh)[query_idx]
        target_w, target_h = map(int, orig_size)
        if len(selected):
            cx, cy, bw, bh = selected.T
            xyxy = np.stack(
                [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1
            ).astype(np.float32)
            xyxy *= np.asarray([target_w, target_h, target_w, target_h], dtype=np.float32)
            xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], 0, target_w)
            xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], 0, target_h)
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)

        selected_masks = None
        if masks is not None:
            mask_logits = np.asarray(masks)[query_idx]
            if mask_logits.ndim == 4 and mask_logits.shape[1] == 1:
                mask_logits = mask_logits[:, 0]
            if mask_logits.ndim != 3:
                raise ValueError(
                    f"RF-DETR ONNX mask output must decode to KxHxW; got {mask_logits.shape}"
                )
            selected_masks = _bilinear_resize_half_pixel(
                mask_logits.astype(np.float32), target_h, target_w
            ) > 0.0

        records = []
        for idx, score in enumerate(scores):
            record = {
                "label": int(labels[idx]),
                "score": float(score),
                "bbox_xyxy": [float(v) for v in xyxy[idx]],
            }
            if selected_masks is not None:
                record["mask"] = selected_masks[idx]
            records.append(record)
        return records

    def predict_batch(self, images_np, orig_sizes):
        if not images_np:
            return []
        batch = np.stack([self._preprocess_one(img) for img in images_np], axis=0).astype(np.float32)
        outputs = self._session.run(None, {self._input_name: batch})
        boxes = outputs[self._box_idx]
        logits = outputs[self._logit_idx]
        masks = outputs[self._mask_idx] if self._mask_idx is not None else None
        return [
            self._decode_one(
                boxes[idx], logits[idx], None if masks is None else masks[idx], orig_sizes[idx]
            )
            for idx in range(len(images_np))
        ]

    def predict_image(self, image_np, orig_size=None):
        if orig_size is None:
            h, w = image_np.shape[:2]
            orig_size = (w, h)
        return self.predict_batch([image_np], [orig_size])[0]
