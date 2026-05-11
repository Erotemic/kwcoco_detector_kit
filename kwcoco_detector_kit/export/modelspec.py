"""
Modelspec sidecar — a small JSON file written next to every exported
``.onnx`` describing the deploy contract:

- input_hw: model's fixed input H/W (for fixed-shape ONNX exports)
- input_channels: 3 for RGB; ``len(channels.split('|'))`` for multispectral
- input_dtype: 'float32' (preprocessed image; not the raw uint8 buffer)
- input_layout: 'NCHW' | 'NHWC'
- preprocess: {scale: 1/255, normalize_mean: [...], normalize_std: [...]}
- postprocess: {score_thresh, nms_iou_thresh, topk}
- modelId: canonical cross-device ID
- meta: variant, category_name, candidate_kind, generated_at, kit_version
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_NORMALIZE_MEAN = (0.0, 0.0, 0.0)
DEFAULT_NORMALIZE_STD = (1.0, 1.0, 1.0)


def write_modelspec(
    onnx_fpath: Path,
    *,
    input_hw: tuple,
    input_channels: int = 3,
    input_dtype: str = "float32",
    input_layout: str = "NCHW",
    preprocess_scale: float = 1.0 / 255.0,
    normalize_mean: Iterable[float] = DEFAULT_NORMALIZE_MEAN,
    normalize_std: Iterable[float] = DEFAULT_NORMALIZE_STD,
    postprocess_score_thresh: float = 0.30,
    postprocess_nms_iou_thresh: float = 0.50,
    postprocess_topk: int = 100,
    model_id: Optional[str] = None,
    variant: str = "",
    category_name: str = "",
    candidate_kind: str = "",
    extra_meta: Optional[dict] = None,
) -> Path:
    """Write ``<onnx_fpath>.modelspec.json`` next to ``onnx_fpath``."""
    from kwcoco_detector_kit import __version__ as _kit_version

    onnx_fpath = Path(onnx_fpath)
    H, W = int(input_hw[0]), int(input_hw[1])
    if model_id is None:
        model_id = f"{variant}-h{H}w{W}" if variant else onnx_fpath.stem

    spec = {
        "modelId": model_id,
        "input": {
            "shape_hw": [H, W],
            "channels": int(input_channels),
            "dtype": str(input_dtype),
            "layout": str(input_layout),
        },
        "preprocess": {
            "scale": float(preprocess_scale),
            "normalize_mean": [float(v) for v in normalize_mean],
            "normalize_std": [float(v) for v in normalize_std],
        },
        "postprocess": {
            "score_thresh": float(postprocess_score_thresh),
            "nms_iou_thresh": float(postprocess_nms_iou_thresh),
            "topk": int(postprocess_topk),
        },
        "meta": {
            "variant": variant,
            "category_name": category_name,
            "candidate_kind": candidate_kind,
            "kit_version": _kit_version,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **(extra_meta or {}),
        },
    }
    out_fpath = onnx_fpath.with_suffix(".modelspec.json")
    out_fpath.write_text(json.dumps(spec, indent=2))
    return out_fpath
