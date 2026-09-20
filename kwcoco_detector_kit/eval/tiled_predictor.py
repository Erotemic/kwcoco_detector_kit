"""Compatibility imports for the generic tiled predictor.

The implementation moved to :mod:`kwcoco_detector_kit.predictors.tiled` because
source-space tiling is an inference primitive, not an evaluation-only concern.

Keep the historical import surface intact while downstream callers migrate.
In particular, an ``import *`` forwarding shim is insufficient here because
Python intentionally omits leading-underscore helpers such as ``_nms_indices``.
"""
from kwcoco_detector_kit.predictors.tiled import (
    TiledPredictor,
    _detections_to_dicts,
    _nms_detections,
    _nms_indices,
    _per_class_nms,
    _per_class_nms_numpy,
    _to_detections,
)

__all__ = [
    "TiledPredictor",
    "_detections_to_dicts",
    "_nms_detections",
    "_nms_indices",
    "_per_class_nms",
    "_per_class_nms_numpy",
    "_to_detections",
]
