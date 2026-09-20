"""Coordinate spaces used by detector prediction.

Detector inference is allowed to run in a scaled *prediction space* for
throughput, while KWCoco prediction annotations are always emitted in the
native source-image coordinate system.  Keeping that transform explicit makes
resolution a first-class part of inference rather than an incidental resize.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def _coerce_scale_xy(value) -> tuple[float, float]:
    if value is None or value == "native":
        return (1.0, 1.0)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"native", "1", "1.0"}:
            return (1.0, 1.0)
        if "," in text:
            parts = [float(p.strip()) for p in text.split(",")]
            if len(parts) != 2:
                raise ValueError(f"prediction scale must be scalar or sx,sy; got {value!r}")
            scale_xy = (parts[0], parts[1])
        else:
            scalar = float(text)
            scale_xy = (scalar, scalar)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parts = list(value)
        if len(parts) != 2:
            raise ValueError(f"prediction scale must be scalar or sx,sy; got {value!r}")
        scale_xy = (float(parts[0]), float(parts[1]))
    else:
        scalar = float(value)
        scale_xy = (scalar, scalar)
    if not all(v > 0 for v in scale_xy):
        raise ValueError(f"prediction scale must be positive; got {value!r}")
    return scale_xy


@dataclass(frozen=True)
class PredictionSpace:
    """Relationship between native image space and detector prediction space.

    ``requested_scale_xy`` is the requested resize. ``scale_xy`` is the exact
    scale after integer output dimensions are chosen, so inverse geometry is
    stable even when dimensions do not divide evenly.
    """

    native_hw: tuple[int, int]
    prediction_hw: tuple[int, int]
    requested_scale_xy: tuple[float, float]
    scale_xy: tuple[float, float]

    @classmethod
    def from_scale(cls, native_hw, scale=1.0):
        native_h, native_w = map(int, native_hw)
        if native_h <= 0 or native_w <= 0:
            raise ValueError(f"native image dimensions must be positive: {native_hw!r}")
        req_sx, req_sy = _coerce_scale_xy(scale)
        pred_w = max(1, int(round(native_w * req_sx)))
        pred_h = max(1, int(round(native_h * req_sy)))
        sx = pred_w / native_w
        sy = pred_h / native_h
        return cls(
            native_hw=(native_h, native_w),
            prediction_hw=(pred_h, pred_w),
            requested_scale_xy=(req_sx, req_sy),
            scale_xy=(sx, sy),
        )

    @property
    def is_native(self) -> bool:
        return self.prediction_hw == self.native_hw

    @property
    def native_from_prediction_xy(self) -> tuple[float, float]:
        sx, sy = self.scale_xy
        return (1.0 / sx, 1.0 / sy)

    def box_to_native_xyxy(self, box):
        inv_x, inv_y = self.native_from_prediction_xy
        x1, y1, x2, y2 = map(float, box)
        return [x1 * inv_x, y1 * inv_y, x2 * inv_x, y2 * inv_y]

    def box_to_prediction_xyxy(self, box):
        sx, sy = self.scale_xy
        x1, y1, x2, y2 = map(float, box)
        return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]

    def warp_multipolygon_to_native(self, mpoly):
        """Warp a kwimage polygon object into native source-image space."""
        if self.is_native:
            return mpoly
        import kwimage

        transform = kwimage.Affine.scale(self.native_from_prediction_xy)
        return mpoly.warp(transform)

    def to_dict(self) -> dict:
        return {
            "native_hw": list(self.native_hw),
            "prediction_hw": list(self.prediction_hw),
            "requested_scale_xy": list(self.requested_scale_xy),
            "actual_scale_xy": list(self.scale_xy),
            "native_from_prediction_xy": list(self.native_from_prediction_xy),
        }
