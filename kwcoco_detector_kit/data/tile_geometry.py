"""Geometry helpers for segmentation-preserving tile extraction.

The tiler has two geometry paths:

* polygon / multipolygon inputs are scaled in continuous coordinates and
  clipped with an exact polygon intersection;
* COCO/kwimage RLE inputs are decoded, resized with nearest-neighbour
  interpolation, cropped, and encoded again.

Both paths return geometry in tile coordinates and derive bbox/area from the
surviving segmentation.  This module deliberately has no project-specific
category or role policy; callers decide whether a surviving fragment is a
usable positive or makes the window unsafe background.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ClippedGeometry:
    """Target geometry surviving a scale-and-crop operation."""

    segmentation: Any | None
    bbox_xywh: list[float]
    area: float
    scaled_source_area: float
    visible_fraction: float
    geometry_kind: str


def segmentation_kind(data: Any) -> str:
    """Return a stable coarse name for a kwcoco segmentation representation."""
    if data is None:
        return "none"
    if isinstance(data, dict):
        keys = set(data)
        if "counts" in keys and ("size" in keys or "shape" in keys):
            return "rle"
        if "exterior" in keys:
            return "polygon"
        return "unknown_dict"
    if isinstance(data, list):
        return "polygon"
    return type(data).__name__


def _jsonable_rle(data: dict) -> dict:
    """Normalize tuple-valued kwimage RLE fields for JSON serialization."""
    out = dict(data)
    for key in ("shape", "size"):
        if isinstance(out.get(key), tuple):
            out[key] = list(out[key])
    counts = out.get("counts")
    if isinstance(counts, bytes):
        out["counts"] = counts.decode("ascii")
    return out


def _mask_bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def _coerce_scale(scale) -> tuple[float, float]:
    if isinstance(scale, (tuple, list)):
        if len(scale) != 2:
            raise ValueError(f"scale must be scalar or length 2, got {scale!r}")
        return float(scale[0]), float(scale[1])
    value = float(scale)
    return value, value


def _polygonal_part(shapely_geom):
    """Drop zero-area line/point debris from a shapely intersection."""
    from shapely.geometry import MultiPolygon, Polygon

    if shapely_geom.is_empty:
        return None
    if isinstance(shapely_geom, (Polygon, MultiPolygon)):
        return shapely_geom if shapely_geom.area > 0 else None
    polys = []
    for part in getattr(shapely_geom, "geoms", []):
        if isinstance(part, Polygon) and part.area > 0:
            polys.append(part)
        elif isinstance(part, MultiPolygon):
            polys.extend(p for p in part.geoms if p.area > 0)
    if not polys:
        return None
    return MultiPolygon(polys) if len(polys) > 1 else polys[0]


def _clip_polygon(segmentation, *, scale, crop_xyxy, source_dims):
    import kwimage
    from shapely import affinity
    from shapely.geometry import box as shapely_box

    sx, sy = _coerce_scale(scale)
    x0, y0, x1, y1 = map(float, crop_xyxy)
    seg = kwimage.Segmentation.coerce(segmentation, dims=source_dims)
    geom = seg.data.to_shapely(fix=True)
    scaled = affinity.scale(geom, xfact=sx, yfact=sy, origin=(0, 0))
    scaled_area = float(scaled.area)
    if scaled_area <= 0:
        return None
    clipped = _polygonal_part(scaled.intersection(shapely_box(x0, y0, x1, y1)))
    if clipped is None:
        return None
    local = affinity.translate(clipped, xoff=-x0, yoff=-y0)
    area = float(local.area)
    if area <= 0:
        return None
    minx, miny, maxx, maxy = local.bounds
    kwgeom = (
        kwimage.Polygon.from_shapely(local)
        if local.geom_type == "Polygon"
        else kwimage.MultiPolygon.from_shapely(local)
    )
    return ClippedGeometry(
        segmentation=kwgeom.to_coco(style="new"),
        bbox_xywh=[float(minx), float(miny), float(maxx - minx), float(maxy - miny)],
        area=area,
        scaled_source_area=scaled_area,
        visible_fraction=area / scaled_area,
        geometry_kind="polygon",
    )

def _clip_rle(segmentation, *, scale, crop_xyxy, source_dims, output_dims):
    import cv2
    import kwimage

    sx, sy = _coerce_scale(scale)
    src_h, src_w = map(int, source_dims)
    out_h, out_w = map(int, output_dims)
    seg = kwimage.Segmentation.coerce(segmentation, dims=(src_h, src_w))
    source_mask = seg.to_mask(dims=(src_h, src_w)).to_c_mask().data.astype(np.uint8)
    scaled_w = max(1, int(round(src_w * sx)))
    scaled_h = max(1, int(round(src_h * sy)))
    scaled_mask = cv2.resize(
        source_mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST
    )
    scaled_area = float(scaled_mask.sum())
    if scaled_area <= 0:
        return None

    x0, y0, x1, y1 = [int(round(v)) for v in crop_xyxy]
    ix0 = max(0, x0)
    iy0 = max(0, y0)
    ix1 = min(scaled_w, x1)
    iy1 = min(scaled_h, y1)
    tile_mask = np.zeros((out_h, out_w), dtype=np.uint8)
    if ix1 > ix0 and iy1 > iy0:
        dx0 = ix0 - x0
        dy0 = iy0 - y0
        dx1 = min(out_w, dx0 + (ix1 - ix0))
        dy1 = min(out_h, dy0 + (iy1 - iy0))
        tile_mask[dy0:dy1, dx0:dx1] = scaled_mask[
            iy0:iy0 + (dy1 - dy0), ix0:ix0 + (dx1 - dx0)
        ]
    area = float(tile_mask.sum())
    if area <= 0:
        return None
    encoded = kwimage.Mask(tile_mask, "c_mask").to_coco(style="new")
    return ClippedGeometry(
        segmentation=_jsonable_rle(encoded),
        bbox_xywh=_mask_bbox(tile_mask),
        area=area,
        scaled_source_area=scaled_area,
        visible_fraction=area / scaled_area,
        geometry_kind="rle",
    )


def clip_segmentation(
    segmentation,
    *,
    scale,
    crop_xyxy,
    source_dims,
    output_dims,
) -> ClippedGeometry | None:
    """Scale and crop a kwcoco segmentation into tile coordinates.

    Args:
        segmentation: Any representation accepted by
            :class:`kwimage.Segmentation`.
        scale: Scalar or ``(sx, sy)`` source-to-scaled-image factor.
        crop_xyxy: Crop extent in the scaled-image coordinate frame.
        source_dims: Source ``(height, width)``.
        output_dims: Tile ``(height, width)`` including any edge padding.

    Returns:
        ``None`` when no positive-area target content intersects the crop;
        otherwise transformed segmentation, bbox, area and visible fraction.
    """
    kind = segmentation_kind(segmentation)
    if kind == "rle":
        return _clip_rle(
            segmentation,
            scale=scale,
            crop_xyxy=crop_xyxy,
            source_dims=source_dims,
            output_dims=output_dims,
        )
    if kind == "polygon":
        return _clip_polygon(
            segmentation,
            scale=scale,
            crop_xyxy=crop_xyxy,
            source_dims=source_dims,
        )
    raise TypeError(f"unsupported segmentation representation: {kind}")


def clip_bbox_geometry(bbox_xywh, *, scale, crop_xyxy) -> ClippedGeometry | None:
    """Fallback geometry for annotations that have a bbox but no mask."""
    sx, sy = _coerce_scale(scale)
    bx, by, bw, bh = map(float, bbox_xywh)
    x0, y0, x1, y1 = map(float, crop_xyxy)
    ax0 = bx * sx
    ay0 = by * sy
    ax1 = (bx + bw) * sx
    ay1 = (by + bh) * sy
    ix0 = max(ax0, x0)
    iy0 = max(ay0, y0)
    ix1 = min(ax1, x1)
    iy1 = min(ay1, y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    area = float((ix1 - ix0) * (iy1 - iy0))
    scaled_area = float(max(0.0, (ax1 - ax0) * (ay1 - ay0)))
    return ClippedGeometry(
        segmentation=None,
        bbox_xywh=[ix0 - x0, iy0 - y0, ix1 - ix0, iy1 - iy0],
        area=area,
        scaled_source_area=scaled_area,
        visible_fraction=(area / scaled_area) if scaled_area else 0.0,
        geometry_kind="bbox",
    )
