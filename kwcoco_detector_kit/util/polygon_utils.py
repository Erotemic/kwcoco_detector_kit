"""Polygon and mask helpers for postprocessing detector/segmenter outputs."""
from __future__ import annotations

import numpy as np


def expand_box_xyxy(box_xyxy, padding, image_shape):
    """Expand an (x1,y1,x2,y2) box by ``padding`` pixels, clamped to image bounds."""
    x1, y1, x2, y2 = map(float, box_xyxy)
    height, width = image_shape[0:2]
    return [
        max(0.0, x1 - padding),
        max(0.0, y1 - padding),
        min(float(width), x2 + padding),
        min(float(height), y2 + padding),
    ]


def _safe_poly_area(poly):
    try:
        return float(poly.area)
    except Exception:
        return 0.0


def mask_to_multi_polygon(
    mask,
    *,
    polygon_simplify: float = 0.0,
    min_component_area: float = 0.0,
    keep_largest_component: bool = True,
):
    """Convert a boolean mask array to a ``kwimage.MultiPolygon``.

    Args:
        mask: 2-D array-like castable to bool.
        polygon_simplify: Simplification tolerance in pixels (0 = off).
        min_component_area: Drop connected components smaller than this area.
        keep_largest_component: Retain only the largest polygon component.

    Returns:
        ``kwimage.MultiPolygon`` (may be empty if no valid contours found).
    """
    import kwimage

    bool_mask = np.asarray(mask).astype(bool)
    mpoly = kwimage.Mask.coerce(bool_mask).to_multi_polygon()
    polygons = [p for p in mpoly.data if _safe_poly_area(p) >= max(0.0, min_component_area)]
    if keep_largest_component and polygons:
        polygons = [max(polygons, key=_safe_poly_area)]
    if polygon_simplify:
        polygons = [p.simplify(polygon_simplify) for p in polygons]
        polygons = [p for p in polygons if _safe_poly_area(p) > 0]
    return kwimage.MultiPolygon(polygons)


def segmentation_to_coco(segmentation):
    """Convert a kwimage segmentation object to a COCO-serialisable dict."""
    if segmentation is None:
        return None
    if hasattr(segmentation, "to_coco"):
        return segmentation.to_coco(style="new")
    return segmentation
