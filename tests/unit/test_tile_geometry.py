"""Focused segmentation geometry tests for tile extraction."""
from __future__ import annotations

import numpy as np


def _decode_mask(segmentation, dims):
    import kwimage

    return (
        kwimage.Segmentation.coerce(segmentation, dims=dims)
        .to_mask(dims=dims)
        .to_c_mask()
        .data.astype(bool)
    )


def test_polygon_crossing_tile_boundary_is_clipped():
    from kwcoco_detector_kit.data.tile_geometry import clip_segmentation

    polygon = {
        "exterior": [[10, 10], [70, 10], [70, 50], [10, 50], [10, 10]],
        "interiors": [],
    }
    got = clip_segmentation(
        polygon,
        scale=1.0,
        crop_xyxy=(40, 0, 80, 40),
        source_dims=(100, 100),
        output_dims=(40, 40),
    )
    assert got is not None
    assert got.geometry_kind == "polygon"
    assert np.allclose(got.bbox_xywh, [0, 10, 30, 30])
    assert np.isclose(got.area, 900.0)
    assert np.isclose(got.visible_fraction, 900.0 / 2400.0)


def test_polygon_scaling_then_crop():
    from kwcoco_detector_kit.data.tile_geometry import clip_segmentation

    polygon = {
        "exterior": [[20, 20], [60, 20], [60, 60], [20, 60], [20, 20]],
        "interiors": [],
    }
    got = clip_segmentation(
        polygon,
        scale=0.5,
        crop_xyxy=(15, 5, 40, 30),
        source_dims=(100, 100),
        output_dims=(25, 25),
    )
    assert got is not None
    # Scaled polygon is [10:30, 10:30], crop origin is (15, 5).
    assert np.allclose(got.bbox_xywh, [0, 5, 15, 20])
    assert np.isclose(got.area, 300.0)


def test_polygon_hole_is_preserved():
    from kwcoco_detector_kit.data.tile_geometry import clip_segmentation

    polygon = {
        "exterior": [[0, 0], [40, 0], [40, 40], [0, 40], [0, 0]],
        "interiors": [[[10, 10], [30, 10], [30, 30], [10, 30], [10, 10]]],
    }
    got = clip_segmentation(
        polygon,
        scale=1.0,
        crop_xyxy=(0, 0, 40, 40),
        source_dims=(40, 40),
        output_dims=(40, 40),
    )
    assert got is not None
    assert np.isclose(got.area, 1200.0)
    assert got.segmentation["interiors"], "the clipped polygon lost its hole"


def test_rle_scale_crop_matches_nearest_reference():
    import cv2
    import kwimage
    from kwcoco_detector_kit.data.tile_geometry import clip_segmentation

    source = np.zeros((12, 16), dtype=np.uint8)
    source[2:10, 4:13] = 1
    source[5:7, 7:9] = 0
    rle = kwimage.Mask(source, "c_mask").to_coco(style="new")

    scale = 1.5
    crop = (5, 4, 17, 14)
    got = clip_segmentation(
        rle,
        scale=scale,
        crop_xyxy=crop,
        source_dims=source.shape,
        output_dims=(10, 12),
    )
    assert got is not None

    scaled = cv2.resize(
        source,
        (round(source.shape[1] * scale), round(source.shape[0] * scale)),
        interpolation=cv2.INTER_NEAREST,
    )
    expected = scaled[crop[1]:crop[3], crop[0]:crop[2]].astype(bool)
    actual = _decode_mask(got.segmentation, (10, 12))
    assert np.array_equal(actual, expected)
    assert got.area == float(expected.sum())


def test_multipolygon_survives_as_multiple_parts():
    import kwimage
    from kwcoco_detector_kit.data.tile_geometry import clip_segmentation

    p1 = kwimage.Polygon(exterior=[[0, 0], [8, 0], [8, 8], [0, 8], [0, 0]])
    p2 = kwimage.Polygon(exterior=[[20, 20], [30, 20], [30, 30], [20, 30], [20, 20]])
    multi = kwimage.MultiPolygon([p1, p2]).to_coco(style="new")
    got = clip_segmentation(
        multi,
        scale=1.0,
        crop_xyxy=(0, 0, 40, 40),
        source_dims=(40, 40),
        output_dims=(40, 40),
    )
    assert got is not None
    coerced = kwimage.Segmentation.coerce(got.segmentation).to_multi_polygon()
    assert len(coerced.data) == 2
