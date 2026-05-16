"""Unit tests for kwcoco_detector_kit.util.polygon_utils."""
from __future__ import annotations

import numpy as np
import pytest


def test_expand_box_xyxy_basic():
    from kwcoco_detector_kit.util.polygon_utils import expand_box_xyxy

    result = expand_box_xyxy([10, 20, 50, 60], padding=5, image_shape=(100, 80))
    assert result == [5.0, 15.0, 55.0, 65.0]


def test_expand_box_xyxy_clamps_to_image():
    from kwcoco_detector_kit.util.polygon_utils import expand_box_xyxy

    result = expand_box_xyxy([2, 3, 78, 97], padding=10, image_shape=(100, 80))
    assert result[0] == 0.0
    assert result[1] == 0.0
    assert result[2] == 80.0
    assert result[3] == 100.0


@pytest.mark.requires_torch
def test_mask_to_multi_polygon_simple():
    from kwcoco_detector_kit.util.polygon_utils import mask_to_multi_polygon

    mask = np.zeros((64, 64), dtype=bool)
    mask[10:30, 10:30] = True
    mpoly = mask_to_multi_polygon(mask, keep_largest_component=True)
    assert len(mpoly.data) == 1
    assert mpoly.data[0].area > 0


@pytest.mark.requires_torch
def test_mask_to_multi_polygon_empty():
    from kwcoco_detector_kit.util.polygon_utils import mask_to_multi_polygon

    mask = np.zeros((32, 32), dtype=bool)
    mpoly = mask_to_multi_polygon(mask)
    assert len(mpoly.data) == 0


@pytest.mark.requires_torch
def test_mask_to_multi_polygon_min_area_filter():
    from kwcoco_detector_kit.util.polygon_utils import mask_to_multi_polygon

    mask = np.zeros((64, 64), dtype=bool)
    mask[5:7, 5:7] = True     # tiny component (~4 px)
    mask[20:40, 20:40] = True  # large component (~400 px)
    mpoly = mask_to_multi_polygon(mask, min_component_area=50, keep_largest_component=False)
    assert len(mpoly.data) == 1  # only large survived


@pytest.mark.requires_torch
def test_segmentation_to_coco_passthrough():
    from kwcoco_detector_kit.util.polygon_utils import segmentation_to_coco

    assert segmentation_to_coco(None) is None
    raw = [[10, 20, 30, 20, 30, 40, 10, 40]]
    assert segmentation_to_coco(raw) is raw
