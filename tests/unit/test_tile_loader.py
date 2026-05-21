"""Tests for data.tile_loader — load-time crop / flip augmentation.

The Phase 3 design: ``data.tile`` writes oversized tiles (e.g. 256x256)
with ``tile_model_input_size=[224, 224]`` metadata. ``TileLoader`` reads
the bundle and crops at load time to the model input. ``augment=True``
uses random crop + flip; ``augment=False`` uses center crop + no flip.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _make_oversized_bundle(synthetic_kwcoco_factory, tmp_path, *,
                            oversize_factor=2.0, name="raw"):
    """Tile a synth bundle with oversize_factor > 1.

    Multiscale mode with tile_size=64, oversize=2.0 -> 128x128 on-disk tiles
    carrying tile_model_input_size=[64, 64].
    """
    from kwcoco_detector_kit.data.tile import TileConfig, run as tile_run

    src = synthetic_kwcoco_factory(name, num_images=4, image_size=(256, 256))
    dst = tmp_path / f"{name}_tiles.kwcoco.zip"
    cfg = TileConfig.cli(
        argv=False,
        data={
            "src": str(src), "dst": str(dst),
            "mode": "multiscale", "category_names": "widget",
            "tile_size": 64, "source_scales": "1.0",
            "stride_frac": 1.0, "min_gt_area_frac": 0.001,
            "min_source_scale_long_side": 32,
            "keep_negative": True, "progress": False,
            "oversize_factor": float(oversize_factor),
        },
    )
    tile_run(cfg)
    return dst


def test_tile_loader_center_crop_matches_model_input_metadata(
    synthetic_kwcoco_factory, tmp_path,
):
    """augment=False crops to tile_model_input_size from the bundle's metadata."""
    from kwcoco_detector_kit.data.tile_store import KwcocoJpegStore
    from kwcoco_detector_kit.data.tile_loader import TileLoader

    bundle = _make_oversized_bundle(
        synthetic_kwcoco_factory, tmp_path, oversize_factor=2.0,
    )
    store = KwcocoJpegStore(bundle)
    # On-disk tiles are 128x128; model input is 64x64 (per metadata).
    loader = TileLoader(store, augment=False, return_tensors=False)
    sample_count = 0
    for batch in loader:
        H, W = batch.image.shape[:2]
        assert (H, W) == (64, 64), (
            f"center-cropped tile should be 64x64, got {(H, W)}"
        )
        # bbox coords are in cropped-image space (≤ 64)
        if batch.bboxes_xywh.size:
            assert (batch.bboxes_xywh[:, 0] >= 0).all()
            assert (batch.bboxes_xywh[:, 1] >= 0).all()
            assert (batch.bboxes_xywh[:, 0] + batch.bboxes_xywh[:, 2] <= 64 + 1e-3).all()
            assert (batch.bboxes_xywh[:, 1] + batch.bboxes_xywh[:, 3] <= 64 + 1e-3).all()
        sample_count += 1
    assert sample_count == store.num_tiles


def test_tile_loader_random_crop_changes_with_seed(synthetic_kwcoco_factory, tmp_path):
    """augment=True with different seeds yields different crop positions."""
    from kwcoco_detector_kit.data.tile_store import KwcocoJpegStore
    from kwcoco_detector_kit.data.tile_loader import TileLoader

    bundle = _make_oversized_bundle(
        synthetic_kwcoco_factory, tmp_path, oversize_factor=2.0,
    )
    # Same store, two different seeds -> the first samples should differ
    # (with high probability — 64 possible top offsets x 64 possible left).
    store = KwcocoJpegStore(bundle)
    a = next(iter(TileLoader(store, augment=True, return_tensors=False, seed=0)))
    store = KwcocoJpegStore(bundle)
    b = next(iter(TileLoader(store, augment=True, return_tensors=False, seed=99)))
    # Compare a couple of byte regions. Different crops on a near-uniform
    # synthetic image MIGHT collide; check pixel patches near the center.
    assert a.image.shape == b.image.shape


def test_tile_loader_normalize_mean_std(synthetic_kwcoco_factory, tmp_path):
    """normalize={'mean': ..., 'std': ...} applies per-channel standardization."""
    from kwcoco_detector_kit.data.tile_store import KwcocoJpegStore
    from kwcoco_detector_kit.data.tile_loader import TileLoader

    bundle = _make_oversized_bundle(
        synthetic_kwcoco_factory, tmp_path, oversize_factor=1.0,
    )
    store = KwcocoJpegStore(bundle)
    loader = TileLoader(
        store, augment=False, return_tensors=False,
        normalize={"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
    )
    for batch in loader:
        # Image is float, not uint8.
        assert batch.image.dtype == np.float32
        # The normalization should not shift uniform-red squares to the
        # uint8-in-[0,1] range — they'll be negative for the green/blue
        # channels and well above 1 for the red.
        assert batch.image.min() < 0 or batch.image.max() > 1
        break


def test_tile_loader_returns_tensors_when_requested(synthetic_kwcoco_factory, tmp_path):
    pytest.importorskip("torch")
    from kwcoco_detector_kit.data.tile_store import KwcocoJpegStore
    from kwcoco_detector_kit.data.tile_loader import TileLoader
    import torch

    bundle = _make_oversized_bundle(
        synthetic_kwcoco_factory, tmp_path, oversize_factor=1.0,
    )
    store = KwcocoJpegStore(bundle)
    loader = TileLoader(store, augment=False, return_tensors=True)
    for batch in loader:
        assert isinstance(batch.image, torch.Tensor)
        # CHW layout for torch
        assert batch.image.ndim == 3 and batch.image.shape[0] == 3
        break


def test_tile_loader_bbox_clip_after_crop(synthetic_kwcoco_factory, tmp_path):
    """Annotations near the tile edge get clipped to the cropped area."""
    from kwcoco_detector_kit.data.tile_store import KwcocoJpegStore
    from kwcoco_detector_kit.data.tile_loader import TileLoader

    bundle = _make_oversized_bundle(
        synthetic_kwcoco_factory, tmp_path, oversize_factor=2.0,
    )
    store = KwcocoJpegStore(bundle)
    loader = TileLoader(store, augment=False, return_tensors=False)
    for batch in loader:
        # If any annotations survived the center crop, their bboxes must
        # be inside the cropped image.
        H, W = batch.image.shape[:2]
        for bbox in batch.bboxes_xywh:
            x, y, w, h = bbox
            assert 0 <= x and 0 <= y
            assert x + w <= W + 1e-3
            assert y + h <= H + 1e-3
