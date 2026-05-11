"""Tests for data.stats.compute_per_channel_stats — Welford correctness."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


class _ConstantStore:
    """Synthetic TileStore yielding N tiles of a fixed pattern, for math tests."""

    def __init__(self, tile_shape, values_per_channel, num_tiles):
        self._tile_shape = tile_shape
        self._values = values_per_channel
        self._n = num_tiles
        self._meta = {"channels": "r|g|b", "categories": []}

    @property
    def num_tiles(self):
        return self._n

    @property
    def metadata(self):
        return self._meta

    def __iter__(self):
        from kwcoco_detector_kit.data.tile_store import TileRecord

        H, W = self._tile_shape
        for _ in range(self._n):
            img = np.zeros((H, W, len(self._values)), dtype=np.uint8)
            for c, v in enumerate(self._values):
                img[..., c] = v
            yield TileRecord(
                image_np=img,
                bboxes_xywh=np.zeros((0, 4), dtype=np.float32),
                category_ids=np.zeros((0,), dtype=np.int64),
                metadata={},
            )


def test_stats_constant_image_returns_uniform_mean_zero_std():
    """A bundle of identical pixels has mean == pixel/255, std == 0."""
    from kwcoco_detector_kit.data.stats import compute_per_channel_stats

    store = _ConstantStore(tile_shape=(8, 8), values_per_channel=[64, 128, 200],
                            num_tiles=3)
    stats = compute_per_channel_stats(store)
    assert stats["num_channels"] == 3
    np.testing.assert_allclose(stats["mean"],
                               [64 / 255.0, 128 / 255.0, 200 / 255.0], atol=1e-6)
    np.testing.assert_allclose(stats["std"], [0.0, 0.0, 0.0], atol=1e-6)
    assert stats["num_tiles_sampled"] == 3
    assert stats["num_pixels"] == 3 * 64


def test_stats_two_tile_means_match_numpy_reference():
    """Welford batch update must match numpy's reference mean/std."""
    from kwcoco_detector_kit.data.stats import compute_per_channel_stats

    # Two tiles with different pixels.
    class _RngStore:
        def __init__(self):
            rng = np.random.RandomState(7)
            self._tiles = [
                (rng.rand(16, 16, 3) * 255).astype(np.uint8) for _ in range(2)
            ]
            self._meta = {"channels": "r|g|b", "categories": []}

        @property
        def num_tiles(self):
            return len(self._tiles)

        @property
        def metadata(self):
            return self._meta

        def __iter__(self):
            from kwcoco_detector_kit.data.tile_store import TileRecord
            for t in self._tiles:
                yield TileRecord(
                    image_np=t,
                    bboxes_xywh=np.zeros((0, 4), dtype=np.float32),
                    category_ids=np.zeros((0,), dtype=np.int64),
                    metadata={},
                )

    store = _RngStore()
    stats = compute_per_channel_stats(store)

    # Reference: stack all pixels and compute mean / std across (N, C).
    all_px = np.concatenate(
        [t.astype(np.float64).reshape(-1, 3) / 255.0 for t in store._tiles],
        axis=0,
    )
    ref_mean = all_px.mean(axis=0)
    ref_std = all_px.std(axis=0, ddof=1)
    np.testing.assert_allclose(stats["mean"], ref_mean, atol=1e-6)
    np.testing.assert_allclose(stats["std"], ref_std, atol=1e-6)


def test_stats_sample_size_caps_iteration():
    """sample_size > 0 stops after that many tiles."""
    from kwcoco_detector_kit.data.stats import compute_per_channel_stats

    store = _ConstantStore(tile_shape=(4, 4), values_per_channel=[100, 100, 100],
                            num_tiles=20)
    stats = compute_per_channel_stats(store, sample_size=5)
    assert stats["num_tiles_sampled"] == 5
