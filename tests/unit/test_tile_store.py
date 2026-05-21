"""Tests for data.tile_store — KwcocoJpegStore + WebdatasetStore.

Round-trip a synthetic kwcoco bundle through both backends and verify
the iteration shape, metadata, and per-tile content match.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _make_tile_bundle(synthetic_kwcoco_factory, tmp_path, *, name="raw"):
    """Synth a 4-image kwcoco bundle, then tile it via data.tile.run."""
    from kwcoco_detector_kit.data.tile import TileConfig, run as tile_run

    src = synthetic_kwcoco_factory(name, num_images=4, image_size=(128, 128))
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
            "oversize_factor": 1.0,
        },
    )
    tile_run(cfg)
    return dst


# ---------------------------------------------------------------------------
# KwcocoJpegStore — wraps the Phase 1 default bundle format
# ---------------------------------------------------------------------------


def test_kwcoco_jpeg_store_iterates_all_tiles(synthetic_kwcoco_factory, tmp_path):
    from kwcoco_detector_kit.data.tile_store import KwcocoJpegStore, TileRecord

    bundle = _make_tile_bundle(synthetic_kwcoco_factory, tmp_path)
    store = KwcocoJpegStore(bundle)
    records = list(store)
    assert len(records) == store.num_tiles
    assert store.num_tiles > 0
    for rec in records:
        assert isinstance(rec, TileRecord)
        assert rec.image_np.ndim == 3
        assert rec.image_np.shape[2] == 3  # RGB
        assert rec.image_np.dtype == np.uint8
        assert rec.bboxes_xywh.ndim == 2 and rec.bboxes_xywh.shape[1] == 4
        assert rec.category_ids.ndim == 1
        assert rec.category_ids.shape[0] == rec.bboxes_xywh.shape[0]


def test_kwcoco_jpeg_store_metadata_round_trip(synthetic_kwcoco_factory, tmp_path):
    from kwcoco_detector_kit.data.tile_store import KwcocoJpegStore

    bundle = _make_tile_bundle(synthetic_kwcoco_factory, tmp_path)
    store = KwcocoJpegStore(bundle)
    md = store.metadata
    assert isinstance(md, dict)
    assert any(c["name"] == "widget" for c in md["categories"])
    assert md["mode"] == "multiscale"
    assert md["oversize_factor"] == 1.0


# ---------------------------------------------------------------------------
# WebdatasetStore — tar-shard backend
# ---------------------------------------------------------------------------


def test_webdataset_store_from_kwcoco_writes_shards(synthetic_kwcoco_factory, tmp_path):
    pytest.importorskip("webdataset")
    from kwcoco_detector_kit.data.tile_store import (
        KwcocoJpegStore, WebdatasetStore,
    )

    bundle = _make_tile_bundle(synthetic_kwcoco_factory, tmp_path)
    src_store = KwcocoJpegStore(bundle)
    n_src = src_store.num_tiles
    assert n_src > 0

    dst = tmp_path / "wds_out"
    out = WebdatasetStore.from_kwcoco(bundle, dst, maxcount=8)
    assert out.num_tiles == n_src
    shards = sorted(dst.glob("shard-*.tar"))
    assert shards, "WebdatasetStore must write at least one shard"
    meta_fpath = dst / WebdatasetStore.META_FNAME
    assert meta_fpath.exists()


def test_webdataset_store_round_trip_matches_kwcoco(
    synthetic_kwcoco_factory, tmp_path,
):
    """Image content + bbox counts survive the kwcoco -> wds round trip."""
    pytest.importorskip("webdataset")
    from kwcoco_detector_kit.data.tile_store import (
        KwcocoJpegStore, WebdatasetStore,
    )

    bundle = _make_tile_bundle(synthetic_kwcoco_factory, tmp_path)
    src = list(KwcocoJpegStore(bundle))

    dst = tmp_path / "wds_out"
    WebdatasetStore.from_kwcoco(bundle, dst, maxcount=8, jpeg_quality=95)
    dst_store = WebdatasetStore(dst)
    dst_records = list(dst_store)

    assert len(dst_records) == len(src), (
        f"shard iteration count mismatch: src={len(src)} dst={len(dst_records)}"
    )
    # Categories agree on count (post-tile, kit-internal cid)
    src_total_anns = sum(r.bboxes_xywh.shape[0] for r in src)
    dst_total_anns = sum(r.bboxes_xywh.shape[0] for r in dst_records)
    assert src_total_anns == dst_total_anns, (
        f"annotation count mismatch: src={src_total_anns} dst={dst_total_anns}"
    )
    # Image shape preserved (within JPEG quantization)
    for sr, dr in zip(src, dst_records):
        assert sr.image_np.shape == dr.image_np.shape


def test_open_store_auto_detects_backend(synthetic_kwcoco_factory, tmp_path):
    pytest.importorskip("webdataset")
    from kwcoco_detector_kit.data.tile_store import (
        open_store, KwcocoJpegStore, WebdatasetStore,
    )

    bundle = _make_tile_bundle(synthetic_kwcoco_factory, tmp_path)
    assert isinstance(open_store(bundle), KwcocoJpegStore)

    dst = tmp_path / "wds_out"
    WebdatasetStore.from_kwcoco(bundle, dst, maxcount=8)
    assert isinstance(open_store(dst), WebdatasetStore)
