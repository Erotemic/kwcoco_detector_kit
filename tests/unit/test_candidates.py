from __future__ import annotations

import json

import kwcoco
import kwimage
import numpy as np


def _odd_source(tmp_path):
    image = np.arange(99 * 101 * 3, dtype=np.uint32).reshape(99, 101, 3)
    image = (image % 251).astype(np.uint8)
    asset = tmp_path / "opaque-source.png"
    kwimage.imwrite(asset, image)
    dset = kwcoco.CocoDataset()
    dset.fpath = tmp_path / "source.kwcoco.zip"
    cid = dset.add_category(name="widget")
    gid = dset.add_image(file_name=str(asset), width=101, height=99, cohort="alpha")
    polygon = kwimage.Polygon(exterior=np.array([[2, 2], [12, 2], [12, 12], [2, 12]]))
    dset.add_annotation(image_id=gid, category_id=cid, segmentation=polygon.to_coco(style="new"), bbox=[2, 2, 10, 10], area=100)
    dset.dump()
    return dset.fpath


def test_virtual_candidates_match_eager_pixels_and_identity(tmp_path):
    from kwcoco_detector_kit.data.candidates import (
        CandidateConfig, enumerate_candidates, load_candidate_index,
        materialize_candidates,
    )
    from kwcoco_detector_kit.data.tile import TileConfig, run as tile_run
    from kwcoco_detector_kit.data.tile_cache import sha256_file

    src = _odd_source(tmp_path)
    fingerprint = sha256_file(src)
    index_path = tmp_path / "candidates.json"
    cfg = CandidateConfig.cli(argv=False, data={
        "src": str(src), "dst": str(index_path), "category_names": "widget",
        "tile_size": 32, "source_scales": "0.37", "stride_frac": 1.0,
        "negative_safety_margin": 0, "min_gt_area_frac": 0.0001,
        "min_source_scale_long_side": 1,
        "source_dataset_fingerprint": fingerprint,
    })
    enumerate_candidates(cfg)
    index = load_candidate_index(index_path)
    assert index["num_candidates"] > 0
    row = index["candidates"][0]
    sx, sy = row["tile_actual_scale_xy"]
    assert sx != sy
    assert row["context"] == {"cohort": "alpha"}

    eager_path = tmp_path / "eager.kwcoco.zip"
    eager_cfg = TileConfig.cli(argv=False, data={
        "src": str(src), "dst": str(eager_path), "mode": "multiscale",
        "category_names": "widget", "tile_size": 32, "source_scales": "0.37",
        "stride_frac": 1.0, "min_gt_area_frac": 0.0001,
        "min_source_scale_long_side": 1,
        "cache_dpath": str(tmp_path / "cache"), "jpeg_quality": 90,
        "source_dataset_fingerprint": fingerprint, "progress": False,
    })
    tile_run(eager_cfg)
    eager = kwcoco.CocoDataset.coerce(eager_path)
    eager_by_id = {img["tile_id"]: img for img in eager.images().objs}
    assert row["tile_id"] in eager_by_id
    virtual = materialize_candidates(
        index, [row], cache_dpath=tmp_path / "cache", jpeg_quality=90,
    )
    got = virtual.images().objs[0]
    expected = eager_by_id[row["tile_id"]]
    assert got["tile_materialization_id"] == expected["tile_materialization_id"]
    assert np.array_equal(kwimage.imread(got["file_name"]), kwimage.imread(expected["file_name"]))


def test_stratification_ignores_opaque_filenames_and_covers_scales():
    from kwcoco_detector_kit.data.mine import stratified_candidate_ids

    items = []
    for source in range(3):
        for scale in ["s10", "s04"]:
            for idx in range(7):
                items.append({
                    "id": len(items) + 1, "tile_id": f"hash-{len(items)}",
                    "tile_source_gid": source, "tile_scale_name": scale,
                    "file_name": f"aa/bb/{'f' * 64}.jpg",
                })
    chosen = stratified_candidate_ids(items, budget=6, seed=3)
    selected = [item for item in items if item["id"] in chosen]
    assert len({item["tile_source_gid"] for item in selected}) == 3
    assert len({item["tile_scale_name"] for item in selected}) == 2


def test_predict_batch_cardinality_mismatch_is_durable_failure(tmp_path, monkeypatch):
    from kwcoco_detector_kit.data.candidates import CandidateConfig, enumerate_candidates
    from kwcoco_detector_kit.data import mine

    src = _odd_source(tmp_path)
    index_path = tmp_path / "candidates.json"
    enumerate_candidates(CandidateConfig.cli(argv=False, data={
        "src": str(src), "dst": str(index_path), "category_names": "widget",
        "tile_size": 32, "source_scales": "0.37", "stride_frac": 1.0,
        "min_source_scale_long_side": 1,
    }))

    class BadPredictor:
        def predict_batch(self, images, sizes):
            return []

    monkeypatch.setattr(mine, "_load_predictor", lambda *args, **kwargs: BadPredictor())
    ledger = tmp_path / "ledger.json"
    cfg = mine.MineConfig.cli(argv=False, data={
        "candidate_index": str(index_path), "workdir": str(tmp_path),
        "dst": str(tmp_path / "hard.kwcoco.zip"), "ledger": str(ledger),
        "cache_dpath": str(tmp_path / "cache"), "trainer": "mock_tiny",
        "allow_failures": True, "progress": False,
    })
    mine.run(cfg)
    doc = json.loads(ledger.read_text())
    assert doc["scan_complete"] is True
    assert doc["scan_successful"] is False
    assert doc["num_failures"] == doc["num_expected"] > 0
    assert all("cardinality mismatch" in row["error"] for row in doc["records"])
