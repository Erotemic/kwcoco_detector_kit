from __future__ import annotations

import json
from pathlib import Path

import kwcoco
import kwimage
import numpy as np
import pytest


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
        materialize_candidate_records, materialize_candidates,
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
    row = materialize_candidate_records(index, limit=1)[0]
    sx, sy = row["tile_actual_scale_xy"]
    assert sx != sy
    assert row["context"] == {"cohort": "alpha"}

    eager_path = tmp_path / "eager.kwcoco.zip"
    eager_cfg = TileConfig.cli(argv=False, data={
        "src": str(src), "dst": str(eager_path), "mode": "multiscale",
        "category_names": "widget", "tile_size": 32, "source_scales": "0.37",
        "stride_frac": 1.0, "min_gt_area_frac": 0.0001,
        "min_source_scale_long_side": 1,
        "cache_dpath": str(tmp_path / "eager-cache"), "jpeg_quality": 90,
        "source_dataset_fingerprint": fingerprint, "progress": False,
    })
    tile_run(eager_cfg)
    eager = kwcoco.CocoDataset.coerce(eager_path)
    eager_by_id = {img["tile_id"]: img for img in eager.images().objs}
    assert row["tile_id"] in eager_by_id
    virtual = materialize_candidates(
        index, [row], cache_dpath=tmp_path / "virtual-cache", jpeg_quality=90,
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
    records = [json.loads(line) for line in Path(doc["records_path"]).read_text().splitlines()]
    assert all("cardinality mismatch" in row["error"] for row in records)

    changed = dict(cfg)
    # Batch size affects the actual scoring execution and remains part of the
    # resumable scan identity.  Final hard-negative threshold/top-K do not.
    changed["batch_size"] = 7
    changed_cfg = mine.MineConfig.cli(argv=False, data=changed)
    with pytest.raises(RuntimeError, match="progress fingerprint mismatch"):
        mine.run(changed_cfg)


def test_streaming_index_is_deterministic_sharded_and_validated(tmp_path):
    from kwcoco_detector_kit.data.candidates import (
        CandidateConfig, enumerate_candidates, iter_candidate_records,
        iter_candidate_records_for_shard, load_candidate_index,
    )

    src = _odd_source(tmp_path)
    roots = [tmp_path / "index1", tmp_path / "index2"]
    for root in roots:
        enumerate_candidates(CandidateConfig.cli(argv=False, data={
            "src": str(src), "dst": str(root), "category_names": "widget",
            "tile_size": 16, "source_scales": "1.0", "stride_frac": .5,
            "min_source_scale_long_side": 1, "rows_per_shard": 5,
        }))
    manifests = [load_candidate_index(root) for root in roots]
    for manifest in manifests:
        manifest.pop("index_dpath")
        assert "candidates" not in manifest
        assert len(manifest["candidate_shards"]) > 1
    assert manifests[0] == manifests[1]
    for shard in manifests[0]["candidate_shards"]:
        assert (roots[0] / shard["name"]).read_bytes() == (roots[1] / shard["name"]).read_bytes()
    ids1 = [row["tile_id"] for row in iter_candidate_records(roots[0])]
    ids2 = [row["tile_id"] for row in iter_candidate_records(roots[1])]
    assert ids1 == ids2
    assert len(ids1) == manifests[0]["num_candidates"]

    # A physical shard iterator does not touch unrelated shard data.
    unrelated = roots[0] / manifests[0]["candidate_shards"][1]["name"]
    original = unrelated.read_bytes()
    unrelated.write_bytes(b"corrupt unrelated shard\n")
    assert list(iter_candidate_records_for_shard(roots[0], 0))
    with pytest.raises(ValueError, match="validation failed|invalid candidate"):
        list(iter_candidate_records(roots[0]))
    unrelated.write_bytes(original[:-7])
    with pytest.raises(ValueError, match="validation failed|invalid candidate"):
        list(iter_candidate_records(roots[0]))


def test_locality_sharding_and_realization_reuse(tmp_path, monkeypatch):
    import kwcoco
    from kwcoco_detector_kit.data.candidates import (
        CandidateConfig, enumerate_candidates, iter_candidate_records,
        iter_realized_candidate_batches, load_candidate_index,
        realize_candidate_arrays,
    )
    from kwcoco_detector_kit.data.mine import iter_candidate_shard_assignments

    src = _odd_source(tmp_path)
    index_path = tmp_path / "local-index"
    enumerate_candidates(CandidateConfig.cli(argv=False, data={
        "src": str(src), "dst": str(index_path), "category_names": "widget",
        "tile_size": 16, "source_scales": "1.0,0.5", "stride_frac": 1.0,
        "min_source_scale_long_side": 1,
    }))
    index = load_candidate_index(index_path)
    rows = list(iter_candidate_records(index))
    assignments = list(iter_candidate_shard_assignments(rows, 4, locality_chunk_size=5))
    assert [row["tile_id"] for row, _rank in assignments] == [row["tile_id"] for row in rows]
    assert len({row["tile_id"] for row, _rank in assignments}) == len(rows)

    for rank in range(4):
        shard_rows = [row for row, assigned in assignments if assigned == rank]
        keys = [(row["tile_source_gid"], tuple(row["tile_actual_scale_xy"])) for row in shard_rows]
        closed = set()
        prior = None
        for key in keys:
            if key != prior:
                assert key not in closed
                if prior is not None:
                    closed.add(prior)
                prior = key

    original = kwcoco.CocoImage.imdelay
    decode_calls = []

    def counted_imdelay(self, *args, **kwargs):
        decode_calls.append(self.img["id"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(kwcoco.CocoImage, "imdelay", counted_imdelay)
    batches = list(iter_realized_candidate_batches(index, iter(rows), batch_size=2))
    got = {
        row["tile_id"]: array
        for batch, arrays, error in batches
        for row, array in zip(batch, arrays)
        if error is None
    }
    num_groups = len({
        (row["tile_source_gid"], tuple(row["tile_actual_scale_xy"])) for row in rows
    })
    assert len(decode_calls) == num_groups
    assert len(batches) > num_groups

    decode_calls.clear()
    expected = realize_candidate_arrays(index, rows)
    assert got.keys() == expected.keys()
    assert all(np.array_equal(got[key], expected[key]) for key in got)


def test_materialization_streams_bounded_crop_batches(tmp_path, monkeypatch):
    from kwcoco_detector_kit.data import candidates

    src = _odd_source(tmp_path)
    index_path = tmp_path / "materialize-index"
    candidates.enumerate_candidates(candidates.CandidateConfig.cli(argv=False, data={
        "src": str(src), "dst": str(index_path), "category_names": "widget",
        "tile_size": 12, "source_scales": "1.0,0.5", "stride_frac": 1.0,
        "min_source_scale_long_side": 1,
    }))
    index = candidates.load_candidate_index(index_path)
    rows = list(candidates.iter_candidate_records(index))
    assert len(rows) > 16

    def forbidden_full_mapping(*args, **kwargs):
        raise AssertionError("materialization must not build the full array mapping")

    monkeypatch.setattr(candidates, "realize_candidate_arrays", forbidden_full_mapping)
    original_batches = candidates.iter_realized_candidate_batches
    observed_batch_sizes = []

    def instrumented_batches(*args, **kwargs):
        for batch, arrays, error in original_batches(*args, **kwargs):
            observed_batch_sizes.append(len(batch))
            yield batch, arrays, error

    monkeypatch.setattr(candidates, "iter_realized_candidate_batches", instrumented_batches)
    out = candidates.materialize_candidates(
        index, iter(rows), cache_dpath=tmp_path / "cache", batch_size=4,
    )
    assert out.n_images == len(rows)
    assert max(observed_batch_sizes) <= 4
    assert len(observed_batch_sizes) > 4
    assert all(Path(img["file_name"]).is_file() for img in out.images().objs)
