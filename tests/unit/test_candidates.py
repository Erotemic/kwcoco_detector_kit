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



def test_virtual_candidate_cache_reuses_raster_after_truth_change(tmp_path, monkeypatch):
    from kwcoco_detector_kit.data import candidates

    src1 = _odd_source(tmp_path)
    dset1 = kwcoco.CocoDataset.coerce(src1)
    index1_path = tmp_path / "index1"
    common = {
        "category_names": "widget", "tile_size": 24,
        "source_scales": "1.0", "stride_frac": 1.0,
        "negative_safety_margin": 0, "min_gt_area_frac": 0.0001,
        "min_source_scale_long_side": 1,
    }
    candidates.enumerate_candidates(candidates.CandidateConfig.cli(argv=False, data={
        "src": str(src1), "dst": str(index1_path), **common,
    }))
    index1 = candidates.load_candidate_index(index1_path)
    rows1 = list(candidates.iter_candidate_records(index1))
    by_raster1 = {row["tile_raster_id"]: row for row in rows1}
    assert by_raster1

    dset2 = dset1.copy()
    ann = next(iter(dset2.anns.values()))
    polygon2 = kwimage.Polygon(
        exterior=np.array([[14, 14], [22, 14], [22, 22], [14, 22]])
    )
    ann["segmentation"] = polygon2.to_coco(style="new")
    ann["bbox"] = [14, 14, 8, 8]
    ann["area"] = 64
    src2 = tmp_path / "source_truth2.kwcoco.zip"
    dset2.fpath = src2
    dset2.dump()
    index2_path = tmp_path / "index2"
    candidates.enumerate_candidates(candidates.CandidateConfig.cli(argv=False, data={
        "src": str(src2), "dst": str(index2_path), **common,
    }))
    index2 = candidates.load_candidate_index(index2_path)
    rows2 = list(candidates.iter_candidate_records(index2))
    by_raster2 = {row["tile_raster_id"]: row for row in rows2}
    common_rasters = sorted(set(by_raster1) & set(by_raster2))
    assert common_rasters
    raster_id = common_rasters[0]
    row1 = by_raster1[raster_id]
    row2 = by_raster2[raster_id]
    assert row1["tile_id"] != row2["tile_id"]

    cache = tmp_path / "cache"
    first = candidates.materialize_candidates(
        index1, [row1], cache_dpath=cache, jpeg_quality=90,
    )
    first_img = first.images().objs[0]

    def forbidden_realization(*args, **kwargs):
        raise AssertionError("warm raster cache hit must not realize source pixels")

    monkeypatch.setattr(candidates, "_realize_scaled_source", forbidden_realization)
    second = candidates.materialize_candidates(
        index2, [row2], cache_dpath=cache, jpeg_quality=90,
    )
    second_img = second.images().objs[0]
    stats = second.dataset["info"][-1]["cache_stats"]
    assert first_img["tile_raster_id"] == second_img["tile_raster_id"]
    assert first_img["tile_materialization_id"] == second_img["tile_materialization_id"]
    assert first_img["file_name"] == second_img["file_name"]
    assert stats == {
        "hits": 1,
        "misses": 0,
        "encoded": 0,
        "published": 0,
        "source_scale_realizations": 0,
    }


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


def test_candidate_selection_is_stable_across_truth_only_rekey(tmp_path):
    from kwcoco_detector_kit.data.candidates import (
        _CandidateIndexWriter, load_candidate_index,
        selected_candidate_record_factory,
    )

    def build(root, truth_prefix):
        writer = _CandidateIndexWriter(root, {
            "source_kwcoco": f"{truth_prefix}.kwcoco.zip",
            "source_dataset_fingerprint": truth_prefix,
            "policy": {},
            "policy_fingerprint": "policy",
        }, rows_per_shard=3)
        for idx in range(12):
            writer.write({
                "tile_id": f"{truth_prefix}-{idx:02d}",
                "tile_raster_id": f"raster-{idx:02d}",
                "tile_source_gid": 1 + (idx // 6),
                "tile_scale_name": "s10" if idx % 2 == 0 else "s05",
                "tile_actual_scale_xy": [1.0, 1.0],
                "tile_scaled_extent_xyxy": [idx, 0, idx + 1, 1],
            })
        writer.close()
        return load_candidate_index(root)

    first = build(tmp_path / "truth1", "truth1")
    second = build(tmp_path / "truth2", "truth2")
    for strategy in ["random", "stratified_by_image"]:
        selected1 = list(selected_candidate_record_factory(
            first, 6, seed=17, strategy=strategy,
        )())
        selected2 = list(selected_candidate_record_factory(
            second, 6, seed=17, strategy=strategy,
        )())
        assert {row["tile_raster_id"] for row in selected1} == {
            row["tile_raster_id"] for row in selected2
        }
        assert {row["tile_id"] for row in selected1} != {
            row["tile_id"] for row in selected2
        }


def test_public_candidate_selection_is_bounded_deterministic_and_stratified(tmp_path):
    from kwcoco_detector_kit.data.candidates import (
        CandidateConfig, enumerate_candidates, load_candidate_index,
        selected_candidate_record_factory,
    )

    src = _odd_source(tmp_path)
    index_path = tmp_path / "selection-index"
    enumerate_candidates(CandidateConfig.cli(argv=False, data={
        "src": str(src), "dst": str(index_path), "category_names": "widget",
        "tile_size": 12, "source_scales": "1.0,0.5", "stride_frac": 0.5,
        "min_source_scale_long_side": 1, "rows_per_shard": 7,
    }))
    index = load_candidate_index(index_path)
    budget = min(12, index["num_candidates"])
    factory = selected_candidate_record_factory(
        index, budget, seed=17, strategy="stratified_by_image",
    )
    rows1 = list(factory())
    rows2 = list(factory())
    assert [r["tile_id"] for r in rows1] == [r["tile_id"] for r in rows2]
    assert len(rows1) == budget
    assert rows1 == sorted(rows1, key=lambda row: (
        row["tile_source_gid"], tuple(row["tile_actual_scale_xy"]),
        row["tile_scaled_extent_xyxy"], row["tile_id"],
    ))
    scales = {row["tile_scale_name"] for row in rows1}
    if budget >= 2:
        assert scales == {"s10", "s05"}

    all_factory = selected_candidate_record_factory(index, 0, seed=17)
    assert sum(1 for _ in all_factory()) == index["num_candidates"]


def test_candidate_enumeration_resumes_from_image_checkpoint(tmp_path, monkeypatch):
    from kwcoco_detector_kit.data import candidates

    image = np.zeros((48, 52, 3), dtype=np.uint8)
    asset = tmp_path / "shared.png"
    kwimage.imwrite(asset, image)
    dset = kwcoco.CocoDataset()
    dset.fpath = tmp_path / "multi.kwcoco.zip"
    dset.add_category(name="widget")
    for idx in range(6):
        dset.add_image(
            file_name=str(asset), width=52, height=48, cohort=f"c{idx}",
        )
    dset.dump()

    common = {
        "src": str(dset.fpath),
        "category_names": "widget",
        "tile_size": 16,
        "source_scales": "1.0,0.5",
        "stride_frac": 1.0,
        "min_source_scale_long_side": 1,
        "rows_per_shard": 7,
        "checkpoint_images": 2,
        "progress": False,
    }

    interrupted = tmp_path / "interrupted"
    original_atomic = candidates._atomic_json
    tripped = {"value": False}

    def interrupt_after_durable_receipt(data, path):
        original_atomic(data, path)
        if Path(path).name == ".candidate-build-resume.json" and not tripped["value"]:
            tripped["value"] = True
            raise RuntimeError("simulated process interruption")

    monkeypatch.setattr(candidates, "_atomic_json", interrupt_after_durable_receipt)
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        candidates.enumerate_candidates(candidates.CandidateConfig.cli(argv=False, data={
            **common, "dst": str(interrupted),
        }))
    receipt = json.loads((interrupted / ".candidate-build-resume.json").read_text())
    assert len(receipt["completed_image_ids"]) == 2

    monkeypatch.setattr(candidates, "_atomic_json", original_atomic)
    candidates.enumerate_candidates(candidates.CandidateConfig.cli(argv=False, data={
        **common, "dst": str(interrupted),
    }))
    assert not (interrupted / ".candidate-build-resume.json").exists()

    clean = tmp_path / "clean"
    candidates.enumerate_candidates(candidates.CandidateConfig.cli(argv=False, data={
        **common, "dst": str(clean), "resume": False,
    }))
    resumed_rows = list(candidates.iter_candidate_records(interrupted))
    clean_rows = list(candidates.iter_candidate_records(clean))
    assert resumed_rows == clean_rows
    assert candidates.load_candidate_index(interrupted)["candidate_content_digest"] == \
        candidates.load_candidate_index(clean)["candidate_content_digest"]

    # A completed build is itself idempotent and should not touch its shards.
    mtimes = {
        path.name: path.stat().st_mtime_ns
        for path in interrupted.glob("candidates-*.jsonl")
    }
    candidates.enumerate_candidates(candidates.CandidateConfig.cli(argv=False, data={
        **common, "dst": str(interrupted),
    }))
    assert mtimes == {
        path.name: path.stat().st_mtime_ns
        for path in interrupted.glob("candidates-*.jsonl")
    }
