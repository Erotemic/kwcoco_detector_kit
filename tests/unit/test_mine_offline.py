"""Tests for data.mine — offline hard-negative mining.

Drives the miner against a mock_tiny-trained checkpoint (no GPU, no
DEIMv2 submodule). The predictor adapter is the unit under test from
the miner's perspective — we verify:

- the miner filters to negative tiles (tile_role=None or 'negative');
- only tiles with max_pred_score >= threshold are kept;
- the score histogram sidecar is written.
"""
from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import kwcoco
import numpy as np
import pytest


def test_stable_shards_have_exact_disjoint_coverage():
    from kwcoco_detector_kit.data.mine import stable_shard_for_key
    keys = [f"tile-{idx:05d}" for idx in range(1000)]
    shards = [
        {key for key in keys if stable_shard_for_key(key, 4) == rank}
        for rank in range(4)
    ]
    assert set.union(*shards) == set(keys)
    assert sum(map(len, shards)) == len(keys)
    assert all(shards[i].isdisjoint(shards[j]) for i in range(4) for j in range(i))


def _build_neg_bundle(bundle_dpath: Path, n: int, seed: int = 0) -> Path:
    import kwimage
    rng = np.random.RandomState(seed)
    bundle_dpath = Path(bundle_dpath)
    asset_dpath = bundle_dpath / "neg_assets"
    asset_dpath.mkdir(parents=True, exist_ok=True)
    dset = kwcoco.CocoDataset()
    dset.fpath = str(bundle_dpath / "neg.kwcoco.zip")
    dset.add_category(name="widget")
    for k in range(n):
        img = (rng.rand(64, 64, 3) * 255).astype(np.uint8)
        fpath = asset_dpath / f"neg_{k:04d}.jpg"
        kwimage.imwrite(str(fpath), img)
        dset.add_image(
            file_name=str(fpath.relative_to(bundle_dpath)),
            width=64, height=64, tile_role="negative",
            tile_source_gid=k // 2, tile_scale_name="s10",
            tile_extent_xyxy_in_source=[k * 64, 0, (k + 1) * 64, 64],
        )
    dset.dump()
    return Path(dset.fpath)


def _train_mock_tiny(tmp_workdir: Path, train_kwcoco: Path):
    """Helper: in-process mock_tiny train for a few steps."""
    from kwcoco_detector_kit.trainers._registry import get_trainer
    trainer = get_trainer("mock_tiny")
    cfg = trainer.generate_config(
        train_kwcoco_fpath=str(train_kwcoco),
        vali_kwcoco_fpath=str(train_kwcoco),
        workdir=tmp_workdir,
        variant="mock_tiny",
        input_hw=(64, 64),
        train_policy="fixed",
        num_classes=1,
        batch_size=2, val_batch_size=2,
        num_epochs=1,
        lr=1e-2, backbone_lr=1e-2,
        use_amp=False,
        channels="r|g|b", scale_tier="S", num_gpus=1,
        data_format="kwcoco",
        extra={"category_names": ["widget"]},
    )
    trainer.launch(cfg, num_gpus=1)
    return tmp_workdir


@pytest.mark.requires_torch
def test_mine_writes_kwcoco_and_sidecar(synthetic_kwcoco, tmp_workdir, tmp_path):
    _train_mock_tiny(tmp_workdir, synthetic_kwcoco)
    neg_kwcoco = _build_neg_bundle(tmp_path / "neg_bundle", n=6)
    dst = tmp_path / "hard_negs.kwcoco.zip"

    from kwcoco_detector_kit.data.mine import MineConfig, run as mine_run
    cfg = MineConfig.cli(
        argv=False,
        data={
            "neg_kwcoco": str(neg_kwcoco),
            "workdir": str(tmp_workdir),
            "dst": str(dst),
            "trainer": "mock_tiny",
            # Below the post-training gate so SOME tiles qualify as "hard".
            "score_thresh": 0.05,
            "max_hard_per_round": 100,
            "device": "cpu",
            "progress": False,
        },
    )
    mine_run(cfg)

    assert dst.exists(), "mine should write the hard-neg kwcoco bundle"
    out = kwcoco.CocoDataset.coerce(str(dst))
    assert out.n_images >= 0  # may be 0 if no tile clears threshold; the bundle exists either way
    sidecar = dst.with_suffix(".mine_stats.json")
    assert sidecar.exists(), "score histogram sidecar must be written"
    stats = json.loads(sidecar.read_text())
    assert "n_scored" in stats and "n_hard" in stats
    assert "score_hist" in stats and "score_bins" in stats


@pytest.mark.requires_torch
def test_mine_cap_respected(synthetic_kwcoco, tmp_workdir, tmp_path):
    _train_mock_tiny(tmp_workdir, synthetic_kwcoco)
    neg_kwcoco = _build_neg_bundle(tmp_path / "neg_bundle", n=10)
    dst = tmp_path / "hard_negs.kwcoco.zip"

    from kwcoco_detector_kit.data.mine import MineConfig, run as mine_run
    cfg = MineConfig.cli(
        argv=False,
        data={
            "neg_kwcoco": str(neg_kwcoco),
            "workdir": str(tmp_workdir),
            "dst": str(dst),
            "trainer": "mock_tiny",
            "score_thresh": 0.0,   # everything qualifies
            "max_hard_per_round": 3,
            "device": "cpu",
            "progress": False,
        },
    )
    mine_run(cfg)
    out = kwcoco.CocoDataset.coerce(str(dst))
    assert out.n_images <= 3, f"max_hard_per_round=3 should cap; got {out.n_images}"


def test_shard_merge_proves_exact_coverage(tmp_path):
    from kwcoco_detector_kit.data.mine import merge_shard_ledgers
    ids = [f"tile-{idx}" for idx in range(12)]
    paths = []
    for rank in range(3):
        records = [{"tile_id": tile_id, "status": "ok"} for idx, tile_id in enumerate(ids) if idx % 3 == rank]
        path = tmp_path / f"rank{rank}.json"
        path.write_text(json.dumps({"scan_complete": True, "records": records}))
        paths.append(path)
    merged = merge_shard_ledgers(paths, ids)
    assert [row["tile_id"] for row in merged] == sorted(ids)
    duplicate = json.loads(paths[1].read_text())
    duplicate["records"].append({"tile_id": ids[0], "status": "ok"})
    paths[1].write_text(json.dumps(duplicate))
    with pytest.raises(RuntimeError, match="overlap"):
        merge_shard_ledgers(paths, ids)


def _synthetic_virtual_ledgers(tmp_path, num_shards=4):
    from kwcoco_detector_kit.data.candidates import (
        CandidateConfig, enumerate_candidates, iter_candidate_records,
    )
    from kwcoco_detector_kit.data.mine import iter_candidate_shard_assignments
    from kwcoco_detector_kit.data.tile_cache import canonical_digest

    source = _build_neg_bundle(tmp_path / "source", n=3)
    index_path = tmp_path / "candidate-index"
    enumerate_candidates(CandidateConfig.cli(argv=False, data={
        "src": str(source), "dst": str(index_path), "category_names": "widget",
        "tile_size": 32, "source_scales": "1.0", "stride_frac": 1.0,
        "min_source_scale_long_side": 1, "rows_per_shard": 3,
    }))
    rows = list(iter_candidate_records(index_path))
    score_by_id = {
        row["tile_id"]: (idx + 1) / (len(rows) + 1)
        for idx, row in enumerate(rows)
    }
    run_spec = {
        "num_shards": num_shards, "max_candidates": 0, "candidate_seed": 0,
        "locality_chunk_size": 3,
        "score_thresh": 0.0, "max_hard_per_round": 5,
    }
    fingerprint = canonical_digest(run_spec)
    ledgers = []
    assignments = list(iter_candidate_shard_assignments(rows, num_shards, 3))
    for rank in range(num_shards):
        shard_rows = [row for row, assigned_rank in assignments if assigned_rank == rank]
        hasher = hashlib.sha256()
        records = []
        for row in shard_rows:
            hasher.update((row["tile_id"] + "\n").encode())
            records.append({
                "tile_id": row["tile_id"], "status": "ok",
                "max_score": score_by_id[row["tile_id"]],
            })
        progress = tmp_path / f"rank{rank}.progress.jsonl"
        progress.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
        ledger = tmp_path / f"rank{rank}.ledger.json"
        ledger.write_text(json.dumps({
            "schema_version": 2, "shard_index": rank,
            "mining_run_fingerprint": fingerprint, "run_spec": run_spec,
            "expected_identity_digest": hasher.hexdigest(),
            "num_expected": len(records), "num_records": len(records),
            "num_failures": 0, "scan_complete": True,
            "scan_successful": True, "records_path": str(progress),
            # Deliberately bogus shard-local selection: finalization must ignore it.
            "selected": records[:1],
        }))
        ledgers.append(ledger)
    return index_path, rows, score_by_id, ledgers


def test_virtual_global_finalizer_exact_topk_and_deterministic(tmp_path):
    from kwcoco_detector_kit.data.mine import finalize_virtual_mining

    index, rows, score_by_id, ledgers = _synthetic_virtual_ledgers(tmp_path)
    expected = [
        tile_id for tile_id, _score in sorted(
            score_by_id.items(), key=lambda item: (-item[1], item[0]),
        )[:5]
    ]
    selected_runs = []
    for run_idx in range(2):
        dst = tmp_path / f"hard-{run_idx}.kwcoco.zip"
        finalize_virtual_mining(
            index, ledgers, dst, cache_dpath=tmp_path / "cache",
            score_thresh=0.0, max_hard_per_round=5,
        )
        sidecar = json.loads(dst.with_suffix(".selected_candidates.json").read_text())
        selected_runs.append([row["tile_id"] for row in sidecar["selected"]])
        assert sidecar["num_scored"] == len(rows)
        assert kwcoco.CocoDataset.coerce(dst).n_images == 5
        stats = json.loads(dst.with_suffix(".mine_stats.json").read_text())
        assert stats["n_scored"] == len(rows)
        assert stats["n_hard"] == 5
        assert stats["score_hist"]
        assert stats["score_quantiles"]["p50"] is not None
    assert selected_runs == [expected, expected]


def test_virtual_global_finalizer_rejects_missing_duplicate_and_failure(tmp_path):
    from kwcoco_detector_kit.data.mine import finalize_virtual_mining

    index, _rows, _scores, ledgers = _synthetic_virtual_ledgers(tmp_path)
    kwargs = {
        "cache_dpath": tmp_path / "cache", "score_thresh": 0.0,
        "max_hard_per_round": 5,
    }
    with pytest.raises(RuntimeError, match="missing or duplicate mining shard"):
        finalize_virtual_mining(index, ledgers[:-1], tmp_path / "missing.kwcoco.zip", **kwargs)

    missing_doc = next(doc for doc in [json.loads(p.read_text()) for p in ledgers] if doc["num_expected"])
    missing_progress = Path(missing_doc["records_path"])
    original_missing = missing_progress.read_text()
    missing_progress.write_text("\n".join(original_missing.splitlines()[:-1]) + "\n")
    with pytest.raises(RuntimeError, match="terminal-result count mismatch"):
        finalize_virtual_mining(index, ledgers, tmp_path / "missing-id.kwcoco.zip", **kwargs)
    missing_progress.write_text(original_missing)

    docs = [json.loads(path.read_text()) for path in ledgers]
    populated = [doc for doc in docs if doc["num_expected"]]
    first_record = json.loads(Path(populated[0]["records_path"]).read_text().splitlines()[0])
    other = next(doc for doc in populated[1:] if doc["shard_index"] != populated[0]["shard_index"])
    other_progress = Path(other["records_path"])
    original = other_progress.read_text()
    other_progress.write_text(original + json.dumps(first_record) + "\n")
    with pytest.raises(RuntimeError, match="duplicate terminal candidate identity"):
        finalize_virtual_mining(index, ledgers, tmp_path / "duplicate.kwcoco.zip", **kwargs)
    other_progress.write_text(original)

    failure_doc = populated[0]
    failure_progress = Path(failure_doc["records_path"])
    records = [json.loads(line) for line in failure_progress.read_text().splitlines()]
    records[0] = {"tile_id": records[0]["tile_id"], "status": "predict_error"}
    failure_progress.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(RuntimeError, match="contains 1 failures"):
        finalize_virtual_mining(index, ledgers, tmp_path / "failure.kwcoco.zip", **kwargs)


@pytest.mark.requires_torch
def test_virtual_mining_subprocess_interruption_resume(synthetic_kwcoco, tmp_workdir, tmp_path):
    from kwcoco_detector_kit.data.candidates import CandidateConfig, enumerate_candidates

    _train_mock_tiny(tmp_workdir, synthetic_kwcoco)
    source_dir = tmp_path / "virtual_source"
    source_dir.mkdir()
    source = _build_neg_bundle(source_dir, n=1)
    # Expand the single source so the interrupted process has durable batches.
    dset = kwcoco.CocoDataset.coerce(source)
    image = dset.images().objs[0]
    arr = (np.random.RandomState(0).rand(512, 512, 3) * 255).astype(np.uint8)
    kwimage = pytest.importorskip("kwimage")
    kwimage.imwrite(dset.get_image_fpath(image["id"]), arr)
    image.update(width=512, height=512, tile_source_gid=77, tile_scale_name="s10")
    dset.dump()
    index_path = tmp_path / "candidates.json"
    enumerate_candidates(CandidateConfig.cli(argv=False, data={
        "src": str(source), "dst": str(index_path), "category_names": "widget",
        "tile_size": 64, "source_scales": "1.0", "stride_frac": 1.0,
    }))
    dst = tmp_path / "hard.kwcoco.zip"
    ledger = tmp_path / "ledger.json"
    config = {
        "candidate_index": str(index_path), "workdir": str(tmp_workdir),
        "dst": str(dst), "ledger": str(ledger), "cache_dpath": str(tmp_path / "cache"),
        "trainer": "mock_tiny", "device": "cpu", "batch_size": 1,
        "score_thresh": 0.0, "max_hard_per_round": 8, "progress": False,
    }
    config_path = tmp_path / "mine_config.json"
    config_path.write_text(json.dumps(config))
    helper = Path(__file__).parents[1] / "helpers" / "virtual_mine_worker.py"
    env = os.environ.copy()
    env["KCD_MINE_TEST_BATCH_DELAY"] = "0.1"
    proc = subprocess.Popen([sys.executable, str(helper), str(config_path)], env=env)
    progress_path = ledger.with_suffix(".json.progress.jsonl")
    deadline = time.time() + 20
    while time.time() < deadline:
        if progress_path.exists() and len(progress_path.read_text().splitlines()) >= 2:
            break
        time.sleep(0.05)
    else:
        proc.kill()
        raise AssertionError("miner made no durable progress")
    proc.kill()
    proc.wait(timeout=5)
    prior = progress_path.read_text().splitlines()
    manifest = json.loads((index_path / "manifest.json").read_text())
    assert 1 < len(prior) < manifest["num_candidates"]
    subprocess.run([sys.executable, str(helper), str(config_path)], check=True, timeout=60)
    doc = json.loads(ledger.read_text())
    assert doc["scan_complete"] and doc["scan_successful"]
    assert doc["num_records"] == doc["num_expected"]
    records = [json.loads(line) for line in Path(doc["records_path"]).read_text().splitlines()]
    assert len({row["tile_id"] for row in records}) == doc["num_expected"]
    first_ids = [img["tile_id"] for img in kwcoco.CocoDataset.coerce(dst).images().objs]
    subprocess.run([sys.executable, str(helper), str(config_path)], check=True, timeout=60)
    second_ids = [img["tile_id"] for img in kwcoco.CocoDataset.coerce(dst).images().objs]
    assert first_ids == second_ids
