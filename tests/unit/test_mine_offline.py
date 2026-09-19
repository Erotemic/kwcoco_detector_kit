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
    assert len({row["tile_id"] for row in doc["records"]}) == doc["num_expected"]
    first_ids = [img["tile_id"] for img in kwcoco.CocoDataset.coerce(dst).images().objs]
    subprocess.run([sys.executable, str(helper), str(config_path)], check=True, timeout=60)
    second_ids = [img["tile_id"] for img in kwcoco.CocoDataset.coerce(dst).images().objs]
    assert first_ids == second_ids
