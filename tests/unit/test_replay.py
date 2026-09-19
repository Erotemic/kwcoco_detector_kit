from __future__ import annotations

from pathlib import Path

import kwcoco


def _bundle(path, rows):
    dset = kwcoco.CocoDataset()
    dset.fpath = str(path)
    for row in rows:
        dset.add_image(
            file_name=str(Path(path).parent / f"{row['tile_id']}.jpg"),
            width=8, height=8, tile_role="negative", **row,
        )
    dset.dump()
    return path


def test_replay_deduplicates_hards_and_keeps_fresh_random(tmp_path):
    from kwcoco_detector_kit.data.replay import ReplayConfig, run

    broad = _bundle(tmp_path / "broad.kwcoco.json", [
        {"tile_id": f"t{i}", "tile_source_gid": i // 2, "tile_scale_name": "s10"}
        for i in range(10)
    ])
    hard1 = _bundle(tmp_path / "hard1.kwcoco.json", [
        {"tile_id": "t0", "tile_source_gid": 0, "tile_scale_name": "s10", "max_pred_score": .7},
        {"tile_id": "t1", "tile_source_gid": 0, "tile_scale_name": "s10", "max_pred_score": .8},
    ])
    hard2 = _bundle(tmp_path / "hard2.kwcoco.json", [
        {"tile_id": "t0", "tile_source_gid": 0, "tile_scale_name": "s10", "max_pred_score": .9},
        {"tile_id": "t2", "tile_source_gid": 1, "tile_scale_name": "s10", "max_pred_score": .6},
    ])
    dst = tmp_path / "replay.kwcoco.json"
    cfg = ReplayConfig.cli(argv=False, data={
        "hard_kwcocos": [str(hard1), str(hard2)], "broad_kwcoco": str(broad),
        "dst": str(dst), "hard_quota": 3, "random_quota": 2, "seed": 11,
    })
    run(cfg)
    out = kwcoco.CocoDataset.coerce(str(dst))
    ids = [img["tile_id"] for img in out.images().objs]
    assert len(ids) == len(set(ids)) == 5
    origins = [img["replay_origin"] for img in out.images().objs]
    assert origins.count("hard") == 3
    assert origins.count("random") == 2
