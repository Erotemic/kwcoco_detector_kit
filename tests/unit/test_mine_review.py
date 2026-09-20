"""Human-review artifacts for hard-negative mining remain source-linked."""
from __future__ import annotations

import json
from pathlib import Path


def test_tile_bbox_maps_back_to_source_coordinates():
    from kwcoco_detector_kit.data.review_mine import _bbox_tile_to_source

    row = {
        "tile_actual_scale_xy": [0.5, 0.25],
        "tile_scaled_extent_xyxy": [100, 50, 200, 150],
    }
    got = _bbox_tile_to_source(row, [10, 20, 30, 40])
    assert got == [220.0, 280.0, 260.0, 360.0]


def test_review_queue_preserves_source_truth_provenance(tmp_path):
    import kwcoco
    import kwimage
    import numpy as np

    from kwcoco_detector_kit.data.candidates import (
        CandidateConfig, enumerate_candidates, iter_candidate_records,
    )
    from kwcoco_detector_kit.data.review_mine import ReviewMineConfig, run
    from kwcoco_detector_kit.data.tile_cache import canonical_digest

    bundle = tmp_path / "source"
    bundle.mkdir()
    dset = kwcoco.CocoDataset()
    dset.fpath = str(bundle / "source.kwcoco.zip")
    cid = dset.add_category(name="poop")
    image_path = bundle / "frame.jpg"
    kwimage.imwrite(str(image_path), np.zeros((128, 128, 3), dtype=np.uint8))
    image_path.with_suffix(".json").write_text("{}\n")
    gid = dset.add_image(file_name="frame.jpg", width=128, height=128, name="frame")
    # Truth elsewhere on the source image: the candidate crop itself remains safe.
    dset.add_annotation(image_id=gid, category_id=cid, bbox=[96, 96, 16, 16])
    dset.dump()

    index_path = tmp_path / "candidates"
    enumerate_candidates(CandidateConfig.cli(argv=False, data={
        "src": str(dset.fpath), "dst": str(index_path),
        "category_names": "poop", "tile_size": 32,
        "source_scales": "1.0", "stride_frac": 1.0,
        "min_source_scale_long_side": 1,
        "negative_safety_margin": 0,
    }))
    rows = list(iter_candidate_records(index_path))
    assert rows
    chosen = rows[:2]

    run_spec = {"test": True}
    fingerprint = canonical_digest(run_spec)
    progress = tmp_path / "rank0.progress.jsonl"
    progress.write_text("".join(
        json.dumps({
            "tile_id": row["tile_id"], "status": "ok",
            "max_score": 0.99 - idx * 0.01,
            "top_label": 0, "top_bbox_xyxy": [4, 5, 20, 21],
        }) + "\n"
        for idx, row in enumerate(chosen)
    ))
    ledger = tmp_path / "rank0.ledger.json"
    ledger.write_text(json.dumps({
        "schema_version": 2,
        "scan_complete": True,
        "scan_successful": True,
        "mining_run_fingerprint": fingerprint,
        "run_spec": run_spec,
        "records_path": str(progress),
    }))
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps({
        "selected": [
            {"tile_id": row["tile_id"], "max_score": 0.99 - idx * 0.01}
            for idx, row in enumerate(chosen)
        ]
    }))

    dst = tmp_path / "review"
    cfg = ReviewMineConfig.cli(argv=False, data={
        "candidate_index": str(index_path),
        "ledgers": [str(ledger)],
        "selected_candidates": str(selected),
        "dst_dpath": str(dst),
        "top_n": 2,
        "per_source": 0,
        "make_previews": False,
    })
    queue_path = run(cfg)
    queue = json.loads(Path(queue_path).read_text())
    assert queue["num_items"] == 2
    item = queue["items"][0]
    assert item["source_gid"] == gid
    assert item["source_fpath"] == str(image_path.resolve())
    assert item["adjacent_json_sidecar"] == str(image_path.with_suffix(".json").resolve())
    assert item["adjacent_json_sidecar_exists"] is True
    assert item["top_bbox_xyxy_in_source"] is not None
    assert item["review_status"] == "unreviewed"

    review_dset = kwcoco.CocoDataset.coerce(str(dst / "review.kwcoco.zip"))
    names = {cat["name"] for cat in review_dset.cats.values()}
    assert "poop" in names
    assert "__hard_negative_review__" in names
    assert (dst / "review_queue.tsv").is_file()
