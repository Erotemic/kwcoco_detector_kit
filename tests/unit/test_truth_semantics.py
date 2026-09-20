from __future__ import annotations

import json


def _make_semantic_dataset(tmp_path):
    import kwcoco
    import kwimage
    import numpy as np

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    fpath = tmp_path / "image.png"
    kwimage.imwrite(str(fpath), image)
    dset = kwcoco.CocoDataset()
    dset.fpath = str(tmp_path / "truth.kwcoco.json")
    cids = {name: dset.add_category(name=name) for name in ["poop", "leaf", "unknown", "ignore"]}
    gid = dset.add_image(file_name=str(fpath), width=64, height=64)
    dset.add_annotation(image_id=gid, category_id=cids["poop"], bbox=[0, 0, 16, 16])
    dset.add_annotation(image_id=gid, category_id=cids["leaf"], bbox=[32, 0, 16, 16])
    dset.add_annotation(image_id=gid, category_id=cids["unknown"], bbox=[0, 32, 16, 16])
    dset.add_annotation(image_id=gid, category_id=cids["ignore"], bbox=[32, 32, 16, 16])

    # KWCoco requires image ``file_name`` values to be unique.  Use a second
    # physical asset for the uncategorized-annotation case rather than
    # registering the same path twice under different image records.
    uncategorized_fpath = tmp_path / "uncategorized.png"
    kwimage.imwrite(str(uncategorized_fpath), image)
    gid2 = dset.add_image(
        file_name=str(uncategorized_fpath),
        width=64,
        height=64,
        name="uncategorized",
    )
    dset.add_annotation(image_id=gid2, category_id=None, bbox=[16, 16, 16, 16])
    dset.dump()
    return dset, gid, gid2


def test_truth_semantics_candidate_and_tile_safety(tmp_path):
    import kwcoco

    from kwcoco_detector_kit.data.candidates import CandidateConfig, enumerate_candidates, materialize_candidate_records
    from kwcoco_detector_kit.data.tile import TileConfig, run as run_tile

    dset, gid, gid2 = _make_semantic_dataset(tmp_path)
    semantics = dict(
        category_names="poop",
        ignore_categories="unknown,ignore",
        uncategorized_annotation_policy="ignore",
        default_non_target_policy="background",
        unclassified_category_policy="ignore",
    )
    cand_dpath = tmp_path / "candidates"
    cfg = CandidateConfig.cli(argv=False, data={
        "src": dset.fpath,
        "dst": str(cand_dpath),
        "tile_size": 32,
        "source_scales": "1.0",
        "stride_frac": 1.0,
        "min_keep_fraction": 0.3,
        "min_gt_area_frac": 0.001,
        "negative_safety_margin": 0,
        "min_source_scale_long_side": 1,
        **semantics,
    }, strict=True)
    enumerate_candidates(cfg)
    rows = materialize_candidate_records(cand_dpath)
    # Only the leaf-only quadrant is trusted background. Unknown/ignore and
    # the uncategorized source are excluded; poop is positive, not negative.
    assert len(rows) == 1
    row = rows[0]
    assert row["tile_source_gid"] == gid
    assert row["tile_scaled_extent_xyxy"][:2] == [32, 0]
    assert row["negative_origin"] == "safe_background_window"
    manifest = json.loads((cand_dpath / "manifest.json").read_text())
    assert manifest["policy"]["truth_semantics"]["ignore_categories"] == ["unknown", "ignore"]

    tiled_fpath = tmp_path / "tiles.kwcoco.zip"
    tile_cfg = TileConfig.cli(argv=False, data={
        "src": dset.fpath,
        "dst": str(tiled_fpath),
        "mode": "multiscale",
        "tile_size": 32,
        "source_scales": "1.0",
        "stride_frac": 1.0,
        "min_keep_fraction": 0.3,
        "min_gt_area_frac": 0.001,
        "min_source_scale_long_side": 1,
        "keep_negative": True,
        "negative_keep_fraction": 1.0,
        **semantics,
    }, strict=True)
    run_tile(tile_cfg)
    tiled = kwcoco.CocoDataset.coerce(str(tiled_fpath))
    roles = [img["tile_role"] for img in tiled.images().objs]
    assert roles.count("positive") == 1
    assert roles.count("negative") == 1
    assert all(img["tile_source_gid"] != gid2 for img in tiled.images().objs)


def test_truth_review_classification(tmp_path):
    from kwcoco_detector_kit.data.truth_review import classify_prediction
    from kwcoco_detector_kit.data.truth_semantics import TruthSemantics

    dset, gid, _ = _make_semantic_dataset(tmp_path)
    sem = TruthSemantics.coerce(
        target_categories=["poop"],
        ignore_categories=["unknown", "ignore"],
        uncategorized_annotation_policy="ignore",
    )
    assert classify_prediction(dset, gid, [0, 0, 16, 16], sem)["classification"] == "matched_target"
    assert classify_prediction(dset, gid, [32, 0, 48, 16], sem)["classification"] == "known_distractor"
    assert classify_prediction(dset, gid, [0, 32, 16, 48], sem)["classification"] == "uncertain_region"
    assert classify_prediction(dset, gid, [48, 16, 63, 31], sem)["classification"] == "unexplained_prediction"

def test_positive_overlapping_uncertain_region_is_dropped(tmp_path):
    import kwcoco
    import kwimage
    import numpy as np

    from kwcoco_detector_kit.data.candidates import CandidateConfig, enumerate_candidates, materialize_candidate_records
    from kwcoco_detector_kit.data.tile import TileConfig, run as run_tile

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    fpath = tmp_path / "mixed.png"
    kwimage.imwrite(str(fpath), image)
    dset = kwcoco.CocoDataset()
    dset.fpath = str(tmp_path / "mixed.kwcoco.json")
    poop_cid = dset.add_category(name="poop")
    unknown_cid = dset.add_category(name="unknown")
    gid = dset.add_image(file_name=str(fpath), width=32, height=32)
    dset.add_annotation(image_id=gid, category_id=poop_cid, bbox=[4, 4, 16, 16])
    dset.add_annotation(image_id=gid, category_id=unknown_cid, bbox=[8, 8, 16, 16])
    dset.dump()

    semantics = dict(
        category_names="poop",
        ignore_categories="unknown",
        uncategorized_annotation_policy="ignore",
        default_non_target_policy="background",
        unclassified_category_policy="ignore",
    )
    cand_dpath = tmp_path / "mixed_candidates"
    enumerate_candidates(CandidateConfig.cli(argv=False, data={
        "src": dset.fpath,
        "dst": str(cand_dpath),
        "tile_size": 32,
        "source_scales": "1.0",
        "stride_frac": 1.0,
        "min_keep_fraction": 0.3,
        "min_gt_area_frac": 0.001,
        "negative_safety_margin": 0,
        "min_source_scale_long_side": 1,
        **semantics,
    }, strict=True))
    assert materialize_candidate_records(cand_dpath) == []

    tiled_fpath = tmp_path / "mixed_tiles.kwcoco.zip"
    run_tile(TileConfig.cli(argv=False, data={
        "src": dset.fpath,
        "dst": str(tiled_fpath),
        "mode": "multiscale",
        "tile_size": 32,
        "source_scales": "1.0",
        "stride_frac": 1.0,
        "min_keep_fraction": 0.3,
        "min_gt_area_frac": 0.001,
        "min_source_scale_long_side": 1,
        "keep_negative": True,
        "negative_keep_fraction": 1.0,
        **semantics,
    }, strict=True))
    tiled = kwcoco.CocoDataset.coerce(str(tiled_fpath))
    assert len(tiled.images()) == 0

