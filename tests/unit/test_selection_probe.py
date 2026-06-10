"""Probe builder: stratified frozen subset with a stable probe_id."""
from __future__ import annotations

import json

from kwcoco_detector_kit.selection.probe import build_probe


def _synthetic_vali(tmp_path):
    import kwcoco
    dset = kwcoco.CocoDataset()
    cid_common = dset.add_category(name="common")
    cid_rare = dset.add_category(name="rare")
    # 20 common-only frames, 4 rare frames, 6 empty frames
    for i in range(30):
        gid = dset.add_image(
            file_name=str(tmp_path / f"img_{i:03d}.jpg"),
            name=f"img_{i:03d}", width=4000, height=3000,
        )
        if i < 20:
            for _ in range(10):
                dset.add_annotation(image_id=gid, category_id=cid_common,
                                    bbox=[10, 10, 50, 50])
        elif i < 24:
            dset.add_annotation(image_id=gid, category_id=cid_rare,
                                bbox=[10, 10, 50, 50])
            dset.add_annotation(image_id=gid, category_id=cid_common,
                                bbox=[80, 80, 50, 50])
        # else: empty
    fpath = tmp_path / "vali.kwcoco.zip"
    dset.fpath = str(fpath)
    dset.dump()
    return fpath


def test_probe_covers_rare_class_and_respects_budget(tmp_path):
    src = _synthetic_vali(tmp_path)
    result = build_probe(src, tmp_path / "probe", frames=10, seed=0,
                         empty_frac=0.2, log=lambda *a: None)
    manifest = result.manifest
    assert manifest["n_images"] == 10
    # rare-positive enrichment: the rare class is covered
    assert manifest["class_support"]["rare"] >= 3
    # empty frames included at the requested fraction
    import kwcoco
    probe = kwcoco.CocoDataset(str(result.probe_kwcoco_fpath))
    per_img = {gid: len(probe.gid_to_aids.get(gid, [])) for gid in probe.images()}
    n_empty = sum(1 for n in per_img.values() if n == 0)
    assert n_empty == 2


def test_probe_is_frozen_and_deterministic(tmp_path):
    src = _synthetic_vali(tmp_path)
    r1 = build_probe(src, tmp_path / "probe", frames=10, seed=0,
                     log=lambda *a: None)
    # second call with same params reuses the frozen manifest verbatim
    r2 = build_probe(src, tmp_path / "probe", frames=10, seed=0,
                     log=lambda *a: None)
    assert r1.probe_id == r2.probe_id
    assert json.loads(r1.manifest_fpath.read_text())["probe_id"] == r1.probe_id


def test_probe_rebuild_with_new_params_is_a_new_identity(tmp_path):
    src = _synthetic_vali(tmp_path)
    r1 = build_probe(src, tmp_path / "p1", frames=10, seed=0,
                     log=lambda *a: None)
    r2 = build_probe(src, tmp_path / "p2", frames=10, seed=7,
                     log=lambda *a: None)
    assert r1.probe_id != r2.probe_id
