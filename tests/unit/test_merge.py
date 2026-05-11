"""Tests for data.merge — round-N positive+negative union."""
from __future__ import annotations

from pathlib import Path

import kwcoco
import numpy as np
import pytest


def _build_role_bundle(bundle_dpath: Path, role: str, n: int,
                        category_name: str = "widget", seed: int = 0) -> Path:
    import kwimage

    rng = np.random.RandomState(seed)
    asset_dpath = bundle_dpath / f"{role}_assets"
    asset_dpath.mkdir(parents=True, exist_ok=True)
    dset = kwcoco.CocoDataset()
    dset.fpath = str(bundle_dpath / f"{role}.kwcoco.zip")
    cid = dset.add_category(name=category_name)
    for k in range(n):
        img = (rng.rand(64, 64, 3) * 255).astype(np.uint8)
        fpath = asset_dpath / f"{role}_{k:04d}.jpg"
        kwimage.imwrite(str(fpath), img)
        gid = dset.add_image(
            file_name=str(fpath.relative_to(bundle_dpath)),
            width=64, height=64, tile_role=role,
        )
        if role == "positive":
            dset.add_annotation(
                image_id=gid, category_id=cid,
                bbox=[10.0, 10.0, 20.0, 20.0], area=400.0, iscrowd=0,
            )
    dset.dump()
    return Path(dset.fpath)


def _merge(pos, neg, dst, *, neg_over_pos, round_index=0, seed=0,
           category_name="widget"):
    from kwcoco_detector_kit.data.merge import MergeConfig, run
    cfg = MergeConfig.cli(
        argv=False,
        data={
            "pos_kwcoco": str(pos), "neg_kwcoco": str(neg), "dst": str(dst),
            "neg_over_pos": neg_over_pos, "round_index": round_index,
            "seed": seed, "category_name": category_name,
        },
    )
    run(cfg)
    return kwcoco.CocoDataset.coerce(str(dst))


def test_merge_round0_ratio_3(tmp_path):
    pos = _build_role_bundle(tmp_path, "positive", 4)
    neg = _build_role_bundle(tmp_path / "neg_dir", "negative", 20)
    dst = tmp_path / "out.kwcoco.zip"
    dset = _merge(pos, neg, dst, neg_over_pos=3.0)
    n_pos = sum(1 for img in dset.images().objs if img.get("tile_role") == "positive")
    n_neg = sum(1 for img in dset.images().objs if img.get("tile_role") == "negative")
    assert n_pos == 4
    # ratio 3 * 4 = 12 negs (capped by 20 available)
    assert n_neg == 12


def test_merge_neg_over_pos_zero_keeps_all_negs(tmp_path):
    pos = _build_role_bundle(tmp_path, "positive", 2)
    neg = _build_role_bundle(tmp_path / "neg_dir", "negative", 7)
    dst = tmp_path / "out.kwcoco.zip"
    dset = _merge(pos, neg, dst, neg_over_pos=0.0)
    n_neg = sum(1 for img in dset.images().objs if img.get("tile_role") == "negative")
    assert n_neg == 7


def test_merge_ratio_capped_by_actual_neg_count(tmp_path):
    pos = _build_role_bundle(tmp_path, "positive", 5)
    neg = _build_role_bundle(tmp_path / "neg_dir", "negative", 3)
    dst = tmp_path / "out.kwcoco.zip"
    dset = _merge(pos, neg, dst, neg_over_pos=10.0)
    n_neg = sum(1 for img in dset.images().objs if img.get("tile_role") == "negative")
    assert n_neg == 3, "expected target 50 negs but only 3 available — caller hit cap"


def test_merge_empty_positives_raises(tmp_path):
    pos = _build_role_bundle(tmp_path, "positive", 0)
    neg = _build_role_bundle(tmp_path / "neg_dir", "negative", 5)
    dst = tmp_path / "out.kwcoco.zip"
    with pytest.raises(RuntimeError):
        _merge(pos, neg, dst, neg_over_pos=1.0)


def test_merge_preserves_positive_annotations(tmp_path):
    pos = _build_role_bundle(tmp_path, "positive", 3)
    neg = _build_role_bundle(tmp_path / "neg_dir", "negative", 2)
    dst = tmp_path / "out.kwcoco.zip"
    dset = _merge(pos, neg, dst, neg_over_pos=1.0)
    # 3 positives x 1 ann each = 3 anns total; negatives have no anns
    assert dset.n_annots == 3
