"""Unit tests for scripts/apply_scheme_to_kwcoco.py.

The script's job: take a kwcoco file with a single 'sealion' category +
a `source_category` field on each annotation, and emit a per-scheme
kwcoco file with the scheme's target_classes assigned to category_ids
1..N. Mirrors what build_scheme_kwcoco.py does but on one file.

This test guards the universal-tile pipeline contract:
    tile(universal_source) -> apply_scheme(scheme=X)
        == build_scheme_kwcoco(source, scheme=X) -> tile(scheme_bundle)
in terms of annotation set + category IDs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("kwcoco")
import kwcoco
import yaml


def _make_universal_fixture(bundle_dpath: Path, *,
                            source_cat_per_ann: list[str]) -> Path:
    """Build a kwcoco bundle that mimics the universal source: one
    'sealion' category, source_category on each annotation."""
    import kwimage
    rng = np.random.RandomState(0)
    asset_dpath = bundle_dpath / "assets"
    asset_dpath.mkdir(parents=True, exist_ok=True)
    dset = kwcoco.CocoDataset()
    dset.fpath = str(bundle_dpath / "universal.kwcoco.zip")
    cid = dset.add_category(name="sealion")
    boxes_per_image = max(1, len(source_cat_per_ann) // 3)
    n_images = max(1, (len(source_cat_per_ann) + boxes_per_image - 1)
                   // boxes_per_image)
    ann_iter = iter(source_cat_per_ann)
    for k in range(n_images):
        img = (rng.rand(64, 64, 3) * 255).astype(np.uint8)
        fpath = asset_dpath / f"img_{k:04d}.jpg"
        kwimage.imwrite(str(fpath), img)
        gid = dset.add_image(
            file_name=str(fpath.relative_to(bundle_dpath)),
            width=64, height=64, name=f"img_{k:04d}",
        )
        for _ in range(boxes_per_image):
            try:
                src = next(ann_iter)
            except StopIteration:
                break
            dset.add_annotation(
                image_id=gid, category_id=cid,
                bbox=[10.0, 10.0, 20.0, 20.0], area=400.0, iscrowd=0,
                source_category=src,
            )
    dset.dump()
    return Path(dset.fpath)


@pytest.fixture
def pup_scheme_yaml(tmp_path) -> Path:
    schemes = {
        "schemes": {
            "pup_vs_nonpup": {
                "description": "2-class pup vs nonpup_sealion",
                "num_classes": 2,
                "target_order": ["pup", "nonpup_sealion"],
                "mapping": {
                    "P": "pup",
                    "B": "nonpup_sealion",
                    "S": "nonpup_sealion",
                    "F": "nonpup_sealion",
                    "J": "nonpup_sealion",
                },
                "drop": ["NFS", "O", "DP", "DN"],
            }
        }
    }
    fpath = tmp_path / "schemes.yaml"
    fpath.write_text(yaml.safe_dump(schemes))
    return fpath


def test_apply_scheme_remaps_categories_in_target_order(tmp_path, pup_scheme_yaml):
    """The new CLI mirrors what build_scheme_kwcoco does on the
    upstream sealion-collapsed bundle, but operates on a single file.
    Same scheme → same category_ids, same drop behavior."""
    src = _make_universal_fixture(
        tmp_path / "src", source_cat_per_ann=["P", "B", "F", "J", "NFS", "O"],
    )
    dst = tmp_path / "out.kwcoco.zip"

    import subprocess, sys
    script = (Path(__file__).resolve().parent.parent.parent /
              "scripts" / "apply_scheme_to_kwcoco.py")
    result = subprocess.run(
        [sys.executable, str(script),
         "--src", str(src),
         "--dst", str(dst),
         "--scheme", "pup_vs_nonpup",
         "--schemes-file", str(pup_scheme_yaml)],
        check=True, capture_output=True, text=True,
    )
    assert dst.exists(), f"output missing: {result.stderr}"

    out = kwcoco.CocoDataset.coerce(str(dst))
    cats = {c["id"]: c["name"] for c in out.dataset["categories"]}
    assert cats == {1: "pup", 2: "nonpup_sealion"}, cats

    counts = {1: 0, 2: 0}
    for ann in out.annots().objs:
        counts[ann["category_id"]] = counts.get(ann["category_id"], 0) + 1
    # P -> pup (1 ann); B, F, J -> nonpup_sealion (3 anns); NFS, O dropped.
    assert counts == {1: 1, 2: 3}, counts


def test_apply_scheme_preserves_source_category(tmp_path, pup_scheme_yaml):
    """source_category survives the apply step so a future audit can
    trace which raw VIAME code produced each remapped annotation."""
    src = _make_universal_fixture(
        tmp_path / "src", source_cat_per_ann=["P", "B"],
    )
    dst = tmp_path / "out.kwcoco.zip"
    import subprocess, sys
    script = (Path(__file__).resolve().parent.parent.parent /
              "scripts" / "apply_scheme_to_kwcoco.py")
    subprocess.run(
        [sys.executable, str(script),
         "--src", str(src), "--dst", str(dst),
         "--scheme", "pup_vs_nonpup",
         "--schemes-file", str(pup_scheme_yaml)],
        check=True,
    )

    out = kwcoco.CocoDataset.coerce(str(dst))
    src_cats = {ann.get("source_category") for ann in out.annots().objs}
    assert "P" in src_cats and "B" in src_cats, src_cats
