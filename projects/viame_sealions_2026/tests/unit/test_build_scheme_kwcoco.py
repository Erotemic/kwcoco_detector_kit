"""Unit tests for scripts/build_scheme_kwcoco.py — synthetic fixtures only.

The script reads kwcoco bundles that have a single ``sealion`` category
with the original class encoded in each annotation's ``source_category``
field (matching the convention of training_ready_v1/*.kwcoco.zip), and
remaps to a multi-class scheme.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("kwcoco")
import kwcoco
import yaml


def _make_collapsed_fixture(bundle_dpath: Path, *,
                            source_cat_per_ann: list[str],
                            n_images: int = 4) -> Path:
    """Build a synthetic kwcoco that mirrors training_ready_v1 layout:
    one 'sealion' category, source_category preserved on each annotation.
    """
    import kwimage
    rng = np.random.RandomState(0)
    asset_dpath = bundle_dpath / "synth_assets"
    asset_dpath.mkdir(parents=True, exist_ok=True)
    dset = kwcoco.CocoDataset()
    dset.fpath = str(bundle_dpath / "synth.kwcoco.zip")
    cid = dset.add_category(name="sealion")
    boxes_per_image = max(1, len(source_cat_per_ann) // n_images)
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
def pup_vs_nonpup_scheme(tmp_path) -> Path:
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


def test_pup_vs_nonpup_remap_assigns_ids_in_order(tmp_path, pup_vs_nonpup_scheme):
    from build_scheme_kwcoco import load_scheme, remap_split, ordered_target_names

    src = _make_collapsed_fixture(
        tmp_path / "src",
        source_cat_per_ann=["P", "B", "F", "J", "NFS", "O"],
        n_images=6,
    )
    scheme = load_scheme(pup_vs_nonpup_scheme, "pup_vs_nonpup")
    assert ordered_target_names(scheme) == ["pup", "nonpup_sealion"]

    dst = tmp_path / "out" / "remapped.kwcoco.zip"
    result = remap_split(src, dst, scheme, dry_run=False)
    dset = kwcoco.CocoDataset.coerce(str(dst))

    cats = {c["id"]: c["name"] for c in dset.dataset["categories"]}
    assert cats == {1: "pup", 2: "nonpup_sealion"}, cats
    assert result["per_target_class"] == {"pup": 1, "nonpup_sealion": 3}
    # NFS + O dropped; DP/DN not in the fixture.
    assert result["dropped_by_source"] == {"NFS": 1, "O": 1}
    assert result["n_unknown_source_categories"] == 0
    # Each annotation in the output should have category_id in {1, 2}.
    cat_ids = {ann["category_id"] for ann in dset.annots().objs}
    assert cat_ids == {1, 2}
    # source_category must be preserved for traceability.
    src_cats = {ann.get("source_category") for ann in dset.annots().objs}
    assert "P" in src_cats and "B" in src_cats


def test_unknown_source_category_is_counted(tmp_path, pup_vs_nonpup_scheme):
    from build_scheme_kwcoco import load_scheme, remap_split

    src = _make_collapsed_fixture(
        tmp_path / "src", source_cat_per_ann=["P", "MYSTERY_CODE"], n_images=2,
    )
    scheme = load_scheme(pup_vs_nonpup_scheme, "pup_vs_nonpup")
    dst = tmp_path / "out.kwcoco.zip"
    result = remap_split(src, dst, scheme)
    # MYSTERY_CODE is not in mapping AND not in drop -> counted as unknown.
    assert result["n_unknown_source_categories"] == 1
    assert result["per_target_class"] == {"pup": 1}


def test_target_order_missing_class_raises(tmp_path):
    """A target referenced in the mapping but absent from target_order
    is a config bug — fail loudly so we don't silently train with the
    wrong class count."""
    from build_scheme_kwcoco import ordered_target_names

    bad_scheme = {
        "target_order": ["pup"],  # missing nonpup_sealion
        "mapping": {"P": "pup", "B": "nonpup_sealion"},
    }
    with pytest.raises(ValueError, match="missing classes"):
        ordered_target_names(bad_scheme)


def test_output_file_names_are_resolvable_from_new_location(
    tmp_path, pup_vs_nonpup_scheme,
):
    """The scheme bundle lives one level deeper than the source bundle
    (training_ready_v1/by_scheme/<scheme>/). A relative file_name in
    the source would point at a nonexistent dir from there — the
    builder must rewrite each image's file_name to an absolute path."""
    from build_scheme_kwcoco import load_scheme, remap_split

    src = _make_collapsed_fixture(
        tmp_path / "src", source_cat_per_ann=["P", "B"], n_images=2,
    )
    scheme = load_scheme(pup_vs_nonpup_scheme, "pup_vs_nonpup")
    dst = tmp_path / "by_scheme" / "pup_vs_nonpup" / "out.kwcoco.zip"
    remap_split(src, dst, scheme)

    out_dset = kwcoco.CocoDataset.coerce(str(dst))
    for img in out_dset.dataset["images"]:
        fname = img["file_name"]
        assert Path(fname).is_absolute(), (
            f"file_name should be absolute so the scheme bundle is "
            f"portable across nested directories; got {fname!r}"
        )
        # And the resolved file must actually exist — i.e. the rewrite
        # used the source bundle's coordinate, not a fresh one.
        resolved = out_dset.get_image_fpath(img["id"])
        assert Path(resolved).exists(), f"resolved path does not exist: {resolved}"


def test_dry_run_does_not_write(tmp_path, pup_vs_nonpup_scheme):
    from build_scheme_kwcoco import load_scheme, remap_split

    src = _make_collapsed_fixture(
        tmp_path / "src", source_cat_per_ann=["P", "B"], n_images=2,
    )
    scheme = load_scheme(pup_vs_nonpup_scheme, "pup_vs_nonpup")
    dst = tmp_path / "out.kwcoco.zip"
    result = remap_split(src, dst, scheme, dry_run=True)
    assert not dst.exists()
    assert result["n_annotations"] == 2
