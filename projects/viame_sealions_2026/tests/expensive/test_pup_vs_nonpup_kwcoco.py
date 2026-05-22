"""Expensive — verifies the real pup_vs_nonpup kwcoco bundles.

Reads training_ready_v1/by_scheme/pup_vs_nonpup/{train,vali,test}.kwcoco.zip
and checks the contracts the trainer relies on:

- Exactly 2 categories named 'pup' and 'nonpup_sealion'.
- Category IDs are stable (1 = pup, 2 = nonpup_sealion).
- No annotation has a category_id outside that set.
- Every annotation has a ``source_category`` field (traceability).
- Class balance matches the scheme_report.json the builder wrote.

Skips if the bundles are not on disk yet (fresh checkout without
having run scripts/build_scheme_kwcoco.py --scheme pup_vs_nonpup).
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

pytest.importorskip("kwcoco")
import kwcoco


SCHEME = "pup_vs_nonpup"
EXPECTED_CATS = {1: "pup", 2: "nonpup_sealion"}


@pytest.fixture(scope="module")
def scheme_dir(training_ready_dir):
    d = training_ready_dir / "by_scheme" / SCHEME
    if not d.exists():
        pytest.skip(
            f"{d} not present; run "
            f"`python3 scripts/build_scheme_kwcoco.py --scheme {SCHEME}` first"
        )
    return d


@pytest.fixture(scope="module")
def scheme_report(scheme_dir):
    fpath = scheme_dir / "scheme_report.json"
    if not fpath.exists():
        pytest.skip(f"{fpath} missing")
    return json.loads(fpath.read_text())


@pytest.mark.parametrize("split", ["train", "vali", "test"])
def test_split_has_expected_categories(scheme_dir, split):
    fpath = scheme_dir / f"{split}.kwcoco.zip"
    if not fpath.exists():
        pytest.skip(f"{fpath} missing")
    dset = kwcoco.CocoDataset.coerce(str(fpath))
    cats = {c["id"]: c["name"] for c in dset.dataset["categories"]}
    assert cats == EXPECTED_CATS, (
        f"{split}: categories {cats!r} != expected {EXPECTED_CATS!r}; "
        "re-run build_scheme_kwcoco.py if the source bundles changed"
    )


@pytest.mark.parametrize("split", ["train", "vali", "test"])
def test_no_annotation_falls_outside_target_classes(scheme_dir, split):
    fpath = scheme_dir / f"{split}.kwcoco.zip"
    if not fpath.exists():
        pytest.skip(f"{fpath} missing")
    dset = kwcoco.CocoDataset.coerce(str(fpath))
    bad = [ann["id"] for ann in dset.annots().objs
           if ann["category_id"] not in EXPECTED_CATS]
    assert not bad, f"{split}: {len(bad)} annotations have unexpected category_id"


@pytest.mark.parametrize("split", ["train", "vali", "test"])
def test_every_annotation_preserves_source_category(scheme_dir, split):
    fpath = scheme_dir / f"{split}.kwcoco.zip"
    if not fpath.exists():
        pytest.skip(f"{fpath} missing")
    dset = kwcoco.CocoDataset.coerce(str(fpath))
    n_missing = sum(
        1 for ann in dset.annots().objs if not ann.get("source_category")
    )
    assert n_missing == 0, (
        f"{split}: {n_missing} annotations missing source_category; "
        "traceability back to the raw VIAME code (B/S/F/J/P/...) is lost"
    )


@pytest.mark.parametrize("split", ["train", "vali", "test"])
def test_per_class_counts_match_scheme_report(scheme_dir, scheme_report, split):
    fpath = scheme_dir / f"{split}.kwcoco.zip"
    if not fpath.exists():
        pytest.skip(f"{fpath} missing")
    dset = kwcoco.CocoDataset.coerce(str(fpath))
    cat_id_to_name = {c["id"]: c["name"] for c in dset.dataset["categories"]}
    counts = collections.Counter(
        cat_id_to_name[ann["category_id"]] for ann in dset.annots().objs
    )
    expected = scheme_report["splits"][split]["per_target_class"]
    assert dict(counts) == expected, (
        f"{split}: on-disk per-class counts {dict(counts)!r} "
        f"!= scheme_report {expected!r}"
    )


@pytest.mark.parametrize("split", ["train", "vali", "test"])
def test_image_paths_resolve_on_disk(scheme_dir, split):
    """Catch the broken-relative-path failure mode: the scheme bundle
    lives one level deeper than the source kwcoco, so a relative
    file_name would resolve to a nonexistent dir from there. We sample
    the first few images per split — sampling, not exhaustive, because
    on a partial dataset (e.g. only some Redacted_Imagery years
    unpacked) some entries may legitimately be missing.
    """
    fpath = scheme_dir / f"{split}.kwcoco.zip"
    if not fpath.exists():
        pytest.skip(f"{fpath} missing")
    dset = kwcoco.CocoDataset.coerce(str(fpath))
    n_check = min(10, dset.n_images)
    sample_gids = list(dset.images())[:n_check]
    missing = []
    for gid in sample_gids:
        try:
            p = dset.get_image_fpath(gid)
        except Exception as exc:
            missing.append((gid, f"resolve error: {exc}"))
            continue
        if not Path(p).exists():
            missing.append((gid, f"path does not exist: {p}"))
    assert not missing, (
        f"{split}: first {n_check} images do not resolve to real files. "
        f"Likely the bundle stores file_name relative to the source kwcoco "
        f"location; rerun `scripts/build_scheme_kwcoco.py --scheme pup_vs_nonpup`. "
        f"First failure: {missing[0]}"
    )


def test_pup_class_is_id_1(scheme_dir):
    """Stable ID assignment matters because the trained model emits
    integer labels and we want label==0 -> pup at inference time. The
    builder uses 1-indexed IDs in the kwcoco; the kit's MSCOCO export
    converts to 0-indexed for training, so this maps to label 0."""
    for split in ("train", "vali", "test"):
        fpath = scheme_dir / f"{split}.kwcoco.zip"
        if not fpath.exists():
            continue
        dset = kwcoco.CocoDataset.coerce(str(fpath))
        cats = {c["id"]: c["name"] for c in dset.dataset["categories"]}
        assert cats[1] == "pup", (
            f"{split}: expected pup at category_id=1, got {cats!r}"
        )
