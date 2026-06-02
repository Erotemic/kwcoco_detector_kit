"""End-to-end tests for the JPEG-path class-balance CLI.

balance_mscoco is the JPEG-path equivalent of the WDS path's
bucket_weights — it resamples an MSCOCO json by duplicating image
entries to hit a target class distribution. The assets on disk are
unchanged; only the json changes. After this, DEIMv2's stock
CocoDetection trains on the balanced composition with no DEIMv2
code changes.

Tests:

  - histogram_matches_target: resampled MSCOCO's per-image bucket
    distribution matches the target within rounding.
  - target_size_overrides_default: explicit target_size changes
    output length; default == input length.
  - new_ids_are_unique_after_oversampling: ids don't collide when
    the same image is duplicated.
  - asset_paths_are_unchanged: file_name still points to the
    original on-disk asset, regardless of duplication.
  - composes_with_cocodetection_path: end-to-end — feed the
    balanced json to DEIMv2's CocoDetection and confirm iteration
    works and yields the balanced count.
  - rejects_unknown_bucket: target_distribution referencing a
    bucket that isn't in the source must fail loudly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


# --- fixtures -----------------------------------------------------------


def _build_mscoco_with_buckets(
    bundle_dpath: Path,
    n_positive_per_class: int = 5,
    n_empty: int = 20,
    category_names=("pup", "nonpup"),
):
    """Synthesize a tiny MSCOCO bundle with predictable bucket sizes.

    Each class gets n_positive_per_class single-class images;
    n_empty images have no annotations.
    """
    import kwimage

    asset_dpath = bundle_dpath / "assets"
    asset_dpath.mkdir(parents=True, exist_ok=True)

    cats = [
        {"id": i, "name": name} for i, name in enumerate(category_names)
    ]
    images = []
    annotations = []
    rng = np.random.RandomState(0)
    next_img_id = 1
    next_ann_id = 1

    # Positive images per class.
    for cid, name in enumerate(category_names):
        for k in range(n_positive_per_class):
            img_pix = (rng.rand(64, 64, 3) * 255).astype(np.uint8)
            fname = f"{name}_{k:03d}.jpg"
            kwimage.imwrite(str(asset_dpath / fname), img_pix)
            images.append({
                "id": next_img_id, "file_name": f"assets/{fname}",
                "width": 64, "height": 64,
            })
            annotations.append({
                "id": next_ann_id, "image_id": next_img_id,
                "category_id": cid, "bbox": [10, 10, 20, 20],
                "area": 400.0, "iscrowd": 0,
            })
            next_img_id += 1
            next_ann_id += 1

    # Empty images.
    for k in range(n_empty):
        img_pix = (rng.rand(64, 64, 3) * 255).astype(np.uint8)
        fname = f"empty_{k:03d}.jpg"
        kwimage.imwrite(str(asset_dpath / fname), img_pix)
        images.append({
            "id": next_img_id, "file_name": f"assets/{fname}",
            "width": 64, "height": 64,
        })
        next_img_id += 1

    mscoco = {"categories": cats, "images": images, "annotations": annotations}
    fpath = bundle_dpath / "src.mscoco.json"
    fpath.write_text(json.dumps(mscoco))
    return fpath, asset_dpath


def _classify_balanced(mscoco: dict):
    """Reproduce balance_mscoco's bucket assignment to count buckets."""
    from kwcoco_detector_kit.data.balance_mscoco import (
        EMPTY_BUCKET, _bucket_of_image,
    )
    cat_name_by_id = {c["id"]: c["name"] for c in mscoco["categories"]}
    category_order = [c["name"] for c in mscoco["categories"]]
    anns_by_image = {}
    for a in mscoco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)
    hist = {}
    for img in mscoco["images"]:
        b = _bucket_of_image(
            img["id"], anns_by_image, cat_name_by_id, category_order,
        )
        hist[b] = hist.get(b, 0) + 1
    return hist


def _import_cocodetection():
    """DEIMv2 stock CocoDetection lives under tpl/DEIMv2/."""
    kit_dpath = Path(__file__).resolve().parents[2]
    deimv2_dpath = kit_dpath / "tpl" / "DEIMv2"
    if not (deimv2_dpath / "engine" / "data" / "dataset"
            / "coco_dataset.py").exists():
        pytest.skip("DEIMv2 submodule not initialised under tpl/DEIMv2/")
    sys.path.insert(0, str(deimv2_dpath))
    try:
        from engine.data.dataset.coco_dataset import (  # noqa: E402
            CocoDetection,
        )
        return CocoDetection
    finally:
        sys.path.pop(0)


# --- tests --------------------------------------------------------------


def test_histogram_matches_target(tmp_path):
    """50/25/25 target produces a 50/25/25 histogram in the output
    (within ±1 due to largest-remainder rounding)."""
    pytest.importorskip("kwimage")
    from kwcoco_detector_kit.data.balance_mscoco import (
        BalanceMSCOCOConfig, run,
    )

    src_fpath, _ = _build_mscoco_with_buckets(
        tmp_path / "src",
        n_positive_per_class=5, n_empty=20,
    )
    dst_fpath = tmp_path / "balanced.mscoco.json"

    cfg = BalanceMSCOCOConfig.cli(argv=False, data=dict(
        src=str(src_fpath),
        dst=str(dst_fpath),
        target_distribution='{"<empty>": 0.5, "pup": 0.25, "nonpup": 0.25}',
        # Source has 5+5+20=30 images; ask for the same size.
        target_size=30,
        seed=0,
    ), strict=True)
    run(cfg)

    out = json.loads(dst_fpath.read_text())
    hist = _classify_balanced(out)

    assert len(out["images"]) == 30
    assert hist.get("<empty>", 0) == 15, (
        f"<empty> bucket = {hist.get('<empty>', 0)}; expected 15 "
        f"(50% of 30). Largest-remainder rounding should give exact "
        f"counts when fractions divide evenly."
    )
    # Per-class buckets: 25% of 30 = 7.5 → 7 or 8; both sum to 15.
    pup = hist.get("pup", 0)
    nonpup = hist.get("nonpup", 0)
    assert pup + nonpup == 15
    assert abs(pup - nonpup) <= 1


def test_target_size_overrides_default(tmp_path):
    """Default target_size = len(src images); explicit overrides it."""
    pytest.importorskip("kwimage")
    from kwcoco_detector_kit.data.balance_mscoco import (
        BalanceMSCOCOConfig, run,
    )

    src_fpath, _ = _build_mscoco_with_buckets(
        tmp_path / "src", n_positive_per_class=4, n_empty=10,
    )
    src = json.loads(src_fpath.read_text())
    src_size = len(src["images"])  # 4+4+10 = 18

    # Default (no target_size): output length == input length.
    dst1 = tmp_path / "default.json"
    cfg1 = BalanceMSCOCOConfig.cli(argv=False, data=dict(
        src=str(src_fpath), dst=str(dst1),
        target_distribution='{"<empty>": 0.5, "pup": 0.5}', seed=0,
    ), strict=True)
    run(cfg1)
    assert len(json.loads(dst1.read_text())["images"]) == src_size

    # Explicit target_size=50: 50 images out.
    dst2 = tmp_path / "explicit.json"
    cfg2 = BalanceMSCOCOConfig.cli(argv=False, data=dict(
        src=str(src_fpath), dst=str(dst2),
        target_distribution='{"<empty>": 0.5, "pup": 0.5}',
        target_size=50, seed=0,
    ), strict=True)
    run(cfg2)
    assert len(json.loads(dst2.read_text())["images"]) == 50


def test_new_ids_are_unique_after_oversampling(tmp_path):
    """When the same source image is duplicated to hit the target,
    each duplicate gets a NEW image_id + ann_id so downstream tools
    don't see id collisions."""
    pytest.importorskip("kwimage")
    from kwcoco_detector_kit.data.balance_mscoco import (
        BalanceMSCOCOConfig, run,
    )

    # 2 pups + 0 nonpups + 100 empties. Asking for 50/50 with
    # target_size=100 forces each pup to be duplicated ~25× — and
    # the test catches any id reuse.
    src_fpath, _ = _build_mscoco_with_buckets(
        tmp_path / "src",
        n_positive_per_class=2, n_empty=100,
        category_names=("pup",),
    )
    dst_fpath = tmp_path / "balanced.json"
    cfg = BalanceMSCOCOConfig.cli(argv=False, data=dict(
        src=str(src_fpath), dst=str(dst_fpath),
        target_distribution='{"<empty>": 0.5, "pup": 0.5}',
        target_size=100, seed=0,
    ), strict=True)
    run(cfg)

    out = json.loads(dst_fpath.read_text())
    img_ids = [img["id"] for img in out["images"]]
    ann_ids = [a["id"] for a in out["annotations"]]
    assert len(img_ids) == len(set(img_ids)), (
        f"duplicate image_ids: {len(img_ids)} entries, "
        f"{len(set(img_ids))} unique"
    )
    assert len(ann_ids) == len(set(ann_ids)), (
        f"duplicate ann_ids: {len(ann_ids)} entries, "
        f"{len(set(ann_ids))} unique"
    )
    # Every annotation must reference an existing new image_id.
    img_id_set = set(img_ids)
    for a in out["annotations"]:
        assert a["image_id"] in img_id_set


def test_asset_paths_are_unchanged(tmp_path):
    """Duplication must not touch on-disk assets — file_name
    references point at the same files."""
    pytest.importorskip("kwimage")
    from kwcoco_detector_kit.data.balance_mscoco import (
        BalanceMSCOCOConfig, run,
    )

    src_fpath, _ = _build_mscoco_with_buckets(
        tmp_path / "src", n_positive_per_class=3, n_empty=5,
    )
    src = json.loads(src_fpath.read_text())
    src_filenames = {img["file_name"] for img in src["images"]}

    dst = tmp_path / "balanced.json"
    cfg = BalanceMSCOCOConfig.cli(argv=False, data=dict(
        src=str(src_fpath), dst=str(dst),
        target_distribution='{"<empty>": 0.5, "pup": 0.25, "nonpup": 0.25}',
        target_size=40, seed=0,
    ), strict=True)
    run(cfg)

    out = json.loads(dst.read_text())
    out_filenames = {img["file_name"] for img in out["images"]}
    # Every output filename is a subset of input (no new ones).
    assert out_filenames <= src_filenames, (
        f"balance_mscoco invented file_name(s) not in source: "
        f"{out_filenames - src_filenames}"
    )
    # The provenance field records the original image id so we can
    # trace duplicates back.
    for img in out["images"]:
        assert "balance_src_image_id" in img
        assert "balance_bucket" in img


def test_composes_with_cocodetection_path(tmp_path):
    """End-to-end: the balanced MSCOCO can be opened by DEIMv2's
    stock CocoDetection — no DEIM code changes needed."""
    pytest.importorskip("kwimage")
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("faster_coco_eval")
    from kwcoco_detector_kit.data.balance_mscoco import (
        BalanceMSCOCOConfig, run,
    )

    src_fpath, _ = _build_mscoco_with_buckets(
        tmp_path / "src", n_positive_per_class=3, n_empty=10,
    )
    dst = tmp_path / "balanced.json"
    cfg = BalanceMSCOCOConfig.cli(argv=False, data=dict(
        src=str(src_fpath), dst=str(dst),
        target_distribution='{"<empty>": 0.5, "pup": 0.25, "nonpup": 0.25}',
        target_size=20, seed=0,
    ), strict=True)
    run(cfg)

    CocoDetection = _import_cocodetection()
    # CocoDetection takes (img_folder, ann_file, transforms,...).
    # File names in the balanced json are relative to the source
    # bundle root.
    ds = CocoDetection(
        img_folder=str(tmp_path / "src"),
        ann_file=str(dst),
        transforms=None,
        return_masks=False,
        remap_mscoco_category=False,
    )
    assert len(ds) == 20

    n_empty = n_positive = 0
    for idx in range(len(ds)):
        _img, target = ds[idx]
        if target["labels"].numel() == 0:
            n_empty += 1
        else:
            n_positive += 1
    assert n_empty + n_positive == 20
    # Target was 50/50 → 10 empty / 10 positive (5 pup + 5 nonpup,
    # ±1 from rounding).
    assert abs(n_empty - 10) <= 1
    assert abs(n_positive - 10) <= 1


def test_rejects_unknown_bucket(tmp_path):
    """target_distribution with a bucket name that doesn't exist in
    the source must raise — silent drops are a reproducibility
    hazard (cf. WDS bucket_weights does_not_read_environ test)."""
    pytest.importorskip("kwimage")
    from kwcoco_detector_kit.data.balance_mscoco import (
        BalanceMSCOCOConfig, run,
    )

    src_fpath, _ = _build_mscoco_with_buckets(
        tmp_path / "src", n_positive_per_class=3, n_empty=5,
    )
    dst = tmp_path / "balanced.json"
    cfg = BalanceMSCOCOConfig.cli(argv=False, data=dict(
        src=str(src_fpath), dst=str(dst),
        target_distribution='{"<empty>": 0.5, "nonexistent_class": 0.5}',
        seed=0,
    ), strict=True)
    with pytest.raises(ValueError, match="nonexistent_class"):
        run(cfg)
