"""Behavior contract tests for the JPEG (non-WDS) training path.

The JPEG path is much simpler than the WebDataset path:

    kwcoco bundle  ---tile.py--->  tiled kwcoco
                              ---coco_export--->  MSCOCO json + asset dir
                                            ---DEIMv2 CocoDetection--->  samples

Unlike the WebDataset adapter (see
:mod:`test_wds_stream_behavior`), there is intentionally
**no skip_empty knob, no bucket_weights, no sampler hook** on
the JPEG path:

* Stock DEIMv2 :class:`CocoDetection` is a vanilla
  :class:`torchvision.datasets.CocoDetection` subclass —
  map-style, uniform-random shuffle, every image in the MSCOCO
  file is yielded including empties.

* Training-set composition is fixed at TILE-WRITE time by
  :class:`~kwcoco_detector_kit.data.tile.TileConfig`:

  * ``keep_negative`` (default True) — emit negative tiles
  * ``min_keep_fraction`` — clip-quality threshold
  * ``min_gt_area_frac`` (multiscale) — positivity threshold

* The kit's :class:`KwcocoDetectionDataset` with
  ``balance_options=[{'attribute': 'contains_annotation'}]``
  EXISTS in :mod:`kwcoco_detector_kit.data.kwcoco_sampler`
  but is dead code — the DEIMv2 trainer
  (:func:`_build_train_yml`) only emits ``CocoDetection`` or
  ``WebDatasetCocoDetection``, never ``KwcocoDetectionDataset``.

These tests pin the contract so anyone designing a gen004-style
class-balanced JPEG run can see the surface area at a glance, and
so an accidental change (e.g. someone wires a sampler in) breaks
a test before it ships.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


# --- fixtures -----------------------------------------------------------


def _build_synth_kwcoco(bundle_dpath: Path, n_images: int = 16,
                        empty_frac: float = 0.0,
                        category_names=("widget",)):
    """A small synthetic kwcoco bundle on disk; some images may have
    no annotations (controls the empty-tile rate).
    """
    import kwcoco
    import kwimage

    asset_dpath = bundle_dpath / "assets"
    asset_dpath.mkdir(parents=True, exist_ok=True)

    dset = kwcoco.CocoDataset()
    dset.fpath = str(bundle_dpath / "synth.kwcoco.json")
    cid_by_name = {n: dset.add_category(name=n) for n in category_names}

    rng = np.random.RandomState(0)
    n_empty = int(n_images * empty_frac)
    empty_idxs = set(rng.choice(n_images, size=n_empty,
                                replace=False).tolist())

    for k in range(n_images):
        img = (rng.rand(96, 96, 3) * 255).astype(np.uint8)
        fpath = asset_dpath / f"img_{k:04d}.jpg"
        kwimage.imwrite(str(fpath), img)
        gid = dset.add_image(
            file_name=str(fpath.relative_to(bundle_dpath)),
            width=96, height=96, name=f"img_{k:04d}",
        )
        if k in empty_idxs:
            continue
        # One small annotation per image; class round-robins.
        cname = category_names[k % len(category_names)]
        dset.add_annotation(
            image_id=gid, category_id=cid_by_name[cname],
            bbox=[10.0, 10.0, 20.0, 20.0],
            area=400.0, iscrowd=0,
        )
    dset.dump()
    return dset, n_images, n_images - n_empty


def _tile_to_mscoco(src_kwcoco_fpath: Path, work_dpath: Path,
                    keep_negative: bool = True,
                    category_names=("widget",)):
    """Run tile.py then coco_export.py to produce the MSCOCO
    json+asset_dir the JPEG trainer reads."""
    from kwcoco_detector_kit.data.tile import TileConfig, run as tile_run
    from kwcoco_detector_kit.data.coco_export import export_mscoco

    # tile.run() expects a FILE path for dst (the output kwcoco
    # bundle), not a directory. Asset dir is derived from the stem.
    work_dpath.mkdir(parents=True, exist_ok=True)
    tiled_fpath = work_dpath / "tiled.kwcoco.json"
    cfg = TileConfig.cli(argv=False, data=dict(
        src=str(src_kwcoco_fpath),
        dst=str(tiled_fpath),
        mode="multiscale",
        category_names=",".join(category_names),
        # Single source scale + small tile so the fixture stays fast.
        source_scales="1.0",
        tile_size=64,
        stride_frac=1.0,
        min_gt_area_frac=0.0005,
        min_keep_fraction=0.20,
        oversize_factor=1.0,
        keep_negative=keep_negative,
        progress=False,
    ), strict=True)
    tile_run(cfg)

    if not tiled_fpath.exists():
        pytest.skip(
            f"tile.py did not produce {tiled_fpath}; output "
            "schema may have changed."
        )

    mscoco_fpath = work_dpath / "data.mscoco.json"
    export_mscoco(
        src=str(tiled_fpath),
        dst=str(mscoco_fpath),
        category_names=list(category_names),
    )
    # tile.py derives the asset dir from the stem: <stem>_assets.
    asset_dpath = work_dpath / (tiled_fpath.stem.replace(".kwcoco", "") + "_assets")
    return mscoco_fpath, asset_dpath


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


def test_tile_keep_negative_true_emits_empty_tiles(tmp_path):
    """``keep_negative=True`` (the kit default) yields tiles that
    have zero annotations after the kwcoco -> MSCOCO export.

    This is the production composition for gen001 v5 single_sealion —
    78.6% empty / 21.4% positive — which gen003 explicitly preserved
    on the WDS side via skip_empty=False.
    """
    pytest.importorskip("kwcoco")
    pytest.importorskip("kwimage")

    src_dset, _, _ = _build_synth_kwcoco(
        tmp_path / "src", n_images=12, empty_frac=0.5,
    )
    mscoco_fpath, _ = _tile_to_mscoco(
        Path(src_dset.fpath), tmp_path / "work",
        keep_negative=True,
    )

    import json
    mscoco = json.loads(Path(mscoco_fpath).read_text())
    images = mscoco["images"]
    anns = mscoco["annotations"]

    # Build "anns per image_id" hist; count empties.
    per_img = {img["id"]: 0 for img in images}
    for a in anns:
        per_img[a["image_id"]] += 1
    n_empty = sum(1 for c in per_img.values() if c == 0)
    n_positive = len(images) - n_empty

    assert len(images) > 0
    assert n_empty > 0, (
        "With keep_negative=True and 50% empty source images, the "
        "MSCOCO output must contain some empty-annotation images. "
        "If this asserts, the tile-side empty filter has changed and "
        "JPEG training composition will silently shift."
    )
    assert n_positive > 0


def test_tile_keep_negative_false_drops_empty_tiles(tmp_path):
    """``keep_negative=False`` filters background-only tiles at
    tile-write time. This is the only "drop empties" knob on the
    JPEG path — there is no skip_empty equivalent at the DataLoader
    layer.
    """
    pytest.importorskip("kwcoco")
    pytest.importorskip("kwimage")

    src_dset, _, _ = _build_synth_kwcoco(
        tmp_path / "src", n_images=12, empty_frac=0.5,
    )
    mscoco_fpath, _ = _tile_to_mscoco(
        Path(src_dset.fpath), tmp_path / "work",
        keep_negative=False,
    )

    import json
    mscoco = json.loads(Path(mscoco_fpath).read_text())
    per_img = {img["id"]: 0 for img in mscoco["images"]}
    for a in mscoco["annotations"]:
        per_img[a["image_id"]] += 1
    n_empty = sum(1 for c in per_img.values() if c == 0)

    assert len(mscoco["images"]) > 0
    assert n_empty == 0, (
        f"keep_negative=False produced {n_empty} empty-annotation "
        f"images; expected 0. The tile-time empty filter has "
        f"regressed."
    )


def test_jpeg_cocodetection_iterates_and_yields_empties(tmp_path):
    """DEIMv2's stock CocoDetection has NO skip_empty knob — every
    image in the MSCOCO json is yielded, including empties. This
    is the opposite contract from the WDS adapter (which exposes
    skip_empty as a constructor arg).

    Pinning this so anyone designing class-balancing for gen004
    can see at a glance that:
      * dropping empties on the JPEG path must happen at
        tile-write time (keep_negative=False), and
      * runtime balancing requires injecting a sampler, which
        the trainer does not currently support.
    """
    pytest.importorskip("kwcoco")
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("faster_coco_eval")

    src_dset, _, _ = _build_synth_kwcoco(
        tmp_path / "src", n_images=12, empty_frac=0.5,
    )
    mscoco_fpath, asset_dpath = _tile_to_mscoco(
        Path(src_dset.fpath), tmp_path / "work",
        keep_negative=True,
    )

    CocoDetection = _import_cocodetection()
    ds = CocoDetection(
        img_folder=str(asset_dpath),
        ann_file=str(mscoco_fpath),
        transforms=None,
        return_masks=False,
        remap_mscoco_category=False,
    )

    # Random-access path: every index resolves.
    total = len(ds)
    assert total > 0

    n_empty_yielded = 0
    n_positive_yielded = 0
    for idx in range(total):
        img, target = ds[idx]
        # Shape contract — gen004 designers will compare against
        # the WDS adapter's target shape.
        assert "boxes" in target
        assert "labels" in target
        assert "image_id" in target
        if target["labels"].numel() == 0:
            n_empty_yielded += 1
        else:
            n_positive_yielded += 1

    assert n_empty_yielded + n_positive_yielded == total
    # The MSCOCO had empties (set up with empty_frac=0.5); they
    # must reach __getitem__ unfiltered.
    assert n_empty_yielded > 0, (
        "DEIMv2 CocoDetection silently filtered empty-annotation "
        "images. If this asserts, somebody wired a skip_empty "
        "shim into the JPEG path — verify it is configurable and "
        "documented, then update this test."
    )


def test_jpeg_cocodetection_has_no_runtime_balance_knobs():
    """Pin the contract: ``CocoDetection.__init__`` exposes no
    sampler, bucket_weights, or class-balance argument.

    Any class-balancing for gen004 on the JPEG path MUST come
    from one of:

      1. Re-tiling with different ``keep_negative``/
         ``min_gt_area_frac`` to change on-disk composition.
      2. Subsampling/oversampling the MSCOCO json post-hoc.
      3. A new sampler wired into the trainer's DataLoader (not
         currently supported by ``_build_train_yml``).

    This test exists so that anyone who later adds a knob to
    CocoDetection has to remove this assertion and also update
    the documentation above. Catching the surface-area change at
    review time matters because gen002's single_sealion
    regression (WDS, journal 2026-06-01) was a silent
    composition shift; we want the JPEG path's analogous shift
    to fail loudly.
    """
    import inspect
    CocoDetection = _import_cocodetection()
    sig = inspect.signature(CocoDetection.__init__)
    params = set(sig.parameters)
    forbidden = {
        "skip_empty", "bucket_weights", "sampler",
        "balance_options", "class_weights", "weighted_sampler",
    }
    found = params & forbidden
    assert not found, (
        f"DEIMv2 CocoDetection now exposes balance-related "
        f"parameters {found}. Update test_jpeg_path_behavior.py "
        f"docs and add iteration tests that exercise them — "
        f"do NOT just delete this assertion."
    )


def test_kit_kwcoco_sampler_is_dead_code():
    """``KwcocoDetectionDataset`` exists in the kit but is NOT
    wired into the deimv2 trainer's _build_train_yml. Pin that
    fact so it's discoverable from the test suite, and so a
    future wire-up forces an explicit test update.
    """
    import inspect
    from kwcoco_detector_kit.trainers import deimv2 as trainer_mod
    src = inspect.getsource(trainer_mod)
    # The trainer must NOT reference KwcocoDetectionDataset anywhere.
    assert "KwcocoDetectionDataset" not in src, (
        "kwcoco_detector_kit.trainers.deimv2 now references "
        "KwcocoDetectionDataset. If this is intentional (e.g. a "
        "gen004 balance-aware path), update this test to assert "
        "the new wiring and add iteration coverage."
    )
    # The WDS path is emitted explicitly with "type":
    # "WebDatasetCocoDetection". The JPEG path inherits "type:
    # CocoDetection" from the upstream coco_detection.yml include
    # chain — that's why we look for img_folder/ann_file as the
    # JPEG-path marker instead of the type string.
    assert "WebDatasetCocoDetection" in src
    assert "img_folder" in src and "ann_file" in src, (
        "trainer no longer emits img_folder/ann_file for the JPEG "
        "path — audit the dataset block and update this test."
    )
