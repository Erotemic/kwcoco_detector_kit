"""
End-to-end smoke test for the WebDataset training data path.

Crosses every layer between a kwcoco source dataset and a
`(PIL.Image, target_dict)` batch yielded by DEIMv2's
`WebDatasetCocoDetection` adapter:

  synthetic kwcoco
    → kwcoco_dataloader.cli.build_detection_webdataset → tar shards
    → kwcoco_dataloader.readers.detection (Sample stream)
    → tpl/DEIMv2/engine/data/dataset/wds_coco_dataset.WebDatasetCocoDetection
    → relabel_detection_sample (source_category → target class index)
    → __iter__ yields (PIL.Image, target_dict)

This single test would have caught (and going forward will catch)
the following gen002 bugs documented in
``dev/benchmark-candidates/webdataset-integration-questions.md``:

  Q1 — img_folder/ann_file YAML-merger leak into __init__ kwargs.
       The test instantiates with both kwargs present and asserts
       the constructor accepts them.
  Q3 — SchemeMapping dataclass kwarg-name mismatch. The test
       reaches __iter__ which constructs SchemeMapping; a kwarg
       mismatch raises TypeError before the first sample yields.
  Q4 — python -m kwcoco_dataloader vs python -m
       kwcoco_dataloader.cli.<entry>. The test invokes the CLI via
       its Python `main()` so a no-__main__.py packaging mistake
       wouldn't be caught here, but a renamed/moved cli module
       would.
  Q5 — module-level import of optional line_profiler dep in
       kwcoco_dataloader.readers.detection. If the dep regresses
       to being import-time-required and unavailable, the
       `from ... import WebDatasetStream` at the top of the
       adapter module crashes.
  Q7 — stale tests pinned to a removed upstream API. The test
       uses `BuildDetectionWebdatasetCLI` directly, so an
       upstream rename surfaces immediately.

The test skips gracefully when its optional runtime deps
(torch, webdataset, wids, the kwcoco_dataloader package, or the
DEIMv2 submodule) aren't installed. In CI / docker the image
installs all of them; the test should always run there.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


def _build_tiny_synth_kwcoco(bundle_dpath: Path):
    """4 images, 4 annotations all labelled raw-source 'W'. The
    detection-writer's bucket attribute will see one bucket."""
    import kwcoco
    import kwimage

    asset_dpath = bundle_dpath / "assets"
    asset_dpath.mkdir(parents=True, exist_ok=True)

    dset = kwcoco.CocoDataset()
    dset.fpath = str(bundle_dpath / "tiny.kwcoco.json")
    cid = dset.add_category(name="widget")

    rng = np.random.RandomState(0)
    for k in range(4):
        img = (rng.rand(64, 64, 3) * 255).astype(np.uint8)
        fpath = asset_dpath / f"img_{k:02d}.jpg"
        kwimage.imwrite(str(fpath), img)
        gid = dset.add_image(
            file_name=str(fpath.relative_to(bundle_dpath)),
            width=64, height=64, name=f"img_{k:02d}",
        )
        dset.add_annotation(
            image_id=gid, category_id=cid,
            bbox=[10.0, 10.0, 20.0, 20.0],
            area=400.0, iscrowd=0,
            # Raw source category — preserved by the writer for
            # downstream relabel. Same shape the kit's tile.py
            # produces via _passthrough_fields stamping.
            source_category="W",
        )
    dset.dump()
    return dset


def _make_shards(in_fpath: Path, out_dpath: Path):
    """Invoke the writer CLI directly (not via subprocess) so an
    import-time regression in the package surfaces in the test
    rather than as a mysterious subprocess exit code."""
    from kwcoco_dataloader.cli.build_detection_webdataset import (
        BuildDetectionWebdatasetCLI,
    )
    BuildDetectionWebdatasetCLI.main(
        argv=False,
        in_fpath=str(in_fpath),
        out_dpath=str(out_dpath),
        bucket_attr="source_category",
        maxcount=10,
        maxsize_mb=1024,
        jpeg_quality=90,
        drop_provenance=False,
        progress=False,
    )


def _import_adapter():
    """Add tpl/DEIMv2 to sys.path and import the adapter. The kit's
    docker image installs DEIMv2 at /opt/.../tpl/DEIMv2 and adds it
    to PYTHONPATH at train time; in local pytest we do the same
    insertion ourselves."""
    kit_dpath = Path(__file__).resolve().parents[2]
    deimv2_dpath = kit_dpath / "tpl" / "DEIMv2"
    if not (deimv2_dpath / "engine" / "data" / "dataset" / "wds_coco_dataset.py").exists():
        pytest.skip("DEIMv2 submodule not initialised under tpl/DEIMv2/")
    sys.path.insert(0, str(deimv2_dpath))
    try:
        from engine.data.dataset.wds_coco_dataset import WebDatasetCocoDetection
        return WebDatasetCocoDetection
    finally:
        sys.path.pop(0)


def test_webdataset_dataset_end_to_end(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("webdataset")
    pytest.importorskip("wids")
    pytest.importorskip("kwcoco_dataloader")
    pytest.importorskip("kwcoco")
    pytest.importorskip("kwimage")
    pytest.importorskip("PIL")

    WebDatasetCocoDetection = _import_adapter()

    # 1. synthetic kwcoco -> shards
    bundle_dpath = tmp_path / "src"
    bundle_dpath.mkdir()
    src_dset = _build_tiny_synth_kwcoco(bundle_dpath)
    shards_dpath = tmp_path / "shards"
    _make_shards(Path(src_dset.fpath), shards_dpath)
    bucket_dirs = sorted(d for d in shards_dpath.iterdir() if d.is_dir())
    assert bucket_dirs, f"no bucket subdirs under {shards_dpath}"
    # The writer mangles the bucket-attr value into a filename-safe
    # token (e.g. 'source_category_EQ_W/'). One bucket for our
    # uniform-class corpus.
    assert any(b.name.endswith("_EQ_W") for b in bucket_dirs), (
        f"expected a 'source_category_EQ_W' bucket; got {[b.name for b in bucket_dirs]}"
    )
    assert any((b / "__header__.json").exists() for b in bucket_dirs)

    # 2. Adapter instantiation. We deliberately pass the kwargs that
    # DEIMv2's YAMLConfig merger leaks through from the parent
    # CocoDetection config (img_folder, ann_file). The adapter must
    # accept-and-ignore both. Q1 from the benchmark-candidates doc.
    ds = WebDatasetCocoDetection(
        shards_dpath=str(shards_dpath),
        category_names=["widget"],
        source_to_target={"W": "widget"},  # raw 'W' -> target 'widget'
        # YAML-merger leaks; must not raise:
        img_folder="/should/not/exist",
        ann_file="/also/should/not/exist",
        return_masks=False,
    )

    # 3. Iterate the first sample. This drives the relabel + target
    # construction path that exercises Q3 (SchemeMapping kwarg names
    # — if those drift, this raises before yielding).
    it = iter(ds)
    img, target = next(it)

    # PIL image, RGB, the synth dimensions.
    from PIL import Image
    assert isinstance(img, Image.Image), f"got {type(img)}"
    assert img.size == (64, 64), f"unexpected size {img.size}"

    # Target shape matches what DEIMv2's BatchImageCollateFunction
    # expects (same shape as torchvision CocoDetection).
    import torch
    for k in ("boxes", "labels", "image_id", "area", "iscrowd",
              "orig_size", "idx"):
        assert k in target, f"missing target['{k}']; have {sorted(target)}"
    assert target["labels"].dtype == torch.int64
    assert target["boxes"].shape[-1] == 4
    assert target["labels"].shape[0] == target["boxes"].shape[0]
    # Single annotation per image, label index 0 in target_order.
    assert target["labels"].shape[0] == 1
    assert int(target["labels"][0]) == 0, (
        f"raw 'W' should map to target_order[0]='widget' (class id 0); "
        f"got {int(target['labels'][0])}"
    )

    # 4. Wrap in a torch DataLoader. PyTorch rejects shuffle=True on
    # IterableDataset (ValueError at DataLoader.__init__). This catches
    # the regression where the kit's generated YAML inherits
    # `train_dataloader.shuffle: True` from the upstream
    # coco_detection.yml without overriding it for the WDS path.
    from torch.utils.data import DataLoader
    # The kit's _build_train_yml emits shuffle=False for the WDS
    # dataset block. Exercise the same invariant here:
    loader = DataLoader(
        ds,
        batch_size=2,
        num_workers=0,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )
    one_batch = next(iter(loader))
    assert isinstance(one_batch, list) and len(one_batch) == 2


def test_webdataset_dataset_drops_unmapped_source_classes(tmp_path):
    """A scheme that doesn't list a source category in `mapping`
    should drop those annotations silently (default unmapped_policy
    in kwcoco_dataloader.readers.detection.SchemeMapping). This is
    how the kit handles the scheme YAML's `drop:` list — distractors
    like NFS are implicit-dropped at relabel time."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("webdataset")
    pytest.importorskip("wids")
    pytest.importorskip("kwcoco_dataloader")
    WebDatasetCocoDetection = _import_adapter()

    import kwcoco
    import kwimage

    # Build a kwcoco with two raw classes: 'W' (kept) and 'X' (dropped).
    bundle_dpath = tmp_path / "src"
    bundle_dpath.mkdir()
    asset_dpath = bundle_dpath / "assets"
    asset_dpath.mkdir()
    dset = kwcoco.CocoDataset()
    dset.fpath = str(bundle_dpath / "tiny.kwcoco.json")
    cid = dset.add_category(name="widget")
    rng = np.random.RandomState(0)
    for k, source in enumerate(["W", "X", "W", "X"]):
        img = (rng.rand(64, 64, 3) * 255).astype(np.uint8)
        fpath = asset_dpath / f"img_{k:02d}.jpg"
        kwimage.imwrite(str(fpath), img)
        gid = dset.add_image(
            file_name=str(fpath.relative_to(bundle_dpath)),
            width=64, height=64, name=f"img_{k:02d}",
        )
        dset.add_annotation(
            image_id=gid, category_id=cid,
            bbox=[10.0, 10.0, 20.0, 20.0],
            area=400.0, iscrowd=0,
            source_category=source,
        )
    dset.dump()

    shards_dpath = tmp_path / "shards"
    _make_shards(Path(dset.fpath), shards_dpath)

    ds = WebDatasetCocoDetection(
        shards_dpath=str(shards_dpath),
        category_names=["widget"],
        # Only 'W' mapped; 'X' falls through to unmapped_policy='drop'.
        source_to_target={"W": "widget"},
    )

    # Drain a small number of samples and check that 'X'-bucket
    # samples either don't appear or appear with their annotations
    # dropped (the adapter skips background-only samples).
    samples = []
    for i, (img, target) in enumerate(ds):
        samples.append(target)
        if i >= 10:
            break
    # We should see ONLY the 'W' samples (with 1 annotation each).
    for tgt in samples:
        assert int(tgt["labels"][0]) == 0
        assert tgt["labels"].shape[0] >= 1
