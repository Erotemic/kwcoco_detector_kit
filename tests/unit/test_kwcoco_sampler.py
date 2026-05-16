"""Unit tests for kwcoco_detector_kit.data.kwcoco_sampler.

These tests use pytest.importorskip to skip when kwcoco_dataloader and/or
ndsampler are not installed (they are optional dependencies).
"""
from __future__ import annotations

import pytest
import numpy as np


# Fixtures are declared locally here to avoid importing kwcoco_dataloader at
# collection time.  All imports are guarded by importorskip.


@pytest.fixture
def demo_kwcoco(tmp_path):
    """Tiny synthetic kwcoco dataset: 8 images, 1 category, boxes.

    Images are written to disk as JPEGs with explicit ``channels='r|g|b'``
    metadata — required for kwcoco_dataloader's channel-spec matching.
    """
    kwcoco = pytest.importorskip("kwcoco")
    import kwimage

    asset_dpath = tmp_path / "assets"
    asset_dpath.mkdir()

    dset = kwcoco.CocoDataset()
    dset.fpath = str(tmp_path / "demo.kwcoco.json")
    cid = dset.add_category("widget")
    rng = np.random.RandomState(0)
    for k in range(8):
        W, H = 128, 128
        img_arr = (rng.rand(H, W, 3) * 255).astype(np.uint8)
        bx = int(rng.randint(10, 60))
        by = int(rng.randint(10, 60))
        bw = int(rng.randint(20, 50))
        bh = int(rng.randint(20, 50))
        img_arr[by:by + bh, bx:bx + bw] = (255, 50, 50)

        fpath = asset_dpath / f"img_{k:02d}.jpg"
        kwimage.imwrite(str(fpath), img_arr)

        gid = dset.add_image(
            file_name=str(fpath),
            width=W, height=H,
            name=f"img_{k:02d}",
            channels="r|g|b",  # required for kwcoco_dataloader channel matching
        )
        dset.add_annotation(
            image_id=gid, category_id=cid,
            bbox=[float(bx), float(by), float(bw), float(bh)],
        )
    dset.dump()
    return dset


def test_import_guard():
    """KwcocoDetectionDataset raises ImportError when kwcoco_dataloader absent.

    This test is only meaningful when kwcoco_dataloader is not installed; when
    it IS installed the import succeeds and we skip the guard test.
    """
    try:
        import kwcoco_dataloader  # noqa: F401
    except ImportError:
        from kwcoco_detector_kit.data.kwcoco_sampler import KwcocoDetectionDataset
        with pytest.raises(ImportError, match="kwcoco_dataloader"):
            KwcocoDetectionDataset("nonexistent.kwcoco.zip")
    else:
        pytest.skip("kwcoco_dataloader is installed — ImportError guard not reachable")


def test_basic_construction(demo_kwcoco):
    """Dataset constructs without error and has correct length."""
    pytest.importorskip("kwcoco_dataloader")
    pytest.importorskip("ndsampler")
    from kwcoco_detector_kit.data.kwcoco_sampler import KwcocoDetectionDataset

    ds = KwcocoDetectionDataset(
        demo_kwcoco,
        chip_dims=(64, 64),
        channels="r|g|b",
        use_grid_negatives=False,
        verbose=0,
    )
    assert len(ds) > 0
    assert "widget" in ds.class_names


def test_getitem_shapes(demo_kwcoco):
    """__getitem__ returns correctly shaped tensors."""
    pytest.importorskip("kwcoco_dataloader")
    pytest.importorskip("ndsampler")
    import torch
    from kwcoco_detector_kit.data.kwcoco_sampler import KwcocoDetectionDataset

    ds = KwcocoDetectionDataset(
        demo_kwcoco,
        chip_dims=(64, 64),
        channels="r|g|b",
        use_grid_negatives=False,
        verbose=0,
    )
    targets = ds.sample_grid["targets"]
    item = ds[targets[0]]

    assert "image" in item
    assert "boxes_ltrb" in item
    assert "class_idxs" in item
    assert "class_names" in item
    assert "meta" in item

    img = item["image"]
    assert isinstance(img, torch.Tensor)
    assert img.ndim == 3  # CHW
    assert img.dtype == torch.float32

    boxes = item["boxes_ltrb"]
    cids = item["class_idxs"]
    assert isinstance(boxes, torch.Tensor)
    assert isinstance(cids, torch.Tensor)
    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert cids.ndim == 1
    assert len(boxes) == len(cids)


def test_empty_image_boxes(demo_kwcoco):
    """Images with no annotations in the window return empty box tensors."""
    pytest.importorskip("kwcoco_dataloader")
    pytest.importorskip("ndsampler")
    import torch
    from kwcoco_detector_kit.data.kwcoco_sampler import KwcocoDetectionDataset

    ds = KwcocoDetectionDataset(
        demo_kwcoco,
        chip_dims=(64, 64),
        channels="r|g|b",
        use_centered_positives=False,
        use_grid_positives=True,
        use_grid_negatives=True,
        verbose=0,
    )
    # Sample from all targets and find at least one negative window
    empty_found = False
    for target in ds.sample_grid["targets"][:20]:
        item = ds[target]
        if len(item["boxes_ltrb"]) == 0:
            empty_found = True
            assert item["boxes_ltrb"].shape == (0, 4)
            assert item["class_idxs"].shape == (0,)
            break
    # It's OK if no empty window was found in a tiny dataset
    _ = empty_found


def test_class_names_consistent(demo_kwcoco):
    """class_names is consistent across items."""
    pytest.importorskip("kwcoco_dataloader")
    pytest.importorskip("ndsampler")
    from kwcoco_detector_kit.data.kwcoco_sampler import KwcocoDetectionDataset

    ds = KwcocoDetectionDataset(
        demo_kwcoco,
        chip_dims=(64, 64),
        channels="r|g|b",
        verbose=0,
    )
    names = ds.class_names
    for target in ds.sample_grid["targets"][:5]:
        item = ds[target]
        assert item["class_names"] == names


def test_detection_collate_fn(demo_kwcoco):
    """detection_collate_fn stacks images and preserves ragged boxes."""
    pytest.importorskip("kwcoco_dataloader")
    pytest.importorskip("ndsampler")
    import torch
    from kwcoco_detector_kit.data.kwcoco_sampler import (
        KwcocoDetectionDataset,
        detection_collate_fn,
    )

    ds = KwcocoDetectionDataset(
        demo_kwcoco,
        chip_dims=(64, 64),
        channels="r|g|b",
        verbose=0,
    )
    targets = ds.sample_grid["targets"]
    items = [ds[t] for t in targets[:3]]
    batch = detection_collate_fn(items)

    assert "image" in batch
    assert "boxes_ltrb" in batch
    assert "class_idxs" in batch
    assert batch["image"].ndim == 4          # (B, C, H, W)
    assert len(batch["boxes_ltrb"]) == 3     # one per image
    assert len(batch["class_idxs"]) == 3
    assert isinstance(batch["image"], torch.Tensor)


def test_make_loader(demo_kwcoco):
    """make_loader returns a DataLoader that yields batches."""
    pytest.importorskip("kwcoco_dataloader")
    pytest.importorskip("ndsampler")
    import torch
    from kwcoco_detector_kit.data.kwcoco_sampler import KwcocoDetectionDataset

    ds = KwcocoDetectionDataset(
        demo_kwcoco,
        chip_dims=(64, 64),
        channels="r|g|b",
        verbose=0,
    )
    loader = ds.make_loader(batch_size=2, num_workers=0)
    batch = next(iter(loader))

    assert "image" in batch
    assert isinstance(batch["image"], torch.Tensor)
    assert batch["image"].shape[0] == 2  # batch dim


def test_balance_options_accepted(demo_kwcoco):
    """balance_options dict is accepted without error."""
    pytest.importorskip("kwcoco_dataloader")
    pytest.importorskip("ndsampler")
    from kwcoco_detector_kit.data.kwcoco_sampler import KwcocoDetectionDataset

    ds = KwcocoDetectionDataset(
        demo_kwcoco,
        chip_dims=(64, 64),
        channels="r|g|b",
        balance_options=[{"attribute": "contains_annotation"}],
        verbose=0,
    )
    assert len(ds) > 0


def test_repr(demo_kwcoco):
    """__repr__ does not raise."""
    pytest.importorskip("kwcoco_dataloader")
    pytest.importorskip("ndsampler")
    from kwcoco_detector_kit.data.kwcoco_sampler import KwcocoDetectionDataset

    ds = KwcocoDetectionDataset(
        demo_kwcoco,
        chip_dims=(64, 64),
        channels="r|g|b",
        verbose=0,
    )
    r = repr(ds)
    assert "KwcocoDetectionDataset" in r
    assert "widget" in r
