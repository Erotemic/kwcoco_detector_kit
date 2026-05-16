"""Tests for the detection-mode path of BuildWebdatasetCLI / LocalWebdatasetBuckets.

All tests skip when kwcoco_dataloader, wids, or webdataset are not installed.
"""
from __future__ import annotations

import pytest
import numpy as np


@pytest.fixture
def demo_vidcoco(tmp_path):
    """
    Tiny synthetic video-style kwcoco dataset with real JPEG images,
    channel metadata, and bounding-box annotations.
    """
    kwcoco = pytest.importorskip("kwcoco")
    import kwimage

    asset_dpath = tmp_path / "assets"
    asset_dpath.mkdir()

    dset = kwcoco.CocoDataset()
    dset.fpath = str(tmp_path / "demo.kwcoco.json")
    vid_id = dset.add_video(name="demo_video", width=96, height=96)
    cid = dset.add_category("widget")

    rng = np.random.RandomState(42)
    for k in range(6):
        W, H = 96, 96
        img_arr = (rng.rand(H, W, 3) * 255).astype(np.uint8)
        bx, by = int(rng.randint(10, 40)), int(rng.randint(10, 40))
        bw, bh = int(rng.randint(20, 40)), int(rng.randint(20, 40))
        img_arr[by:by + bh, bx:bx + bw] = (200, 80, 80)

        fpath = asset_dpath / f"img_{k:02d}.jpg"
        kwimage.imwrite(str(fpath), img_arr)

        gid = dset.add_image(
            file_name=str(fpath),
            width=W, height=H,
            name=f"img_{k:02d}",
            video_id=vid_id,
            frame_index=k,
            channels="r|g|b",
        )
        dset.add_annotation(
            image_id=gid, category_id=cid,
            bbox=[float(bx), float(by), float(bw), float(bh)],
        )
    dset.dump()
    return dset


def test_custom_subset_detection(demo_vidcoco):
    """custom_subset_detection serializes correct fields from a T=1 frame item."""
    pytest.importorskip("kwcoco_dataloader")
    pytest.importorskip("ndsampler")
    from kwcoco_dataloader.cli.build_webdataset import BuildWebdatasetCLI
    from kwcoco_dataloader.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDataset

    torch_dataset = KWCocoVideoDataset(
        demo_vidcoco, mode='test',
        time_steps=1, output_type='rgb',
        window_dims=(64, 64),
        requested_tasks={'boxes': True, 'class': False, 'saliency': False, 'change': False},
        verbose=0,
    )
    target = torch_dataset.sample_grid['targets'][0]
    batch_item = torch_dataset[target]
    item = batch_item._frame_collated()

    sample = BuildWebdatasetCLI.custom_subset_detection(item)
    assert set(sample.keys()) == {'__key__', 'meta.pyd', 'det.npz', 'non_collatable.pyd'}

    det = sample['det.npz']
    assert 'imdata_chw' in det
    assert 'box_ltrb' in det
    assert 'box_cidxs' in det

    assert det['imdata_chw'].ndim == 3           # (C, H, W)
    assert det['box_ltrb'].ndim == 2
    assert det['box_ltrb'].shape[1] == 4
    assert det['box_cidxs'].ndim == 1
    assert len(det['box_ltrb']) == len(det['box_cidxs'])
    assert det['imdata_chw'].dtype == np.float32
    assert det['box_ltrb'].dtype == np.float32
    assert det['box_cidxs'].dtype == np.int64


def test_build_detection_webdataset_writes_shards(demo_vidcoco, tmp_path):
    """BuildWebdatasetCLI.main with task_type='detection' writes readable shards."""
    pytest.importorskip("kwcoco_dataloader")
    pytest.importorskip("webdataset")
    pytest.importorskip("wids")
    from kwcoco_dataloader.cli.build_webdataset import BuildWebdatasetCLI

    out_dpath = tmp_path / "det_wds"
    out_dpath.mkdir()

    data_config = {
        'window_dims': (64, 64),
        'channels': 'r|g|b',
    }
    BuildWebdatasetCLI.main(
        argv=0,
        in_fpath=demo_vidcoco.fpath,
        out_dpath=out_dpath,
        data_config=data_config,
        maxcount=20,
        num_workers=0,
        task_type='detection',
    )

    # Shards must have been written
    tars = list(out_dpath.rglob("*.tar"))
    assert len(tars) > 0, "No shard files written"


def test_readback_detection_webdataset(demo_vidcoco, tmp_path):
    """Detection shards decode to {imdata_chw, box_ltrb, box_cidxs} without error."""
    pytest.importorskip("kwcoco_dataloader")
    pytest.importorskip("webdataset")
    pytest.importorskip("wids")
    from kwcoco_dataloader.cli.build_webdataset import BuildWebdatasetCLI, LocalWebdatasetBuckets

    out_dpath = tmp_path / "det_wds"
    out_dpath.mkdir()

    BuildWebdatasetCLI.main(
        argv=0,
        in_fpath=demo_vidcoco.fpath,
        out_dpath=out_dpath,
        data_config={'window_dims': (64, 64), 'channels': 'r|g|b'},
        maxcount=20,
        num_workers=0,
        task_type='detection',
    )

    ds = LocalWebdatasetBuckets(out_dpath)
    assert len(ds) > 0

    sample = ds[0]
    det = sample.get('.det.npz')
    assert det is not None, f"No .det.npz key in sample: {list(sample.keys())}"
    assert det['imdata_chw'].ndim == 3
    assert det['box_ltrb'].ndim == 2 and det['box_ltrb'].shape[1] == 4
    assert det['box_cidxs'].ndim == 1


def test_detection_augmenter_crops_image_and_boxes(demo_vidcoco, tmp_path):
    """TimeSpaceAugmenter with space_dims crops imdata_chw and clips box_ltrb."""
    pytest.importorskip("kwcoco_dataloader")
    pytest.importorskip("webdataset")
    pytest.importorskip("wids")
    from kwcoco_dataloader.cli.build_webdataset import BuildWebdatasetCLI, LocalWebdatasetBuckets

    out_dpath = tmp_path / "det_wds_aug"
    out_dpath.mkdir()

    BuildWebdatasetCLI.main(
        argv=0,
        in_fpath=demo_vidcoco.fpath,
        out_dpath=out_dpath,
        data_config={'window_dims': (64, 64), 'channels': 'r|g|b'},
        maxcount=20,
        num_workers=0,
        task_type='detection',
    )

    crop_size = 32
    ds = LocalWebdatasetBuckets(out_dpath, augment={'space_dims': crop_size})
    assert len(ds) > 0

    for i in range(len(ds)):
        sample = ds[i]
        det = sample['.det.npz']
        assert det['imdata_chw'].shape == (3, crop_size, crop_size), (
            f"Expected (3, {crop_size}, {crop_size}) got {det['imdata_chw'].shape}"
        )
        boxes = det['box_ltrb']
        if len(boxes) > 0:
            # All clipped coords must be within [0, crop_size]
            assert (boxes >= 0).all()
            assert (boxes[:, [0, 2]] <= crop_size).all()
            assert (boxes[:, [1, 3]] <= crop_size).all()
            # All kept boxes must have positive area
            assert ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])).all()
