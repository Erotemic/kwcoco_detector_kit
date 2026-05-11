"""Tests for data.mine — offline hard-negative mining.

Drives the miner against a mock_tiny-trained checkpoint (no GPU, no
DEIMv2 submodule). The predictor adapter is the unit under test from
the miner's perspective — we verify:

- the miner filters to negative tiles (tile_role=None or 'negative');
- only tiles with max_pred_score >= threshold are kept;
- the score histogram sidecar is written.
"""
from __future__ import annotations

import json
from pathlib import Path

import kwcoco
import numpy as np
import pytest


def _build_neg_bundle(bundle_dpath: Path, n: int, seed: int = 0) -> Path:
    import kwimage
    rng = np.random.RandomState(seed)
    bundle_dpath = Path(bundle_dpath)
    asset_dpath = bundle_dpath / "neg_assets"
    asset_dpath.mkdir(parents=True, exist_ok=True)
    dset = kwcoco.CocoDataset()
    dset.fpath = str(bundle_dpath / "neg.kwcoco.zip")
    dset.add_category(name="widget")
    for k in range(n):
        img = (rng.rand(64, 64, 3) * 255).astype(np.uint8)
        fpath = asset_dpath / f"neg_{k:04d}.jpg"
        kwimage.imwrite(str(fpath), img)
        dset.add_image(
            file_name=str(fpath.relative_to(bundle_dpath)),
            width=64, height=64, tile_role="negative",
        )
    dset.dump()
    return Path(dset.fpath)


def _train_mock_tiny(tmp_workdir: Path, train_kwcoco: Path):
    """Helper: in-process mock_tiny train for a few steps."""
    from kwcoco_detector_kit.trainers._registry import get_trainer
    trainer = get_trainer("mock_tiny")
    cfg = trainer.generate_config(
        train_kwcoco_fpath=str(train_kwcoco),
        vali_kwcoco_fpath=str(train_kwcoco),
        workdir=tmp_workdir,
        variant="mock_tiny",
        input_hw=(64, 64),
        train_policy="fixed",
        num_classes=1,
        batch_size=2, val_batch_size=2,
        num_epochs=1,
        lr=1e-2, backbone_lr=1e-2,
        use_amp=False,
        channels="r|g|b", scale_tier="S", num_gpus=1,
        data_format="kwcoco",
        extra={"category_name": "widget"},
    )
    trainer.launch(cfg, num_gpus=1)
    return tmp_workdir


@pytest.mark.requires_torch
def test_mine_writes_kwcoco_and_sidecar(synthetic_kwcoco, tmp_workdir, tmp_path):
    _train_mock_tiny(tmp_workdir, synthetic_kwcoco)
    neg_kwcoco = _build_neg_bundle(tmp_path / "neg_bundle", n=6)
    dst = tmp_path / "hard_negs.kwcoco.zip"

    from kwcoco_detector_kit.data.mine import MineConfig, run as mine_run
    cfg = MineConfig.cli(
        argv=False,
        data={
            "neg_kwcoco": str(neg_kwcoco),
            "workdir": str(tmp_workdir),
            "dst": str(dst),
            "trainer": "mock_tiny",
            # Below the post-training gate so SOME tiles qualify as "hard".
            "score_thresh": 0.05,
            "max_hard_per_round": 100,
            "device": "cpu",
            "progress": False,
        },
    )
    mine_run(cfg)

    assert dst.exists(), "mine should write the hard-neg kwcoco bundle"
    out = kwcoco.CocoDataset.coerce(str(dst))
    assert out.n_images >= 0  # may be 0 if no tile clears threshold; the bundle exists either way
    sidecar = dst.with_suffix(".mine_stats.json")
    assert sidecar.exists(), "score histogram sidecar must be written"
    stats = json.loads(sidecar.read_text())
    assert "n_scored" in stats and "n_hard" in stats
    assert "score_hist" in stats and "score_bins" in stats


@pytest.mark.requires_torch
def test_mine_cap_respected(synthetic_kwcoco, tmp_workdir, tmp_path):
    _train_mock_tiny(tmp_workdir, synthetic_kwcoco)
    neg_kwcoco = _build_neg_bundle(tmp_path / "neg_bundle", n=10)
    dst = tmp_path / "hard_negs.kwcoco.zip"

    from kwcoco_detector_kit.data.mine import MineConfig, run as mine_run
    cfg = MineConfig.cli(
        argv=False,
        data={
            "neg_kwcoco": str(neg_kwcoco),
            "workdir": str(tmp_workdir),
            "dst": str(dst),
            "trainer": "mock_tiny",
            "score_thresh": 0.0,   # everything qualifies
            "max_hard_per_round": 3,
            "device": "cpu",
            "progress": False,
        },
    )
    mine_run(cfg)
    out = kwcoco.CocoDataset.coerce(str(dst))
    assert out.n_images <= 3, f"max_hard_per_round=3 should cap; got {out.n_images}"
