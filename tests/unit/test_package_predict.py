"""Tests for package-build and package-aware predict."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.requires_torch
def test_mock_tiny_package_zip_predict_roundtrip(synthetic_kwcoco, tmp_workdir, tmp_path):
    import kwcoco
    import yaml

    from kwcoco_detector_kit.export.package import (
        build_model_package,
        open_package,
        suggest_package_out,
    )
    from kwcoco_detector_kit.predict import predict_kwcoco
    from kwcoco_detector_kit.trainers._registry import get_trainer

    trainer = get_trainer("mock_tiny")
    cfg = trainer.generate_config(
        train_kwcoco_fpath=str(synthetic_kwcoco),
        vali_kwcoco_fpath=str(synthetic_kwcoco),
        workdir=tmp_workdir,
        variant="mock_tiny",
        input_hw=(64, 64),
        train_policy="fixed",
        num_classes=1,
        batch_size=2,
        val_batch_size=2,
        num_epochs=1,
        lr=1e-2,
        backbone_lr=1e-2,
        use_amp=False,
        channels="r|g|b",
        scale_tier="S",
        num_gpus=1,
        data_format="kwcoco",
        extra={"category_name": "widget", "score_thresh": 0.01},
    )
    trainer.launch(cfg, num_gpus=1)

    suggested = suggest_package_out(
        out_root=tmp_path / "packages",
        dataset_slug="synthetic",
        experiment_slug="unit",
        variant="mock_tiny",
        run_id="run0",
        username="alice",
        hostname="node0",
    )
    assert suggested.parts[-8:-1] == (
        "synthetic", "unit", "users", "alice", "hosts", "node0", "run0",
    )
    assert suggested.name == "mock_tiny.zip"

    package_zip = tmp_path / "mock_tiny_package.zip"
    build_model_package(
        workdir=tmp_workdir,
        out=package_zip,
        trainer="mock_tiny",
        variant="mock_tiny",
        category_name="widget",
        dataset_slug="synthetic",
        experiment_slug="unit",
        train_kwcoco=str(synthetic_kwcoco),
        vali_kwcoco=str(synthetic_kwcoco),
        test_kwcoco=str(synthetic_kwcoco),
        username="alice",
        hostname="node0",
    )

    with open_package(package_zip) as (root, manifest):
        assert manifest["schema"] == "kwcoco_detector_kit.package.v1"
        assert manifest["trainer"] == "mock_tiny"
        assert manifest["provenance"]["username"] == "alice"
        assert manifest["provenance"]["hostname"] == "node0"
        assert manifest["artifacts"]["checkpoint"] == "weights/checkpoint.pth"
        assert (root / manifest["artifacts"]["checkpoint"]).exists()
        assert not Path(manifest["artifacts"]["checkpoint"]).is_absolute()
        yaml.safe_dump(manifest)

    pred_fpath = tmp_path / "pred.kwcoco.zip"
    predict_kwcoco(
        package=package_zip,
        src=synthetic_kwcoco,
        dst=pred_fpath,
        device="cpu",
        score_thresh=0.05,
    )

    pred = kwcoco.CocoDataset.coerce(str(pred_fpath))
    assert pred.n_images == kwcoco.CocoDataset.coerce(str(synthetic_kwcoco)).n_images
    assert pred.n_cats == 1
    assert pred.n_annots > 0
