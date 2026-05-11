"""Tests for trainers.mock_tiny + the CPU smoke train -> export -> predict path."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_mock_tiny_is_registered():
    from kwcoco_detector_kit.trainers._registry import get_trainer
    t = get_trainer("mock_tiny")
    assert t.name == "mock_tiny"
    assert "mock_tiny" in t.variants


def test_mock_tiny_generate_config_writes_yaml(synthetic_kwcoco, tmp_workdir):
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
        batch_size=2, val_batch_size=2,
        num_epochs=1,
        lr=1e-2, backbone_lr=1e-2,
        use_amp=False,
        channels="r|g|b", scale_tier="S", num_gpus=1,
        data_format="kwcoco",
        extra={"category_name": "widget"},
    )
    assert Path(cfg).exists()


@pytest.mark.requires_torch
def test_mock_tiny_train_export_predict_cycle(synthetic_kwcoco, tmp_workdir):
    """End-to-end on CPU: train -> export -> build predictor -> predict_image."""
    import numpy as np

    from kwcoco_detector_kit.trainers._registry import get_trainer
    from kwcoco_detector_kit.export.onnx import export_onnx

    trainer = get_trainer("mock_tiny")
    cfg = trainer.generate_config(
        train_kwcoco_fpath=str(synthetic_kwcoco),
        vali_kwcoco_fpath=str(synthetic_kwcoco),
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

    # find_checkpoint
    ckpt = trainer.find_checkpoint(tmp_workdir)
    assert ckpt.exists()

    # export ONNX
    onnx_fpath = export_onnx(trainer=trainer, workdir=tmp_workdir, input_hw=(64, 64))
    assert onnx_fpath.exists()
    assert onnx_fpath.with_suffix(".modelspec.json").exists()

    # predictor
    predictor = trainer.build_predictor(tmp_workdir, device="cpu")
    dummy = (np.ones((64, 64, 3)) * 128).astype(np.uint8)
    detections = predictor.predict_image(dummy, (64, 64))
    assert isinstance(detections, list)
