from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_rfdetr_registered_without_importing_upstream():
    from kwcoco_detector_kit.trainers._registry import get_trainer, list_trainers
    assert "rfdetr" in list_trainers()
    assert get_trainer("rfdetr").name == "rfdetr"


def test_rfdetr_config_and_segmentation_dataset_layout(synthetic_kwcoco, tmp_path):
    from kwcoco_detector_kit.trainers._registry import get_trainer

    trainer = get_trainer("rfdetr")
    cfg_path = trainer.generate_config(
        synthetic_kwcoco, synthetic_kwcoco, tmp_path / "run",
        variant="seg_2xlarge", input_hw=(768, 768), train_policy="fixed",
        num_classes=1, batch_size=2, val_batch_size=4, num_epochs=3,
        lr=1e-4, backbone_lr=1.5e-4, use_amp=True, channels="r|g|b",
        scale_tier="XL", num_gpus=4, data_format="kwcoco",
        extra={"category_names": ["widget"], "grad_accum_steps": 2},
    )
    cfg = json.loads(cfg_path.read_text())
    assert cfg["runtime"] == {"distributed": True, "num_gpus": 4}
    assert cfg["train"]["grad_accum_steps"] == 2
    assert cfg["train"]["multi_scale"] is False
    root = Path(cfg["train"]["dataset_dir"])
    for split in ["train", "valid"]:
        coco = json.loads((root / split / "_annotations.coco.json").read_text())
        assert coco["categories"][0]["name"] == "widget"
        assert all(Path(img["file_name"]).is_absolute() for img in coco["images"])
    launcher = (cfg_path.parent / "launch_rfdetr.py").read_text()
    assert 'train_kwargs["devices"]' in launcher


def test_rfdetr_rejects_invalid_segmentation_resolution(synthetic_kwcoco, tmp_path):
    from kwcoco_detector_kit.trainers._registry import get_trainer
    with pytest.raises(ValueError, match="divisible by 24"):
        get_trainer("rfdetr").generate_config(
            synthetic_kwcoco, synthetic_kwcoco, tmp_path / "run",
            variant="seg_2xlarge", input_hw=(770, 770), num_classes=1,
            batch_size=1, val_batch_size=1, num_epochs=1, lr=1e-4,
            backbone_lr=1e-5, use_amp=True, channels="r|g|b",
            scale_tier="XL", num_gpus=1, data_format="kwcoco",
            extra={"category_names": ["widget"]},
        )
