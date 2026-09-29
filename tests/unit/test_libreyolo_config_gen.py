from __future__ import annotations

import json
from pathlib import Path

import yaml


def _generate(trainer, src, workdir, *, variant="deimv2_atto", **overrides):
    kwargs = {
        "variant": variant,
        "input_hw": (320, 320) if variant == "deimv2_atto" else (640, 640),
        "train_policy": "fixed",
        "num_classes": 1,
        "batch_size": 3,
        "val_batch_size": 4,
        "num_epochs": 2,
        "lr": 2e-4,
        "backbone_lr": 2e-5,
        "use_amp": True,
        "channels": "r|g|b",
        "scale_tier": "M",
        "num_gpus": 2,
        "data_format": "kwcoco",
        "extra": {"category_names": ["widget"], "num_workers": 3},
    }
    kwargs.update(overrides)
    return trainer.generate_config(src, src, workdir, **kwargs)


def test_libreyolo_registered_without_importing_model_families():
    from kwcoco_detector_kit.trainers._registry import get_trainer, list_trainers

    assert "libreyolo" in list_trainers()
    trainer = get_trainer("libreyolo")
    assert trainer.name == "libreyolo"
    assert {"deimv2_atto", "dfine_x_seg", "rfdetr_xx_seg", "yolo9_c", "gtr_x_seg", "tinyformer_xl"} <= set(trainer.variants)


def test_libreyolo_config_uses_native_coco_bridge_without_copying_images(synthetic_kwcoco, tmp_path):
    from kwcoco_detector_kit.trainers._registry import get_trainer

    trainer = get_trainer("libreyolo")
    cfg_path = _generate(trainer, synthetic_kwcoco, tmp_path / "run")
    cfg = json.loads(cfg_path.read_text())
    policy = json.loads((tmp_path / "run" / "policy.json").read_text())

    assert cfg["trainer"] == "libreyolo"
    assert cfg["model"]["family"] == "deimv2"
    assert cfg["model"]["model_ref"] == "LibreDEIMv2atto.pt"
    assert cfg["runtime"] == {"distributed": True, "num_gpus": 2, "scale_tier": "M"}
    assert policy["data_contract"] == {
        "source_of_truth": "kwcoco",
        "training_bridge": "native_coco",
        "prepared_images_copied": False,
        "source_space_evaluation": "kdk",
    }
    assert policy["framework"]["family"] == "deimv2"

    data = yaml.safe_load(Path(cfg["data"]["yaml"]).read_text())
    train_coco = json.loads(Path(data["annotations"]["train"]).read_text())
    assert train_coco["categories"] == [{"id": 0, "name": "widget", "supercategory": "widget"}]
    assert all(Path(img["file_name"]).is_absolute() for img in train_coco["images"])
    bridge_dir = Path(cfg["data"]["yaml"]).parent
    assert not any(p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in bridge_dir.rglob("*"))


def test_libreyolo_segmentation_variant_requests_segmentation_export(synthetic_kwcoco, tmp_path, monkeypatch):
    from kwcoco_detector_kit.trainers import libreyolo as mod
    from kwcoco_detector_kit.trainers._registry import get_trainer

    calls = []

    def fake_export(src, dst, **kwargs):
        calls.append(kwargs)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_text('{"images": [], "annotations": [], "categories": []}')

    monkeypatch.setattr("kwcoco_detector_kit.data.coco_export.export_mscoco", fake_export)
    trainer = get_trainer("libreyolo")
    _generate(
        trainer,
        synthetic_kwcoco,
        tmp_path / "run",
        variant="rfdetr_xx_seg",
        input_hw=(768, 768),
        batch_size=1,
        num_gpus=1,
    )
    assert len(calls) == 2
    assert all(call["include_segmentations"] is True for call in calls)
    assert mod.VARIANTS["rfdetr_xx_seg"]["supports_masks"] is True


def test_libreyolo_train_translation_uses_internal_multi_gpu_spawn(synthetic_kwcoco, tmp_path):
    from kwcoco_detector_kit.trainers import libreyolo as mod
    from kwcoco_detector_kit.trainers._registry import get_trainer

    cfg_path = _generate(
        get_trainer("libreyolo"),
        synthetic_kwcoco,
        tmp_path / "run",
        variant="deimv2_s",
        train_policy="multiscale",
    )
    cfg = json.loads(cfg_path.read_text())
    kwargs = mod._build_train_kwargs(cfg, num_gpus=2)
    assert kwargs["device"] == [0, 1]
    assert kwargs["multi_scale"] is True
    assert kwargs["backbone_lr_mult"] == 0.1
    launcher = (cfg_path.parent / "launch_libreyolo.py").read_text()
    assert "torch.distributed" not in launcher
    assert "_launcher_main" in launcher


def test_libreyolo_rejects_non_rgb_model_input(synthetic_kwcoco, tmp_path):
    import pytest
    from kwcoco_detector_kit.trainers._registry import get_trainer

    with pytest.raises(ValueError, match="RGB tensors"):
        _generate(
            get_trainer("libreyolo"),
            synthetic_kwcoco,
            tmp_path / "run",
            channels="r|g|b|nir",
        )


def test_libreyolo_coerces_unsupported_multiscale_policy(synthetic_kwcoco, tmp_path):
    import pytest
    from kwcoco_detector_kit.trainers._registry import get_trainer

    trainer = get_trainer("libreyolo")
    with pytest.warns(UserWarning, match="coercing train_policy"):
        cfg_path = _generate(
            trainer,
            synthetic_kwcoco,
            tmp_path / "run",
            variant="yolo9_c",
            train_policy="multiscale",
        )
    cfg = json.loads(cfg_path.read_text())
    policy = json.loads((tmp_path / "run" / "policy.json").read_text())
    assert cfg["train"]["requested_policy"] == "multiscale"
    assert cfg["train"]["policy"] == "fixed"
    assert policy["train_policy_requested"] == "multiscale"
    assert policy["train_policy"] == "fixed"


def test_libreyolo_rejects_exact_range_multiscale_policy(synthetic_kwcoco, tmp_path):
    import pytest
    from kwcoco_detector_kit.trainers._registry import get_trainer

    with pytest.raises(ValueError, match="Exact KDK multiscale range"):
        _generate(
            get_trainer("libreyolo"),
            synthetic_kwcoco,
            tmp_path / "run",
            variant="deimv2_s",
            train_policy="multiscale_480_800",
        )
