"""DEIMv2 trainer: kwcoco -> MSCOCO conversion happens automatically inside generate_config."""
from __future__ import annotations

import json
from pathlib import Path

import yaml


def test_kwcoco_input_is_converted_to_mscoco(synthetic_kwcoco, tmp_path):
    """Given a kwcoco path as train_kwcoco_fpath, the generated YAML's
    ann_file points at a .mscoco.json that exists on disk."""
    from kwcoco_detector_kit.trainers._registry import get_trainer

    trainer = get_trainer("deimv2")
    workdir = tmp_path / "wd"
    cfg_fpath = trainer.generate_config(
        train_kwcoco_fpath=str(synthetic_kwcoco),
        vali_kwcoco_fpath=str(synthetic_kwcoco),
        workdir=workdir,
        variant="deimv2_hgnetv2_atto",
        input_hw=(256, 256),
        train_policy="fixed",
        num_classes=1, batch_size=4, val_batch_size=4, num_epochs=1,
        lr=5e-4, backbone_lr=2.5e-5, use_amp=False,
        channels="r|g|b", scale_tier="S", num_gpus=1,
        data_format="kwcoco",
        extra={"category_names": ["widget"]},
    )
    cfg = yaml.safe_load(Path(cfg_fpath).read_text())
    train_ann = cfg["train_dataloader"]["dataset"]["ann_file"]
    val_ann = cfg["val_dataloader"]["dataset"]["ann_file"]
    assert train_ann.endswith(".mscoco.json"), (
        f"DEIMv2 train ann_file should be MSCOCO; got {train_ann}"
    )
    assert val_ann.endswith(".mscoco.json"), (
        f"DEIMv2 val ann_file should be MSCOCO; got {val_ann}"
    )
    assert Path(train_ann).exists() and Path(val_ann).exists()
    # Quick MSCOCO schema sanity
    payload = json.loads(Path(train_ann).read_text())
    assert "images" in payload and "annotations" in payload and "categories" in payload
    assert payload["categories"][0]["name"] == "widget"


def test_mscoco_input_is_passed_through(synthetic_kwcoco, tmp_path):
    """If the caller hands DEIMv2 an already-MSCOCO file, it's used verbatim."""
    from kwcoco_detector_kit.trainers._registry import get_trainer
    from kwcoco_detector_kit.data.coco_export import export_mscoco

    # Pre-build an MSCOCO file.
    pre_mscoco = tmp_path / "pre.mscoco.json"
    export_mscoco(synthetic_kwcoco, pre_mscoco, category_names=["widget"],
                  include_segmentations=False, category_id_start=0)

    trainer = get_trainer("deimv2")
    workdir = tmp_path / "wd"
    cfg_fpath = trainer.generate_config(
        train_kwcoco_fpath=str(pre_mscoco),
        vali_kwcoco_fpath=str(pre_mscoco),
        workdir=workdir,
        variant="deimv2_hgnetv2_atto",
        input_hw=(256, 256),
        train_policy="fixed",
        num_classes=1, batch_size=4, val_batch_size=4, num_epochs=1,
        lr=5e-4, backbone_lr=2.5e-5, use_amp=False,
        channels="r|g|b", scale_tier="S", num_gpus=1,
        data_format="kwcoco",
        extra={"category_names": ["widget"]},
    )
    cfg = yaml.safe_load(Path(cfg_fpath).read_text())
    train_ann = cfg["train_dataloader"]["dataset"]["ann_file"]
    # Same path as we passed in.
    assert train_ann == str(pre_mscoco)
