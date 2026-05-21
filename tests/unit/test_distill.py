"""Unit tests for kwcoco_detector_kit.data.distill."""
from __future__ import annotations

import json

import pytest


@pytest.mark.requires_torch
def test_pseudo_label_kwcoco_roundtrip(synthetic_kwcoco, tmp_workdir, tmp_path):
    """End-to-end pseudo-labelling with mock_tiny as teacher."""
    import kwcoco
    from kwcoco_detector_kit.export.package import build_model_package
    from kwcoco_detector_kit.trainers._registry import get_trainer
    from kwcoco_detector_kit.data.distill import pseudo_label_kwcoco

    # Train a mock_tiny teacher
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
        extra={"category_names": ["widget"], "score_thresh": 0.01},
    )
    trainer.launch(cfg, num_gpus=1)

    teacher_pkg = tmp_path / "teacher.zip"
    build_model_package(
        workdir=tmp_workdir,
        out=teacher_pkg,
        trainer="mock_tiny",
        variant="mock_tiny",
        category_names=["widget"],
        dataset_slug="test",
        experiment_slug="distill",
        username="test",
        hostname="testhost",
    )

    # Generate pseudo-labels from the teacher over the same synthetic data
    dst = tmp_path / "pseudo.kwcoco.zip"
    out = pseudo_label_kwcoco(
        teacher_package=teacher_pkg,
        src=synthetic_kwcoco,
        dst=dst,
        device="cpu",
        score_thresh=0.01,
        min_annotations=1,
    )

    assert out.exists()
    pl = kwcoco.CocoDataset.coerce(str(out))
    # Every annotation should be tagged as pseudo_label
    for ann in pl.anns.values():
        assert ann.get("role") == "pseudo_label"
        assert "pseudo_label_score" in ann


@pytest.mark.requires_torch
def test_pseudo_label_min_annotations_filter(synthetic_kwcoco, tmp_workdir, tmp_path):
    """min_annotations=999 should drop all images (no image has 999 predictions)."""
    import kwcoco
    from kwcoco_detector_kit.export.package import build_model_package
    from kwcoco_detector_kit.trainers._registry import get_trainer
    from kwcoco_detector_kit.data.distill import pseudo_label_kwcoco

    trainer = get_trainer("mock_tiny")
    cfg = trainer.generate_config(
        train_kwcoco_fpath=str(synthetic_kwcoco),
        vali_kwcoco_fpath=str(synthetic_kwcoco),
        workdir=tmp_workdir,
        variant="mock_tiny",
        input_hw=(64, 64),
        train_policy="fixed",
        num_classes=1, batch_size=2, val_batch_size=2, num_epochs=1,
        lr=1e-2, backbone_lr=1e-2, use_amp=False,
        channels="r|g|b", scale_tier="S", num_gpus=1, data_format="kwcoco",
        extra={"category_names": ["widget"], "score_thresh": 0.01},
    )
    trainer.launch(cfg, num_gpus=1)
    pkg = tmp_path / "pkg.zip"
    build_model_package(workdir=tmp_workdir, out=pkg, trainer="mock_tiny",
                        variant="mock_tiny", category_names=["widget"],
                        dataset_slug="t", experiment_slug="t",
                        username="u", hostname="h")

    dst = tmp_path / "pl.kwcoco.zip"
    out = pseudo_label_kwcoco(
        teacher_package=pkg, src=synthetic_kwcoco, dst=dst,
        device="cpu", score_thresh=0.01, min_annotations=999,
    )
    pl = kwcoco.CocoDataset.coerce(str(out))
    assert pl.n_images == 0


@pytest.mark.requires_torch
def test_generate_distill_policy(tmp_path):
    from kwcoco_detector_kit.data.distill import generate_distill_policy

    out = generate_distill_policy(
        teacher_package=tmp_path / "fake_teacher.zip",
        student_variant="deimv2_hgnetv2_n",
        workdir=tmp_path / "student_workdir",
        distill_alpha=0.7,
        temperature=6.0,
    )
    assert out.exists()
    policy = json.loads(out.read_text())
    assert policy["distill_mode"] == "soft_label"
    assert policy["student_variant"] == "deimv2_hgnetv2_n"
    assert policy["distill_alpha"] == 0.7
    assert policy["temperature"] == 6.0
