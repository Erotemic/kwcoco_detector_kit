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
        extra={"category_names": ["widget"], "score_thresh": 0.01},
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
        category_names=["widget"],
        dataset_slug="synthetic",
        experiment_slug="unit",
        train_kwcoco=str(synthetic_kwcoco),
        vali_kwcoco=str(synthetic_kwcoco),
        test_kwcoco=str(synthetic_kwcoco),
        username="alice",
        hostname="node0",
    )

    with open_package(package_zip) as (root, manifest):
        assert manifest["schema"] == "kwcoco_detector_kit.package.v2"
        assert manifest["trainer"] == "mock_tiny"
        assert manifest["provenance"]["username"] == "alice"
        assert manifest["provenance"]["hostname"] == "node0"
        assert manifest["artifacts"]["checkpoint"] == "weights/best_stg2.pth"
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


def test_predict_config_windowed_key_value_boolean():
    """The kwconf boolean is one tri-state option, not a --no-* twin."""
    from kwcoco_detector_kit.predict import PredictConfig

    common = [
        "--model=dummy.zip",
        "--src=src.kwcoco.zip",
        "--dst=pred.kwcoco.zip",
    ]

    config = PredictConfig.cli(argv=common + ["--windowed=false"], strict=True)
    assert config.windowed is False

    config = PredictConfig.cli(argv=common + ["--windowed=true"], strict=True)
    assert config.windowed is True

    config = PredictConfig.cli(argv=common, strict=True)
    assert config.windowed is None

    config = PredictConfig.cli(argv=common + ["--pipeline=false"], strict=True)
    assert config.pipeline is False

    config = PredictConfig.cli(argv=common + ["--pipeline=true"], strict=True)
    assert config.pipeline is True

    config = PredictConfig.cli(argv=common + ["--prediction-scale=0.4"], strict=True)
    assert config.prediction_scale == 0.4

    config = PredictConfig.cli(argv=common + ["--resume=false"], strict=True)
    assert config.resume is False

    config = PredictConfig.cli(
        argv=common + ["--checkpoint-every=17", "--checkpoint-seconds=12.5"],
        strict=True,
    )
    assert config.checkpoint_every == 17
    assert config.checkpoint_seconds == 12.5

@pytest.mark.requires_torch
def test_predict_resume_after_prepare_failure(synthetic_kwcoco, tmp_workdir, tmp_path, monkeypatch):
    """A mid-run source-preparation failure must leave resumable committed work."""
    import json

    import kwcoco

    from kwcoco_detector_kit.export.package import build_model_package
    from kwcoco_detector_kit.predict import (
        _clone_for_predictions,
        _prediction_resume_journal_dir,
        _prediction_resume_paths,
        predict_kwcoco,
    )
    from kwcoco_detector_kit.predictors.source_window import SourceWindowReader
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
        extra={"category_names": ["widget"], "score_thresh": 0.01},
    )
    trainer.launch(cfg, num_gpus=1)

    package_zip = tmp_path / "resume_mock_tiny_package.zip"
    build_model_package(
        workdir=tmp_workdir,
        out=package_zip,
        trainer="mock_tiny",
        variant="mock_tiny",
        category_names=["widget"],
        dataset_slug="synthetic",
        experiment_slug="resume-unit",
        train_kwcoco=str(synthetic_kwcoco),
        vali_kwcoco=str(synthetic_kwcoco),
        test_kwcoco=str(synthetic_kwcoco),
        username="alice",
        hostname="node0",
    )

    true = kwcoco.CocoDataset.coerce(str(synthetic_kwcoco))
    source_gids = list(map(int, true.images()))
    assert len(source_gids) >= 3, "resume regression needs at least three source images"

    dst = tmp_path / "resume.pred.kwcoco.zip"
    partial_fpath, state_fpath = _prediction_resume_paths(dst)

    original_prepare = SourceWindowReader.prepare
    prepare_calls = {"n": 0}

    def fail_after_two(self, *args, **kwargs):
        prepare_calls["n"] += 1
        if prepare_calls["n"] == 3:
            raise RuntimeError("intentional resume regression failure")
        return original_prepare(self, *args, **kwargs)

    monkeypatch.setattr(SourceWindowReader, "prepare", fail_after_two)
    with pytest.raises(RuntimeError, match="failed to prepare gid"):
        predict_kwcoco(
            package=package_zip,
            src=synthetic_kwcoco,
            dst=dst,
            device="cpu",
            score_thresh=0.05,
            pipeline=False,
            resume=True,
            checkpoint_every=1,
            checkpoint_seconds=0,
        )

    journal_dpath = _prediction_resume_journal_dir(dst)
    assert not partial_fpath.exists(), "v2 checkpoints must not rewrite a whole KWCoco"
    assert state_fpath.exists()
    assert journal_dpath.exists()
    state = json.loads(state_fpath.read_text())
    assert state["schema"] == "kwcoco_detector_kit.predict_resume.v2"
    assert state["shards"]
    completed_before_resume = list(map(int, state["completed_gids"]))
    assert completed_before_resume
    assert len(completed_before_resume) < len(source_gids)

    shard_records = []
    for shard_name in state["shards"]:
        payload = json.loads((journal_dpath / shard_name).read_text())
        shard_records.extend(payload["records"])
    assert {int(r["gid"]) for r in shard_records} == set(completed_before_resume)

    # Exercise compatibility with the already-deployed v1 checkpoint format:
    # synthesize its one immutable partial KWCoco from the v2 shard, then make
    # the state claim v1. The next run must migrate it once and never rewrite
    # that KWCoco again.
    from kwcoco_detector_kit.data.postprocess import add_prediction_annotations

    legacy = _clone_for_predictions(true, partial_fpath)
    for record in shard_records:
        add_prediction_annotations(
            legacy, int(record["gid"]), record["anns"], "mock_tiny:torch"
        )
    legacy.dump()
    legacy_state = {
        "schema": "kwcoco_detector_kit.predict_resume.v1",
        "identity": state["identity"],
        "identity_sha256": state["identity_sha256"],
        "completed_gids": completed_before_resume,
        "completed_images": len(completed_before_resume),
        "total_images": len(source_gids),
        "partial_prediction": str(partial_fpath),
        "reason": "test-legacy-migration",
    }
    state_fpath.write_text(json.dumps(legacy_state, indent=2) + "\n")
    import shutil
    shutil.rmtree(journal_dpath)

    # Restore normal preparation and rerun the identical command. The resume
    # identity must match, the v1 base must migrate to v2 shards, committed
    # gids must be skipped, and success must clean up all checkpoint artifacts.
    monkeypatch.setattr(SourceWindowReader, "prepare", original_prepare)
    predict_kwcoco(
        package=package_zip,
        src=synthetic_kwcoco,
        dst=dst,
        device="cpu",
        score_thresh=0.05,
        pipeline=False,
        resume=True,
        checkpoint_every=1,
        checkpoint_seconds=0,
    )

    final = kwcoco.CocoDataset.coerce(str(dst))
    assert final.n_images == true.n_images
    assert final.n_annots > 0
    assert not partial_fpath.exists()
    assert not state_fpath.exists()
    assert not journal_dpath.exists()


@pytest.mark.requires_torch
def test_predict_resume_rejects_changed_identity(synthetic_kwcoco, tmp_workdir, tmp_path, monkeypatch):
    """A partial run must never be resumed with different inference settings."""
    from kwcoco_detector_kit.export.package import build_model_package
    from kwcoco_detector_kit.predict import predict_kwcoco
    from kwcoco_detector_kit.predictors.source_window import SourceWindowReader
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
        extra={"category_names": ["widget"], "score_thresh": 0.01},
    )
    trainer.launch(cfg, num_gpus=1)
    package_zip = tmp_path / "identity_mock_tiny_package.zip"
    build_model_package(
        workdir=tmp_workdir,
        out=package_zip,
        trainer="mock_tiny",
        variant="mock_tiny",
        category_names=["widget"],
        dataset_slug="synthetic",
        experiment_slug="resume-unit",
        train_kwcoco=str(synthetic_kwcoco),
        vali_kwcoco=str(synthetic_kwcoco),
        test_kwcoco=str(synthetic_kwcoco),
        username="alice",
        hostname="node0",
    )

    dst = tmp_path / "identity.pred.kwcoco.zip"
    original_prepare = SourceWindowReader.prepare
    calls = {"n": 0}

    def fail_after_one(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("intentional resume identity failure")
        return original_prepare(self, *args, **kwargs)

    monkeypatch.setattr(SourceWindowReader, "prepare", fail_after_one)
    with pytest.raises(RuntimeError, match="failed to prepare gid"):
        predict_kwcoco(
            package=package_zip,
            src=synthetic_kwcoco,
            dst=dst,
            device="cpu",
            score_thresh=0.05,
            pipeline=False,
            resume=True,
            checkpoint_every=1,
            checkpoint_seconds=0,
        )

    monkeypatch.setattr(SourceWindowReader, "prepare", original_prepare)
    with pytest.raises(RuntimeError, match="resume checkpoint does not match this run"):
        predict_kwcoco(
            package=package_zip,
            src=synthetic_kwcoco,
            dst=dst,
            device="cpu",
            score_thresh=0.10,
            pipeline=False,
            resume=True,
            checkpoint_every=1,
            checkpoint_seconds=0,
        )
