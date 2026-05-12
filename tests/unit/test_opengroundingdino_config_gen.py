"""
OpenGroundingDINO trainer plugin — config-gen structural tests.

Mirrors the DEIMv2 test style: drive generate_config() without the
upstream submodule present, then inspect the on-disk artifacts
(detector_prepared/, generated_configs/, policy.json) for structural
correctness.

Acceptance: covers both registered variants × {fixed, multiscale}
policies. Does NOT depend on the OpenGroundingDINO submodule being
installed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


VARIANTS = ["opengroundingdino_swint", "opengroundingdino_swinb"]


def _get_trainer():
    from kwcoco_detector_kit.trainers._registry import get_trainer
    return get_trainer("opengroundingdino")


def _generate(trainer, tmp_path, src_kwcoco, *, variant, input_hw=(800, 800),
              num_classes=1, batch_size=4, num_epochs=2, label_list=None):
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True, exist_ok=True)
    cfg_fpath = trainer.generate_config(
        train_kwcoco_fpath=str(src_kwcoco),
        vali_kwcoco_fpath=str(src_kwcoco),
        workdir=workdir,
        variant=variant,
        input_hw=tuple(input_hw),
        num_classes=int(num_classes),
        batch_size=int(batch_size),
        val_batch_size=int(batch_size),
        num_epochs=int(num_epochs),
        lr=1e-4, backbone_lr=1e-5, use_amp=True,
        channels="r|g|b", scale_tier="L", num_gpus=1,
        data_format="kwcoco",
        extra={"category_name": "widget", "label_list": label_list or ["widget"]},
    )
    return workdir, cfg_fpath


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_opengroundingdino_is_registered():
    trainer = _get_trainer()
    assert trainer.name == "opengroundingdino"
    assert set(trainer.variants) == set(VARIANTS)


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_supports_dynamic_input(variant):
    trainer = _get_trainer()
    assert trainer.supports_dynamic_input(variant) is True


def test_unknown_variant_raises():
    trainer = _get_trainer()
    with pytest.raises(KeyError):
        trainer.supports_dynamic_input("not_a_variant")


def test_does_not_support_onnx_or_webdataset():
    trainer = _get_trainer()
    assert trainer.supports_onnx_export is False
    assert trainer.supports_webdataset_input() is False


# ---------------------------------------------------------------------------
# generate_config writes the expected artifacts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_generate_config_writes_cfg_and_policy_and_datasets(
    variant, synthetic_kwcoco, tmp_path,
):
    trainer = _get_trainer()
    workdir, cfg = _generate(trainer, tmp_path, synthetic_kwcoco, variant=variant)
    assert Path(cfg).exists()
    assert (workdir / "policy.json").exists()
    assert (workdir / "detector_prepared" / "datasets.json").exists()
    assert (workdir / "detector_prepared" / "train.mscoco.json").exists()
    assert (workdir / "detector_prepared" / "vali.mscoco.json").exists()
    assert (workdir / "detector_prepared" / "label_map.json").exists()


@pytest.mark.parametrize("variant", VARIANTS)
def test_datasets_json_structure(variant, synthetic_kwcoco, tmp_path):
    """datasets.json has train (ODVG) + val (COCO) sections per the v9 contract."""
    trainer = _get_trainer()
    workdir, _ = _generate(trainer, tmp_path, synthetic_kwcoco, variant=variant)
    payload = json.loads((workdir / "detector_prepared" / "datasets.json").read_text())
    assert "train" in payload and isinstance(payload["train"], list) and len(payload["train"]) >= 1
    assert "val" in payload and isinstance(payload["val"], list) and len(payload["val"]) >= 1
    train_entry = payload["train"][0]
    assert train_entry["dataset_mode"] == "odvg"
    val_entry = payload["val"][0]
    assert val_entry["dataset_mode"] == "coco"


@pytest.mark.parametrize("variant", VARIANTS)
def test_policy_json_records_input_hw_and_label_list(
    variant, synthetic_kwcoco, tmp_path,
):
    trainer = _get_trainer()
    workdir, _ = _generate(
        trainer, tmp_path, synthetic_kwcoco, variant=variant,
        input_hw=(640, 640), label_list=["alpha", "beta"],
    )
    policy = json.loads((workdir / "policy.json").read_text())
    assert policy["variant"] == variant
    assert policy["export_input_h"] == 640
    assert policy["export_input_w"] == 640
    assert policy["candidate_kind"] == "real"
    assert policy["label_list"] == ["alpha", "beta"]


def test_label_map_reflects_train_categories(synthetic_kwcoco, tmp_path):
    """label_map.json maps {id: name} extracted from the train MSCOCO."""
    trainer = _get_trainer()
    workdir, _ = _generate(trainer, tmp_path, synthetic_kwcoco,
                           variant="opengroundingdino_swint")
    label_map = json.loads(
        (workdir / "detector_prepared" / "label_map.json").read_text()
    )
    assert "widget" in label_map.values()


def test_stub_config_has_override_keys_when_repo_missing(
    synthetic_kwcoco, tmp_path, monkeypatch,
):
    """Without $KCD_OPENGROUNDINGDINO_REPO_DPATH, generate_config writes
    a stub Python config containing the kit's override keys."""
    monkeypatch.delenv("KCD_OPENGROUNDINGDINO_REPO_DPATH", raising=False)
    trainer = _get_trainer()
    workdir, cfg_fpath = _generate(
        trainer, tmp_path, synthetic_kwcoco,
        variant="opengroundingdino_swint",
        batch_size=8, num_epochs=10, label_list=["foo"],
    )
    text = Path(cfg_fpath).read_text()
    assert "label_list" in text
    assert "batch_size = 8" in text
    assert "epochs = 10" in text
    assert "use_coco_eval = False" in text


def test_launch_failure_writes_train_log_tail(tmp_path, monkeypatch):
    trainer = _get_trainer()
    repo = tmp_path / "repo"
    repo.mkdir()
    train_sh = repo / "train_dist.sh"
    train_sh.write_text(
        "#!/usr/bin/env bash\n"
        "echo first line\n"
        "echo upstream failure detail\n"
        "exit 7\n"
    )
    train_sh.chmod(0o755)
    monkeypatch.setenv("KCD_OPENGROUNDINGDINO_REPO_DPATH", str(repo))

    workdir = tmp_path / "work"
    cfg_dpath = workdir / "generated_configs"
    cfg_dpath.mkdir(parents=True)
    prep_dpath = workdir / "detector_prepared"
    prep_dpath.mkdir()
    cfg_fpath = cfg_dpath / "ogdino_cfg.py"
    cfg_fpath.write_text("# fake config\n")
    (prep_dpath / "datasets.json").write_text("{}\n")

    with pytest.raises(RuntimeError) as exc:
        trainer.launch(cfg_fpath, num_gpus=1)

    assert (workdir / "train.log").exists()
    assert (workdir / "train_command.json").exists()
    assert "upstream failure detail" in (workdir / "train.log").read_text()
    assert "exit 7" in str(exc.value)
    assert "upstream failure detail" in str(exc.value)


# ---------------------------------------------------------------------------
# memory_tier_default_batch
# ---------------------------------------------------------------------------


def test_memory_tier_default_batch_shrinks_at_larger_input():
    trainer = _get_trainer()
    b_small = trainer.memory_tier_default_batch(
        "opengroundingdino_swint", (640, 640), 48.0,
    )
    b_large = trainer.memory_tier_default_batch(
        "opengroundingdino_swint", (1024, 1024), 48.0,
    )
    assert b_small >= b_large


def test_memory_tier_default_batch_grows_with_vram():
    trainer = _get_trainer()
    b48 = trainer.memory_tier_default_batch("opengroundingdino_swint", (800, 800), 48.0)
    b96 = trainer.memory_tier_default_batch("opengroundingdino_swint", (800, 800), 96.0)
    assert b96 >= b48
