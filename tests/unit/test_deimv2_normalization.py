"""Input-normalization contract for DEIMv2.

Upstream normalizes DINOv3 inputs with ImageNet statistics in the config
(``deimv2_dinov3_x_coco.yml:77``) and in every inference tool
(``tools/inference/{torch,onnx,trt}_inf.py``). The kit emitted it NOWHERE, so a
COCO detector optimised for normalized DINO inputs was fed raw [0, 1] tensors
on every run this project has done.

Normalization is DINOv3-ONLY. All four ``dinov3_*_coco.yml`` normalize; all
eight ``hgnetv2_*_coco.yml`` do not. Applying it to HGNetv2 would be the same
mistake in the other direction and would break the sea-lion recipe.

The parity rule these tests defend: train, val, predictor and the exported
modelspec must all report the same contract, and that contract is READ from the
run's own generated config rather than re-derived from the variant -- so a
checkpoint trained before this change is scored and exported without
normalization, automatically.
"""
import pytest

from kwcoco_detector_kit.trainers.deimv2 import (
    DINO_NORMALIZE_MEAN, DINO_NORMALIZE_STD, _amp_dtype_is_bf16,
    _normalize_from_cfg)


def _ops(cfg, which="train_dataloader"):
    return cfg[which]["dataset"]["transforms"]["ops"]


def _normalize_ops(ops):
    return [o for o in ops if isinstance(o, dict) and o.get("type") == "Normalize"]


# ---------------------------------------------------------------------------
# Emission, gated on family
# ---------------------------------------------------------------------------


def test_dinov3_train_and_val_are_normalized(tmp_path):
    from tests.unit.test_train_config_gen import _generate, _get_trainer
    cfg = _generate(_get_trainer(), tmp_path, variant="deimv2_dinov3_x",
                    input_hw=(1024, 1024), num_epochs=14)
    for which in ("train_dataloader", "val_dataloader"):
        got = _normalize_ops(_ops(cfg, which))
        assert len(got) == 1, f"{which} should carry exactly one Normalize"
        assert got[0]["mean"] == list(DINO_NORMALIZE_MEAN)
        assert got[0]["std"] == list(DINO_NORMALIZE_STD)


def test_hgnetv2_is_not_normalized(tmp_path):
    """Regression guard for the sea-lion recipe.

    base/deimv2.yml:104-105 goes straight from ConvertPILImage to
    ConvertBoxes. Normalizing HGNetv2 would hand ITS COCO checkpoint an input
    distribution it never trained on.
    """
    from tests.unit.test_train_config_gen import _generate, _get_trainer
    cfg = _generate(_get_trainer(), tmp_path, variant="deimv2_hgnetv2_n",
                    input_hw=(640, 640), num_epochs=14)
    for which in ("train_dataloader", "val_dataloader"):
        assert _normalize_ops(_ops(cfg, which)) == []


def test_normalize_sits_between_convertpilimage_and_convertboxes(tmp_path):
    """Order matters: it must apply to the scaled float tensor."""
    from tests.unit.test_train_config_gen import _generate, _get_trainer
    cfg = _generate(_get_trainer(), tmp_path, variant="deimv2_dinov3_x",
                    input_hw=(1024, 1024), num_epochs=14)
    kinds = [o.get("type") for o in _ops(cfg) if isinstance(o, dict)]
    assert kinds.index("ConvertPILImage") < kinds.index("Normalize")
    assert kinds.index("Normalize") < kinds.index("ConvertBoxes")


# ---------------------------------------------------------------------------
# Recovery: the contract is read back from the config, not the variant
# ---------------------------------------------------------------------------


def _cfg_with(ops):
    return {"val_dataloader": {"dataset": {"transforms": {"ops": ops}}}}


def test_recovers_the_dino_contract():
    cfg = _cfg_with([{"type": "Resize"},
                     {"type": "Normalize", "mean": [0.485, 0.456, 0.406],
                      "std": [0.229, 0.224, 0.225]}])
    mean, std = _normalize_from_cfg(cfg)
    assert mean == [0.485, 0.456, 0.406]
    assert std == [0.229, 0.224, 0.225]


@pytest.mark.parametrize("cfg", [
    _cfg_with([{"type": "Resize"}, {"type": "ConvertPILImage"}]),   # hgnetv2
    _cfg_with([]),
    {},                                                            # legacy run
])
def test_absent_normalize_means_identity(cfg):
    """Old checkpoints and HGNetv2 legitimately have no Normalize."""
    assert _normalize_from_cfg(cfg) == ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])


@pytest.mark.parametrize("bad", [
    {"type": "Normalize", "std": [1, 1, 1]},                  # no mean
    {"type": "Normalize", "mean": [0.5, 0.5, 0.5]},           # no std
    {"type": "Normalize", "mean": [0.5, 0.5], "std": [1, 1]},  # 2 channels
    {"type": "Normalize", "mean": [0.5, 0.5, 0.5], "std": [1, 0, 1]},  # zero std
    {"type": "Normalize", "mean": "nope", "std": [1, 1, 1]},
])
def test_malformed_normalize_raises_rather_than_degrading(bad):
    """Present-but-broken must NOT fall back to identity.

    Absence is a legitimate contract; corruption is not. Degrading silently
    would run a DINO model under preprocessing it never trained with, which is
    the class of mismatch this whole change exists to remove.
    """
    with pytest.raises(ValueError):
        _normalize_from_cfg(_cfg_with([{"type": "Resize"}, bad]))


# ---------------------------------------------------------------------------
# GradScaler follows the AMP dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype,expect_bf16", [
    ("bfloat16", True), ("bf16", True), ("BFloat16", True),
    ("float16", False), ("fp16", False), ("", False),
])
def test_amp_dtype_detection(dtype, expect_bf16, monkeypatch):
    if dtype:
        monkeypatch.setenv("KCD_AMP_DTYPE", dtype)
    else:
        monkeypatch.delenv("KCD_AMP_DTYPE", raising=False)
    assert _amp_dtype_is_bf16() is expect_bf16


@pytest.mark.parametrize("dtype,enabled", [("bfloat16", False), ("float16", True)])
def test_gradscaler_is_disabled_under_bf16(dtype, enabled, tmp_path, monkeypatch):
    """GradScaler compensates for fp16 underflow and is a vestige under bf16.

    Disabled rather than removed, so YAMLConfig.scaler still returns an object
    and det_engine stays in its autocast branch.
    """
    monkeypatch.setenv("KCD_AMP_DTYPE", dtype)
    from tests.unit.test_train_config_gen import _generate, _get_trainer
    cfg = _generate(_get_trainer(), tmp_path, variant="deimv2_dinov3_x",
                    input_hw=(1024, 1024), num_epochs=14)
    assert cfg["scaler"]["type"] == "GradScaler"
    assert cfg["scaler"]["enabled"] is enabled
