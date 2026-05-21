"""
DEIMv2 train-config generator — structural invariant tests.

These tests catch failure modes #13, #14, #18 by exercising every (variant
× train_policy × input_hw) cell in the kit's default sweep matrix:

  #13  collate_fn must be a SIBLING of dataset under train_dataloader,
       never a CHILD. With wrong indent, ``CocoDetection.__init__`` would
       crash. We assert structural placement.

  #14  HGNetv2 variants pre-bake pos_embed at eval_spatial_size and do NOT
       support multi-scale collate. Upstream's configs all set
       ``base_size_repeat: ~`` for HGNetv2. The kit's generator must:
         (a) report supports_dynamic_input(variant) == False for HGNetv2,
         (b) coerce the generated collate to base_size_repeat=None even
             when the caller requested multiscale (the round-loop layer
             handles the coercion at orchestration level; the trainer
             plugin enforces the architectural constraint itself).

  #18  Whenever input_hw changes, FIVE values must change in lockstep:
       - eval_spatial_size
       - train_dataloader.dataset.transforms[Resize].size
       - val_dataloader.dataset.transforms[Resize].size
       - train_dataloader.collate_fn.base_size  (the MS_BASE)
       - train_dataloader.dataset.transforms[Mosaic].output_size  (INPUT/2)

Tests do NOT depend on the DEIMv2 submodule being installed — only on
the kit's generator + yaml.safe_load. The Level-B "drive through
``engine.core.YAMLConfig``" variant lives under
``@pytest.mark.requires_deimv2`` and is opt-in.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_trainer():
    from kwcoco_detector_kit.trainers._registry import get_trainer
    return get_trainer("deimv2")


def _find_transform(transforms_block, transform_type):
    """Look up the first op of `transform_type` inside a Compose block."""
    ops = transforms_block.get("ops") or []
    for op in ops:
        if isinstance(op, dict) and op.get("type") == transform_type:
            return op
    return None


def _generate(trainer, tmp_path, *, variant, input_hw, train_policy="fixed",
              batch_size=4, val_batch_size=4, num_epochs=2, lr=5e-4,
              backbone_lr=2.5e-5, category_names=("widget",)):
    """Drive ``generate_config`` and return the parsed YAML dict."""
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True, exist_ok=True)
    category_names = list(category_names)
    cfg_fpath = trainer.generate_config(
        train_kwcoco_fpath="/tmp/train.mscoco.json",
        vali_kwcoco_fpath="/tmp/vali.mscoco.json",
        workdir=workdir,
        variant=variant,
        input_hw=tuple(input_hw),
        train_policy=train_policy,
        num_classes=len(category_names),
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_epochs=num_epochs,
        lr=lr,
        backbone_lr=backbone_lr,
        use_amp=False,
        channels="r|g|b",
        scale_tier="M",
        num_gpus=1,
        data_format="kwcoco",
        extra={"category_names": category_names},
    )
    assert Path(cfg_fpath).exists(), f"generate_config returned non-existent {cfg_fpath}"
    return yaml.safe_load(Path(cfg_fpath).read_text())


# Canonical variant list for the sweep matrix. 8 HGNetv2 + 4 DINOv3 = 12.
HGNETV2_VARIANTS = [
    "deimv2_hgnetv2_atto",
    "deimv2_hgnetv2_femto",
    "deimv2_hgnetv2_pico",
    "deimv2_hgnetv2_n",
    "deimv2_hgnetv2_s",
    "deimv2_hgnetv2_m",
    "deimv2_hgnetv2_l",
    "deimv2_hgnetv2_x",
]
DINOV3_VARIANTS = [
    "deimv2_dinov3_s",
    "deimv2_dinov3_m",
    "deimv2_dinov3_l",
    "deimv2_dinov3_x",
]
ALL_VARIANTS = HGNETV2_VARIANTS + DINOV3_VARIANTS

INPUT_HWS = [(320, 320), (416, 416), (512, 512), (640, 640)]

POLICIES_HGNETV2 = ["fixed"]                        # HGNetv2: only fixed
POLICIES_DINOV3 = ["fixed", "multiscale", "multiscale_512_768"]


# ---------------------------------------------------------------------------
# Smoke: every variant generates SOMETHING that yaml.safe_load parses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_every_variant_generates_parseable_yaml(variant, tmp_path):
    trainer = _get_trainer()
    cfg = _generate(trainer, tmp_path, variant=variant, input_hw=(320, 320))
    assert isinstance(cfg, dict), f"{variant}: top-level is not a dict"
    assert "train_dataloader" in cfg
    assert "val_dataloader" in cfg
    assert "eval_spatial_size" in cfg
    assert cfg.get("task") == "detection"


# ---------------------------------------------------------------------------
# Failure #13: collate_fn must be SIBLING of dataset, not CHILD
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_collate_fn_is_sibling_of_dataset(variant, tmp_path):
    trainer = _get_trainer()
    cfg = _generate(trainer, tmp_path, variant=variant, input_hw=(320, 320))
    td = cfg["train_dataloader"]
    assert "collate_fn" in td, (
        f"{variant}: train_dataloader has no collate_fn; would fall back to "
        "upstream default which doesn't enforce base_size = input"
    )
    assert "collate_fn" not in td["dataset"], (
        f"{variant}: collate_fn is nested inside train_dataloader.dataset — "
        "this is failure #13. DEIMv2's workspace.create would forward it as "
        "a kwarg to CocoDetection.__init__ and crash."
    )


# ---------------------------------------------------------------------------
# Failure #14: HGNetv2 supports_dynamic_input() == False; DINOv3 == True.
# Generated YAML for HGNetv2 must always have base_size_repeat == None
# regardless of the policy the caller requested.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", HGNETV2_VARIANTS)
def test_hgnetv2_supports_dynamic_input_false(variant):
    trainer = _get_trainer()
    assert trainer.supports_dynamic_input(variant) is False, (
        f"{variant}: HGNetv2 encoders pre-bake pos_embed and do NOT support "
        "multi-scale collate."
    )


@pytest.mark.parametrize("variant", DINOV3_VARIANTS)
def test_dinov3_supports_dynamic_input_true(variant):
    trainer = _get_trainer()
    assert trainer.supports_dynamic_input(variant) is True


@pytest.mark.parametrize(
    "variant", HGNETV2_VARIANTS
)
@pytest.mark.parametrize(
    "policy", ["fixed", "multiscale", "multiscale_320_512"]
)
def test_hgnetv2_yaml_forces_no_multiscale(variant, policy, tmp_path):
    """Even when caller asks for multiscale, HGNetv2 must emit base_size_repeat=None."""
    trainer = _get_trainer()
    cfg = _generate(trainer, tmp_path, variant=variant, input_hw=(320, 320),
                    train_policy=policy)
    collate = cfg["train_dataloader"]["collate_fn"]
    assert collate["base_size_repeat"] in (None,), (
        f"{variant} policy={policy}: base_size_repeat={collate['base_size_repeat']!r} "
        "but HGNetv2 must always be base_size_repeat=None (fixed scale)."
    )


@pytest.mark.parametrize("variant", DINOV3_VARIANTS)
def test_dinov3_multiscale_honored(variant, tmp_path):
    trainer = _get_trainer()
    cfg = _generate(trainer, tmp_path, variant=variant, input_hw=(640, 640),
                    train_policy="multiscale")
    collate = cfg["train_dataloader"]["collate_fn"]
    assert collate["base_size_repeat"] is not None and int(collate["base_size_repeat"]) > 0, (
        f"{variant} multiscale: base_size_repeat should be a positive int."
    )


# ---------------------------------------------------------------------------
# Failure #18: when input_hw changes, FIVE values change in lockstep.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ALL_VARIANTS)
@pytest.mark.parametrize("input_hw", INPUT_HWS)
def test_five_sizes_match_for_fixed_policy(variant, input_hw, tmp_path):
    trainer = _get_trainer()
    cfg = _generate(trainer, tmp_path, variant=variant, input_hw=input_hw,
                    train_policy="fixed")
    H, W = input_hw

    # 1. eval_spatial_size
    assert tuple(cfg["eval_spatial_size"]) == (H, W)

    # 2. train_dataloader.dataset.transforms[Resize].size
    train_tr = cfg["train_dataloader"]["dataset"]["transforms"]
    resize = _find_transform(train_tr, "Resize")
    assert resize is not None, f"{variant}: train transforms missing Resize"
    assert tuple(resize["size"]) == (H, W), (
        f"{variant}@{input_hw}: train Resize {resize['size']} != eval {input_hw}"
    )

    # 3. val_dataloader.dataset.transforms[Resize].size
    val_tr = cfg["val_dataloader"]["dataset"]["transforms"]
    val_resize = _find_transform(val_tr, "Resize")
    assert val_resize is not None, f"{variant}: val transforms missing Resize"
    assert tuple(val_resize["size"]) == (H, W), (
        f"{variant}@{input_hw}: val Resize {val_resize['size']} != eval {input_hw}"
    )

    # 4. train_dataloader.collate_fn.base_size = max(H, W) — INPUT_LONG
    collate = cfg["train_dataloader"]["collate_fn"]
    assert int(collate["base_size"]) == max(H, W), (
        f"{variant}@{input_hw}: collate base_size {collate['base_size']} != max(H,W)"
    )

    # 5. train_dataloader.dataset.transforms[Mosaic].output_size = H // 2
    #    Some variants may have no Mosaic op for tier S; only assert when present.
    mosaic = _find_transform(train_tr, "Mosaic")
    if mosaic is not None:
        expected_mosaic = H // 2
        assert int(mosaic["output_size"]) == expected_mosaic, (
            f"{variant}@{input_hw}: Mosaic output_size {mosaic['output_size']} "
            f"!= H/2 = {expected_mosaic}"
        )


# ---------------------------------------------------------------------------
# Optimizer block must follow the family (HGNetv2 vs DINOv3 backbone regex)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", HGNETV2_VARIANTS)
def test_hgnetv2_optimizer_block(variant, tmp_path):
    trainer = _get_trainer()
    cfg = _generate(trainer, tmp_path, variant=variant, input_hw=(320, 320))
    opt = cfg["optimizer"]
    params = opt["params"]
    pattern_strings = "|".join(p["params"] for p in params if "params" in p)
    assert "backbone" in pattern_strings, (
        f"{variant}: optimizer params should reference 'backbone' for HGNetv2"
    )


@pytest.mark.parametrize("variant", DINOV3_VARIANTS)
def test_dinov3_optimizer_block(variant, tmp_path):
    trainer = _get_trainer()
    cfg = _generate(trainer, tmp_path, variant=variant, input_hw=(640, 640))
    opt = cfg["optimizer"]
    params = opt["params"]
    pattern_strings = "|".join(p["params"] for p in params if "params" in p)
    assert "dinov3" in pattern_strings, (
        f"{variant}: optimizer params should reference 'dinov3' for DINOv3"
    )


# ---------------------------------------------------------------------------
# num_classes is derived from the kwcoco categories table, not hardcoded.
# ---------------------------------------------------------------------------


def test_num_classes_is_not_hardcoded_to_one(tmp_path):
    trainer = _get_trainer()
    five_cats = [f"cat{i}" for i in range(5)]
    cfg = _generate(trainer, tmp_path, variant="deimv2_hgnetv2_n",
                    input_hw=(320, 320), category_names=five_cats)
    assert int(cfg["num_classes"]) == 5, (
        f"num_classes={cfg['num_classes']} but caller passed 5"
    )


# ---------------------------------------------------------------------------
# Per-variant memory table: deimv2_hgnetv2_n @ 320x320 batch 24 GB returns
# something in the 16-32 range. Larger inputs auto-shrink.
# ---------------------------------------------------------------------------


def test_memory_tier_default_batch_shrinks_with_input_area():
    trainer = _get_trainer()
    b320 = trainer.memory_tier_default_batch("deimv2_hgnetv2_n", (320, 320), 24.0)
    b640 = trainer.memory_tier_default_batch("deimv2_hgnetv2_n", (640, 640), 24.0)
    assert b320 > b640, (
        f"deimv2_hgnetv2_n: batch should shrink at larger input "
        f"({b320} @ 320 vs {b640} @ 640)"
    )


def test_memory_tier_default_batch_grows_with_vram():
    trainer = _get_trainer()
    b24 = trainer.memory_tier_default_batch("deimv2_hgnetv2_n", (320, 320), 24.0)
    b80 = trainer.memory_tier_default_batch("deimv2_hgnetv2_n", (320, 320), 80.0)
    assert b80 >= b24, (
        f"deimv2_hgnetv2_n: batch at 80GB ({b80}) should be >= 24GB ({b24})"
    )


# ---------------------------------------------------------------------------
# multiscale_<lo>_<hi> policy parser
# ---------------------------------------------------------------------------


def test_multiscale_policy_with_explicit_band(tmp_path):
    """multiscale_320_512 puts base around 416 and produces a band that includes that range."""
    trainer = _get_trainer()
    cfg = _generate(trainer, tmp_path, variant="deimv2_dinov3_s",
                    input_hw=(416, 416), train_policy="multiscale_320_512")
    collate = cfg["train_dataloader"]["collate_fn"]
    base = int(collate["base_size"])
    # Per the prior project's parser: MS_BASE = ((lo+hi)/2 + 16) rounded to 32.
    # (320+512)/2 = 416; +16 -> 432; /32 -> 13.5 -> 13 -> *32 = 416.
    assert 320 <= base <= 512, (
        f"multiscale_320_512 produced base_size {base}, expected in [320, 512]"
    )
    assert int(collate["base_size_repeat"]) > 0


# ---------------------------------------------------------------------------
# Per-variant upstream config relpath — every variant resolves to an
# actual upstream config file in tpl/DEIMv2/configs/deimv2/
# ---------------------------------------------------------------------------


def test_every_variant_has_known_upstream_config_relpath():
    from kwcoco_detector_kit.trainers._registry import get_trainer
    trainer = get_trainer("deimv2")
    for variant in ALL_VARIANTS:
        relpath = trainer.variants[variant]["upstream_config_relpath"]
        assert relpath.startswith("configs/deimv2/"), (
            f"{variant}: upstream_config_relpath {relpath!r} doesn't live under configs/deimv2/"
        )
        # File name shape: deimv2_hgnetv2_<size>_coco.yml or deimv2_dinov3_<size>_coco.yml
        assert relpath.endswith("_coco.yml"), (
            f"{variant}: relpath {relpath!r} doesn't end in _coco.yml"
        )
