"""DEIMv2: PostProcessor.num_top_queries must be <= num_queries * num_classes (lesson #26).

Upstream's default of 300 assumes COCO ``num_classes=91``; with the
kit's ``num_classes=1`` override the topk axis collapses to
``num_queries`` and the trainer crashes inside ``evaluate()`` with
``RuntimeError: selected index k out of range`` after the first epoch.
Test every variant × num_classes combination.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml


HGNETV2_VARIANTS = [
    "deimv2_hgnetv2_atto", "deimv2_hgnetv2_femto", "deimv2_hgnetv2_pico",
    "deimv2_hgnetv2_n", "deimv2_hgnetv2_s", "deimv2_hgnetv2_m",
    "deimv2_hgnetv2_l", "deimv2_hgnetv2_x",
]
DINOV3_VARIANTS = [
    "deimv2_dinov3_s", "deimv2_dinov3_m",
    "deimv2_dinov3_l", "deimv2_dinov3_x",
]
ALL_VARIANTS = HGNETV2_VARIANTS + DINOV3_VARIANTS


def _generate(trainer, tmp_path, *, variant, num_classes, input_hw=(256, 256)):
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True, exist_ok=True)
    category_names = [f"cat{i}" for i in range(int(num_classes))]
    cfg_fpath = trainer.generate_config(
        train_kwcoco_fpath="/tmp/train.mscoco.json",
        vali_kwcoco_fpath="/tmp/vali.mscoco.json",
        workdir=workdir,
        variant=variant,
        input_hw=tuple(input_hw),
        train_policy="fixed",
        num_classes=int(num_classes),
        batch_size=2, val_batch_size=2, num_epochs=1,
        lr=5e-4, backbone_lr=2.5e-5, use_amp=False,
        channels="r|g|b", scale_tier="S", num_gpus=1,
        data_format="kwcoco",
        extra={"category_names": category_names},
    )
    return yaml.safe_load(Path(cfg_fpath).read_text())


@pytest.mark.parametrize("variant", ALL_VARIANTS)
@pytest.mark.parametrize("num_classes", [1, 5, 20])
def test_num_top_queries_does_not_exceed_num_queries_times_num_classes(
    variant, num_classes, tmp_path,
):
    """The invariant that prevents the topk crash."""
    from kwcoco_detector_kit.trainers._registry import get_trainer
    trainer = get_trainer("deimv2")

    info = trainer.variants[variant]
    num_queries = int(info["num_queries"])

    cfg = _generate(trainer, tmp_path, variant=variant, num_classes=num_classes)
    assert "PostProcessor" in cfg, f"{variant}: missing PostProcessor override"
    topk = int(cfg["PostProcessor"]["num_top_queries"])
    assert topk >= 1
    assert topk <= num_queries * num_classes, (
        f"{variant} num_classes={num_classes}: num_top_queries={topk} > "
        f"num_queries*num_classes={num_queries * num_classes}; "
        "evaluate() will raise RuntimeError: selected index k out of range"
    )


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_num_top_queries_matches_coco_default_for_91_classes(variant, tmp_path):
    """At ``num_classes=91`` (COCO), the kit's override should match the
    upstream default of 300 — we don't accidentally shrink the topk on
    COCO setups."""
    from kwcoco_detector_kit.trainers._registry import get_trainer
    trainer = get_trainer("deimv2")
    cfg = _generate(trainer, tmp_path, variant=variant, num_classes=91)
    topk = int(cfg["PostProcessor"]["num_top_queries"])
    assert topk == 300, (
        f"{variant} num_classes=91: expected topk=300 (upstream default); got {topk}"
    )


def test_per_variant_num_queries_table_is_complete():
    """Every registered variant has a ``num_queries`` entry."""
    from kwcoco_detector_kit.trainers._registry import get_trainer
    trainer = get_trainer("deimv2")
    for name in ALL_VARIANTS:
        info = trainer.variants[name]
        assert "num_queries" in info, f"{name}: missing num_queries"
        assert isinstance(info["num_queries"], int) and info["num_queries"] > 0


def test_per_variant_num_queries_matches_upstream():
    """Spot-check the small-variant overrides (the kit's hand-maintained table)."""
    from kwcoco_detector_kit.trainers._registry import get_trainer
    trainer = get_trainer("deimv2")
    # Atto/femto/pico explicitly override upstream; the rest inherit
    # configs/base/deimv2.yml's default of 300.
    assert trainer.variants["deimv2_hgnetv2_atto"]["num_queries"] == 100
    assert trainer.variants["deimv2_hgnetv2_femto"]["num_queries"] == 150
    assert trainer.variants["deimv2_hgnetv2_pico"]["num_queries"] == 200
    for v in ("deimv2_hgnetv2_n", "deimv2_hgnetv2_s", "deimv2_hgnetv2_x",
              "deimv2_dinov3_s", "deimv2_dinov3_l", "deimv2_dinov3_x"):
        assert trainer.variants[v]["num_queries"] == 300


def test_effective_num_top_queries_helper():
    """Direct sanity-check of the helper function."""
    from kwcoco_detector_kit.trainers.deimv2 import _effective_num_top_queries

    # Below upstream default — keep all queries
    assert _effective_num_top_queries(num_queries=100, num_classes=1) == 100
    # Above upstream default — cap at default
    assert _effective_num_top_queries(num_queries=100, num_classes=91) == 300
    # Edge: zero classes (should never happen but the helper must clamp to >= 1)
    assert _effective_num_top_queries(num_queries=100, num_classes=0) == 1
