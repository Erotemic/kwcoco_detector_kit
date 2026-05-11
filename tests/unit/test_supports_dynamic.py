"""Tests for the supports_dynamic_input flag (failure #14).

DEIMv2 HGNetv2 family -> False (pre-bakes pos_embed at eval_spatial_size).
DEIMv2 DINOv3 family  -> True  (per-batch positional-embedding interpolation).
mock_tiny             -> True  (architecturally shape-agnostic).
"""
from __future__ import annotations

import pytest

from kwcoco_detector_kit.trainers._registry import get_trainer


HGNETV2_VARIANTS = [
    "deimv2_hgnetv2_atto", "deimv2_hgnetv2_femto", "deimv2_hgnetv2_pico",
    "deimv2_hgnetv2_n", "deimv2_hgnetv2_s", "deimv2_hgnetv2_m",
    "deimv2_hgnetv2_l", "deimv2_hgnetv2_x",
]
DINOV3_VARIANTS = [
    "deimv2_dinov3_s", "deimv2_dinov3_m",
    "deimv2_dinov3_l", "deimv2_dinov3_x",
]


@pytest.mark.parametrize("variant", HGNETV2_VARIANTS)
def test_deimv2_hgnetv2_returns_false(variant):
    trainer = get_trainer("deimv2")
    assert trainer.supports_dynamic_input(variant) is False


@pytest.mark.parametrize("variant", DINOV3_VARIANTS)
def test_deimv2_dinov3_returns_true(variant):
    trainer = get_trainer("deimv2")
    assert trainer.supports_dynamic_input(variant) is True


def test_mock_tiny_returns_true():
    trainer = get_trainer("mock_tiny")
    assert trainer.supports_dynamic_input("mock_tiny") is True


def test_unknown_variant_raises():
    trainer = get_trainer("deimv2")
    with pytest.raises(KeyError):
        trainer.supports_dynamic_input("deimv2_nonsense_xyz")
