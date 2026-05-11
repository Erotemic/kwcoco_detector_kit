"""Tests for trainers._tier — auto-tier detection + AMP defaults."""
from __future__ import annotations

import pytest

from kwcoco_detector_kit.trainers._tier import (
    detect_tier, use_amp_for_tier, TierInfo,
)


def test_detect_tier_force_overrides():
    info = detect_tier(force="cluster")
    assert info.tier == "cluster"


def test_detect_tier_on_cpu_returns_tier_S():
    """On a CPU host with no CUDA, aggregate_vram_gb=0 -> tier S."""
    info = detect_tier()
    assert info.tier == "S"
    assert info.aggregate_vram_gb == 0.0


def test_use_amp_for_tier_default():
    assert use_amp_for_tier("S") is False
    assert use_amp_for_tier("M") is True
    assert use_amp_for_tier("L") is True
    assert use_amp_for_tier("XL") is True
    assert use_amp_for_tier("cluster") is True
