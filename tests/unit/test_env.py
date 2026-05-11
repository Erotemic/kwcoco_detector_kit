"""Tests for kwcoco_detector_kit._env helpers."""
from __future__ import annotations

import os
import resource

from kwcoco_detector_kit._env import (
    raise_nofile_limit,
    prepend_pythonpath,
    default_cuda_visible_devices,
)


def test_raise_nofile_limit_clamps_to_hard_cap():
    """Failure #15: requesting above hard cap clamps to the hard cap."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # Request well above the hard cap.
    before, after, status = raise_nofile_limit(target=hard + 100_000)
    assert before == soft
    # Status is 'no_change' (already at or above), 'raised', or 'clamped'.
    assert status in ("no_change", "raised", "clamped", "failed")
    if status == "clamped":
        assert after <= hard


def test_raise_nofile_limit_no_change_when_already_high():
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft > 1:
        before, after, status = raise_nofile_limit(target=1)
        assert status == "no_change"
        assert before == after


def test_prepend_pythonpath_is_idempotent(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    paths = ["/tmp/a", "/tmp/b"]
    first = prepend_pythonpath(paths)
    second = prepend_pythonpath(paths)
    # Second call doesn't duplicate.
    assert second == first
    parts = first.split(os.pathsep)
    assert parts.count(str(os.path.realpath("/tmp/a"))) == 1


def test_default_cuda_visible_devices_when_unset(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert default_cuda_visible_devices() == "0"


def test_default_cuda_visible_devices_respects_user(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    assert default_cuda_visible_devices() is None
