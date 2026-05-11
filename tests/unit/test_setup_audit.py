"""Tests for orchestration.setup_audit."""
from __future__ import annotations

from kwcoco_detector_kit.orchestration.setup_audit import (
    PROBES, probe_env, Probe,
)


def test_probes_cover_all_19_failure_modes_classes():
    """PROBES should at least name onnx, onnxscript, onnxsim (#9 + #10),
    faster_coco_eval (#11), and a torch presence check."""
    names = {p.module for p in PROBES}
    expected_required_minimum = {
        "onnx", "onnxscript", "onnxsim", "onnxruntime",
        "faster_coco_eval", "transformers", "tensorboard", "scipy",
        "torch", "kwcoco", "kwimage", "scriptconfig", "ubelt", "yaml",
    }
    missing = expected_required_minimum - names
    assert not missing, f"setup_audit must probe: {missing}"


def test_core_group_lacks_no_critical_dep():
    """Probing 'core' should pass on the test env (where pyproject.toml's
    install_requires has been satisfied)."""
    missing = probe_env(groups=["core"])
    # kwcoco / kwimage / scriptconfig / torch / cv2 / yaml all in install_requires.
    assert not missing, f"core probes failed: {[p.module for p in missing]}"


def test_group_filter_restricts_probes():
    onnx_only = probe_env(groups=["onnx"])
    for p in onnx_only:
        assert p.group == "onnx"
