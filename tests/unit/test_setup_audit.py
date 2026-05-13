"""Tests for orchestration.setup_audit."""
from __future__ import annotations

from kwcoco_detector_kit.orchestration.setup_audit import (
    PROBES, probe_env, Probe,
    _strict_import, _parse_groups, _hint_for_error, _satisfies_version_spec,
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


def test_opengroundingdino_pins_transformers_before_v5():
    probes = [p for p in PROBES if p.group == "opengroundingdino" and p.module == "transformers"]
    assert probes
    assert probes[0].version_spec == ">=4.35,<4.47"
    assert probes[0].pip_name == "transformers>=4.35,<4.47"


# ---------------------------------------------------------------------------
# _strict_import — catches version conflicts that find_spec misses
# ---------------------------------------------------------------------------


def test_strict_import_returns_true_for_real_module():
    ok, err = _strict_import("kwcoco")
    assert ok is True
    assert err is None


def test_strict_import_returns_false_for_missing_module():
    ok, err = _strict_import("nonexistent_module_xyz_12345")
    assert ok is False
    assert err is not None
    assert "ModuleNotFoundError" in err or "ImportError" in err


def test_version_spec_check_for_installed_core_module():
    ok, err = _satisfies_version_spec("yaml", ">0")
    assert ok is True
    assert err is None


def test_version_spec_check_reports_impossible_constraint():
    ok, err = _satisfies_version_spec("yaml", "<0")
    assert ok is False
    assert "does not satisfy" in err


# ---------------------------------------------------------------------------
# _parse_groups — tolerates both string and list inputs (scriptconfig smartcast)
# ---------------------------------------------------------------------------


def test_parse_groups_from_comma_string():
    assert _parse_groups("core,onnx,deimv2") == ["core", "onnx", "deimv2"]


def test_parse_groups_from_list():
    assert _parse_groups(["core", "onnx"]) == ["core", "onnx"]


def test_parse_groups_from_stringified_list():
    """scriptconfig's smartcast can deliver str([list]) — `[\"'core'\", \"'onnx'\"]`."""
    assert _parse_groups("['core', 'onnx']") == ["core", "onnx"]
    # And the actual scriptconfig-mangled form: a list whose elements are
    # already quoted strings like "'core'".
    assert _parse_groups(["'core'", "'onnx'"]) == ["core", "onnx"]


# ---------------------------------------------------------------------------
# _hint_for_error — actionable fix lines for known transitive conflicts
# ---------------------------------------------------------------------------


def test_hint_for_transformers_huggingface_hub_conflict():
    """The error reported by the user's host smoke."""
    err = (
        "ImportError: huggingface-hub>=0.34.0,<1.0 is required for a normal "
        "functioning of this module, but found huggingface-hub==1.14.0."
    )
    hint = _hint_for_error("transformers", err)
    assert hint is not None
    assert "transformers" in hint or "huggingface-hub" in hint


def test_hint_for_error_returns_none_for_unknown_module():
    assert _hint_for_error("widget_module", "ImportError: something weird") is None
