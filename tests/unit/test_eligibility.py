"""Tests for orchestration.eligibility — four-class state machine + candidate_kind filter."""
from __future__ import annotations

import pytest

from kwcoco_detector_kit.orchestration.eligibility import (
    Row, classify_row,
    NOT_READY, HOST_PROMISING, DEPLOY_ELIGIBLE, DEPLOY_INELIGIBLE,
    _infer_candidate_kind,
)


def _ok_row(**overrides):
    base = dict(
        candidate_id="c0", variant="deimv2_hgnetv2_n",
        checkpoint_path="best_stg2.pth", onnx_path="x.onnx",
        modelspec_path="x.modelspec.json",
        test_ap=0.30,
        desktop_latency_ms_mean=50.0,
    )
    base.update(overrides)
    return Row(**base)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


def test_not_ready_when_no_checkpoint():
    row = _ok_row(checkpoint_path="")
    classify_row(row, max_desktop_ms=80, allow_missing_desktop_bench=False)
    assert row.eligibility_class == NOT_READY
    assert row.status == "no_checkpoint"


def test_not_ready_when_no_onnx():
    row = _ok_row(onnx_path="")
    classify_row(row, max_desktop_ms=80, allow_missing_desktop_bench=False)
    assert row.eligibility_class == NOT_READY
    assert row.status == "no_onnx"


def test_not_ready_when_no_eval():
    row = _ok_row(test_ap=None)
    classify_row(row, max_desktop_ms=80, allow_missing_desktop_bench=False)
    assert row.eligibility_class == NOT_READY
    assert row.status == "no_eval"


def test_not_ready_when_no_desktop_bench_and_strict():
    row = _ok_row(desktop_latency_ms_mean=None)
    classify_row(row, max_desktop_ms=80, allow_missing_desktop_bench=False)
    assert row.eligibility_class == NOT_READY


def test_host_promising_when_no_desktop_bench_but_lenient():
    row = _ok_row(desktop_latency_ms_mean=None)
    classify_row(row, max_desktop_ms=80, allow_missing_desktop_bench=True)
    assert row.eligibility_class == HOST_PROMISING


def test_host_promising_when_desktop_under_gate_and_no_device_data():
    row = _ok_row(desktop_latency_ms_mean=30.0)
    classify_row(row, max_desktop_ms=80, allow_missing_desktop_bench=False)
    assert row.eligibility_class == HOST_PROMISING
    assert row.desktop_eligible == "yes"


def test_deploy_ineligible_when_desktop_over_gate():
    row = _ok_row(desktop_latency_ms_mean=120.0)
    classify_row(row, max_desktop_ms=80, allow_missing_desktop_bench=False)
    assert row.eligibility_class == DEPLOY_INELIGIBLE
    assert row.desktop_eligible == "no"


def test_deploy_ineligible_when_device_fps_below_gate():
    row = _ok_row(desktop_latency_ms_mean=30.0)
    row.device_fps = 5.0
    row.device_eligible = "no"
    classify_row(row, max_desktop_ms=80, allow_missing_desktop_bench=False)
    assert row.eligibility_class == DEPLOY_INELIGIBLE


def test_deploy_eligible_when_device_fps_at_gate():
    row = _ok_row(desktop_latency_ms_mean=30.0)
    row.device_fps = 20.0
    row.device_eligible = "yes"
    classify_row(row, max_desktop_ms=80, allow_missing_desktop_bench=False)
    assert row.eligibility_class == DEPLOY_ELIGIBLE


# ---------------------------------------------------------------------------
# candidate_kind inference — smoke vs real
# ---------------------------------------------------------------------------


def test_candidate_kind_explicit_smoke():
    assert _infer_candidate_kind({"candidate_kind": "smoke"}) == "smoke"


def test_candidate_kind_explicit_real():
    assert _infer_candidate_kind({"candidate_kind": "real"}) == "real"


def test_candidate_kind_falls_back_to_variant_prefix_mock_tiny():
    assert _infer_candidate_kind({"variant": "mock_tiny"}) == "smoke"


def test_candidate_kind_falls_back_to_variant_prefix_v4_mock():
    """Backwards-compat for legacy variant strings from the prior project."""
    assert _infer_candidate_kind({"variant": "v4_mock_tiny"}) == "smoke"


def test_candidate_kind_defaults_to_real():
    assert _infer_candidate_kind({"variant": "deimv2_hgnetv2_n"}) == "real"


def test_candidate_kind_unknown_treated_as_real():
    assert _infer_candidate_kind({"variant": "future_variant"}) == "real"
