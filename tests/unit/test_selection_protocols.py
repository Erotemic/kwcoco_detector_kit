"""Protocol resolution, fingerprint composition, comparability identity."""
from __future__ import annotations

import pytest

from kwcoco_detector_kit.eval.protocols import (
    DatasetBinding,
    TRUE_TILED_V1,
    WHOLE_RESIZE_V1,
    fingerprint,
    get_protocol,
    protocol_id,
    resolve_protocol,
)


def test_registry_protocols_are_parameterized_families():
    # unresolved constants refuse to hash (forces explicit resolution)
    with pytest.raises(ValueError):
        TRUE_TILED_V1.to_jsonable()
    assert not TRUE_TILED_V1.is_resolved()


def test_resolve_replaces_params_and_is_stable():
    p1 = resolve_protocol(TRUE_TILED_V1, train_input_hw=(640, 640))
    p2 = resolve_protocol(TRUE_TILED_V1, train_input_hw=[640, 640])
    assert p1.is_resolved()
    assert p1.regime.window == (640, 640)
    assert protocol_id(p1) == protocol_id(p2)


def test_resolution_params_enter_the_fingerprint():
    """tiled@640 and tiled@1280 are different comparison spaces."""
    ds = DatasetBinding(role="vali", dataset_id="abc123")
    p640 = resolve_protocol(TRUE_TILED_V1, train_input_hw=(640, 640))
    p1280 = resolve_protocol(TRUE_TILED_V1, train_input_hw=(1280, 1280))
    assert fingerprint(p640, ds) != fingerprint(p1280, ds)


def test_fingerprint_is_protocol_compose_dataset():
    p = resolve_protocol(TRUE_TILED_V1, train_input_hw=(640, 640))
    ds_a = DatasetBinding(role="probe", dataset_id="aaaa")
    ds_b = DatasetBinding(role="probe", dataset_id="bbbb")
    assert fingerprint(p, ds_a) != fingerprint(p, ds_b)
    # n_images is informational, never part of the identity
    ds_a2 = DatasetBinding(role="probe", dataset_id="aaaa", n_images=50)
    assert fingerprint(p, ds_a) == fingerprint(p, ds_a2)
    # role is informational too: identity is purely content, so in-loop
    # scores on a file satisfy a rerank axis bound to the same file
    ds_a3 = DatasetBinding(role="vali_full", dataset_id="aaaa")
    assert fingerprint(p, ds_a) == fingerprint(p, ds_a3)


def test_probe_is_same_protocol_different_dataset():
    """The central factorization: proxy differs from target only in dataset."""
    p = resolve_protocol(TRUE_TILED_V1, train_input_hw=(640, 640))
    probe = DatasetBinding(role="probe", dataset_id="probe123")
    full = DatasetBinding(role="vali_full", dataset_id="vali456")
    assert protocol_id(p) == protocol_id(p)
    assert fingerprint(p, probe) != fingerprint(p, full)


def test_whole_and_tiled_never_share_a_fingerprint():
    ds = DatasetBinding(role="vali", dataset_id="abc")
    tiled = resolve_protocol(TRUE_TILED_V1, train_input_hw=(640, 640))
    whole = resolve_protocol(WHOLE_RESIZE_V1, train_input_hw=(640, 640))
    assert fingerprint(tiled, ds) != fingerprint(whole, ds)


def test_unknown_protocol_name_raises():
    with pytest.raises(KeyError):
        get_protocol("does_not_exist")


def test_missing_param_raises():
    with pytest.raises(KeyError):
        resolve_protocol(TRUE_TILED_V1, wrong_param=(640, 640))
