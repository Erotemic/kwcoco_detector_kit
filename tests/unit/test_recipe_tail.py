"""An absolute stage-2 / no-aug tail.

Proportional scaling preserves upstream's shape, and for the tail that is the
wrong invariant: DINOv3-X ends with 8 epochs past stop_epoch at 58 epochs,
which scales to 4 at 28 epochs and 2 at 14. The phase that consolidates
shrinks exactly as the schedule gets shorter, so the runs with the least
training get the least consolidation.
"""
import pytest

from kwcoco_detector_kit.trainers._deimv2_recipe import (
    extract_recipe, retarget_tail, scale_recipe)
from kwcoco_detector_kit.trainers.deimv2 import _resolve_upstream_cfg_fpath


def _x():
    import os
    fpath = _resolve_upstream_cfg_fpath("deimv2_dinov3_x")
    if not os.path.exists(fpath):
        pytest.skip(f"DEIMv2 submodule not present: {fpath}")
    return extract_recipe(fpath)


def test_upstream_tail_is_eight_epochs():
    """The number gen007 preserves, pinned so a change upstream is visible."""
    r = _x()
    assert r.total_epochs - r.stop_epoch == 8


def test_the_gen007_schedule_is_pinned():
    s = retarget_tail(scale_recipe(_x(), 28), 8)
    assert s.aug_policy_epochs == (2, 12, 20)
    assert s.flat_epoch == 12
    assert s.no_aug_epoch == 8
    assert s.stop_epoch == 20
    assert s.matcher_change_epoch == 18


def test_proportional_scaling_shrinks_the_tail_and_retarget_does_not():
    """The motivating comparison, stated as a test."""
    for n in (14, 28):
        assert scale_recipe(_x(), n).total_epochs - scale_recipe(_x(), n).stop_epoch < 8
        assert retarget_tail(scale_recipe(_x(), n), 8).stop_epoch == n - 8


@pytest.mark.parametrize("n,tail", [(14, 8), (20, 8), (28, 8), (40, 8), (28, 4)])
def test_invariants_hold(n, tail):
    s = retarget_tail(scale_recipe(_x(), n), tail)
    e0, e1, e2 = s.aug_policy_epochs
    assert s.stop_epoch == n - tail
    assert 0 < e0 <= e1 <= e2 == s.stop_epoch
    assert s.matcher_change_epoch < n
    assert 1 <= s.flat_epoch <= n - 1


def test_flat_epoch_tracks_the_middle_boundary():
    """Upstream's convention -- see _repair_flat_epoch."""
    s = retarget_tail(scale_recipe(_x(), 28), 8)
    assert s.flat_epoch == s.aug_policy_epochs[1]


def test_a_tail_that_leaves_no_primary_phase_is_rejected():
    with pytest.raises(ValueError):
        retarget_tail(scale_recipe(_x(), 10), 10)


@pytest.mark.parametrize("bad", [0, -1])
def test_nonpositive_tail_is_rejected(bad):
    with pytest.raises(ValueError):
        retarget_tail(scale_recipe(_x(), 28), bad)


def test_weight_decay_and_mixup_are_untouched():
    """retarget_tail moves landmarks only; it must not restyle the recipe."""
    base = scale_recipe(_x(), 28)
    s = retarget_tail(base, 8)
    assert s.weight_decay == base.weight_decay
    assert s.mixup_epochs == base.mixup_epochs
    assert s.copyblend_epochs == base.copyblend_epochs
