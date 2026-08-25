"""The generated schedule must match the selected upstream config.

These tests exist because the kit used to carry its own copy of DEIMv2's
recipe, and the copy drifted without anything noticing:

  * ``_UPSTREAM_AUG_POLICY_EPOCHS = (4, 78, 148)`` over 150 epochs was the
    HGNETV2-N recipe applied universally, with its true total of 160
    mis-transcribed as 150. DINOv3-X is ``[4, 29, 50]`` over 58.
  * ``weight_decay`` was hardcoded to 1e-4, but DINOv3-L and -X use 1.25e-4.
  * ``mixup_epochs`` / ``copyblend_epochs`` / ``matcher_change_epoch`` were
    never emitted, so short runs silently inherited upstream's absolute values
    and never terminated augmentation or reached the matcher change.

Each test below fails if any of those regress.
"""
import pytest

from kwcoco_detector_kit.trainers._deimv2_recipe import (
    extract_recipe, load_upstream_config, scale_recipe)
from kwcoco_detector_kit.trainers.deimv2 import _resolve_upstream_cfg_fpath

DINOV3 = ["deimv2_dinov3_s", "deimv2_dinov3_m",
          "deimv2_dinov3_l", "deimv2_dinov3_x"]
HGNETV2 = ["deimv2_hgnetv2_" + s for s in
           ("atto", "femto", "pico", "n", "s", "m", "l", "x")]


def _cfg(variant):
    fpath = _resolve_upstream_cfg_fpath(variant)
    import os
    if not os.path.exists(fpath):
        pytest.skip(f"DEIMv2 submodule not present: {fpath}")
    return fpath


@pytest.mark.parametrize("variant", DINOV3 + HGNETV2)
def test_extracted_recipe_matches_the_vendored_yaml(variant):
    """The extraction must agree with the file, field by field."""
    fpath = _cfg(variant)
    recipe = extract_recipe(fpath)
    raw = load_upstream_config(fpath)
    collate = raw["train_dataloader"]["collate_fn"]
    assert recipe.total_epochs == raw["epoches"]
    if raw["flat_epoch"] <= raw["epoches"]:
        assert recipe.flat_epoch == raw["flat_epoch"]
    else:
        # hgnetv2_n only -- see test_upstream_flat_epoch_typo_is_repaired.
        assert recipe.flat_epoch == recipe.aug_policy_epochs[1]
    assert recipe.no_aug_epoch == raw["no_aug_epoch"]
    assert list(recipe.aug_policy_epochs) == list(
        raw["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"])
    assert list(recipe.mixup_epochs) == list(collate["mixup_epochs"])
    assert list(recipe.copyblend_epochs) == list(collate["copyblend_epochs"])
    assert recipe.stop_epoch == collate["stop_epoch"]
    assert recipe.matcher_change_epoch == \
        raw["DEIMCriterion"]["matcher"]["matcher_change_epoch"]
    assert recipe.weight_decay == raw["optimizer"]["weight_decay"]


def test_dinov3_x_is_the_58_epoch_recipe_not_a_150_epoch_one():
    """Regression for the phantom (4, 78, 148)/150 constant."""
    r = extract_recipe(_cfg("deimv2_dinov3_x"))
    assert r.total_epochs == 58
    assert r.aug_policy_epochs == (4, 29, 50)
    assert r.stop_epoch == 50
    assert r.matcher_change_epoch == 45


@pytest.mark.parametrize("variant,expected", [
    ("deimv2_dinov3_s", 1e-4), ("deimv2_dinov3_m", 1e-4),
    ("deimv2_dinov3_l", 1.25e-4), ("deimv2_dinov3_x", 1.25e-4),
])
def test_weight_decay_is_variant_specific(variant, expected):
    """Regression for flattening every DINOv3 variant to 1e-4."""
    assert extract_recipe(_cfg(variant)).weight_decay == pytest.approx(expected)


def test_variants_do_not_share_one_schedule():
    """A single hardcoded constant could not have covered these."""
    totals = {v: extract_recipe(_cfg(v)).total_epochs for v in DINOV3}
    assert len(set(totals.values())) == len(totals), totals


def test_scaling_is_identity_at_the_native_schedule():
    r = extract_recipe(_cfg("deimv2_dinov3_x"))
    assert scale_recipe(r, r.total_epochs) == r


def test_dinov3_x_scaled_to_14_epochs():
    """The gen006 schedule, pinned."""
    s = scale_recipe(extract_recipe(_cfg("deimv2_dinov3_x")), 14)
    assert s.aug_policy_epochs == (1, 7, 12)
    assert s.mixup_epochs == (1, 7)
    assert s.copyblend_epochs == (1, 12)
    assert s.stop_epoch == 12
    assert s.flat_epoch == 7
    assert s.no_aug_epoch == 2
    assert s.matcher_change_epoch == 11
    assert s.weight_decay == pytest.approx(1.25e-4)


@pytest.mark.parametrize("num_epochs", [1, 2, 3, 5, 8, 12, 14, 24, 36, 58])
def test_scaled_invariants_hold(num_epochs):
    r = extract_recipe(_cfg("deimv2_dinov3_x"))
    s = scale_recipe(r, num_epochs)
    e0, e1, e2 = s.aug_policy_epochs
    assert 0 < e0 <= e1 <= e2 <= max(1, num_epochs - 1)
    assert 1 <= s.stop_epoch <= num_epochs
    assert s.matcher_change_epoch < num_epochs
    assert 1 <= s.flat_epoch <= max(1, num_epochs - 1)
    # NOTE: stop_epoch == e2 and mixup == (e0, e1) hold for DINOv3 but are NOT
    # universal -- hgnetv2_n's copyblend is (4, 78) against e2=148, and
    # atto/femto/pico put stop_epoch at 468 against e2=400. Asserting them here
    # is what let a scaler that reconstructed those fields look correct.


def test_loader_has_no_shared_mutable_state():
    """Upstream's load_config(file_path, cfg=dict()) accumulates across calls.

    Resolving two variants in one process must not blend them.
    """
    x = load_upstream_config(_cfg("deimv2_dinov3_x"))
    s = load_upstream_config(_cfg("deimv2_dinov3_s"))
    x2 = load_upstream_config(_cfg("deimv2_dinov3_x"))
    assert x["epoches"] == x2["epoches"] == 58
    assert s["epoches"] == 132
    assert x["epoches"] != s["epoches"]


def test_upstream_flat_epoch_typo_is_repaired():
    """deimv2_hgnetv2_n_coco.yml sets flat_epoch 7800 against epoches 160.

    Under FlatCosineLR that means the LR never anneals. Every other variant
    sets flat_epoch to the MIDDLE augmentation boundary, so the intended value
    is 78. Propagating 7800 faithfully would hand every HGNetv2 run -- the
    sea-lion project's variant -- a constant learning rate for its whole
    schedule, which is a regression the kit's old `num_epochs // 2` did not
    have.
    """
    raw = load_upstream_config(_cfg("deimv2_hgnetv2_n"))
    assert raw["flat_epoch"] == 7800 and raw["epoches"] == 160, (
        "upstream changed; re-check whether the repair is still needed")
    r = extract_recipe(_cfg("deimv2_hgnetv2_n"))
    assert r.flat_epoch == 78 == r.aug_policy_epochs[1]


def test_the_phantom_constant_belonged_to_hgnetv2_n():
    """(4, 78, 148) was real -- it is HGNetv2-N's, not DINOv3's.

    The kit applied it to every variant, with N's true total of 160
    mis-transcribed as 150. Pinning the provenance so the two are never
    conflated again.
    """
    n = extract_recipe(_cfg("deimv2_hgnetv2_n"))
    x = extract_recipe(_cfg("deimv2_dinov3_x"))
    assert n.aug_policy_epochs == (4, 78, 148) and n.total_epochs == 160
    assert x.aug_policy_epochs == (4, 29, 50) and x.total_epochs == 58


@pytest.mark.parametrize("variant", DINOV3 + HGNETV2)
def test_flat_epoch_is_always_inside_the_schedule(variant):
    r = extract_recipe(_cfg(variant))
    assert 1 <= r.flat_epoch <= r.total_epochs


# ---------------------------------------------------------------------------
# Per-field scaling. An earlier scaler rebuilt mixup/copyblend/stop_epoch from
# the policy boundaries, which is right for DINOv3 and wrong for four of the
# twelve variants.
# ---------------------------------------------------------------------------

ALL_VARIANTS = DINOV3 + ["deimv2_hgnetv2_" + s for s in
                         ("atto", "femto", "pico", "n", "s", "m", "l", "x")]


@pytest.mark.parametrize("variant", ALL_VARIANTS)
@pytest.mark.parametrize("num_epochs", [2, 5, 14, 24, 58])
def test_disabled_augmentation_never_becomes_enabled(variant, num_epochs):
    """atto/femto/pico ship mixup and copyblend as (40000, 15000).

    Start after end is upstream's 'never run this'. Clamping such a window into
    range -- which a naive scaler does -- turns an augmentation the recipe
    disabled back on.
    """
    from kwcoco_detector_kit.trainers._deimv2_recipe import augmentation_is_disabled
    r = extract_recipe(_cfg(variant))
    s = scale_recipe(r, num_epochs)
    for field in ("mixup_epochs", "copyblend_epochs"):
        if augmentation_is_disabled(getattr(r, field)):
            assert augmentation_is_disabled(getattr(s, field)), (
                f"{variant} {field} was disabled upstream "
                f"{getattr(r, field)} and became {getattr(s, field)}")


def test_hgnetv2_n_copyblend_is_not_rebuilt_from_the_policy():
    """Its copyblend ends at 78, not at e2=148."""
    r = extract_recipe(_cfg("deimv2_hgnetv2_n"))
    assert r.copyblend_epochs == (4, 78)
    assert r.aug_policy_epochs[2] == 148
    s = scale_recipe(r, 14)
    # 78/160 * 14 = 6.8 -> 7, NOT 148/160 * 14 = 13
    assert s.copyblend_epochs[1] == 7


def test_stop_epoch_is_not_forced_to_equal_the_final_boundary():
    """atto decouples them: stop_epoch 468 against e2 400."""
    r = extract_recipe(_cfg("deimv2_hgnetv2_atto"))
    assert r.stop_epoch == 468 and r.aug_policy_epochs[2] == 400
    s = scale_recipe(r, 14)
    assert s.stop_epoch != s.aug_policy_epochs[2]


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_identity_at_native_schedule_for_every_variant(variant):
    r = extract_recipe(_cfg(variant))
    assert scale_recipe(r, r.total_epochs) == r


@pytest.mark.parametrize("variant", ALL_VARIANTS)
@pytest.mark.parametrize("num_epochs", [2, 5, 14, 24])
def test_enabled_window_never_starts_in_the_warmup_epoch(variant, num_epochs):
    from kwcoco_detector_kit.trainers._deimv2_recipe import augmentation_is_disabled
    s = scale_recipe(extract_recipe(_cfg(variant)), num_epochs)
    for field in ("mixup_epochs", "copyblend_epochs"):
        w = getattr(s, field)
        if not augmentation_is_disabled(w):
            assert w[0] >= 1, f"{variant} {field}={w} would fire during warmup"
