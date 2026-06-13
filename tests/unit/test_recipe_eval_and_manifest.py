"""Tests for KCD-CFG-01 (${VAR} interpolation), KCD-EVAL-01 (first-class eval
mode), and KCD-DATA-01 (data manifest + expectation guard)."""
import pytest

from kwcoco_detector_kit.configs import expand_env_vars
from kwcoco_detector_kit.orchestration.recipe import (
    _build_sweep_data, _resolve_eval_block, _training_is_tiled,
)
from kwcoco_detector_kit.data.manifest import compute_manifest, assert_expected


# ---------- KCD-CFG-01: env interpolation ----------

def test_expand_env_basic(monkeypatch):
    monkeypatch.setenv("KCD_TEST_ROOT", "/ssd/work")
    assert expand_env_vars("a: ${KCD_TEST_ROOT}/x") == "a: /ssd/work/x"


def test_expand_env_default_used_when_unset(monkeypatch):
    monkeypatch.delenv("KCD_MISSING_X", raising=False)
    assert expand_env_vars("v: ${KCD_MISSING_X:-fallback}") == "v: fallback"


def test_expand_env_missing_raises(monkeypatch):
    monkeypatch.delenv("KCD_MISSING_Y", raising=False)
    with pytest.raises(KeyError) as ei:
        expand_env_vars("v: ${KCD_MISSING_Y}", source="recipe.yaml")
    assert "KCD_MISSING_Y" in str(ei.value)


def test_expand_env_leaves_bare_dollar_alone():
    # bare $VAR (no braces) must be untouched (don't mangle regex/shell-ish text)
    assert expand_env_vars("price is $5 and $HOME stays") == "price is $5 and $HOME stays"


# ---------- KCD-EVAL-01: first-class eval mode ----------

def _recipe(**overrides):
    base = {
        "schema": "recipe.v1",
        "name": "t",
        "data": {"train_kwcoco": "tr", "vali_kwcoco": "va", "test_kwcoco": "te"},
        "workspace": {"kcd_root": "/tmp/k"},
        "sweep": {
            "trainer": "deimv2",
            "matrix": [{"variant": "deimv2_pico", "input_hw": [768, 768],
                        "train_policy": "fixed"}],
            "category_names": "poop",
        },
    }
    base.update(overrides)
    return base


def test_legacy_sweep_tiled_eval_now_passes_through():
    # This is the exact v13 bug: sweep.tiled_eval was silently dropped.
    recipe = _recipe()
    recipe["sweep"]["tiled_eval"] = True
    recipe["sweep"]["tiled_eval_overlap"] = 0.2
    sweep_data = _build_sweep_data(recipe, {})
    assert sweep_data["tiled_eval"] is True
    assert sweep_data["tiled_eval_overlap"] == 0.2


def test_eval_block_explicit_tiled():
    recipe = _recipe(eval={"mode": "tiled", "overlap": 0.3, "device": "cuda"})
    sweep_data = _build_sweep_data(recipe, {})
    assert sweep_data["tiled_eval"] is True
    assert sweep_data["tiled_eval_overlap"] == 0.3
    assert sweep_data["eval_device"] == "cuda"


def test_eval_block_explicit_whole_image():
    recipe = _recipe(eval={"mode": "whole_image"})
    sweep_data = _build_sweep_data(recipe, {})
    assert sweep_data["tiled_eval"] is False


def test_eval_auto_resolves_tiled_for_corpus_data():
    # data.tiled set (pre-built tile corpus, train_policy=fixed) -> tiled.
    recipe = _recipe(eval={"mode": "auto"})
    recipe["data"]["tiled"] = True
    sweep_data = _build_sweep_data(recipe, {})
    assert sweep_data["tiled_eval"] is True


def test_eval_auto_resolves_tiled_for_multiscale_policy():
    recipe = _recipe(eval={"mode": "auto"})
    recipe["sweep"]["matrix"][0]["train_policy"] = "multiscale_512_768"
    assert _training_is_tiled(recipe)
    sweep_data = _build_sweep_data(recipe, {})
    assert sweep_data["tiled_eval"] is True


def test_eval_auto_resolves_whole_image_otherwise():
    recipe = _recipe(eval={"mode": "auto"})
    assert _training_is_tiled(recipe) is None
    sweep_data = _build_sweep_data(recipe, {})
    assert sweep_data["tiled_eval"] is False


def test_eval_block_bad_mode_raises():
    recipe = _recipe(eval={"mode": "nonsense"})
    with pytest.raises(ValueError):
        _build_sweep_data(recipe, {})


# ---------- KCD-DATA-01: data manifest ----------

def test_compute_manifest_counts(synthetic_kwcoco_factory):
    bundle = synthetic_kwcoco_factory("m", num_images=5, boxes_per_image=2,
                                      category_names=("poop",))
    man = compute_manifest(bundle)
    assert man["exists"] is True
    assert man["n_images"] == 5
    assert man["n_annots"] == 10
    assert man["categories"] == ["poop"]
    assert man["per_category"]["poop"] == 10
    assert len(man["content_hash"]) == 16


def test_compute_manifest_missing_file(tmp_path):
    man = compute_manifest(tmp_path / "nope.kwcoco.zip")
    assert man["exists"] is False


def test_assert_expected_passes(synthetic_kwcoco_factory):
    bundle = synthetic_kwcoco_factory("ok", num_images=4, category_names=("poop",))
    man = compute_manifest(bundle)
    assert assert_expected(man, {"n_images": 4, "categories": ["poop"]}) == []


def test_assert_expected_raises_on_mismatch(synthetic_kwcoco_factory):
    # the "filenames lie" bug: declared 7350, bundle actually has 4.
    bundle = synthetic_kwcoco_factory("lie", num_images=4, category_names=("poop",))
    man = compute_manifest(bundle)
    with pytest.raises(ValueError) as ei:
        assert_expected(man, {"n_images": 7350}, source="recipe.data.expect")
    assert "n_images" in str(ei.value)


def test_assert_expected_warn_mode_no_raise(synthetic_kwcoco_factory):
    bundle = synthetic_kwcoco_factory("warn", num_images=4, category_names=("poop",))
    man = compute_manifest(bundle)
    mismatches = assert_expected(man, {"n_images": 7350}, strict=False)
    assert mismatches and "n_images" in mismatches[0]
