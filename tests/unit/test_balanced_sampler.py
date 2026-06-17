"""Dataloader-level balanced sampling: weights pipeline + DDP sampler."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from kwcoco_detector_kit.data.balanced_sampler import (
    EMPTY_KEY,
    DistributedWeightedRandomSampler,
    build_sample_grid_from_mscoco,
    compute_index_weights,
    load_balance_weights,
    sampler_from_weights_file,
    write_balance_weights,
)


def _mscoco(tmp_path, n_pup=2, n_bull=3, n_empty=4):
    cats = [{"id": 1, "name": "pup"}, {"id": 2, "name": "bull"}]
    images, anns = [], []
    aid = 1
    for i in range(n_pup + n_bull + n_empty):
        images.append({"id": i + 1, "file_name": f"im_{i}.jpg",
                       "width": 640, "height": 640})
    for i in range(n_pup):
        anns.append({"id": aid, "image_id": i + 1, "category_id": 1,
                     "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0})
        aid += 1
    for i in range(n_bull):
        anns.append({"id": aid, "image_id": n_pup + i + 1, "category_id": 2,
                     "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0})
        aid += 1
    fpath = tmp_path / "train.mscoco.json"
    fpath.write_text(json.dumps(
        {"images": images, "annotations": anns, "categories": cats}))
    return fpath


class StubForest:
    """Stands in for BalancedSampleForest until .index_weights() ships."""

    def __init__(self, sample_grid, rng=None):
        self.grid = sample_grid
        self.subdivide_calls = []

    def subdivide(self, key, weights=None, default_weight=0):
        self.subdivide_calls.append((key, weights))

    def index_weights(self):
        # upweight rare strata: weight 1/|stratum|
        from collections import Counter
        strata = [tuple(sorted(item["classes"])) for item in self.grid]
        sizes = Counter(strata)
        return [1.0 / sizes[s] for s in strata]


def test_grid_order_and_empty_sentinel(tmp_path):
    fpath = _mscoco(tmp_path, n_pup=1, n_bull=1, n_empty=2)
    grid = build_sample_grid_from_mscoco(fpath)
    assert [g["classes"] for g in grid] == [
        {"pup": 1}, {"bull": 1}, {EMPTY_KEY: 1}, {EMPTY_KEY: 1},
    ]


def test_compute_index_weights_via_forest_contract(tmp_path):
    fpath = _mscoco(tmp_path, n_pup=1, n_bull=2, n_empty=4)
    made = {}

    def factory(grid, rng):
        made["forest"] = StubForest(grid, rng)
        return made["forest"]

    target = {"pup": 0.5, "bull": 0.3, EMPTY_KEY: 0.2}
    w = compute_index_weights(
        fpath, class_weights=target, forest_factory=factory)
    # class_weights reach the 'classes' subdivision
    assert made["forest"].subdivide_calls == [("classes", target)]
    # normalized; rare stratum (pup, 1 image) outweighs common (empty, 4)
    assert sum(w) == pytest.approx(1.0)
    assert w[0] > w[3]


def test_max_oversample_cap(tmp_path):
    # StubForest gives equal mass per STRATUM; 1 pup tile vs 99 empty tiles
    # -> pup stratum (1 tile) and empty stratum (99 tiles) each get 0.5 mass
    # -> pup tile weight = 0.5, each empty tile weight = 0.5/99 ≈ 0.005
    fpath = _mscoco(tmp_path, n_pup=1, n_bull=0, n_empty=99)

    def factory(grid, rng):
        return StubForest(grid, rng)

    w_uncapped = compute_index_weights(fpath, forest_factory=factory)
    assert w_uncapped[0] == pytest.approx(0.5)       # pup gets half the mass

    # max_oversample=1: cap at 1/100 per index, then renormalize.
    # Pup (0.5) gets capped down to 0.01; empty tiles (each ~0.005) are
    # below the cap and rise slightly after renorm.
    w_capped = compute_index_weights(
        fpath, forest_factory=factory, max_oversample=1)
    assert sum(w_capped) == pytest.approx(1.0)
    # pup was capped to exactly the cap boundary (1/100), before renorm
    assert w_capped[0] < w_uncapped[0]           # cap reduced pup's share
    assert w_capped[0] <= 1.0 / 100 + 1e-9  # at or below the cap


def test_missing_index_weights_is_a_clear_error(tmp_path):
    fpath = _mscoco(tmp_path)

    class OldForest:
        """Today's BalancedSampleForest: subdivide but no index_weights."""
        def __init__(self, grid):
            pass

        def subdivide(self, key, weights=None, default_weight=0):
            pass

    with pytest.raises(NotImplementedError, match="index_weights"):
        compute_index_weights(fpath, forest_factory=lambda g, rng: OldForest(g))


def test_weights_file_roundtrip_and_mismatch_guard(tmp_path):
    fpath = write_balance_weights(
        tmp_path / "w.json", [0.25, 0.25, 0.5], meta={"seed": 0})
    assert load_balance_weights(fpath) == [0.25, 0.25, 0.5]
    sampler = sampler_from_weights_file(fpath, dataset_len=3)
    assert len(sampler) == 3
    with pytest.raises(ValueError, match="different annotation file"):
        sampler_from_weights_file(fpath, dataset_len=99)


def test_sampler_is_deterministic_and_epoch_varying():
    w = [1.0, 1.0, 1.0, 1.0]
    s1 = DistributedWeightedRandomSampler(w, seed=7, rank=0, world_size=1)
    s2 = DistributedWeightedRandomSampler(w, seed=7, rank=0, world_size=1)
    assert list(s1) == list(s2)              # same (seed, epoch, rank)
    s2.set_epoch(1)
    assert list(s1) != list(s2)              # epoch reseeds


def test_sampler_ranks_draw_independent_streams_with_split_length():
    w = [1.0] * 10
    r0 = DistributedWeightedRandomSampler(w, seed=0, rank=0, world_size=4)
    r1 = DistributedWeightedRandomSampler(w, seed=0, rank=1, world_size=4)
    assert len(r0) == len(r1) == 3            # ceil(10 / 4)
    assert list(r0) != list(r1)


def test_sampler_respects_weights():
    # index 0 carries 90% of the mass
    w = [0.9] + [0.1 / 99] * 99
    s = DistributedWeightedRandomSampler(
        w, num_samples_total=5000, seed=0, rank=0, world_size=1)
    counts = Counter(s)
    assert counts[0] / 5000 == pytest.approx(0.9, abs=0.03)


def test_sampler_epoch_length_override():
    s = DistributedWeightedRandomSampler(
        [1.0] * 100, num_samples_total=10, seed=0, rank=0, world_size=2)
    assert len(s) == 5


def test_sampler_rejects_bad_weights():
    with pytest.raises(ValueError):
        DistributedWeightedRandomSampler([], rank=0, world_size=1)
    with pytest.raises(ValueError):
        DistributedWeightedRandomSampler([0.0, 0.0], rank=0, world_size=1)
    with pytest.raises(ValueError):
        DistributedWeightedRandomSampler([1.0, -0.5], rank=0, world_size=1)


# ---------------------------------------------------------------------------
# Config plumbing: generate_config -> train.yml kcd_sample_* keys
# ---------------------------------------------------------------------------

def _gen_yml(tmp_path, extra):
    import yaml
    from kwcoco_detector_kit.trainers._registry import get_trainer
    trainer = get_trainer("deimv2")
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True, exist_ok=True)
    cfg_fpath = trainer.generate_config(
        train_kwcoco_fpath="/tmp/train.mscoco.json",
        vali_kwcoco_fpath="/tmp/vali.mscoco.json",
        workdir=workdir,
        variant="deimv2_dinov3_s",
        input_hw=(64, 64),
        train_policy="fixed",
        num_classes=1,
        batch_size=2, val_batch_size=2, num_epochs=2,
        lr=5e-4, backbone_lr=2.5e-5, use_amp=False,
        channels="r|g|b", scale_tier="M", num_gpus=1,
        data_format="kwcoco",
        extra={"category_names": ["widget"], **extra},
    )
    return yaml.safe_load(open(cfg_fpath).read())


def test_generate_config_threads_sampler_keys(tmp_path):
    yml = _gen_yml(tmp_path, {
        "balance_weights_fpath": "/x/balance_weights.json",
        "balance_epoch_length": 5000,
        "balance_seed": 7,
    })
    assert yml["kcd_sample_weights_fpath"] == "/x/balance_weights.json"
    assert yml["kcd_sample_epoch_length"] == 5000
    assert yml["kcd_sample_seed"] == 7


def test_generate_config_omits_sampler_keys_by_default(tmp_path):
    yml = _gen_yml(tmp_path, {})
    assert "kcd_sample_weights_fpath" not in yml
    assert "kcd_sample_epoch_length" not in yml


def test_generate_config_rejects_sampler_plus_wds(tmp_path):
    with pytest.raises(ValueError, match="WebDataset"):
        _gen_yml(tmp_path, {
            "balance_weights_fpath": "/x/w.json",
            "train_wds_shards_dpath": "/x/shards",
        })
