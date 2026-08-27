"""One global draw without replacement, partitioned across ranks.

Both properties matter and neither is visible from a single rank:

  * no index twice within an epoch -- with replacement, drawing 96,000 of
    495,514 puts ~9% of draws on tiles already seen that epoch, and
    reweighting concentrates the mass so it gets worse;
  * no index on two ranks at once -- independent per-rank streams let two GPUs
    spend the same synchronised optimizer step on the same tile.

A sampler that satisfied only the first would still waste a quarter of a
4-GPU step, so the cross-rank disjointness is tested explicitly rather than
inferred.
"""
import pytest

torch = pytest.importorskip("torch")

from kwcoco_detector_kit.data.balanced_sampler import (  # noqa: E402
    DistributedWeightedNoReplacementSampler, sampler_from_weights_file,
    write_balance_weights)


def _ranks(weights, *, world_size, epoch=0, total=None, seed=0):
    out = []
    for r in range(world_size):
        s = DistributedWeightedNoReplacementSampler(
            weights, num_samples_total=total, seed=seed,
            rank=r, world_size=world_size)
        s.set_epoch(epoch)
        out.append(list(s))
    return out


def test_no_index_repeats_within_an_epoch():
    got = list(DistributedWeightedNoReplacementSampler(
        [1.0] * 100, num_samples_total=40, rank=0, world_size=1))
    assert len(got) == 40
    assert len(set(got)) == 40


def test_ranks_never_share_an_index():
    parts = _ranks([1.0] * 400, world_size=4, total=200)
    flat = [i for p in parts for i in p]
    assert len(set(flat)) == len(flat), "two ranks drew the same tile"
    for p in parts:
        assert len(p) == 50


def test_the_union_across_ranks_is_the_global_draw():
    """Partitioning must not silently drop part of the draw."""
    parts = _ranks([1.0] * 400, world_size=4, total=200)
    assert sum(len(p) for p in parts) == 200


def test_weights_bias_the_draw():
    w = [10.0] * 10 + [0.01] * 990
    drawn = set(DistributedWeightedNoReplacementSampler(
        w, num_samples_total=50, rank=0, world_size=1))
    assert len(drawn & set(range(10))) >= 8


def test_zero_weight_indices_are_never_drawn():
    """empty_weight=0 must actually exclude, not merely deprioritise."""
    w = [0.0] * 50 + [1.0] * 50
    drawn = set(DistributedWeightedNoReplacementSampler(
        w, num_samples_total=50, rank=0, world_size=1))
    assert drawn == set(range(50, 100))


def test_epoch_changes_the_draw_and_the_seed_reproduces_it():
    a = _ranks([1.0] * 200, world_size=1, epoch=0, total=50)[0]
    b = _ranks([1.0] * 200, world_size=1, epoch=1, total=50)[0]
    again = _ranks([1.0] * 200, world_size=1, epoch=0, total=50)[0]
    assert a != b, "every epoch must be a fresh draw"
    assert a == again, "a resumed run must replay the same epoch"


def test_all_ranks_agree_on_the_global_order():
    """The partition is only disjoint if every rank draws the same order."""
    orders = []
    for r in range(4):
        s = DistributedWeightedNoReplacementSampler(
            [1.0] * 300, num_samples_total=100, rank=r, world_size=4)
        orders.append(s._global_order().tolist())
    assert all(o == orders[0] for o in orders)


def test_epoch_longer_than_the_corpus_is_clamped_not_padded():
    """Padding with repeats would make this the with-replacement sampler."""
    s = DistributedWeightedNoReplacementSampler(
        [1.0] * 10, num_samples_total=50, rank=0, world_size=1)
    got = list(s)
    assert len(got) == 10 and len(set(got)) == 10


def test_epoch_longer_than_the_positive_support_is_clamped():
    s = DistributedWeightedNoReplacementSampler(
        [1.0] * 5 + [0.0] * 95, num_samples_total=40, rank=0, world_size=1)
    assert sorted(s) == list(range(5))


@pytest.mark.parametrize("bad", [[], [-1.0, 1.0], [0.0, 0.0]])
def test_invalid_weights_are_rejected(bad):
    with pytest.raises(ValueError):
        DistributedWeightedNoReplacementSampler(bad)


def test_factory_selects_the_mode(tmp_path):
    from kwcoco_detector_kit.data.balanced_sampler import (
        DistributedWeightedRandomSampler)
    fpath = write_balance_weights(tmp_path / "w.json", [0.25, 0.25, 0.5])
    with_repl = sampler_from_weights_file(fpath, dataset_len=3, replacement=True)
    without = sampler_from_weights_file(fpath, dataset_len=3, replacement=False)
    assert isinstance(with_repl, DistributedWeightedRandomSampler)
    assert isinstance(without, DistributedWeightedNoReplacementSampler)


def test_factory_still_guards_the_count_mismatch(tmp_path):
    fpath = write_balance_weights(tmp_path / "w.json", [0.5, 0.5])
    with pytest.raises(ValueError):
        sampler_from_weights_file(fpath, dataset_len=3, replacement=False)
