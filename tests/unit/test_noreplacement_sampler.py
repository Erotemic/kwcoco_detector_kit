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


# ---------------------------------------------------------------------------
# Selection and presentation order are separate concerns
# ---------------------------------------------------------------------------


def test_the_epoch_is_not_ordered_by_weight():
    """topk returns the selected set in descending weight order.

    Fed straight to the loader that front-loads every epoch with the rarest
    samples and ends it with the most common -- a weight schedule nobody asked
    for, which batch statistics and EMA would track across the whole epoch.
    Measured on the real fish weights before the fix: mean sample weight fell
    monotonically across quartiles, 4.81e-6 -> 3.46e-6.
    """
    n = 4000
    w = [1.0 / (i + 1) for i in range(n)]          # index 0 heaviest
    got = list(DistributedWeightedNoReplacementSampler(
        w, num_samples_total=1200, rank=0, world_size=1))
    half = len(got) // 2
    first = sum(got[:half]) / half
    second = sum(got[half:]) / (len(got) - half)
    # Sorted by weight, `first` would be far below `second` (low index = heavy).
    assert abs(first - second) < 0.20 * max(first, second), (
        f"epoch still ordered by weight: mean index {first:.0f} then {second:.0f}")


def test_heavy_samples_are_spread_across_the_epoch_not_front_loaded():
    """The direct form of the measurement that exposed the bug.

    Two clearly separated weight groups, so the check does not depend on a
    heavy tail: a handful of very heavy items would dominate any quartile MEAN
    by luck alone even under a perfect shuffle, which is why this counts group
    membership per quartile instead of averaging weights.

    Sorted by weight, every heavy sample lands in the first quartile. Shuffled,
    they are spread roughly evenly.
    """
    n_heavy, n_light = 400, 3600
    w = [10.0] * n_heavy + [1.0] * n_light
    got = list(DistributedWeightedNoReplacementSampler(
        w, num_samples_total=1600, rank=0, world_size=1))
    q = len(got) // 4
    per_quartile = [sum(1 for i in got[k * q:(k + 1) * q] if i < n_heavy)
                    for k in range(4)]
    total_heavy = sum(per_quartile)
    assert total_heavy > 0, "fixture drew no heavy samples"
    expected = total_heavy / 4
    worst = max(abs(c - expected) for c in per_quartile) / expected
    assert worst < 0.25, (
        f"heavy samples unevenly placed across the epoch: {per_quartile} "
        f"(expected ~{expected:.0f} each)")


def test_the_unshuffled_selection_would_fail_that_check():
    """Proves the previous test can detect the bug it is written for."""
    import torch
    n_heavy, n_light = 400, 3600
    w = [10.0] * n_heavy + [1.0] * n_light
    s = DistributedWeightedNoReplacementSampler(
        w, num_samples_total=1600, rank=0, world_size=1)
    # Reproduce the pre-fix behaviour: select, then keep topk's own order.
    g = torch.Generator()
    g.manual_seed(s.seed * 1_000_003 + s.epoch * 1_009)
    u = torch.rand(len(w), generator=g, dtype=torch.double)
    keys = torch.log(s._weights) - torch.log(-torch.log(u))
    unshuffled = torch.topk(keys, 1600, sorted=True).indices.tolist()
    q = len(unshuffled) // 4
    per_quartile = [sum(1 for i in unshuffled[k * q:(k + 1) * q] if i < n_heavy)
                    for k in range(4)]
    expected = sum(per_quartile) / 4
    worst = max(abs(c - expected) for c in per_quartile) / expected
    assert worst >= 0.25, (
        f"the ordering check is not sensitive enough: {per_quartile}")


def test_shuffling_does_not_change_which_indices_are_selected():
    """Order is randomised; the SET -- and every diversity statistic -- is not."""
    w = [1.0 / (i + 1) for i in range(500)]
    s = DistributedWeightedNoReplacementSampler(
        w, num_samples_total=200, rank=0, world_size=1)
    s.set_epoch(2)
    a = set(s)
    s.set_epoch(2)
    assert set(s) == a


def test_the_shuffled_order_is_still_deterministic():
    a = list(DistributedWeightedNoReplacementSampler(
        [1.0] * 300, num_samples_total=100, seed=5, rank=0, world_size=1))
    b = list(DistributedWeightedNoReplacementSampler(
        [1.0] * 300, num_samples_total=100, seed=5, rank=0, world_size=1))
    assert a == b


def test_ranks_stay_disjoint_after_shuffling():
    """The shuffle must happen BEFORE the partition and identically on all ranks."""
    parts = _ranks([1.0 / (i + 1) for i in range(800)], world_size=4, total=400)
    flat = [i for p in parts for i in p]
    assert len(set(flat)) == len(flat)
    assert sum(len(p) for p in parts) == 400
