"""Contract for sequence/track-aware sampling weights.

The corpus this exists for: 439 sequences, 495,514 tiles, 448x length
imbalance, effective sequence count 81. Uniform sampling draws from something
that behaves like 81 sequences, so the headline property under test is that
weighting RAISES the effective count without deleting anything.

Two silent failure modes are pinned hardest, because neither crashes:

  * a weighting that flattens sequences by starving tracks (measured: at
    seq_alpha=1.0 effective tracks FALL from 4,473 to 1,729); and
  * a weighting that flattens by oversampling one tile dozens of times per
    epoch, which is memorisation wearing a balance costume.
"""
import math

import pytest

from kwcoco_detector_kit.data.sequence_balance import (
    cap_oversample, combine_weights, compute_sequence_weights,
    flatten_weights, oversample_profile, summarize_groups)


def _effective(weights, groups):
    """Inverse Simpson over group MASS -- the metric the design was chosen on."""
    mass = {}
    for w, g in zip(weights, groups):
        mass[g] = mass.get(g, 0.0) + w
    total = sum(mass.values())
    return 1.0 / sum((m / total) ** 2 for m in mass.values())


def _skewed(sizes):
    """A corpus shaped like the real one: a few long sequences, many short."""
    sequences, tracks = [], []
    for si, n in enumerate(sizes):
        for j in range(n):
            sequences.append(f"seq{si}")
            tracks.append([f"seq{si}_t{j % max(1, n // 10)}"])
    return {"sequences": sequences, "tracks": tracks,
            "source_frames": list(range(len(sequences))),
            "roles": ["positive"] * len(sequences)}


# ---------------------------------------------------------------------------
# summarize_groups
# ---------------------------------------------------------------------------


def test_uniform_groups_are_reported_as_balanced():
    s = summarize_groups(["a", "a", "b", "b", "c", "c"])
    assert s["n_groups"] == 3
    assert s["gini"] == pytest.approx(0.0, abs=1e-9)
    assert s["effective_count"] == pytest.approx(3.0)
    assert s["imbalance_ratio"] == 1.0


def test_effective_count_collapses_under_domination():
    """97 items in one group, 3 singletons: nominally 4 groups, effectively ~1."""
    s = summarize_groups(["big"] * 97 + ["a", "b", "c"])
    assert s["n_groups"] == 4
    assert s["effective_count"] < 1.1
    assert s["gini"] > 0.7


def test_effective_count_is_between_one_and_n_groups():
    for groups in (["a"] * 10, ["a"] * 9 + ["b"], list(range(10))):
        s = summarize_groups(groups)
        assert 1.0 - 1e-9 <= s["effective_count"] <= s["n_groups"] + 1e-9


def test_empty_input_is_an_error_not_a_zero():
    with pytest.raises(ValueError):
        summarize_groups([])


# ---------------------------------------------------------------------------
# flatten_weights
# ---------------------------------------------------------------------------


def test_alpha_zero_changes_nothing():
    """The default must be a genuine no-op, so enabling balance is the only
    thing that alters sampling."""
    groups = ["a"] * 5 + ["b"]
    assert flatten_weights(groups, 0.0) == [1.0] * 6


def test_alpha_one_equalises_group_mass():
    groups = ["a"] * 10 + ["b"] * 2
    w = flatten_weights(groups, 1.0)
    assert sum(w[:10]) == pytest.approx(sum(w[10:]))


def test_alpha_half_damps_without_equalising():
    """sqrt damping: a 100x longer group keeps 10x the mass, not 1x and not 100x."""
    groups = ["a"] * 100 + ["b"]
    w = flatten_weights(groups, 0.5)
    assert sum(w[:100]) / sum(w[100:]) == pytest.approx(10.0)


def test_negative_alpha_is_rejected():
    with pytest.raises(ValueError):
        flatten_weights(["a"], -0.5)


# ---------------------------------------------------------------------------
# cap_oversample -- the guard on the cure
# ---------------------------------------------------------------------------


def test_cap_bounds_the_largest_weight():
    w = cap_oversample([0.9] + [0.1 / 99] * 99, 4.0)
    assert max(w) <= 4.0 / 100 + 1e-9
    assert sum(w) == pytest.approx(1.0)


def test_cap_keeps_every_index_reachable():
    """Capping must never zero anything -- that would be discarding data."""
    w = cap_oversample([0.99] + [0.01 / 49] * 49, 2.0)
    assert all(x > 0 for x in w)


def test_cap_terminates_on_a_pathological_input():
    """Renormalisation can push capped weights back over; the loop is bounded."""
    w = cap_oversample([1.0] + [0.0] * 9, 1.0)
    assert sum(w) == pytest.approx(1.0)
    assert all(math.isfinite(x) for x in w)


@pytest.mark.parametrize("bad", [0, -1])
def test_nonpositive_cap_is_rejected(bad):
    with pytest.raises(ValueError):
        cap_oversample([0.5, 0.5], bad)


# ---------------------------------------------------------------------------
# combine_weights
# ---------------------------------------------------------------------------


def test_combine_multiplies_and_normalises():
    w = combine_weights([1.0, 2.0], [1.0, 3.0])
    assert w == pytest.approx([1 / 7, 6 / 7])


def test_length_disagreement_is_loud():
    """Silent truncation here would misalign every downstream weight."""
    with pytest.raises(ValueError):
        combine_weights([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# compute_sequence_weights -- the properties the design was chosen for
# ---------------------------------------------------------------------------


def test_balancing_raises_effective_sequence_count():
    idx = _skewed([1000, 500, 100, 50, 30])
    base = _effective([1.0] * len(idx["sequences"]), idx["sequences"])
    got = _effective(
        compute_sequence_weights(idx, seq_alpha=0.5, track_alpha=0.5,
                                 max_oversample=8),
        idx["sequences"])
    assert got > base * 1.3, f"{base:.2f} -> {got:.2f}"


def test_half_flattening_beats_full_flattening_on_track_diversity():
    """The measured reason alpha is 0.5 and not 1.0.

    Full sequence flattening pours mass into short sequences, and short
    sequences are short because they contain few tracks -- so it buys sequence
    diversity with track diversity. Half flattening improved both on the real
    corpus (81->238 sequences, 1454->4473 tracks).
    """
    idx = _skewed([2000, 1000, 300, 80, 40, 30])
    flat_tracks = [(s, t) for s, ts in zip(idx["sequences"], idx["tracks"])
                   for t in ts]

    def track_eff(alpha):
        w = compute_sequence_weights(idx, seq_alpha=alpha, track_alpha=0.5,
                                     max_oversample=8)
        return _effective(w, flat_tracks)

    assert track_eff(0.5) > track_eff(1.0)


def test_nothing_is_ever_discarded():
    """Rebalancing, not pruning: every tile keeps a strictly positive weight."""
    idx = _skewed([5000, 10])
    w = compute_sequence_weights(idx, seq_alpha=1.0, track_alpha=1.0,
                                 max_oversample=8)
    assert all(x > 0 for x in w)
    assert sum(w) == pytest.approx(1.0)


def test_alpha_zero_everywhere_is_uniform():
    """With every knob off, the sampler must reproduce today's behaviour."""
    idx = _skewed([100, 10, 5])
    w = compute_sequence_weights(idx, seq_alpha=0.0, track_alpha=0.0,
                                 frame_alpha=0.0, max_oversample=None)
    assert w == pytest.approx([1.0 / len(w)] * len(w))


def test_empty_tiles_are_anchored_to_the_mean_annotated_tile():
    """empty_weight must keep one meaning as track_alpha varies."""
    idx = {"sequences": ["s"] * 4, "tracks": [["t1"], ["t1"], ["t2"], []],
           "source_frames": [0, 1, 2, 3], "roles": ["x"] * 4}
    w = compute_sequence_weights(idx, seq_alpha=0, track_alpha=0.5,
                                 empty_weight=1.0, max_oversample=None)
    annotated_mean = sum(w[:3]) / 3
    assert w[3] == pytest.approx(annotated_mean, rel=1e-9)


def test_empty_weight_zero_suppresses_negatives_without_erroring():
    idx = {"sequences": ["s"] * 3, "tracks": [["t"], ["t"], []],
           "source_frames": [0, 1, 2], "roles": ["x"] * 3}
    w = compute_sequence_weights(idx, seq_alpha=0, track_alpha=0,
                                 empty_weight=0.0, max_oversample=None)
    assert w[2] == pytest.approx(0.0)


def test_untracked_annotations_do_not_merge_into_one_pseudo_track():
    """Annotations lacking track_id must not all be lumped together.

    Lumping would create a single enormous pseudo-track and suppress every
    tile it touches -- the opposite of the intent.
    """
    from kwcoco_detector_kit.data.sequence_balance import load_tile_index
    import json
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    tiles = {
        "images": [{"id": i, "tile_source_gid": 1} for i in (1, 2, 3)],
        "annotations": [{"id": 10 + i, "image_id": i} for i in (1, 2, 3)],
    }
    (d / "t.json").write_text(json.dumps(tiles))
    idx = load_tile_index(d / "t.json")
    seen = {t for ts in idx["tracks"] for t in ts}
    assert len(seen) == 3, f"untracked annotations were merged: {seen}"


def test_oversample_profile_flags_within_epoch_repetition():
    n = 1000
    w = [0.5] + [0.5 / (n - 1)] * (n - 1)
    p = oversample_profile(w, epoch_length=1000)
    assert p["expected_draws_max"] == pytest.approx(500.0)
    assert p["expected_draws_mean"] == pytest.approx(1.0)


def test_sidecar_schema_matches_the_existing_reader():
    """The weights must load through balanced_sampler's own reader.

    This module deliberately does not add a second sidecar format; the whole
    point is to reuse sampler_from_weights_file and the patched solver.
    """
    import json
    import tempfile
    from pathlib import Path

    from kwcoco_detector_kit.data.balanced_sampler import load_balance_weights

    idx = _skewed([10, 5])
    w = compute_sequence_weights(idx, seq_alpha=0.5, track_alpha=0.5)
    fpath = Path(tempfile.mkdtemp()) / "w.json"
    fpath.write_text(json.dumps({"weights": w, "meta": {}}))
    assert load_balance_weights(fpath) == pytest.approx(w)
