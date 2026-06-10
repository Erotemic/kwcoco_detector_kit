"""The journal fold: boards, retention, anchors, fail-retentive GC."""
from __future__ import annotations

from kwcoco_detector_kit.selection.boards import BucketSpec, bucket_id, fold

FP_PROBE = "fp_probe_aaa"
FP_WHOLE = "fp_whole_bbb"
INLOOP = [FP_PROBE, FP_WHOLE]
BUCKETS = [
    BucketSpec(fingerprint=FP_PROBE, metric="AP@0.5", k=2, label="probe.ap"),
    BucketSpec(fingerprint=FP_WHOLE, metric="AP@0.5", k=1, label="whole.ap"),
]


def _staged(epoch):
    return {"event": "epoch_staged", "epoch": epoch,
            "ckpt": f"staging/epoch_{epoch:04d}.pth"}


def _score(epoch, fp, ap):
    return {"event": "score_record", "epoch": epoch, "fingerprint": fp,
            "measures": {"AP@0.5": ap}}


def test_boards_rank_and_truncate_with_tie_to_earlier_epoch():
    events = [_staged(e) for e in range(4)]
    events += [_score(0, FP_PROBE, 0.5), _score(1, FP_PROBE, 0.7),
               _score(2, FP_PROBE, 0.7), _score(3, FP_PROBE, 0.6)]
    state = fold(events, buckets=BUCKETS, inloop_fingerprints=[FP_PROBE])
    board = state.boards[bucket_id(BUCKETS[0])]
    # tie at 0.7: earlier epoch (1) first; k=2 truncates
    assert [(e.epoch, e.value) for e in board] == [(1, 0.7), (2, 0.7)]


def test_comparability_scores_under_other_fingerprints_never_enter():
    events = [_staged(0), _score(0, FP_WHOLE, 0.99)]
    state = fold(events, buckets=[BUCKETS[0]], inloop_fingerprints=INLOOP)
    assert state.boards[bucket_id(BUCKETS[0])] == []


def test_pending_excludes_scored_and_below_min_epoch():
    events = [_staged(e) for e in range(4)] + [_score(2, FP_PROBE, 0.5)]
    state = fold(events, buckets=BUCKETS, inloop_fingerprints=INLOOP,
                 min_epoch=2)
    # epochs 0,1 below floor -> never scored; epoch 2 probe done
    assert (2, FP_PROBE) not in state.pending
    assert (2, FP_WHOLE) in state.pending
    assert (3, FP_PROBE) in state.pending
    assert all(e >= 2 for e, _ in state.pending)


def test_gc_is_fail_retentive():
    """Unscored epochs are never deletable, even off every board."""
    events = [_staged(0), _staged(1),
              _score(0, FP_PROBE, 0.1)]   # whole score missing for epoch 0
    state = fold(events, buckets=BUCKETS, inloop_fingerprints=INLOOP)
    assert state.deletable == []          # nothing fully scored yet
    events += [_score(0, FP_WHOLE, 0.1), _score(1, FP_PROBE, 0.9),
               _score(1, FP_WHOLE, 0.9)]
    state = fold(events, buckets=BUCKETS, inloop_fingerprints=INLOOP)
    # both epochs are on boards (k=2 probe board holds both) -> no deletes
    assert 1 in state.retained
    # epoch 0 is on the probe board too (k=2, only two epochs)
    assert state.deletable == []


def test_displaced_epoch_becomes_deletable():
    events = [_staged(e) for e in range(4)]
    for e, ap in [(0, 0.1), (1, 0.5), (2, 0.6), (3, 0.7)]:
        events += [_score(e, FP_PROBE, ap), _score(e, FP_WHOLE, ap)]
    state = fold(events, buckets=BUCKETS, inloop_fingerprints=INLOOP)
    # probe k=2 -> {3,2}; whole k=1 -> {3}; union {2,3}
    assert state.retained == {2, 3}
    assert state.deletable == [0, 1]


def test_below_floor_epochs_deletable_without_scores():
    events = [_staged(0), _staged(5)]
    state = fold(events, buckets=BUCKETS, inloop_fingerprints=INLOOP,
                 min_epoch=3)
    assert 0 in state.deletable
    assert 5 not in state.deletable


def test_anchors_are_top_m_of_primary_and_strip_targets_the_rest():
    events = [_staged(e) for e in range(4)]
    for e, ap in [(0, 0.4), (1, 0.9), (2, 0.8), (3, 0.7)]:
        events += [_score(e, FP_PROBE, ap), _score(e, FP_WHOLE, 1.0 - ap)]
    buckets = [
        BucketSpec(fingerprint=FP_PROBE, metric="AP@0.5", k=3, label="p"),
        BucketSpec(fingerprint=FP_WHOLE, metric="AP@0.5", k=1, label="w"),
    ]
    state = fold(events, buckets=buckets, inloop_fingerprints=INLOOP,
                 anchor_bucket=buckets[0], anchor_top_m=2)
    # probe board: 1,2,3 ; whole board: 0 (1-ap max) ; union {0,1,2,3}
    assert state.anchors == {1, 2}
    assert set(state.strippable) == {0, 3}


def test_gc_events_persist_state_and_are_not_reissued():
    events = [_staged(e) for e in range(3)]
    for e, ap in [(0, 0.1), (1, 0.8), (2, 0.9)]:
        events += [_score(e, FP_PROBE, ap), _score(e, FP_WHOLE, ap)]
    buckets = [BucketSpec(fingerprint=FP_PROBE, metric="AP@0.5", k=2, label="p")]
    state = fold(events, buckets=buckets, inloop_fingerprints=INLOOP,
                 anchor_bucket=buckets[0], anchor_top_m=1)
    assert state.deletable == [0] and set(state.strippable) == {1}
    events += [{"event": "gc", "action": "delete", "epoch": 0},
               {"event": "gc", "action": "strip", "epoch": 1}]
    state = fold(events, buckets=buckets, inloop_fingerprints=INLOOP,
                 anchor_bucket=buckets[0], anchor_top_m=1)
    assert state.deletable == [] and state.strippable == []
    assert 0 in state.deleted and 1 in state.stripped


def test_train_complete_and_rerank_done_flags():
    events = [_staged(0), {"event": "train_complete"}]
    state = fold(events, buckets=[], inloop_fingerprints=[])
    assert state.train_complete and not state.rerank_done
    events.append({"event": "rerank_result", "winner_epoch": 0})
    state = fold(events, buckets=[], inloop_fingerprints=[])
    assert state.rerank_done
