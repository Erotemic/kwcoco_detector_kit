"""Leaderboards, union retention, and GC decisions as a pure fold.

``fold(events, plan)`` reduces the run journal to a :class:`FoldState`:
which epochs are staged, scored, on which boards, retained, anchored —
and which GC actions are now safe. The fold is pure and deterministic, so
selection state is always recomputable from the journal (crash-safe
resume for free).

Key invariants (spec: docs/planning/checkpoint_selection.md):

- **Comparability**: a board is keyed by fingerprint; values from
  different fingerprints never meet.
- **Fail-retentive GC**: an epoch is deletable only when scored under
  *every* in-loop fingerprint (or it is below ``min_epoch``, where boards
  are inert and the score could never matter) and it sits on no board.
- **Monotone displacement**: scores are immutable, so an epoch that falls
  off a board (or out of the anchor top-M) can never climb back —
  deletion and optimizer-stripping are safe the moment they trigger.
- **Anchors**: the top-M of the primary board keep the full payload
  (+optimizer) as collapse-recovery resume points; everything else
  retained is slim (model+EMA).

Ties on a board sort by *earlier epoch first* (conservative, pre-collapse
bias), deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

__all__ = ["BucketSpec", "BoardEntry", "FoldState", "fold", "bucket_id"]


@dataclass(frozen=True)
class BucketSpec:
    """One leaderboard: top-``k`` epochs by ``metric`` within ``fingerprint``."""
    fingerprint: str
    metric: str
    k: int
    label: str = ""          # human-readable, e.g. "true_tiled.probe.AP@0.5"


def bucket_id(spec: BucketSpec) -> str:
    return f"{spec.fingerprint}:{spec.metric}"


@dataclass
class BoardEntry:
    epoch: int
    value: float


@dataclass
class FoldState:
    staged: Dict[int, str] = field(default_factory=dict)       # epoch -> ckpt relpath
    deleted: Set[int] = field(default_factory=set)
    stripped: Set[int] = field(default_factory=set)
    train_complete: bool = False
    # (epoch, fingerprint) -> measures
    scores: Dict[Tuple[int, str], Dict[str, float]] = field(default_factory=dict)
    ckpt_hashes: Dict[int, str] = field(default_factory=dict)
    boards: Dict[str, List[BoardEntry]] = field(default_factory=dict)
    retained: Set[int] = field(default_factory=set)
    anchors: Set[int] = field(default_factory=set)
    # actionable, in deterministic order
    pending: List[Tuple[int, str]] = field(default_factory=list)
    deletable: List[int] = field(default_factory=list)
    strippable: List[int] = field(default_factory=list)
    rerank_done: bool = False


def fold(
    events: Sequence[Dict[str, Any]],
    *,
    buckets: Sequence[BucketSpec],
    inloop_fingerprints: Sequence[str],
    anchor_bucket: Optional[BucketSpec] = None,
    anchor_top_m: int = 2,
    min_epoch: int = 0,
) -> FoldState:
    state = FoldState()

    for ev in events:
        kind = ev.get("event")
        if kind == "epoch_staged":
            state.staged[int(ev["epoch"])] = str(ev["ckpt"])
        elif kind == "train_complete":
            state.train_complete = True
        elif kind == "score_record":
            key = (int(ev["epoch"]), str(ev["fingerprint"]))
            # scores are immutable: first write wins
            state.scores.setdefault(key, dict(ev.get("measures") or {}))
            if ev.get("ckpt_hash"):
                state.ckpt_hashes.setdefault(int(ev["epoch"]), str(ev["ckpt_hash"]))
        elif kind == "gc":
            epoch = int(ev["epoch"])
            if ev.get("action") == "delete":
                state.deleted.add(epoch)
            elif ev.get("action") == "strip":
                state.stripped.add(epoch)
        elif kind == "rerank_result":
            state.rerank_done = True

    # ---- boards (deleted epochs excluded defensively; the GC invariant
    # means they were never board members when deleted) ----
    for spec in buckets:
        entries: List[BoardEntry] = []
        for (epoch, fp), measures in state.scores.items():
            if fp != spec.fingerprint or epoch in state.deleted:
                continue
            if epoch < min_epoch:
                continue
            value = measures.get(spec.metric)
            if value is None:
                continue
            entries.append(BoardEntry(epoch=epoch, value=float(value)))
        entries.sort(key=lambda e: (-e.value, e.epoch))
        state.boards[bucket_id(spec)] = entries[: max(0, int(spec.k))]

    state.retained = {
        e.epoch for board in state.boards.values() for e in board
    }

    if anchor_bucket is not None:
        primary = state.boards.get(bucket_id(anchor_bucket), [])
        state.anchors = {e.epoch for e in primary[: max(0, int(anchor_top_m))]}

    # ---- actionable work, deterministic order ----
    inloop = list(inloop_fingerprints)
    live = sorted(set(state.staged) - state.deleted)
    for epoch in live:
        if epoch < min_epoch:
            continue   # boards inert below min_epoch: never score
        for fp in inloop:
            if (epoch, fp) not in state.scores:
                state.pending.append((epoch, fp))

    for epoch in live:
        if epoch in state.retained:
            continue
        below_floor = epoch < min_epoch
        fully_scored = all((epoch, fp) in state.scores for fp in inloop)
        if below_floor or fully_scored:
            state.deletable.append(epoch)

    for epoch in sorted(state.retained):
        if epoch in state.anchors or epoch in state.stripped:
            continue
        if epoch in state.deleted or epoch not in state.staged:
            continue
        state.strippable.append(epoch)

    return state
