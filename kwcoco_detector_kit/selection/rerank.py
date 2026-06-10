"""Final multi-objective re-rank over the retained union.

Output is a ``candidate × axis`` **matrix**, not a single ranking. An
axis is either a primitive ``(fingerprint, metric)`` pair (scored by a
real eval pass on the full-validation binding) or a **derived metric**
(a named, versioned composition of primitive axes — e.g. ``combined_v1``
= harmonic mean — whose provenance embeds its input axes).

Selection policies over the matrix:

- ``argmax``    — winner = best on the declared primary axis (default).
- ``aggregate`` — winner = best on a declared derived axis.
- ``pareto``    — compute the non-dominated frontier; auto-pick by the
  closest-to-ideal tiebreak on min-max-normalized axes.

Whatever the policy, the full matrix **and** the Pareto frontier are
persisted, so a jack-of-all-trades pick stays available later with zero
recompute. Ties break toward the earlier epoch, deterministically.

The re-rank dataset is the **full validation split, never test**:
re-rank selects; test is reserved for reporting the selected model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Axis", "DerivedMetricDef", "DERIVED_METRICS",
    "build_matrix", "pareto_front", "select", "RerankResult",
]


@dataclass(frozen=True)
class Axis:
    """One column of the matrix."""
    fingerprint: str = ""        # primitive axes
    metric: str = ""
    derived: str = ""            # derived axes (name in DERIVED_METRICS)
    label: str = ""

    @property
    def axis_id(self) -> str:
        if self.derived:
            return f"derived:{self.derived}"
        return f"{self.fingerprint}:{self.metric}"


def _hmean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    if any(v <= 0 for v in vals):
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


@dataclass(frozen=True)
class DerivedMetricDef:
    """A named, versioned composition of primitive axes."""
    name: str
    version: int
    fn_name: str            # "hmean" | "mean"

    def compute(self, values: Sequence[float]) -> float:
        if self.fn_name == "hmean":
            return _hmean(values)
        if self.fn_name == "mean":
            vals = [float(v) for v in values]
            return sum(vals) / len(vals) if vals else 0.0
        raise KeyError(f"unknown derived fn {self.fn_name!r}")


DERIVED_METRICS: Dict[str, DerivedMetricDef] = {
    # punishes weakness on either axis — the generalist signal
    "combined_v1": DerivedMetricDef(name="combined_v1", version=1, fn_name="hmean"),
}


def build_matrix(
    scores: Dict[Tuple[int, str], Dict[str, float]],
    candidates: Sequence[int],
    axes: Sequence[Axis],
    derived_inputs: Optional[Dict[str, List[Axis]]] = None,
) -> Dict[int, Dict[str, Optional[float]]]:
    """``{epoch: {axis_id: value}}``; derived axes computed from primitives.

    ``derived_inputs`` maps a derived-metric name to the primitive axes it
    composes (declared explicitly in config — auditable, never inferred).
    """
    derived_inputs = derived_inputs or {}
    matrix: Dict[int, Dict[str, Optional[float]]] = {}
    for epoch in sorted(candidates):
        row: Dict[str, Optional[float]] = {}
        for axis in axes:
            if axis.derived:
                ddef = DERIVED_METRICS.get(axis.derived)
                inputs = derived_inputs.get(axis.derived, [])
                if ddef is None or not inputs:
                    row[axis.axis_id] = None
                    continue
                vals = [row.get(a.axis_id) for a in inputs]
                if any(v is None for v in vals):
                    vals = [
                        scores.get((epoch, a.fingerprint), {}).get(a.metric)
                        for a in inputs
                    ]
                if any(v is None for v in vals):
                    row[axis.axis_id] = None
                else:
                    row[axis.axis_id] = ddef.compute([float(v) for v in vals])
            else:
                row[axis.axis_id] = scores.get(
                    (epoch, axis.fingerprint), {}
                ).get(axis.metric)
        matrix[epoch] = row
    return matrix


def _complete_rows(matrix, axis_ids):
    return {
        e: row for e, row in matrix.items()
        if all(row.get(a) is not None for a in axis_ids)
    }


def pareto_front(
    matrix: Dict[int, Dict[str, Optional[float]]],
    axes: Sequence[Axis],
) -> List[int]:
    """Non-dominated epochs (maximize all axes); rows with gaps excluded."""
    axis_ids = [a.axis_id for a in axes]
    rows = _complete_rows(matrix, axis_ids)
    front = []
    for e, row in rows.items():
        dominated = False
        for e2, row2 in rows.items():
            if e2 == e:
                continue
            ge_all = all(row2[a] >= row[a] for a in axis_ids)
            gt_any = any(row2[a] > row[a] for a in axis_ids)
            if ge_all and gt_any:
                dominated = True
                break
        if not dominated:
            front.append(e)
    return sorted(front)


def _closest_to_ideal(matrix, axes, candidates) -> Optional[int]:
    """Tiebreak: min-max normalize each axis, pick min euclidean distance
    to the ideal point (1, 1, ...); ties -> earlier epoch."""
    axis_ids = [a.axis_id for a in axes]
    rows = {e: matrix[e] for e in candidates}
    lo = {a: min(rows[e][a] for e in rows) for a in axis_ids}
    hi = {a: max(rows[e][a] for e in rows) for a in axis_ids}
    best, best_d = None, None
    for e in sorted(rows):
        d = 0.0
        for a in axis_ids:
            span = hi[a] - lo[a]
            norm = 1.0 if span <= 0 else (rows[e][a] - lo[a]) / span
            d += (1.0 - norm) ** 2
        d = math.sqrt(d)
        if best_d is None or d < best_d:
            best, best_d = e, d
    return best


@dataclass
class RerankResult:
    winner_epoch: Optional[int]
    policy: str
    primary_axis_id: str
    matrix: Dict[int, Dict[str, Optional[float]]]
    frontier: List[int]
    notes: List[str] = field(default_factory=list)

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "winner_epoch": self.winner_epoch,
            "policy": self.policy,
            "primary_axis_id": self.primary_axis_id,
            "matrix": {str(e): row for e, row in sorted(self.matrix.items())},
            "frontier": self.frontier,
            "notes": self.notes,
        }


def select(
    matrix: Dict[int, Dict[str, Optional[float]]],
    *,
    axes: Sequence[Axis],
    policy: str,
    primary: Axis,
) -> RerankResult:
    notes: List[str] = []
    frontier = pareto_front(matrix, axes) if len(axes) > 1 else sorted(matrix)

    def _argmax(axis: Axis) -> Optional[int]:
        scored = [
            (row[axis.axis_id], -e)
            for e, row in matrix.items() if row.get(axis.axis_id) is not None
        ]
        if not scored:
            return None
        best = max(scored)
        return -best[1]

    if policy == "argmax":
        winner = _argmax(primary)
    elif policy == "aggregate":
        winner = _argmax(primary)
        if not primary.derived:
            notes.append("aggregate policy with a non-derived primary axis")
    elif policy == "pareto":
        if frontier:
            winner = _closest_to_ideal(matrix, axes, frontier)
        else:
            winner = _argmax(primary)
            notes.append("empty/complete-row-less frontier; fell back to argmax")
    else:
        raise KeyError(f"unknown rerank policy {policy!r}")

    if winner is None:
        notes.append("no candidate had a primary-axis score")

    return RerankResult(
        winner_epoch=winner,
        policy=policy,
        primary_axis_id=primary.axis_id,
        matrix=matrix,
        frontier=frontier,
        notes=notes,
    )
