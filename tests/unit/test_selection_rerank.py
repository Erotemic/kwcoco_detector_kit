"""Matrix building, derived metrics, pareto frontier, policies."""
from __future__ import annotations

import pytest

from kwcoco_detector_kit.selection.rerank import (
    Axis,
    build_matrix,
    pareto_front,
    select,
)

FP_T = "fp_tiled"
FP_W = "fp_whole"
AX_T = Axis(fingerprint=FP_T, metric="AP@0.5", label="tiled")
AX_W = Axis(fingerprint=FP_W, metric="AP@0.5", label="whole")
AX_C = Axis(derived="combined_v1", label="combined")


def _scores():
    # epoch 5: tiled specialist; epoch 3: whole specialist; epoch 4: generalist
    return {
        (3, FP_T): {"AP@0.5": 0.60}, (3, FP_W): {"AP@0.5": 0.80},
        (4, FP_T): {"AP@0.5": 0.75}, (4, FP_W): {"AP@0.5": 0.75},
        (5, FP_T): {"AP@0.5": 0.85}, (5, FP_W): {"AP@0.5": 0.55},
    }


def test_matrix_with_derived_combined_v1():
    matrix = build_matrix(
        _scores(), [3, 4, 5], [AX_T, AX_W, AX_C],
        derived_inputs={"combined_v1": [AX_T, AX_W]},
    )
    # harmonic mean punishes one-axis weakness: generalist epoch 4 wins it
    assert matrix[4][AX_C.axis_id] == pytest.approx(0.75)
    assert matrix[4][AX_C.axis_id] > matrix[3][AX_C.axis_id]
    assert matrix[4][AX_C.axis_id] > matrix[5][AX_C.axis_id]


def test_pareto_front_keeps_all_non_dominated():
    matrix = build_matrix(_scores(), [3, 4, 5], [AX_T, AX_W])
    assert pareto_front(matrix, [AX_T, AX_W]) == [3, 4, 5]
    # add a dominated epoch
    scores = _scores()
    scores[(2, FP_T)] = {"AP@0.5": 0.50}
    scores[(2, FP_W)] = {"AP@0.5": 0.50}
    matrix = build_matrix(scores, [2, 3, 4, 5], [AX_T, AX_W])
    assert pareto_front(matrix, [AX_T, AX_W]) == [3, 4, 5]


def test_argmax_policy_and_tie_to_earlier_epoch():
    scores = _scores()
    scores[(6, FP_T)] = {"AP@0.5": 0.85}     # tie with epoch 5 on tiled
    scores[(6, FP_W)] = {"AP@0.5": 0.55}
    matrix = build_matrix(scores, [3, 4, 5, 6], [AX_T, AX_W])
    result = select(matrix, axes=[AX_T, AX_W], policy="argmax", primary=AX_T)
    assert result.winner_epoch == 5            # earlier epoch wins the tie
    assert result.frontier                     # frontier persisted regardless


def test_pareto_policy_tiebreaks_closest_to_ideal():
    matrix = build_matrix(_scores(), [3, 4, 5], [AX_T, AX_W])
    result = select(matrix, axes=[AX_T, AX_W], policy="pareto", primary=AX_T)
    assert result.winner_epoch == 4            # the generalist
    assert result.frontier == [3, 4, 5]


def test_aggregate_policy_on_derived_primary():
    matrix = build_matrix(
        _scores(), [3, 4, 5], [AX_T, AX_W, AX_C],
        derived_inputs={"combined_v1": [AX_T, AX_W]},
    )
    result = select(matrix, axes=[AX_T, AX_W, AX_C], policy="aggregate",
                    primary=AX_C)
    assert result.winner_epoch == 4


def test_missing_scores_leave_holes_not_crashes():
    scores = {(3, FP_T): {"AP@0.5": 0.6}}      # whole never scored
    matrix = build_matrix(scores, [3], [AX_T, AX_W])
    assert matrix[3][AX_W.axis_id] is None
    result = select(matrix, axes=[AX_T, AX_W], policy="argmax", primary=AX_W)
    assert result.winner_epoch is None
    assert result.notes


def test_unknown_policy_raises():
    matrix = build_matrix(_scores(), [3], [AX_T])
    with pytest.raises(KeyError):
        select(matrix, axes=[AX_T], policy="bogus", primary=AX_T)
