"""Checkpoint selection & retention (spec: docs/planning/checkpoint_selection.md).

The trainer appends to a run journal; a detached worker scores staged
checkpoints under in-loop fingerprints, folds leaderboards, GCs staging,
and runs the final multi-objective re-rank.
"""
