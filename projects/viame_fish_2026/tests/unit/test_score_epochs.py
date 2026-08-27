"""Guards on the epoch-selection driver.

Two things here can silently produce a plausible-looking but meaningless
result, so both are pinned:

1. Enumerating the checkpoints. A run with per-epoch staging must yield every
   staged epoch, in epoch order; a run without staging must yield exactly one
   autoselect row. Getting this wrong does not crash, it just quietly scores
   fewer checkpoints than the summary claims to cover.

2. Keeping stage 1 and stage 2 apart. Stage 1 ranks epochs on every 8th image;
   stage 2 measures B on all 35,111. Both write under the same output root, so
   if the stride were not part of the path a stage-2 run would reuse -- or be
   reused as -- a stage-1 result, and a subsample AP would end up reported as
   B.
"""
import importlib.util
import pathlib

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"


def _load():
    fpath = SCRIPTS / "score_epochs.py"
    if not fpath.exists():
        pytest.skip(f"missing {fpath}")
    spec = importlib.util.spec_from_file_location("score_epochs", fpath)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as ex:                      # torch/kwcoco absent
        pytest.skip(f"kit deps unavailable: {ex}")
    return mod


def test_staged_epochs_are_returned_in_epoch_order(tmp_path):
    mod = _load()
    staging = tmp_path / "staging"
    staging.mkdir()
    # Created out of order on purpose: sorting must come from the zero-padded
    # name det_solver writes, not from directory order.
    for e in (13, 0, 4, 10):
        (staging / f"epoch_{e:04d}.pth").touch()
    got = mod.checkpoints_for(tmp_path)
    assert [label for label, _ in got] == [
        "epoch_0000", "epoch_0004", "epoch_0010", "epoch_0013"]
    assert all(ckpt is not None for _, ckpt in got)


def test_zero_padding_is_what_makes_the_sort_correct(tmp_path):
    """epoch_10 would sort before epoch_4 without it -- pin the convention."""
    mod = _load()
    staging = tmp_path / "staging"
    staging.mkdir()
    for e in range(14):
        (staging / f"epoch_{e:04d}.pth").touch()
    labels = [label for label, _ in mod.checkpoints_for(tmp_path)]
    assert labels == [f"epoch_{e:04d}" for e in range(14)]


def test_a_run_without_staging_scores_its_autoselected_checkpoint(tmp_path):
    """gen001/gen003 predate per-epoch staging and must still contribute."""
    mod = _load()
    got = mod.checkpoints_for(tmp_path)
    assert got == [("autoselect", None)], (
        "None is the signal to let the trainer autoselect")


def test_an_empty_staging_dir_falls_back_rather_than_scoring_nothing(tmp_path):
    mod = _load()
    (tmp_path / "staging").mkdir()
    assert mod.checkpoints_for(tmp_path) == [("autoselect", None)]


def test_non_epoch_files_in_staging_are_ignored(tmp_path):
    mod = _load()
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "epoch_0004.pth").touch()
    (staging / "best_stg1.pth").touch()
    (staging / "notes.txt").touch()
    assert [l for l, _ in mod.checkpoints_for(tmp_path)] == ["epoch_0004"]


def test_stride_one_uses_the_full_split_untouched(tmp_path):
    """No subsample bundle is written, and no kwcoco import is needed."""
    mod = _load()
    assert mod.subsampled_target("/data/vali.kwcoco.json", 1, tmp_path) == \
        "/data/vali.kwcoco.json"
    assert list(tmp_path.iterdir()) == []
