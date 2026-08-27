"""A caller must be able to score a SPECIFIC checkpoint.

``build_predictor`` used to always call ``find_checkpoint``, which autoselects
best_stg2 > best_stg1 > last. A run that stages a checkpoint per epoch
therefore had no way to ask "how does epoch 4 score?" through the kit's own
eval path, and per-epoch comparison had to bypass ``run_kwcoco_eval`` --
losing tiled inference, the protocol fingerprint and the distractor sidecar
along the way.

The failure these tests guard against is the quiet one: a pinned checkpoint
that is ignored or silently falls back to autoselection. Fourteen epochs would
then produce fourteen results that are all secretly the same weights, and the
epoch curve would look flat for a reason that has nothing to do with training.
"""
import pytest

from kwcoco_detector_kit.trainers._interface import _pin_checkpoint


class _Trainer:
    """Stands in for any trainer: only find_checkpoint is consulted."""

    def __init__(self, auto):
        self.auto = auto
        self.calls = 0

    def find_checkpoint(self, workdir):
        self.calls += 1
        return self.auto


def test_none_defers_to_autoselection(tmp_path):
    auto = tmp_path / "best_stg1.pth"
    auto.touch()
    t = _Trainer(auto)
    assert _pin_checkpoint(t, tmp_path, None) == auto
    assert t.calls == 1


def test_an_explicit_checkpoint_wins_and_autoselect_is_never_consulted(tmp_path):
    auto = tmp_path / "best_stg1.pth"
    auto.touch()
    pinned = tmp_path / "staging" / "epoch_0004.pth"
    pinned.parent.mkdir()
    pinned.touch()
    t = _Trainer(auto)
    assert _pin_checkpoint(t, tmp_path, pinned) == pinned
    assert t.calls == 0, "autoselection must not run when a checkpoint is pinned"


def test_a_missing_pinned_checkpoint_raises_rather_than_falling_back(tmp_path):
    """The dangerous failure mode, made loud.

    Falling back would score the autoselected checkpoint while reporting it
    under the pinned checkpoint's label.
    """
    auto = tmp_path / "best_stg1.pth"
    auto.touch()
    t = _Trainer(auto)
    with pytest.raises(FileNotFoundError):
        _pin_checkpoint(t, tmp_path, tmp_path / "staging" / "epoch_0099.pth")
    assert t.calls == 0


def test_a_string_path_is_accepted(tmp_path):
    pinned = tmp_path / "epoch_0004.pth"
    pinned.touch()
    assert _pin_checkpoint(_Trainer(None), tmp_path, str(pinned)) == pinned


@pytest.mark.parametrize("name", ["deimv2", "opengroundingdino", "mock_tiny"])
def test_every_trainer_exposes_the_parameter(name):
    """The eval path passes checkpoint= unconditionally.

    A trainer that did not accept it would fail at call time, deep inside a
    scoring job, rather than here.
    """
    import inspect

    import kwcoco_detector_kit.trainers  # noqa: F401 -- registers the plugins
    from kwcoco_detector_kit.trainers._registry import get_trainer

    sig = inspect.signature(get_trainer(name).build_predictor)
    assert "checkpoint" in sig.parameters
    assert sig.parameters["checkpoint"].default is None


def test_run_kwcoco_eval_forwards_the_checkpoint(tmp_path, monkeypatch):
    """End of the wire: the parameter must reach build_predictor."""
    from kwcoco_detector_kit.eval import kwcoco_eval

    seen = {}

    class _T:
        name = "spy"

        def build_predictor(self, workdir, *, device="cpu", checkpoint=None):
            seen["checkpoint"] = checkpoint
            raise _Stop()

    class _Stop(Exception):
        pass

    pinned = tmp_path / "epoch_0004.pth"
    pinned.touch()
    with pytest.raises(_Stop):
        kwcoco_eval.run_kwcoco_eval(
            trainer=_T(),
            workdir=tmp_path,
            test_kwcoco="unused.kwcoco.json",
            kcd_root=tmp_path / "out",
            candidate_id="spy",
            category_names=["fish"],
            checkpoint=pinned,
        )
    assert seen["checkpoint"] == pinned
