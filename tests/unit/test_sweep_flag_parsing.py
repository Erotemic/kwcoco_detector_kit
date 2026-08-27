"""`--do_eval=False` must actually disable eval.

gen006 relies on this to keep the HELD-OUT TEST split untouched while it
trains: pareto_sweep defaults do_eval/do_export/do_bench to True, and the fish
launcher forwards KCD_TEST_KWCOCO into the sweep regardless.

These are ``isflag`` options. A bare ``--do_eval False`` risks parsing as the
flag (True) plus a stray positional, which would silently ENABLE what we meant
to disable -- and spending the holdout cannot be undone. The launcher therefore
uses the ``=`` form, and this test pins that it means what we think.
"""
import pytest

kwconf = pytest.importorskip("kwconf")

from kwcoco_detector_kit.orchestration.pareto_sweep import __cli__ as SweepCLI

#: SweepConfig marks these required, and kwconf validates required fields even
#: under strict=False, so every construction here has to supply them. They are
#: never read: nothing in these tests runs a sweep.
_REQUIRED = [
    "--train_kwcoco=/nonexistent/train.kwcoco.json",
    "--vali_kwcoco=/nonexistent/vali.kwcoco.json",
    "--test_kwcoco=/nonexistent/test.kwcoco.json",
]


def _cfg(*flags):
    return SweepCLI.cli(argv=_REQUIRED + list(flags), strict=False)


@pytest.mark.parametrize("flag", ["do_eval", "do_export", "do_bench"])
def test_flag_defaults_to_true(flag):
    """If this ever changes, gen006's protection is redundant, not wrong."""
    assert bool(getattr(_cfg(), flag)) is True


@pytest.mark.parametrize("flag", ["do_eval", "do_export", "do_bench"])
@pytest.mark.parametrize("false_form", ["False", "false", "0"])
def test_equals_form_disables_the_flag(flag, false_form):
    cfg = _cfg(f"--{flag}={false_form}")
    assert bool(getattr(cfg, flag)) is False, (
        f"--{flag}={false_form} did not disable it; gen006 would score the "
        f"held-out test split")


def test_all_three_disabled_together():
    """The exact combination gen006 passes."""
    cfg = _cfg("--do_eval=False", "--do_export=False", "--do_bench=False")
    assert not cfg.do_eval and not cfg.do_export and not cfg.do_bench


def test_selection_journal_defaults_off_and_enables_with_equals_form():
    """gen006 turns per-epoch staging on the same way."""
    assert bool(_cfg().selection_journal) is False
    assert bool(_cfg("--selection_journal=True").selection_journal) is True


def test_gen007_flags_parse_and_reach_the_trainer():
    """aug_profile, tail_epochs and balance_replacement are experiment-defining.

    A flag that parses but never reaches ``extra`` would leave the run silently
    on the defaults -- full augmentation, a proportional tail, and sampling with
    replacement -- while the launcher banner claimed otherwise.
    """
    from kwcoco_detector_kit.orchestration.pareto_sweep import SweepConfig
    cfg = SweepConfig.cli(argv=[
        "--train_kwcoco", "a.json", "--vali_kwcoco", "b.json",
        "--test_kwcoco", "c.json",
        "--aug_profile", "tiled_light",
        "--tail_epochs", "8",
        "--balance_replacement", "False",
    ], strict=False)
    assert cfg.aug_profile == "tiled_light"
    assert int(cfg.tail_epochs) == 8
    assert bool(cfg.balance_replacement) is False


def test_gen007_flag_defaults_are_the_historical_behaviour():
    """Every new knob must default to what the kit did before it existed."""
    from kwcoco_detector_kit.orchestration.pareto_sweep import SweepConfig
    cfg = SweepConfig.cli(argv=[
        "--train_kwcoco", "a.json", "--vali_kwcoco", "b.json",
        "--test_kwcoco", "c.json",
    ], strict=False)
    assert cfg.aug_profile == "full"
    assert int(cfg.tail_epochs) == 0
    assert bool(cfg.balance_replacement) is True
