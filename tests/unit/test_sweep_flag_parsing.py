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


@pytest.mark.parametrize("flag", ["do_eval", "do_export", "do_bench"])
def test_flag_defaults_to_true(flag):
    """If this ever changes, gen006's protection is redundant, not wrong."""
    cfg = SweepCLI.cli(argv=[], strict=False)
    assert bool(getattr(cfg, flag)) is True


@pytest.mark.parametrize("flag", ["do_eval", "do_export", "do_bench"])
@pytest.mark.parametrize("false_form", ["False", "false", "0"])
def test_equals_form_disables_the_flag(flag, false_form):
    cfg = SweepCLI.cli(argv=[f"--{flag}={false_form}"], strict=False)
    assert bool(getattr(cfg, flag)) is False, (
        f"--{flag}={false_form} did not disable it; gen006 would score the "
        f"held-out test split")


def test_all_three_disabled_together():
    """The exact combination gen006 passes."""
    cfg = SweepCLI.cli(
        argv=["--do_eval=False", "--do_export=False", "--do_bench=False"],
        strict=False)
    assert not cfg.do_eval and not cfg.do_export and not cfg.do_bench
