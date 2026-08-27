"""Every ``kcd_sample_*`` key the kit emits must be consumed by the solver.

The gap this exists for: the trainer emitted ``kcd_sample_replacement`` into
the generated YAML, the sweep flag parsed it, the sidecar recorded it, the
launcher banner announced it -- and ``_solver.py`` called
``sampler_from_weights_file()`` without forwarding it, so the factory fell back
to ``replacement=True``. Every existing test passed: the factory worked, the
CLI carried the flag, and the config contained the key. Nothing checked that
anything ever READ it.

A key that is written but never read is invisible from both ends. These tests
close the loop at the two places it can break -- the source-level contract, and
the actual object the config produces.
"""
import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SOLVER = REPO / "tpl" / "DEIMv2" / "engine" / "solver" / "_solver.py"
TRAINER = REPO / "kwcoco_detector_kit" / "trainers" / "deimv2.py"


def _emitted_sample_keys():
    """Every ``yml["kcd_sample_*"] = ...`` the trainer writes."""
    src = TRAINER.read_text()
    return set(re.findall(r'yml\[\s*[\'"](kcd_sample_[a-z_]+)[\'"]\s*\]\s*=', src))


def test_the_trainer_emits_the_keys_we_think_it_does():
    """Guard the guard: if this shrinks, the test below stops checking anything."""
    keys = _emitted_sample_keys()
    assert keys >= {"kcd_sample_epoch_length", "kcd_sample_seed",
                    "kcd_sample_replacement"}, keys


@pytest.mark.skipif(not SOLVER.exists(), reason="DEIMv2 submodule not present")
def test_every_emitted_sample_key_is_read_by_the_solver():
    """The contract that would have caught the replacement bug directly."""
    solver_src = SOLVER.read_text()
    unread = sorted(k for k in _emitted_sample_keys() if k not in solver_src)
    assert not unread, (
        f"{unread} are written into the generated config but never read by "
        f"{SOLVER.relative_to(REPO)} -- the run would silently use the "
        "factory defaults")


@pytest.mark.skipif(not SOLVER.exists(), reason="DEIMv2 submodule not present")
def test_the_solver_forwards_replacement_to_the_factory():
    """Reading the key is not enough; it has to reach the call."""
    tree = ast.parse(SOLVER.read_text())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "sampler_from_weights_file"]
    assert calls, "sampler_from_weights_file is not called in _solver.py"
    for call in calls:
        kwargs = {kw.arg for kw in call.keywords}
        assert "replacement" in kwargs, (
            f"sampler_from_weights_file called with {sorted(kwargs)} -- without "
            "replacement= the factory silently defaults to with-replacement "
            "sampling")


# ---------------------------------------------------------------------------
# The behavioural half: config dict -> the solver's own lookups -> the object
# ---------------------------------------------------------------------------


def _sampler_for(cfg, weights_fpath, dataset_len):
    """Replay exactly the lookups _solver.py performs on the generated config."""
    from kwcoco_detector_kit.data.balanced_sampler import sampler_from_weights_file
    return sampler_from_weights_file(
        weights_fpath,
        dataset_len=dataset_len,
        epoch_length=cfg.get("kcd_sample_epoch_length") or None,
        seed=int(cfg.get("kcd_sample_seed", 0) or 0),
        replacement=bool(cfg.get("kcd_sample_replacement", True)),
    )


@pytest.mark.parametrize("replacement,expected", [
    (False, "DistributedWeightedNoReplacementSampler"),
    (True, "DistributedWeightedRandomSampler"),
])
def test_the_generated_config_selects_the_right_sampler(replacement, expected,
                                                        tmp_path):
    pytest.importorskip("torch")
    from kwcoco_detector_kit.data.balanced_sampler import write_balance_weights
    fpath = write_balance_weights(tmp_path / "w.json", [0.2, 0.3, 0.5])
    cfg = {"kcd_sample_weights_fpath": str(fpath),
           "kcd_sample_epoch_length": 2,
           "kcd_sample_seed": 0,
           "kcd_sample_replacement": replacement}
    sampler = _sampler_for(cfg, fpath, dataset_len=3)
    assert type(sampler).__name__ == expected


def test_a_config_without_the_key_keeps_the_historical_behaviour(tmp_path):
    """Runs predating the flag must be unaffected."""
    pytest.importorskip("torch")
    from kwcoco_detector_kit.data.balanced_sampler import write_balance_weights
    fpath = write_balance_weights(tmp_path / "w.json", [0.5, 0.5])
    sampler = _sampler_for({"kcd_sample_seed": 0}, fpath, dataset_len=2)
    assert type(sampler).__name__ == "DistributedWeightedRandomSampler"
