"""Guards on the test-scoring driver.

The failure modes here are quiet ones. A test number is only interpretable if
the checkpoint was chosen somewhere else, and only comparable if every run was
measured the same way -- neither property announces itself when broken.
"""
import importlib.util
import json
import pathlib

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"


def _load():
    fpath = SCRIPTS / "score_test_once.py"
    if not fpath.exists():
        pytest.skip(f"missing {fpath}")
    spec = importlib.util.spec_from_file_location("score_test_once", fpath)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as ex:
        pytest.skip(f"kit deps unavailable: {ex}")
    return mod


def _summary(tmp_path, rows, stride=8):
    p = tmp_path / "summary.json"
    p.write_text(json.dumps({"stride": stride, "rows": rows}))
    return p


def test_winner_is_the_highest_scoring_row_per_run(tmp_path):
    mod = _load()
    p = _summary(tmp_path, [
        {"run": "a", "label": "epoch_0001", "ap": 0.70},
        {"run": "a", "label": "epoch_0006", "ap": 0.73},
        {"run": "a", "label": "epoch_0009", "ap": 0.72},
        {"run": "b", "label": "autoselect", "ap": 0.77},
    ])
    got = mod.winners_from_vali(p)
    assert got["a"][0] == "epoch_0006"
    assert got["b"][0] == "autoselect"


def test_the_stride_is_carried_so_the_output_can_name_it(tmp_path):
    """A stride-8 ranking is a fine way to CHOOSE and never a reportable AP."""
    mod = _load()
    p = _summary(tmp_path, [{"run": "a", "label": "epoch_0001", "ap": 0.7}], stride=8)
    assert mod.winners_from_vali(p)["a"][2] == 8


def test_a_missing_summary_yields_no_winners(tmp_path):
    mod = _load()
    assert mod.winners_from_vali(tmp_path / "nope.json") == {}


def test_autoselect_defers_to_the_trainer(tmp_path):
    mod = _load()
    assert mod.checkpoint_for(tmp_path, "autoselect") is None


def test_a_named_epoch_resolves_to_its_staged_file(tmp_path):
    mod = _load()
    (tmp_path / "staging").mkdir()
    ckpt = tmp_path / "staging" / "epoch_0006.pth"
    ckpt.touch()
    assert mod.checkpoint_for(tmp_path, "epoch_0006") == ckpt


def test_a_missing_staged_file_raises_rather_than_falling_back(tmp_path):
    """Falling back would score a different checkpoint than the record names."""
    mod = _load()
    (tmp_path / "staging").mkdir()
    (tmp_path / "best_stg1.pth").touch()
    with pytest.raises(FileNotFoundError):
        mod.checkpoint_for(tmp_path, "epoch_0006")
