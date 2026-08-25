"""The sweep, the trainer and RunJournal must agree on one layout.

The three derive their paths differently, which is how they silently
disagreed:

    RunJournal(workdir)   journal = workdir/journal/train.jsonl
                          staging = workdir/staging
    det_solver(J)         journal = J/train.jsonl
                          staging = J.parent/'staging'
    pareto_sweep          passes J

So J must be ``workdir/"journal"``. It was ``workdir/"staging"``, and that
almost worked: staging resolved to ``staging.parent/'staging'`` -- the same
directory -- so every epoch checkpoint was written to the right place and only
the journal landed at ``staging/train.jsonl``, where nothing looks. A one-shot
run would have staged all 14 epochs and the reranker would have found none of
them.
"""
import ast
from pathlib import Path

from kwcoco_detector_kit.selection.journal import RunJournal

_SWEEP = Path("kwcoco_detector_kit/orchestration/pareto_sweep.py")
_SOLVER = Path("tpl/DEIMv2/engine/solver/det_solver.py")


def _sweep_journal_suffix():
    """The literal the sweep appends to workdir for selection_journal_dpath."""
    tree = ast.parse(_SWEEP.read_text())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Dict)):
            continue
        for key, val in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "selection_journal_dpath":
                for sub in ast.walk(val):
                    if (isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Div)
                            and isinstance(sub.right, ast.Constant)):
                        return sub.right.value
    raise AssertionError("could not find selection_journal_dpath in the sweep")


def test_sweep_passes_the_journal_dir_not_the_staging_dir(tmp_path):
    suffix = _sweep_journal_suffix()
    rj = RunJournal(tmp_path)
    assert tmp_path / suffix == rj.journal_dpath, (
        f"sweep passes workdir/{suffix!r}, but RunJournal reads "
        f"{rj.journal_dpath.name!r}")
    assert suffix != "staging", (
        "passing the staging dir writes the journal to staging/train.jsonl, "
        "which the reranker never reads -- while the .pth files still land "
        "correctly, so it looks like it works")


def test_solver_derivation_matches_runjournal(tmp_path):
    """Replay det_solver's own arithmetic against RunJournal's."""
    rj = RunJournal(tmp_path)
    journal_dir = tmp_path / _sweep_journal_suffix()
    # det_solver.py: _kcd_staging = _kcd_journal.parent / 'staging'
    assert journal_dir.parent / "staging" == rj.staging_dpath
    # det_solver.py: open(_kcd_journal / 'train.jsonl', 'a')
    assert journal_dir / "train.jsonl" == rj.train_fpath


def test_solver_still_derives_staging_from_the_journal_parent():
    """Pin the coupling this test exists to protect.

    If the fork stops deriving staging from the journal dir's parent, the
    sweep's single path stops determining both and this whole contract needs
    revisiting.
    """
    if not _SOLVER.exists():
        import pytest
        pytest.skip("DEIMv2 submodule not present")
    src = _SOLVER.read_text()
    assert "_kcd_journal.parent / 'staging'" in src
    assert "_kcd_journal / 'train.jsonl'" in src
