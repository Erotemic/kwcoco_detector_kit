"""orchestration.pareto_sweep must exit non-zero when any cell fails.

This regression test pins the behaviour that surfaced in the host_smoke
log: a single mock_tiny cell failed at the export stage, but the sweep
exited 0 and the smoke driver reported `[PASS]`. The fix: even with
`--keep_going`, the sweep raises ``SystemExit(N)`` after writing the
index TSV when any row's status starts with `fail_`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def test_sweep_exits_nonzero_when_a_cell_fails(synthetic_kwcoco, tmp_path):
    """Force a failure by pointing sweep at a non-existent test kwcoco for the eval stage."""
    # We invoke the sweep CLI as a subprocess so we can read the exit code.
    cmd = [
        sys.executable, "-m", "kwcoco_detector_kit", "sweep",
        "--train_kwcoco", str(synthetic_kwcoco),
        "--vali_kwcoco",  str(synthetic_kwcoco),
        "--test_kwcoco",  "/nonexistent/path/that/will/break/eval.kwcoco.zip",
        "--kcd_root", str(tmp_path),
        "--trainer", "mock_tiny", "--variant", "mock_tiny",
        "--input_hw", "64,64", "--train_policy", "fixed",
        "--num_epochs", "1", "--batch_size", "2", "--val_batch_size", "2",
        "--category_names", "widget",
        "--scale_tier", "S",
        "--keep_going",  # value of True (default) — confirm exit-nonzero even with keep_going
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0, (
        f"sweep should exit non-zero when a cell fails; got rc=0\n"
        f"stdout:\n{result.stdout[-1000:]}\n"
        f"stderr:\n{result.stderr[-1000:]}"
    )

    # The sweep should still have written the index TSV.
    indexes = list(tmp_path.rglob("index.tsv"))
    assert indexes, "sweep should write index.tsv even on cell failure"
    rows = Path(indexes[0]).read_text().splitlines()
    # Header + at least one row of fail_*.
    assert any("fail_" in r for r in rows[1:]), (
        f"expected at least one fail_* row in {indexes[0]}; got:\n{chr(10).join(rows)}"
    )
