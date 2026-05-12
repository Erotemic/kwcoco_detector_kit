from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_follow_module():
    root = Path(__file__).resolve().parents[2]
    fpath = root / "smoketests" / "dino_v2_4x" / "slurm" / "follow_job.py"
    spec = importlib.util.spec_from_file_location("kcd_follow_job", fpath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_show_job_expands_stdout_template():
    mod = _load_follow_module()
    text = (
        "JobId=12345 JobName=kcd-dino2-01 UserId=jon(1000) "
        "JobState=RUNNING StdOut=/tmp/logs/%x-%j.out StdErr=/tmp/logs/%x-%j.err"
    )
    data = mod.parse_show_job(text)
    assert data["StdOut"] == "/tmp/logs/kcd-dino2-01-12345.out"
    assert data["StdErr"] == "/tmp/logs/kcd-dino2-01-12345.err"


def test_tailer_waits_then_reads(tmp_path, capsys):
    mod = _load_follow_module()
    fpath = tmp_path / "job.out"
    tailer = mod.Tailer(fpath)
    try:
        assert tailer.read_available() is False
        fpath.write_text("hello\n")
        assert tailer.read_available() is True
        assert tailer.read_available() is False
    finally:
        tailer.close()
    captured = capsys.readouterr()
    assert "hello" in captured.out
    assert "waiting for log" in captured.err
