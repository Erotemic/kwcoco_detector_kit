"""Graceful sweep restart / retry filtering."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def test_sweep_marks_fully_skipped_cell_ok_resumed(tmp_path, monkeypatch):
    import kwcoco_detector_kit.orchestration.pareto_sweep as ps

    cell = {"variant": "mock_tiny", "input_hw": [64, 64], "train_policy": "fixed"}
    candidate_id = ps._candidate_id(cell)
    workdir = tmp_path / "runs" / candidate_id
    export_dpath = workdir / "export"
    metrics_dpath = tmp_path / "eval" / candidate_id / "eval"
    export_dpath.mkdir(parents=True)
    metrics_dpath.mkdir(parents=True)
    (workdir / "best_stg2.pth").write_bytes(b"checkpoint")
    # A skippable checkpoint must come from a COMPLETED training — mark it.
    (workdir / ps._TRAIN_COMPLETE_MARKER).write_text("ok\n")
    (export_dpath / "mock_tiny_h64_w64.onnx").write_bytes(b"x" * 262144)
    (export_dpath / "mock_tiny_h64_w64.bench.json").write_text("{}")
    (metrics_dpath / "detect_metrics.json").write_text("{}")

    monkeypatch.setattr(ps, "get_trainer", lambda name: object())

    def _should_not_run(*args, **kwargs):
        raise AssertionError("stage should have been skipped")

    monkeypatch.setattr(ps, "_run_train", _should_not_run)
    monkeypatch.setattr(ps, "_run_export", _should_not_run)
    monkeypatch.setattr(ps, "_run_eval", _should_not_run)
    monkeypatch.setattr(ps, "_run_bench", _should_not_run)

    cfg = ps.SweepConfig.cli(
        argv=False,
        data={
            "train_kwcoco": "/unused/train.kwcoco.zip",
            "vali_kwcoco": "/unused/vali.kwcoco.zip",
            "test_kwcoco": "/unused/test.kwcoco.zip",
            "kcd_root": str(tmp_path),
            "trainer": "mock_tiny",
            "variant": "mock_tiny",
            "input_hw": [64, 64],
            "train_policy": "fixed",
            "do_export": True,
            "do_eval": True,
            "do_bench": True,
        },
    )
    index_fpath = ps.run(cfg)

    rows = list(csv.DictReader(Path(index_fpath).open(), delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == candidate_id
    assert rows[0]["status"] == "ok_resumed"


def test_sweep_retrains_when_checkpoint_lacks_completion_marker(tmp_path, monkeypatch):
    """A best_*.pth from a crashed/killed run (no completion marker) must
    trigger a RETRAIN, not skip-to-eval on the partial checkpoint."""
    import kwcoco_detector_kit.orchestration.pareto_sweep as ps

    cell = {"variant": "mock_tiny", "input_hw": [64, 64], "train_policy": "fixed"}
    candidate_id = ps._candidate_id(cell)
    workdir = tmp_path / "runs" / candidate_id
    workdir.mkdir(parents=True)
    # Simulate a killed run: best_stg2.pth present but NO .train_complete marker
    # (DEIMv2 writes best_stg2 after the first eval, so an epoch-0 kill leaves one).
    (workdir / "best_stg2.pth").write_bytes(b"partial-checkpoint")

    calls = {"train": 0, "eval": 0}

    def _fake_train(trainer, *, config, cell, workdir, candidate_id):
        calls["train"] += 1  # _mark_train_complete is called by the sweep after this

    def _record_eval(*args, **kwargs):
        calls["eval"] += 1
        return tmp_path / "eval" / "x" / "eval" / "detect_metrics.json"

    monkeypatch.setattr(ps, "get_trainer", lambda name: object())
    monkeypatch.setattr(ps, "_run_train", _fake_train)
    monkeypatch.setattr(ps, "_run_eval", _record_eval)
    # export/bench off so the test stays focused on the train/eval gating
    cfg = ps.SweepConfig.cli(
        argv=False,
        data={
            "train_kwcoco": "/unused/train.kwcoco.zip",
            "vali_kwcoco": "/unused/vali.kwcoco.zip",
            "test_kwcoco": "/unused/test.kwcoco.zip",
            "kcd_root": str(tmp_path),
            "trainer": "mock_tiny",
            "variant": "mock_tiny",
            "input_hw": [64, 64],
            "train_policy": "fixed",
            "do_export": False,
            "do_eval": True,
            "do_bench": False,
        },
    )
    ps.run(cfg)

    assert calls["train"] == 1, "partial checkpoint (no marker) must retrain"
    # eval runs AFTER the retrain (on the fresh model), and the marker now exists
    assert (workdir / ps._TRAIN_COMPLETE_MARKER).exists()


def test_retry_failed_filters_prior_ok_cells(tmp_path):
    import kwcoco_detector_kit.orchestration.pareto_sweep as ps

    prior = tmp_path / "prior.tsv"
    rows = [
        {"candidate_id": "a_64x64_fixed", "status": "ok"},
        {"candidate_id": "b_64x64_fixed", "status": "fail_eval"},
    ]
    with prior.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "status"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    matrix = [
        {"variant": "a", "input_hw": [64, 64], "train_policy": "fixed"},
        {"variant": "b", "input_hw": [64, 64], "train_policy": "fixed"},
        {"variant": "c", "input_hw": [64, 64], "train_policy": "fixed"},
    ]
    kept = ps._filter_retry_failed(matrix, prior)
    assert [ps._candidate_id(c) for c in kept] == [
        "b_64x64_fixed",
        "c_64x64_fixed",
    ]
