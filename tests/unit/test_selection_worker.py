"""End-to-end worker: journal in, scored/GC'd/reranked state out.

A fake trainer stages tiny checkpoint files + journal rows; a fake scorer
returns a deterministic per-epoch AP curve. No GPU, no torch.
"""
from __future__ import annotations

import json

import pytest

from kwcoco_detector_kit.eval.protocols import DatasetBinding
from kwcoco_detector_kit.selection.config import (
    SelectionConfig,
    default_selection_config,
    resolve_plan,
)
from kwcoco_detector_kit.selection.journal import RunJournal
from kwcoco_detector_kit.selection.worker import SelectionWorker

# probe AP peaks at epoch 6; whole AP peaks at epoch 3 (divergent boards)
PROBE_AP = {0: 0.05, 1: 0.20, 2: 0.40, 3: 0.55, 4: 0.62, 5: 0.70, 6: 0.78, 7: 0.74}
WHOLE_AP = {0: 0.10, 1: 0.30, 2: 0.50, 3: 0.60, 4: 0.55, 5: 0.50, 6: 0.45, 7: 0.40}
PUP_AP = {e: v * 0.5 for e, v in PROBE_AP.items()}


def fake_scorer(ckpt_fpath, binding):
    epoch = int(ckpt_fpath.stem.split("_")[1])
    if binding.protocol.name == "true_tiled":
        return {"AP@0.5": PROBE_AP[epoch], "ap/pup": PUP_AP[epoch]}
    return {"AP@0.5": WHOLE_AP[epoch]}


def fake_strip(fpath):
    fpath.write_text(fpath.read_text() + ":stripped")


def _make_plan(num_epochs=8, config=None):
    config = config or default_selection_config(trains_on_tiles=True)
    roles = {"probe", "vali", "vali_full"}
    return resolve_plan(
        config,
        train_input_hw=(640, 640),
        dataset_fpaths={r: f"/unused/{r}.kwcoco.zip" for r in roles},
        dataset_ids={r: f"id_{r}" for r in roles},
        num_epochs=num_epochs,
        class_support={"probe": {"pup": 500}, "vali": {"pup": 5000}},
    )


def _stage(journal, epoch):
    journal.staging_dpath.mkdir(parents=True, exist_ok=True)
    ckpt = journal.staged_ckpt_fpath(epoch)
    ckpt.write_text(f"ckpt-{epoch}")
    journal.append_train({
        "event": "epoch_staged", "epoch": epoch,
        "ckpt": f"staging/{ckpt.name}",
    })


def test_worker_full_run(tmp_path):
    plan = _make_plan()
    journal = RunJournal(tmp_path)
    worker = SelectionWorker(tmp_path, plan, fake_scorer,
                             strip_fn=fake_strip, log=lambda *a: None)

    # trainer emits epochs 0..7 (min_epoch = 0.1*8 = 1 -> epoch 0 inert)
    for epoch in range(8):
        _stage(journal, epoch)
        worker.step()

    journal.append_train({"event": "train_complete"})
    worker.step()

    state = worker._fold()
    assert state.rerank_done

    # boards: probe k=3 -> {6,7,5}; whole k=2 -> {3,4(0.55)... whole curve
    # peaks at 3 (0.60) then 4 (0.55)} ; union {3,4,5,6,7}
    assert state.retained == {3, 4, 5, 6, 7}
    # anchors = top-2 of the FIRST bucket (probe board): epochs 6, 7
    assert state.anchors == {6, 7}

    # epoch 0 was below min_epoch: deleted without scoring
    assert not journal.staged_ckpt_fpath(0).exists()
    assert all((0, fp) not in state.scores
               for fp in [b.fingerprint for b in plan.inloop_bindings])
    # displaced epochs deleted from disk
    for epoch in (1, 2):
        assert not journal.staged_ckpt_fpath(epoch).exists()
    # retained-not-anchor stripped; anchors untouched
    for epoch in (3, 4, 5):
        assert journal.staged_ckpt_fpath(epoch).read_text().endswith(":stripped")
    for epoch in (6, 7):
        assert journal.staged_ckpt_fpath(epoch).read_text() == f"ckpt-{epoch}"

    # rerank: primary = true_tiled.vali_full AP@0.5 (same curve) -> epoch 6
    rerank = json.loads((journal.journal_dpath / "rerank.json").read_text())
    assert rerank["winner_epoch"] == 6
    assert rerank["frontier"]                      # persisted frontier
    assert rerank["provenance"]["candidates"] == [3, 4, 5, 6, 7]

    # the definitions store has every fingerprint used
    defs = journal.definitions()
    for b in [*plan.inloop_bindings, *plan.rerank_bindings]:
        assert b.fingerprint in defs
        assert defs[b.fingerprint]["protocol"]["name"] == b.protocol.name


def test_worker_is_idempotent_and_resumable(tmp_path):
    """Re-running the worker over a complete journal does nothing new."""
    plan = _make_plan()
    journal = RunJournal(tmp_path)
    worker = SelectionWorker(tmp_path, plan, fake_scorer,
                             strip_fn=fake_strip, log=lambda *a: None)
    for epoch in range(8):
        _stage(journal, epoch)
    journal.append_train({"event": "train_complete"})
    while worker.step():
        pass
    n_events = len(journal.read_worker())

    # a brand-new worker (crash/restart) re-folds and finds nothing to do
    worker2 = SelectionWorker(tmp_path, plan, fake_scorer,
                              strip_fn=fake_strip, log=lambda *a: None)
    assert worker2.step() is False
    assert len(journal.read_worker()) == n_events


def test_worker_lag_never_deletes_unscored(tmp_path):
    """Fail-retentive: a dead scorer means nothing is ever deleted."""
    plan = _make_plan()
    journal = RunJournal(tmp_path)

    def dead_scorer(ckpt, binding):
        raise RuntimeError("scorer down")

    worker = SelectionWorker(tmp_path, plan, dead_scorer,
                             strip_fn=fake_strip, log=lambda *a: None)
    for epoch in range(4):
        _stage(journal, epoch)
    worker.step()
    # epoch 0 is below min_epoch (floor=1): deletable without scores.
    # everything else must survive.
    for epoch in range(1, 4):
        assert journal.staged_ckpt_fpath(epoch).exists()


def test_per_class_bucket_disabled_below_support_floor():
    config = default_selection_config(trains_on_tiles=True)
    config.buckets.append(
        {"protocol": "true_tiled", "dataset": "probe",
         "metric": "ap/dead_nonpup", "k": 3})
    plan = resolve_plan(
        config,
        train_input_hw=(640, 640),
        dataset_fpaths={r: f"/unused/{r}" for r in ("probe", "vali", "vali_full")},
        dataset_ids={r: f"id_{r}" for r in ("probe", "vali", "vali_full")},
        num_epochs=30,
        class_support={"probe": {"pup": 500, "dead_nonpup": 7}},
        log=lambda *a: None,
    )
    labels = [b.label for b in plan.buckets]
    assert not any("dead_nonpup" in lbl for lbl in labels)
    assert any("dead_nonpup" in d for d in plan.disabled_buckets)


def test_whole_image_project_is_todays_behavior():
    """The strict-special-case claim: one bucket, k=1, whole lens only."""
    config = default_selection_config(trains_on_tiles=False)
    plan = resolve_plan(
        config,
        train_input_hw=(640, 640),
        dataset_fpaths={"vali": "/u/v", "vali_full": "/u/vf"},
        dataset_ids={"vali": "idv", "vali_full": "idvf"},
        num_epochs=30,
    )
    assert len(plan.buckets) == 1
    assert plan.buckets[0].k == 1
    assert plan.anchor_top_m == 1
    assert len(plan.inloop_bindings) == 1
    assert plan.inloop_bindings[0].protocol.name == "whole_resize"


def test_unknown_dataset_role_raises():
    config = SelectionConfig(
        inloop=[{"protocol": "true_tiled", "dataset": "probe"}],
        buckets=[{"protocol": "true_tiled", "dataset": "probe",
                  "metric": "AP@0.5", "k": 1}],
        rerank={"axes": [{"protocol": "true_tiled", "dataset": "vali_full",
                          "metric": "AP@0.5"}],
                "policy": "argmax",
                "primary": {"protocol": "true_tiled", "dataset": "vali_full",
                            "metric": "AP@0.5"}},
    )
    with pytest.raises(KeyError):
        resolve_plan(
            config, train_input_hw=(640, 640),
            dataset_fpaths={"vali": "/u/v"}, dataset_ids={"vali": "x"},
            num_epochs=10,
        )
