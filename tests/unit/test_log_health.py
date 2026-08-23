"""Regression tests for run-health diagnosis.

Each failure mode here was paid for in GPU-days on the fish project:

  nan_zombie   job 489 -- 11 epochs at AP 0.000 with loss exactly 0.0000
  stall        job 489 -- 2 h 03 m frozen mid-epoch; job 293 -- 2 days
  ap_collapse  job 293 -- epoch 4 fell to 0.000 and recovered at 5
  nan_abort    the guard added to det_engine.py after 489

The fixtures are synthesized rather than excerpted, so the suite runs in the
Docker build gate where /data is not mounted. Every line format below was
copied from a real log and verified against the parser. ``test_real_logs_*``
runs the parser over the genuine articles when they happen to be on disk, and
skips otherwise -- the per-project convention for expensive tests.
"""
import os
import textwrap

import pytest

from kwcoco_detector_kit.monitoring import log_health as lh


def _progress(ts, epoch, step, total=3924, loss="51.7700", avg="59.8067"):
    return (f"[{ts}] Epoch: [{epoch}]  [{step:5d}/{total}]  eta: 0:41:29  "
            f"lr: 0.000001  loss: {loss} ({avg})  loss_bbox: 0.4490 (0.6747)")


def _epoch_end(ts, epoch, hms="0:47:27"):
    return f"[{ts}] Epoch: [{epoch}] Total time: {hms} (0.7254 s / it) loss:"


def _ap(ts, value):
    return (f"[{ts}]  Average Precision  (AP) @[ IoU=0.50:0.95 | "
            f"area=   all | maxDets=100 ] = {value}")


def _healthy_epoch(epoch, hour, ap="0.533"):
    """One complete healthy epoch: progress, epoch total, eval."""
    h = f"{hour:02d}"
    return [
        _progress(f"2026-08-22T{h}:00:00", epoch, 0),
        _progress(f"2026-08-22T{h}:06:00", epoch, 500),
        _progress(f"2026-08-22T{h}:12:00", epoch, 1000),
        _epoch_end(f"2026-08-22T{h}:47:00", epoch),
        _ap(f"2026-08-22T{h}:52:00", ap),
        (f"[2026-08-22T{h}:52:01] best_stat: {{'epoch': {epoch}, "
         f"'coco_eval_bbox': {ap}}}"),
    ]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parses_epochs_ap_and_wall_time():
    lines = _healthy_epoch(0, 16, ap="0.523") + _healthy_epoch(1, 17, ap="0.533")
    run = lh.parse_log(lines)
    assert [r.epoch for r in run.epochs] == [0, 1]
    assert run.epochs[0].wall_seconds == pytest.approx(47 * 60 + 27)
    assert run.epochs[0].aps == [0.523]
    assert run.epochs[1].aps == [0.533]
    assert run.best_stat == (1, 0.533)


def test_multiple_evals_per_epoch_are_all_captured():
    """DEIMv2 can eval both the raw model and the EMA model in one epoch."""
    lines = [
        _epoch_end("2026-08-22T17:47:00", 3),
        _ap("2026-08-22T17:52:00", "0.539"),
        _ap("2026-08-22T17:53:00", "0.623"),
    ]
    run = lh.parse_log(lines)
    assert run.epochs[0].aps == [0.539, 0.623]
    assert run.epochs[0].best_ap == 0.623


def test_healthy_log_has_no_findings():
    lines = []
    for i, hour in enumerate(range(16, 20)):
        lines += _healthy_epoch(i, hour)
    run, findings = lh.health_of_lines(lines)
    assert findings == [], [f.render() for f in findings]


# ---------------------------------------------------------------------------
# nan_zombie -- job 489
# ---------------------------------------------------------------------------


def test_detects_nan_zombie_from_exactly_zero_loss():
    lines = _healthy_epoch(0, 16) + [
        _progress("2026-08-22T17:00:00", 1, 7500, loss="31.2780", avg="31.5063"),
        _progress("2026-08-22T17:06:00", 1, 8000, loss="0.0000", avg="29.8094"),
        _progress("2026-08-22T17:12:00", 1, 8500, loss="0.0000", avg="28.0561"),
    ]
    run, findings = lh.health_of_lines(lines)
    codes = [f.code for f in findings]
    assert "nan_zombie" in codes
    z = next(f for f in findings if f.code == "nan_zombie")
    assert z.severity == "fatal"
    assert "epoch 1 step 8000" in z.detail


def test_single_zero_loss_sample_is_not_a_zombie():
    """One sample could be a formatting fluke; two consecutive cannot."""
    lines = [
        _progress("2026-08-22T17:06:00", 1, 8000, loss="0.0000", avg="29.8"),
        _progress("2026-08-22T17:12:00", 1, 8500, loss="31.2", avg="30.1"),
    ]
    _, findings = lh.health_of_lines(lines)
    assert "nan_zombie" not in [f.code for f in findings]


def test_nan_dump_without_guard_is_flagged():
    lines = [
        _progress("2026-08-22T17:00:00", 1, 7500),
        "[2026-08-22T17:01:00] tensor([[[nan, nan, nan, nan],",
        "         [nan, nan, nan, nan],",
    ]
    _, findings = lh.health_of_lines(lines)
    assert "nan_dump" in [f.code for f in findings]


def test_nan_dump_is_not_double_reported_when_the_guard_fired():
    """With the abort guard present, nan_abort is the finding that matters."""
    lines = [
        "[2026-08-22T17:01:00] tensor([[[nan, nan, nan, nan],",
        "[2026-08-22T17:01:01] RuntimeError: Non-finite pred_boxes at epoch 1 "
        "step 7700 (global_step 11624, amp dtype torch.bfloat16).",
    ]
    _, findings = lh.health_of_lines(lines)
    codes = [f.code for f in findings]
    assert "nan_abort" in codes
    assert "nan_dump" not in codes


# ---------------------------------------------------------------------------
# stall -- job 489's 2 h freeze, job 293's 2 days
# ---------------------------------------------------------------------------


def test_detects_mid_epoch_stall():
    lines = [
        _progress("2026-08-22T06:19:24", 1, 4000, total=10464),
        _progress("2026-08-22T08:22:18", 1, 4500, total=10464),
    ]
    _, findings = lh.health_of_lines(lines)
    stall = next(f for f in findings if f.code == "stall")
    assert "2.05 h" in stall.message
    assert "step 4000 -> 4500" in stall.detail


def test_epoch_boundary_gap_is_not_a_stall():
    """The eval + 819 MB checkpoint save between epochs is legitimate.

    Restricting stall detection to consecutive progress lines *within* one
    epoch is what prevents this false positive.
    """
    lines = [
        _progress("2026-08-22T16:12:00", 0, 3500),
        _epoch_end("2026-08-22T16:47:00", 0),
        _ap("2026-08-22T16:52:00", "0.523"),
        _progress("2026-08-22T17:00:00", 1, 0),      # 48 min after the last
        _progress("2026-08-22T17:06:00", 1, 500),
    ]
    _, findings = lh.health_of_lines(lines)
    assert "stall" not in [f.code for f in findings]


def test_stall_threshold_is_configurable():
    lines = [
        _progress("2026-08-22T06:00:00", 1, 4000),
        _progress("2026-08-22T06:20:00", 1, 4500),   # 20 min
    ]
    _, findings = lh.health_of_lines(lines, stall_seconds=1800)
    assert "stall" not in [f.code for f in findings]
    _, findings = lh.health_of_lines(lines, stall_seconds=600)
    assert "stall" in [f.code for f in findings]


# ---------------------------------------------------------------------------
# ap_collapse and launcher signatures
# ---------------------------------------------------------------------------


def test_detects_ap_collapse_after_a_healthy_epoch():
    lines = _healthy_epoch(0, 16, ap="0.538") + [
        _epoch_end("2026-08-22T21:47:00", 1),
        _ap("2026-08-22T21:52:00", "0.000"),
    ]
    _, findings = lh.health_of_lines(lines)
    collapse = next(f for f in findings if f.code == "ap_collapse")
    assert collapse.severity == "fatal"
    assert "epoch 1" in collapse.message


def test_first_epoch_at_zero_is_not_a_collapse():
    """Nothing healthy preceded it, so there is nothing to have collapsed."""
    lines = [_epoch_end("2026-08-22T16:47:00", 0),
             _ap("2026-08-22T16:52:00", "0.000")]
    _, findings = lh.health_of_lines(lines)
    assert "ap_collapse" not in [f.code for f in findings]


@pytest.mark.parametrize("code,line", [
    ("nccl_watchdog",
     "[rank0] Watchdog caught collective operation timeout: "
     "WorkNCCL(SeqNum=108, OpType=ALLREDUCE, NumelIn=1)"),
    ("oom", "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate"),
    ("crash", "torch.distributed.elastic.multiprocessing.errors."
              "ChildFailedError:"),
])
def test_launcher_failure_signatures(code, line):
    _, findings = lh.health_of_lines([line])
    assert code in [f.code for f in findings]


def test_findings_are_sorted_fatal_first():
    lines = [
        _progress("2026-08-22T06:00:00", 1, 4000),
        _progress("2026-08-22T08:30:00", 1, 4500),          # stall (warning)
        _progress("2026-08-22T08:36:00", 1, 5000, loss="0.0000", avg="1.0"),
        _progress("2026-08-22T08:42:00", 1, 5500, loss="0.0000", avg="1.0"),
    ]
    _, findings = lh.health_of_lines(lines)
    severities = [f.severity for f in findings]
    assert severities == sorted(severities, key=lambda s: s != "fatal")


# ---------------------------------------------------------------------------
# Reporting and watch
# ---------------------------------------------------------------------------


def test_report_includes_remaining_estimate():
    lines = _healthy_epoch(0, 16) + _healthy_epoch(1, 17)
    run, findings = lh.health_of_lines(lines)
    text = lh.format_report(run, findings, num_epochs=24)
    assert "remaining" in text
    assert "no findings" in text


def test_watch_emits_one_event_per_epoch_and_dedupes(tmp_path):
    log = tmp_path / "run-490.out"
    log.write_text("\n".join(_healthy_epoch(0, 16)) + "\n")
    events = []
    lh.watch(log, poll=0, num_epochs=24, emit=events.append, _max_polls=1)
    assert len(events) == 1 and events[0].startswith("epoch 0 done in")
    assert "23 left" in events[0]

    # A second call over an unchanged file must not re-emit; a grown file must.
    log.write_text("\n".join(_healthy_epoch(0, 16) + _healthy_epoch(1, 17)) + "\n")
    events.clear()
    lh.watch(log, poll=0, num_epochs=24, emit=events.append, _max_polls=1)
    assert len(events) == 2


def test_watch_reports_a_live_stall(tmp_path):
    log = tmp_path / "run-490.out"
    log.write_text("\n".join(_healthy_epoch(0, 16)) + "\n")
    clock = iter([0.0, 0.0, 5000.0, 5000.0])
    events = []
    lh.watch(log, poll=0, emit=events.append, stall_seconds=1800,
             now=lambda: next(clock), _max_polls=2)
    assert any("stall_live" in e for e in events)


# ---------------------------------------------------------------------------
# The genuine articles, when they are on disk
# ---------------------------------------------------------------------------


_REAL = {
    "nan_zombie": ("fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen002_warmstart-489.out",
                   {"nan_zombie", "ap_collapse", "stall"}),
    "recovered": ("fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen001-293.out",
                  {"nan_zombie", "ap_collapse"}),
}


@pytest.mark.parametrize("key", sorted(_REAL))
def test_real_logs_reproduce_their_known_findings(key):
    dpath = os.environ.get("KCD_SLURM_LOG_DPATH", "/data/users/jon.crall/slurm_logs")
    name, expected = _REAL[key]
    fpath = os.path.join(dpath, name)
    if not os.path.exists(fpath):
        pytest.skip(f"real log not on this host: {fpath}")
    with open(fpath, errors="replace") as fh:
        _, findings = lh.health_of_lines(fh)
    assert expected <= {f.code for f in findings}
