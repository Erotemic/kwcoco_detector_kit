"""Diagnose the health of a DEIMv2 training run from its slurm log.

``follow_job.py`` makes ``sbatch`` feel like a foreground command by printing
the log verbatim. That is the right tool when you are watching a run start.
It is the wrong tool for a 24-hour job, for two reasons:

1. Volume. fish job 489 emitted 500,000 lines / 13.9 MB, almost all of it
   ``tensor([[[nan, nan, ...`` dumps.
2. It has no notion of *health*. A run can be alive, burning four GPUs, and
   producing nothing -- and a verbatim tail looks exactly the same as a
   healthy one.

This module is the missing half: pure functions over log text that answer
"is this run OK, and if not, what went wrong". Pure because the interesting
failures are all reconstructable after the fact -- which means they can be
unit-tested against logs we already have on disk.

## The failure modes this encodes

Each of these cost real GPU-days on this project, and until now the knowledge
lived only in journal entries and commit messages:

``nan_zombie``
    The forward goes NaN and DEIMv2 keeps training. Every loss reads exactly
    ``0.0000``, so the gradients are 0 and *finite*: GradScaler finds nothing
    to skip and upstream's ``math.isfinite(loss_value)`` guard never fires.

    This is not always terminal -- DEIMv2 reloads ``best_stg1.pth`` whenever an
    eval fails to improve (det_solver.py:213-217), which is how fish job 293
    escaped at epoch 5 after going NaN at epoch 4. But every zombie epoch is
    wasted, and when the trigger reproduces the run never escapes: job 489
    reloaded good weights at the end of each epoch and re-NaN'd inside the
    next one, burning 11 epochs at AP 0.000. Either way it wants a human.
    Newer images abort instead (``nan_abort``); old images and archived logs
    still need diagnosing.

``stall``
    Training stops advancing without dying. fish job 293 sat in a spinning
    NCCL all-reduce for two days; job 489 froze for 2 h 03 m between steps
    4000 and 4500 of epoch 1. Detected from the log's own timestamps by
    comparing consecutive progress lines *within an epoch*, so it does not
    need a live clock and does not trip on the eval + checkpoint gap between
    epochs.

``ap_collapse``
    Validation AP falls to 0.000 after a healthy epoch. The visible symptom of
    ``nan_zombie``, but also catches divergence that leaves the weights finite.

``nccl_watchdog`` / ``crash`` / ``oom``
    Emitted by the shared ``_sbatch_train.sh`` launcher, so they apply to every
    project in the kit, not just this one.

Dependency-free on purpose (stdlib only): it has to run on a login node,
before any Docker image starts, exactly like ``follow_job.py``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, List, Optional, Sequence


# --------------------------------------------------------------------------
# Line grammar. All of these come from DEIMv2's MetricLogger / det_solver and
# the kit's `[%Y-%m-%dT%H:%M:%S]` stamping in _sbatch_train.sh.
# --------------------------------------------------------------------------

_TS = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\]")
_PROGRESS = re.compile(r"Epoch: \[(\d+)\]\s+\[\s*(\d+)/(\d+)\]")
_LOSS = re.compile(r"\bloss:\s+([0-9.]+)\s+\(([0-9.]+)\)")
_EPOCH_TOTAL = re.compile(
    r"Epoch: \[(\d+)\] Total time: (\d+):(\d{2}):(\d{2})")
_AP_ALL = re.compile(
    r"Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=\s+all \| "
    r"maxDets=100 \]\s+=\s+(-?[0-9.]+)")
_BEST_STAT = re.compile(
    r"best_stat: \{'epoch': (\d+), 'coco_eval_bbox': ([0-9.]+)\}")

# (code, severity, human message, compiled pattern)
_SIGNATURES = (
    ("nan_abort", "fatal",
     "training aborted on non-finite pred_boxes (the kit's NaN guard fired)",
     re.compile(r"Non-finite pred_boxes")),
    ("nccl_watchdog", "fatal",
     "NCCL watchdog aborted a collective that exceeded its timeout",
     re.compile(r"Watchdog caught collective operation timeout")),
    ("oom", "fatal",
     "CUDA out of memory",
     re.compile(r"CUDA out of memory|torch\.OutOfMemoryError|OUT_OF_MEMORY")),
    ("crash", "fatal",
     "a rank died and torchrun reported the failure",
     re.compile(r"ChildFailedError|srun: error|RuntimeError")),
    ("traceback", "fatal",
     "python traceback in the log",
     re.compile(r"^Traceback \(most recent call last\)")),
)

# A NaN tensor dump from DEIMv2's (pre-guard) detector. Distinct from
# nan_abort: this one means the run KEPT GOING.
_NAN_DUMP = re.compile(r"tensor\(\[\[\[nan|\[nan, nan, nan, nan\]")


# --------------------------------------------------------------------------
# Parsed structures
# --------------------------------------------------------------------------


@dataclass
class ProgressSample:
    """One MetricLogger progress line."""
    line_no: int
    timestamp: Optional[datetime]
    epoch: int
    step: int
    total_steps: int
    loss: Optional[float]        # window median, the first of the two numbers
    loss_avg: Optional[float]


@dataclass
class EpochRecord:
    epoch: int
    #: Training time only, as logged by "Epoch: [N] Total time". EXCLUDES the
    #: eval that follows it, so never use this alone for an ETA -- use
    #: cycle_seconds, which is measured end-to-end.
    wall_seconds: Optional[float] = None
    #: Wall time from the previous epoch's end to this one's, so it includes
    #: eval and the checkpoint save. None for the first epoch seen.
    cycle_seconds: Optional[float] = None
    #: A list because nothing guarantees one eval per epoch; in practice
    #: DEIMv2 emits a single AP block per epoch here.
    aps: List[float] = field(default_factory=list)

    @property
    def best_ap(self) -> Optional[float]:
        return max(self.aps) if self.aps else None


@dataclass
class Finding:
    code: str
    severity: str                # 'fatal' | 'warning'
    message: str
    line_no: Optional[int] = None
    detail: str = ""

    def render(self) -> str:
        where = f" (line {self.line_no})" if self.line_no else ""
        extra = f" -- {self.detail}" if self.detail else ""
        return f"[{self.severity.upper()}] {self.code}: {self.message}{where}{extra}"


@dataclass
class RunLog:
    progress: List[ProgressSample] = field(default_factory=list)
    epochs: List[EpochRecord] = field(default_factory=list)
    best_stat: Optional[tuple] = None      # (epoch, ap)
    signature_hits: List[Finding] = field(default_factory=list)
    nan_dump_lines: List[int] = field(default_factory=list)
    first_timestamp: Optional[datetime] = None
    last_timestamp: Optional[datetime] = None
    num_lines: int = 0


def _parse_ts(line: str) -> Optional[datetime]:
    m = _TS.match(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def parse_log(lines: Iterable[str]) -> RunLog:
    """Parse a DEIMv2 slurm log into a :class:`RunLog`.

    Streams line-by-line and keeps only summaries, so a 500k-line NaN-dump log
    costs no more memory than a healthy one.
    """
    out = RunLog()
    by_epoch = {}
    pending_ap_epoch = 0
    prev_epoch_end = None

    for idx, raw in enumerate(lines, start=1):
        out.num_lines = idx
        line = raw.rstrip("\n")
        ts = _parse_ts(line)
        if ts is not None:
            if out.first_timestamp is None:
                out.first_timestamp = ts
            out.last_timestamp = ts

        m = _PROGRESS.search(line)
        if m:
            loss = loss_avg = None
            lm = _LOSS.search(line)
            if lm:
                loss, loss_avg = float(lm.group(1)), float(lm.group(2))
            out.progress.append(ProgressSample(
                line_no=idx, timestamp=ts, epoch=int(m.group(1)),
                step=int(m.group(2)), total_steps=int(m.group(3)),
                loss=loss, loss_avg=loss_avg))
            continue

        m = _EPOCH_TOTAL.search(line)
        if m:
            ep = int(m.group(1))
            secs = (int(m.group(2)) * 3600 + int(m.group(3)) * 60
                    + int(m.group(4)))
            rec = by_epoch.setdefault(ep, EpochRecord(epoch=ep))
            rec.wall_seconds = float(secs)
            if ts is not None:
                if prev_epoch_end is not None:
                    rec.cycle_seconds = (ts - prev_epoch_end).total_seconds()
                prev_epoch_end = ts
            pending_ap_epoch = ep
            continue

        m = _AP_ALL.search(line)
        if m:
            rec = by_epoch.setdefault(pending_ap_epoch,
                                      EpochRecord(epoch=pending_ap_epoch))
            rec.aps.append(float(m.group(1)))
            continue

        m = _BEST_STAT.search(line)
        if m:
            out.best_stat = (int(m.group(1)), float(m.group(2)))
            continue

        if _NAN_DUMP.search(line):
            # Only keep the first few -- job 489 had ~380,000 of these.
            if len(out.nan_dump_lines) < 5:
                out.nan_dump_lines.append(idx)
            continue

        for code, severity, message, pat in _SIGNATURES:
            if pat.search(line):
                # One finding per signature; the first hit is the informative
                # one and later ones are usually cascade noise.
                if not any(f.code == code for f in out.signature_hits):
                    out.signature_hits.append(Finding(
                        code=code, severity=severity, message=message,
                        line_no=idx, detail=line.strip()[:200]))
                break

    out.epochs = [by_epoch[k] for k in sorted(by_epoch)]
    return out


# --------------------------------------------------------------------------
# Diagnosis
# --------------------------------------------------------------------------


#: A stall must clear the longest legitimate pause. Progress prints every
#: ``print_freq`` steps (500 in every kit config), which is ~3.5-13 min
#: depending on batch size. 30 min leaves generous headroom while still
#: catching job 489's 2 h 03 m freeze.
DEFAULT_STALL_SECONDS = 1800.0

#: Losses are O(30) in this stack, so an exact 0.0 is never legitimate. Two
#: consecutive samples rules out a one-off formatting fluke.
ZOMBIE_MIN_SAMPLES = 2


def diagnose(run: RunLog, *,
             stall_seconds: float = DEFAULT_STALL_SECONDS,
             zombie_min_samples: int = ZOMBIE_MIN_SAMPLES) -> List[Finding]:
    """Return findings, most severe first."""
    findings: List[Finding] = list(run.signature_hits)

    # -- nan_zombie: consecutive progress lines whose loss is exactly 0 ------
    run_len = 0
    zombie_start = None
    for s in run.progress:
        if s.loss is not None and s.loss == 0.0:
            run_len += 1
            if zombie_start is None:
                zombie_start = s
        else:
            run_len = 0
            zombie_start = None
        if run_len >= zombie_min_samples:
            findings.append(Finding(
                code="nan_zombie", severity="fatal",
                message=("loss is exactly 0.0000 -- the weights are NaN and "
                         "every epoch from here is wasted"),
                line_no=zombie_start.line_no,
                detail=(f"first seen at epoch {zombie_start.epoch} "
                        f"step {zombie_start.step}")))
            break

    # -- stall: gap between consecutive progress lines *within* an epoch -----
    # Restricting to a single epoch is what keeps the eval + checkpoint-save
    # gap between epochs from reading as a stall.
    worst = None
    for prev, cur in zip(run.progress, run.progress[1:]):
        if prev.epoch != cur.epoch:
            continue
        if prev.timestamp is None or cur.timestamp is None:
            continue
        gap = (cur.timestamp - prev.timestamp).total_seconds()
        if gap >= stall_seconds and (worst is None or gap > worst[0]):
            worst = (gap, prev, cur)
    if worst is not None:
        gap, prev, cur = worst
        findings.append(Finding(
            code="stall", severity="warning",
            message=f"training stalled for {gap / 3600.0:.2f} h mid-epoch",
            line_no=cur.line_no,
            detail=(f"epoch {prev.epoch} step {prev.step} -> {cur.step} "
                    f"took {gap:.0f}s")))

    # -- ap_collapse: AP fell to 0 after a healthy epoch ---------------------
    seen_healthy = False
    for rec in run.epochs:
        best = rec.best_ap
        if best is None:
            continue
        if best > 0.1:
            seen_healthy = True
        elif seen_healthy and best <= 0.0:
            findings.append(Finding(
                code="ap_collapse", severity="fatal",
                message=(f"vali AP collapsed to {best:.3f} at epoch "
                         f"{rec.epoch} after a healthy epoch")))
            break

    # -- nan dumps without our guard: an old image still running the old path
    if run.nan_dump_lines and not any(f.code == "nan_abort" for f in findings):
        findings.append(Finding(
            code="nan_dump", severity="warning",
            message=("NaN tensor dumps present without an abort -- this image "
                     "predates the kit's NaN guard"),
            line_no=run.nan_dump_lines[0]))

    order = {"fatal": 0, "warning": 1}
    findings.sort(key=lambda f: (order.get(f.severity, 2), f.line_no or 0))
    return findings


def _fmt_hms(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    s = int(seconds)
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def format_report(run: RunLog, findings: Sequence[Finding], *,
                  num_epochs: Optional[int] = None) -> str:
    """Render a compact human-readable health summary."""
    out = []
    last = run.progress[-1] if run.progress else None
    out.append(f"lines parsed:  {run.num_lines}")
    if run.first_timestamp and run.last_timestamp:
        span = (run.last_timestamp - run.first_timestamp).total_seconds()
        out.append(f"log span:      {_fmt_hms(span)} "
                   f"({run.first_timestamp} -> {run.last_timestamp})")
    if last is not None:
        out.append(f"last progress: epoch {last.epoch} "
                   f"step {last.step}/{last.total_steps} "
                   f"loss {last.loss}")
    if run.best_stat:
        out.append(f"best so far:   epoch {run.best_stat[0]} "
                   f"AP {run.best_stat[1]:.4f}")

    if run.epochs:
        out.append("")
        out.append("  epoch  wall       AP (per eval)")
        for rec in run.epochs:
            aps = "  ".join(f"{a:.3f}" for a in rec.aps) or "-"
            out.append(f"  {rec.epoch:5d}  {_fmt_hms(rec.wall_seconds):9s}  {aps}")
        done = [r for r in run.epochs if r.wall_seconds]
        cycles = [r.cycle_seconds for r in run.epochs if r.cycle_seconds]
        if done and num_epochs:
            left = max(0, int(num_epochs) - len(done))
            # Prefer the end-to-end cycle: "Total time" is training only and
            # omits the eval, which understates a 24-epoch ETA by hours.
            if cycles:
                mean = sum(cycles) / len(cycles)
                label = "mean cycle (incl. eval)"
            else:
                mean = sum(r.wall_seconds for r in done) / len(done)
                label = "mean epoch (train only)"
            out.append(f"  {label} {_fmt_hms(mean)}; {left} left "
                       f"=> ~{_fmt_hms(mean * left)} remaining")

    out.append("")
    if findings:
        for f in findings:
            out.append(f.render())
    else:
        out.append("no findings -- run looks healthy")
    return "\n".join(out)


def health_of_lines(lines: Iterable[str], **kw):
    """Convenience: parse and diagnose in one call."""
    run = parse_log(lines)
    return run, diagnose(run, **kw)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def watch(log_fpath, *, poll: float = 60.0, num_epochs=None,
          stall_seconds: float = DEFAULT_STALL_SECONDS, emit=print,
          now=None, _max_polls=None) -> int:
    """Poll a growing log and emit one line per NEW event.

    Designed to be the ``command`` of a background monitor: every line written
    to stdout becomes one notification, so this deliberately emits only
    completed epochs and newly-appeared findings -- never raw log content.

    The whole file is re-parsed each poll rather than tracked incrementally.
    That is not the clever approach, but parsing is ~0.6 s for the largest log
    this project has ever produced (500k lines), and it makes the emitted set
    a pure function of the file, so there is no incremental-state bug to have.

    Liveness is judged by file growth: a log that stops growing for
    ``stall_seconds`` gets one ``stall_live`` event. This is the case a
    verbatim tail cannot report, because silence and health look identical.
    """
    import time
    from pathlib import Path

    now = now or (lambda: time.time())
    path = Path(str(log_fpath))
    seen_epochs = set()
    seen_findings = set()
    last_size = -1
    last_growth = now()
    stall_emitted = False
    polls = 0

    while True:
        try:
            size = path.stat().st_size
        except OSError:
            size = last_size

        if size != last_size:
            last_size = size
            last_growth = now()
            stall_emitted = False
            try:
                with open(path, errors="replace") as fh:
                    run, findings = health_of_lines(
                        fh, stall_seconds=stall_seconds)
            except OSError:
                run, findings = None, []

            if run is not None:
                for rec in run.epochs:
                    if rec.epoch in seen_epochs or rec.wall_seconds is None:
                        continue
                    seen_epochs.add(rec.epoch)
                    aps = " ".join(f"{a:.4f}" for a in rec.aps) or "no-eval"
                    line = (f"epoch {rec.epoch} done in "
                            f"{_fmt_hms(rec.wall_seconds)}  AP {aps}")
                    if num_epochs:
                        left = max(0, int(num_epochs) - (rec.epoch + 1))
                        line += (f"  ({left} left, "
                                 f"~{_fmt_hms(rec.wall_seconds * left)})")
                    emit(line)
                for f in findings:
                    key = (f.code, f.line_no)
                    if key not in seen_findings:
                        seen_findings.add(key)
                        emit(f.render())
        elif not stall_emitted and (now() - last_growth) > stall_seconds:
            stall_emitted = True
            emit(f"[WARNING] stall_live: {path.name} has not grown for "
                 f"{(now() - last_growth) / 60.0:.0f} min")

        polls += 1
        if _max_polls is not None and polls >= _max_polls:
            return 0
        time.sleep(poll)


try:
    import kwconf

    class RunHealthConfig(kwconf.Config):
        """Diagnose (or watch) the health of a DEIMv2 training run's slurm log.

        Reports per-epoch AP and wall time, and flags the failure modes that
        have cost this project GPU-days: NaN zombies, mid-epoch stalls, AP
        collapse, NCCL watchdog aborts, OOM and rank crashes.

        Examples:
            python -m kwcoco_detector_kit run-health $KCD_SLURM_LOG_DPATH/myrun-490.out
            python -m kwcoco_detector_kit run-health run-490.out --num_epochs 24
            python -m kwcoco_detector_kit run-health run-490.out --watch --poll 60
        """

        log = kwconf.Value(None, position=1, required=True,
                           help="path to the slurm .out for the run")
        num_epochs = kwconf.Value(None, help="planned epoch count; enables the "
                                             "remaining-time estimate")
        stall_seconds = kwconf.Value(
            DEFAULT_STALL_SECONDS,
            help="mid-epoch gap that counts as a stall (default 1800)")
        watch = kwconf.Value(False, isflag=True,
                             help="poll and emit one line per new event "
                                  "instead of printing a one-shot report")
        poll = kwconf.Value(60.0, help="seconds between polls in --watch mode")
        fail_on_findings = kwconf.Value(
            False, isflag=True,
            help="exit non-zero if any fatal finding is present")

        @classmethod
        def main(cls, argv=1, **kwargs):
            from pathlib import Path
            config = cls.cli(argv=argv, data=kwargs, strict=True)
            log_fpath = Path(str(config.log))
            if not log_fpath.exists():
                print(f"ERROR: no such log: {log_fpath}")
                return 2
            num_epochs = (int(config.num_epochs)
                          if config.num_epochs is not None else None)
            if config.watch:
                return watch(log_fpath, poll=float(config.poll),
                             num_epochs=num_epochs,
                             stall_seconds=float(config.stall_seconds))
            with open(log_fpath, errors="replace") as fh:
                run, findings = health_of_lines(
                    fh, stall_seconds=float(config.stall_seconds))
            print(format_report(run, findings, num_epochs=num_epochs))
            if config.fail_on_findings and any(
                    f.severity == "fatal" for f in findings):
                return 1
            return 0

    __cli__ = RunHealthConfig
except ModuleNotFoundError:  # pragma: no cover
    # kwconf is a hard dependency of the kit, so this only fires when the
    # parsing half is used standalone -- e.g. triaging an archived log on a
    # host without the kit installed. Narrow to ModuleNotFoundError so a typo
    # in the config body above surfaces as itself rather than as a missing CLI.
    __cli__ = None


def _argparse_main(argv=None) -> int:
    """Stdlib-only entry point for `python3 path/to/log_health.py <log>`.

    The kit is not pip-installed on every host that has the logs -- aiq's
    login node runs everything heavy inside Docker -- which is why
    ``follow_job.py`` is invoked by path with the system interpreter. This
    mirrors that so triage never requires a container or an install.
    """
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Diagnose the health of a DEIMv2 training run's slurm log.")
    parser.add_argument("log", help="path to the slurm .out for the run")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="planned epoch count; enables the ETA line")
    parser.add_argument("--stall_seconds", type=float,
                        default=DEFAULT_STALL_SECONDS)
    parser.add_argument("--watch", action="store_true",
                        help="poll and emit one line per new event")
    parser.add_argument("--poll", type=float, default=60.0)
    parser.add_argument("--fail_on_findings", action="store_true")
    args = parser.parse_args(argv)

    log_fpath = Path(args.log)
    if not log_fpath.exists():
        print(f"ERROR: no such log: {log_fpath}")
        return 2
    if args.watch:
        return watch(log_fpath, poll=args.poll, num_epochs=args.num_epochs,
                     stall_seconds=args.stall_seconds)
    with open(log_fpath, errors="replace") as fh:
        run, findings = health_of_lines(fh, stall_seconds=args.stall_seconds)
    print(format_report(run, findings, num_epochs=args.num_epochs))
    if args.fail_on_findings and any(f.severity == "fatal" for f in findings):
        return 1
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(_argparse_main())
