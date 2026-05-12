#!/usr/bin/env python
"""
Follow a Slurm job's stdout until the job finishes.

This intentionally stays dependency-free so it can run on a login node before
the Docker image starts. It makes ``sbatch`` feel close to a foreground command:
print the log as it is written, poll Slurm for liveness, and return the job's
final exit code when Slurm accounting exposes it.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple


TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}

FAILURE_STATES = TERMINAL_STATES - {"COMPLETED"}


def run_text(cmd, *, check=False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def parse_show_job(text: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for part in re.split(r"\s+", text.strip()):
        if "=" in part:
            key, value = part.split("=", 1)
            data[key] = value
    job_name = data.get("JobName", "")
    job_id = data.get("JobId", "")
    for key in ["StdOut", "StdErr"]:
        if key in data:
            data[key] = data[key].replace("%x", job_name).replace("%j", job_id)
    return data


def scontrol_job_info(jobid: str) -> Dict[str, str]:
    proc = run_text(["scontrol", "show", "job", str(jobid)])
    if proc.returncode != 0:
        return {}
    if "Invalid job id" in proc.stdout or "slurm_load_jobs error" in proc.stderr:
        return {}
    return parse_show_job(proc.stdout)


def squeue_state(jobid: str) -> Optional[str]:
    proc = run_text(["squeue", "-h", "-j", str(jobid), "-o", "%T"])
    if proc.returncode != 0:
        return None
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def sacct_state_exit(jobid: str) -> Tuple[Optional[str], Optional[int]]:
    proc = run_text([
        "sacct",
        "-n",
        "-P",
        "-j",
        str(jobid),
        "--format=JobID,State,ExitCode",
    ])
    if proc.returncode != 0:
        return None, None
    best_state = None
    best_exit = None
    for line in proc.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue
        jid, state, exit_code = parts[:3]
        if "." in jid:
            continue
        best_state = state.split()[0]
        try:
            best_exit = int(exit_code.split(":", 1)[0])
        except ValueError:
            best_exit = None
    return best_state, best_exit


class Tailer:
    def __init__(self, fpath: Path):
        self.fpath = fpath
        self.file = None
        self.missing_announced = False

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    def read_available(self) -> bool:
        if self.file is None:
            if not self.fpath.exists():
                if not self.missing_announced:
                    print(f"[slurm-follow] waiting for log: {self.fpath}", file=sys.stderr)
                    self.missing_announced = True
                return False
            self.file = self.fpath.open("rb")
        emitted = False
        while True:
            where = self.file.tell()
            line = self.file.readline()
            if not line:
                self.file.seek(where)
                break
            sys.stdout.write(line.decode("utf8", errors="replace"))
            sys.stdout.flush()
            emitted = True
        return emitted


def infer_stdout_path(jobid: str, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        return Path(explicit).expanduser()
    info = scontrol_job_info(jobid)
    stdout = info.get("StdOut")
    if stdout and stdout != "/dev/null":
        return Path(stdout).expanduser()
    return None


def follow(jobid: str, *, stdout_path: Optional[str] = None,
           poll_interval: float = 2.0, grace_polls: int = 3) -> int:
    jobid = str(jobid).split(";", 1)[0]
    log_fpath = infer_stdout_path(jobid, stdout_path)
    if log_fpath is None:
        print(f"[slurm-follow] could not determine StdOut for job {jobid}", file=sys.stderr)
        return 2

    print(f"[slurm-follow] job {jobid}", file=sys.stderr)
    print(f"[slurm-follow] log {log_fpath}", file=sys.stderr)

    absent_polls = 0
    last_state = None
    tailer = Tailer(log_fpath)
    try:
        while True:
            emitted = tailer.read_available()
            state = squeue_state(jobid)
            if state:
                absent_polls = 0
                if state != last_state:
                    print(f"[slurm-follow] state {state}", file=sys.stderr)
                    last_state = state
            else:
                absent_polls += 1
                sacct_state, sacct_exit = sacct_state_exit(jobid)
                if sacct_state:
                    tailer.read_available()
                    print(f"[slurm-follow] final state {sacct_state}", file=sys.stderr)
                    if sacct_exit is not None:
                        return int(sacct_exit)
                    return 1 if sacct_state in FAILURE_STATES else 0
                if absent_polls >= grace_polls:
                    tailer.read_available()
                    print("[slurm-follow] job left squeue; sacct did not report a final state",
                          file=sys.stderr)
                    return 0
            if not emitted:
                time.sleep(poll_interval)
    finally:
        tailer.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobid", help="Slurm job id returned by sbatch --parsable")
    parser.add_argument("--stdout", help="Known stdout path; skips scontrol path lookup")
    parser.add_argument("--poll", type=float, default=float(os.environ.get("KCD_SLURM_FOLLOW_POLL", "2.0")))
    args = parser.parse_args(argv)
    return follow(args.jobid, stdout_path=args.stdout, poll_interval=args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
