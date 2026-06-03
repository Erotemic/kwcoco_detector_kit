#!/usr/bin/env python3
"""Resource utilization snapshot for kit training jobs.

Focuses on the metrics that distinguish a healthy run from a thrashing
one — especially disk I/O, which most operators monitor poorly:

  * Per-training-process CPU%, RSS, disk read/write rate, GPU memory.
  * System CPU + RAM + page cache + disk r/w + GPU utilization.
  * The diagnostic ratio: high disk read + low GPU util = data-starved.

By default it auto-discovers training procs by matching a glob against
the cmdline (defaults catch DEIMv2 / kit sweep launches). Pass
``--pattern`` to scope to a specific job, or ``--pids`` to monitor
exact PIDs.

Usage:

    # one-shot snapshot
    python dev/check_resources.py

    # continuous, 2s refresh, until Ctrl-C
    python dev/check_resources.py --watch 2

    # scope to one job
    python dev/check_resources.py --pattern pup_vs_nonpup

    # explicit PIDs
    python dev/check_resources.py --pids 12345,12346
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import psutil
except ImportError:
    sys.stderr.write(
        "psutil is required. uv pip install psutil\n"
    )
    sys.exit(1)


DEFAULT_PATTERNS = (
    "tpl/DEIMv2/train.py",
    "kwcoco_detector_kit sweep",
    "torch.distributed.run",
)


# -------------------- per-process sampling -----------------------


@dataclass
class ProcSample:
    pid: int
    cmdline: str
    cpu_pct: float = 0.0
    rss_mb: float = 0.0
    read_mb_s: float = 0.0
    write_mb_s: float = 0.0
    nthreads: int = 0
    gpu_index: Optional[int] = None
    gpu_mem_mb: float = 0.0


@dataclass
class ProcSampler:
    """Holds last-IO-counter values to compute deltas."""
    last_read_bytes: Dict[int, int] = field(default_factory=dict)
    last_write_bytes: Dict[int, int] = field(default_factory=dict)
    last_time: float = 0.0

    def sample(self, procs: List[psutil.Process]) -> List[ProcSample]:
        now = time.monotonic()
        dt = max(0.001, now - self.last_time)
        out: List[ProcSample] = []
        for p in procs:
            try:
                with p.oneshot():
                    io = p.io_counters()
                    rb = io.read_bytes
                    wb = io.write_bytes
                    s = ProcSample(
                        pid=p.pid,
                        cmdline=_short_cmdline(p),
                        cpu_pct=p.cpu_percent(interval=None),
                        rss_mb=p.memory_info().rss / 1e6,
                        nthreads=p.num_threads(),
                    )
                    if p.pid in self.last_read_bytes and self.last_time:
                        s.read_mb_s = (rb - self.last_read_bytes[p.pid]) / dt / 1e6
                        s.write_mb_s = (wb - self.last_write_bytes[p.pid]) / dt / 1e6
                    self.last_read_bytes[p.pid] = rb
                    self.last_write_bytes[p.pid] = wb
                    out.append(s)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.last_time = now
        return out


def _short_cmdline(p: psutil.Process, maxlen: int = 60) -> str:
    try:
        cmd = " ".join(p.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return f"<pid {p.pid}>"
    # collapse long paths to basename
    parts = cmd.split(" ")
    for i, part in enumerate(parts):
        if "/" in part and len(part) > 30:
            parts[i] = ".../" + os.path.basename(part)
    cmd = " ".join(parts)
    if len(cmd) > maxlen:
        cmd = cmd[: maxlen - 1] + "…"
    return cmd


# -------------------- GPU sampling -----------------------


def sample_gpu() -> Dict[int, Dict[str, float]]:
    """Returns gpu_index -> {util%, mem_mb_used, mem_mb_total}.

    Empty dict if nvidia-smi isn't available.
    """
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return {}
    try:
        out = subprocess.check_output(
            [
                nvsmi,
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}
    info: Dict[int, Dict[str, float]] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            idx, util, used, total = (
                int(parts[0]),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
            )
        except ValueError:
            continue
        info[idx] = {"util": util, "used": used, "total": total}
    return info


def sample_gpu_procs() -> Dict[int, int]:
    """Returns gpu_index -> pid for processes using each GPU."""
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return {}
    try:
        out = subprocess.check_output(
            [
                nvsmi,
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}
    # gpu_uuid → index mapping
    try:
        idx_out = subprocess.check_output(
            [nvsmi, "--query-gpu=uuid,index", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}
    uuid_to_idx = {}
    for line in idx_out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            uuid_to_idx[parts[0]] = int(parts[1])

    pid_to_gpu: Dict[int, int] = {}
    pid_to_mem: Dict[int, float] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        uuid, pid_s, mem_s = parts
        try:
            pid = int(pid_s)
            mem = float(mem_s)
        except ValueError:
            continue
        if uuid in uuid_to_idx:
            pid_to_gpu[pid] = uuid_to_idx[uuid]
            pid_to_mem[pid] = mem
    return {"gpu_index": pid_to_gpu, "gpu_mem_mb": pid_to_mem}


# -------------------- system sampling --------------------


@dataclass
class SysSample:
    cpu_pct: float
    ram_used_gb: float
    ram_total_gb: float
    ram_cache_gb: float
    disk_read_mb_s: float
    disk_write_mb_s: float
    # Linux PSI (Pressure Stall Information): % of time tasks are
    # stalled on I/O / CPU / memory. 0 = no pressure, >10% in avg10
    # is meaningful, >20% in avg60 is real congestion.
    io_pressure_some_10: float = 0.0
    io_pressure_some_60: float = 0.0
    cpu_pressure_some_10: float = 0.0
    mem_pressure_some_10: float = 0.0


class SysSampler:
    def __init__(self) -> None:
        self.last_io = psutil.disk_io_counters()
        self.last_time = time.monotonic()

    def sample(self) -> SysSample:
        now = time.monotonic()
        dt = max(0.001, now - self.last_time)
        io = psutil.disk_io_counters()
        if io is None or self.last_io is None:
            r = w = 0.0
        else:
            r = (io.read_bytes - self.last_io.read_bytes) / dt / 1e6
            w = (io.write_bytes - self.last_io.write_bytes) / dt / 1e6
        self.last_io = io
        self.last_time = now

        mem = psutil.virtual_memory()
        # Linux only: page cache is `cached`. On other OSes we surface 0.
        cache = getattr(mem, "cached", 0)

        # PSI — direct measure of how much time tasks waited on
        # a resource (cgroup-aware kernel metric, kernel >= 4.20).
        # Far more meaningful than throughput numbers for "is the
        # system pressured?" Read once per sample; cheap.
        io_some_10, io_some_60 = _read_pressure("io", "some")
        cpu_some_10, _ = _read_pressure("cpu", "some")
        mem_some_10, _ = _read_pressure("memory", "some")

        return SysSample(
            cpu_pct=psutil.cpu_percent(interval=None),
            ram_used_gb=(mem.total - mem.available) / 1e9,
            ram_total_gb=mem.total / 1e9,
            ram_cache_gb=cache / 1e9,
            disk_read_mb_s=r,
            disk_write_mb_s=w,
            io_pressure_some_10=io_some_10,
            io_pressure_some_60=io_some_60,
            cpu_pressure_some_10=cpu_some_10,
            mem_pressure_some_10=mem_some_10,
        )


def _read_pressure(resource: str, line_prefix: str) -> tuple[float, float]:
    """Parse /proc/pressure/<resource> for the given line ('some' or 'full').

    Returns (avg10, avg60). Both 0.0 if the file isn't readable
    (older kernels, non-Linux, restricted containers).
    """
    try:
        with open(f"/proc/pressure/{resource}") as f:
            for line in f:
                if line.startswith(line_prefix):
                    parts = line.split()
                    avg10 = float(parts[1].split("=")[1])
                    avg60 = float(parts[2].split("=")[1])
                    return avg10, avg60
    except (FileNotFoundError, PermissionError, IndexError, ValueError):
        pass
    return 0.0, 0.0


# -------------------- discovery + rendering -----------------------


def find_procs(pattern_argv: List[str], pids: List[int]) -> List[psutil.Process]:
    if pids:
        out = []
        for pid in pids:
            try:
                out.append(psutil.Process(pid))
            except psutil.NoSuchProcess:
                continue
        return out
    pats = pattern_argv or list(DEFAULT_PATTERNS)
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any(pat in cmd for pat in pats):
            out.append(p)
    return out


def render(sys_s: SysSample, procs: List[ProcSample], gpus: Dict[int, dict]) -> str:
    lines = []
    lines.append(
        f"  SYS  cpu={sys_s.cpu_pct:5.1f}%   "
        f"ram={sys_s.ram_used_gb:5.1f} / {sys_s.ram_total_gb:5.1f} GB "
        f"(cache={sys_s.ram_cache_gb:5.1f} GB)   "
        f"disk r={sys_s.disk_read_mb_s:7.2f} MB/s  "
        f"w={sys_s.disk_write_mb_s:6.2f} MB/s"
    )
    # PSI line — only show if any pressure exists, else it's noise.
    if (sys_s.io_pressure_some_10 or sys_s.cpu_pressure_some_10
            or sys_s.mem_pressure_some_10):
        lines.append(
            f"  PSI  io  10s={sys_s.io_pressure_some_10:5.1f}%  60s={sys_s.io_pressure_some_60:5.1f}%   "
            f"cpu 10s={sys_s.cpu_pressure_some_10:5.1f}%   "
            f"mem 10s={sys_s.mem_pressure_some_10:5.1f}%   "
            f"(some-stall %)"
        )
    if gpus:
        for idx, g in sorted(gpus.items()):
            lines.append(
                f"  GPU{idx} util={g['util']:5.1f}%  mem={g['used']:6.0f} / {g['total']:6.0f} MB"
            )
    if not procs:
        lines.append("  no training procs match the pattern")
        return "\n".join(lines)
    lines.append("")
    lines.append(
        f"  {'PID':>7}  {'CPU%':>6}  {'RSS':>9}  "
        f"{'disk_r':>10}  {'disk_w':>10}  {'gpu':>3}  "
        f"{'gpu_mem':>8}  cmd"
    )
    lines.append("  " + "-" * 92)
    for p in sorted(procs, key=lambda x: x.pid):
        lines.append(
            f"  {p.pid:>7}  {p.cpu_pct:>5.1f}%  {p.rss_mb:>6.0f} MB  "
            f"{p.read_mb_s:>6.2f} MB/s  {p.write_mb_s:>6.2f} MB/s  "
            f"{p.gpu_index if p.gpu_index is not None else '-':>3}  "
            f"{p.gpu_mem_mb:>5.0f} MB  {p.cmdline}"
        )
    return "\n".join(lines)


def diagnose(sys_s: SysSample, procs: List[ProcSample], gpus: Dict[int, dict]) -> List[str]:
    notes = []
    # Disk-starved GPU
    if gpus and procs:
        active_gpus = [g for g in gpus.values() if g["used"] > 1000]
        if active_gpus:
            avg_util = sum(g["util"] for g in active_gpus) / len(active_gpus)
            total_disk_r = sys_s.disk_read_mb_s
            if avg_util < 30 and total_disk_r > 50:
                notes.append(
                    f"data-starved GPU? avg util={avg_util:.0f}% but "
                    f"disk read={total_disk_r:.1f} MB/s — workers may be IO-bound"
                )
            elif avg_util < 30 and total_disk_r < 5:
                notes.append(
                    f"low GPU util ({avg_util:.0f}%) AND low disk ({total_disk_r:.1f} MB/s) "
                    "— page cache serving warm, GPU is the bottleneck (consider larger "
                    "batch / model)"
                )
    # RAM pressure
    ram_free_gb = sys_s.ram_total_gb - sys_s.ram_used_gb
    if ram_free_gb < 5:
        notes.append(
            f"RAM nearly full ({ram_free_gb:.1f} GB free) — page cache may be evicting"
        )
    # CPU saturation
    if sys_s.cpu_pct > 90:
        notes.append(
            f"CPU saturated ({sys_s.cpu_pct:.0f}%) — dataloader workers may be "
            "queuing; check `data:` field in train iter prints"
        )
    # PSI thresholds — meaningful signal that goes beyond just %util.
    if sys_s.io_pressure_some_60 > 20:
        notes.append(
            f"sustained I/O pressure (PSI io some 60s={sys_s.io_pressure_some_60:.1f}%) "
            "— tasks have been stalled on disk; check page cache fit + shard storage"
        )
    elif sys_s.io_pressure_some_10 > 30:
        notes.append(
            f"recent I/O pressure spike (PSI io some 10s={sys_s.io_pressure_some_10:.1f}%) — "
            "may settle if it was a one-off (checkpoint save?)"
        )
    if sys_s.mem_pressure_some_10 > 10:
        notes.append(
            f"memory pressure (PSI mem some 10s={sys_s.mem_pressure_some_10:.1f}%) — "
            "page cache thrashing, reduce per-job RAM use or workers"
        )
    return notes


# -------------------- main -----------------------


def main():
    ap = argparse.ArgumentParser(
        description="Resource snapshot for kit training jobs."
    )
    ap.add_argument(
        "--pattern", action="append", default=[],
        help="cmdline substring; can pass multiple. Default matches DEIMv2 + kit sweep + torch.distributed.run.",
    )
    ap.add_argument(
        "--pids", default="",
        help="comma-separated PIDs to monitor explicitly",
    )
    ap.add_argument(
        "--watch", type=float, default=0,
        help="refresh interval in seconds; 0 = one-shot",
    )
    args = ap.parse_args()

    pids = [int(p) for p in args.pids.split(",") if p.strip()] if args.pids else []

    proc_sampler = ProcSampler()
    sys_sampler = SysSampler()

    # prime cpu_percent counters so the first reading isn't 0
    psutil.cpu_percent(interval=None)
    procs = find_procs(args.pattern, pids)
    for p in procs:
        try:
            p.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if args.watch > 0:
        try:
            while True:
                time.sleep(args.watch)
                _emit(args, proc_sampler, sys_sampler, pids, header=True)
        except KeyboardInterrupt:
            pass
    else:
        time.sleep(0.5)   # short window so cpu_percent is meaningful
        _emit(args, proc_sampler, sys_sampler, pids, header=False)


def _emit(args, proc_sampler, sys_sampler, pids, header: bool):
    procs = find_procs(args.pattern, pids)
    proc_samples = proc_sampler.sample(procs)
    sys_sample = sys_sampler.sample()
    gpus = sample_gpu()
    gpu_pid_info = sample_gpu_procs()
    pid_to_gpu_idx = gpu_pid_info.get("gpu_index", {})
    pid_to_gpu_mem = gpu_pid_info.get("gpu_mem_mb", {})
    for s in proc_samples:
        s.gpu_index = pid_to_gpu_idx.get(s.pid)
        s.gpu_mem_mb = pid_to_gpu_mem.get(s.pid, 0.0)

    if header:
        print(f"\n=== {time.strftime('%H:%M:%S')} ===")
    print(render(sys_sample, proc_samples, gpus))
    notes = diagnose(sys_sample, proc_samples, gpus)
    if notes:
        print()
        for n in notes:
            print(f"  [hint] {n}")


if __name__ == "__main__":
    main()
