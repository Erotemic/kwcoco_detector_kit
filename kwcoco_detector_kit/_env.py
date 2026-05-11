"""
Cross-module env / sys-level helpers.

Centralised so they're easy to mock in tests:

- RLIMIT_NOFILE raise + clamp (failure #15).
- PYTHONPATH idempotent prepend (failure section in prior project's setup_env.sh).
- CUDA visibility default (failure #17, PCIe-link-width mismatch).
"""
from __future__ import annotations

import os
import resource
import sys
from pathlib import Path
from typing import Iterable, Optional


def raise_nofile_limit(target: int = 65536) -> tuple[int, int, str]:
    """Raise the soft FD limit to `target`, clamped to the kernel hard cap.

    Returns ``(soft_before, soft_after, status)``. Status is one of:

      ``'raised'``      target reached or exceeded.
      ``'clamped'``     hard cap < target; soft raised to hard cap.
      ``'no_change'``   soft already >= target.
      ``'failed'``      OSError raising — soft unchanged.

    Failure #15: non-root users can't raise the soft limit above ``ulimit -Hn``.
    The right escape is documented as ``KCD_TORCH_MP_SHARING=file_system`` —
    switches torch IPC from FD-per-tensor to filesystem-backed shared memory.
    """
    try:
        soft_before, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ValueError, OSError):
        return (-1, -1, "failed")

    if soft_before >= int(target):
        return (soft_before, soft_before, "no_change")

    soft_after = min(int(target), hard)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft_after, hard))
    except (ValueError, OSError):
        return (soft_before, soft_before, "failed")

    status = "raised" if soft_after >= int(target) else "clamped"
    return (soft_before, soft_after, status)


def prepend_pythonpath(paths: Iterable[str | Path]) -> str:
    """Prepend `paths` to PYTHONPATH idempotently — no duplicate entries.

    Mirrors the prior project's setup_env.sh idempotent PYTHONPATH prepend.
    Sourcing twice should not duplicate entries.
    """
    existing = os.environ.get("PYTHONPATH", "")
    existing_parts = [p for p in existing.split(os.pathsep) if p]
    new_parts: list[str] = []
    for p in paths:
        s = str(Path(p).expanduser().resolve())
        if s and s not in existing_parts and s not in new_parts:
            new_parts.append(s)
    combined = os.pathsep.join(new_parts + existing_parts)
    os.environ["PYTHONPATH"] = combined
    for s in reversed(new_parts):
        if s not in sys.path:
            sys.path.insert(0, s)
    return combined


def default_cuda_visible_devices() -> Optional[str]:
    """Return the CUDA_VISIBLE_DEVICES the kit should default to on this host.

    For single-host non-cluster setups with a known PCIe-link-width mismatch
    (failure #17), DDP all-reduce is bottlenecked by the slowest peer. The
    safe default is to expose only the highest-bandwidth GPU. Callers that
    want explicit multi-GPU set the env var before invoking the trainer.

    Returns:
        ``"0"`` if not already set; ``None`` if already set (respect user).
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return None
    return "0"
