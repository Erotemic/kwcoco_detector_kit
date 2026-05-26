"""
Provenance capture for trained-artifact traceability.

Every trained workdir + eval-metrics file should be able to answer
"which version of the kit + DEIMv2 + Open-GroundingDino produced this
artifact?" at a single glance, without depending on shell history or
external bookkeeping.

This module is the **single source of truth** for that information.
Callers read ``provenance_dict()`` and stamp the result into their
output JSONs (policy.json, detect_metrics.json, etc.).

Resolution order for the kit SHA:

  1. ``$KCD_PROVENANCE_KIT_SHA`` env var (set by docker build via
     a label-derived /etc/kcd_provenance.json).
  2. ``/etc/kcd_provenance.json`` file (read once, cached).
  3. ``git -C <kit_root> rev-parse HEAD`` of the editable install.
  4. ``"<unknown>"`` if everything fails.

Same fallback chain for the DEIMv2 and Open-GroundingDino submodules.
"""
from __future__ import annotations

import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


_PROVENANCE_FILE = Path("/etc/kcd_provenance.json")


@lru_cache(maxsize=1)
def _read_provenance_file() -> Dict[str, Any]:
    """Read /etc/kcd_provenance.json (written by docker build) if present."""
    if _PROVENANCE_FILE.exists():
        try:
            return json.loads(_PROVENANCE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _git_rev(repo: Path) -> Optional[str]:
    """`git rev-parse HEAD` for `repo` if it's a git working tree.

    Uses ``-c safe.directory=*`` so bind-mounted repos in docker (which
    have a UID mismatch with the in-container root user) don't trip
    git's dubious-ownership guard.
    """
    if not repo.exists():
        return None
    try:
        out = subprocess.check_output(
            ["git", "-c", "safe.directory=*", "-C", str(repo),
             "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.strip()
    except Exception:
        return None


def _git_dirty(repo: Path) -> Optional[bool]:
    """True if `repo` has uncommitted changes. None if not a git repo."""
    if not repo.exists():
        return None
    try:
        out = subprocess.check_output(
            ["git", "-c", "safe.directory=*", "-C", str(repo),
             "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return bool(out.strip())
    except Exception:
        return None


@lru_cache(maxsize=1)
def _kit_repo_root() -> Optional[Path]:
    """The editable install root of kwcoco_detector_kit, or None."""
    try:
        import kwcoco_detector_kit
        pkg_init = Path(kwcoco_detector_kit.__file__).resolve()
        candidate = pkg_init.parent.parent
        if (candidate / ".git").exists():
            return candidate
        return None
    except Exception:
        return None


def _resolve(env_key: str, file_key: str, fallback_repo: Optional[Path]) -> str:
    """env -> file -> git fallback -> '<unknown>'."""
    if (v := os.environ.get(env_key)):
        return v
    if (v := _read_provenance_file().get(file_key)):
        return v
    if fallback_repo and (v := _git_rev(fallback_repo)):
        return v
    return "<unknown>"


def provenance_dict() -> Dict[str, Any]:
    """Return a dict describing the kit + submodule SHAs.

    Keys (always present, "<unknown>" if not resolvable):

      kit_sha            kwcoco_detector_kit HEAD
      kit_dirty          uncommitted changes flag (True/False/None)
      deimv2_sha         tpl/DEIMv2 HEAD
      opengroundingdino_sha   tpl/Open-GroundingDino HEAD
      source             where the SHAs came from
                         ('env' | 'file' | 'git' | 'mixed' | 'unknown')

    Callers should stamp this verbatim into their output JSON under a
    "provenance" key.
    """
    kit_root = _kit_repo_root()
    deimv2 = (kit_root / "tpl" / "DEIMv2") if kit_root else None
    ogdino = (kit_root / "tpl" / "Open-GroundingDino") if kit_root else None

    # Track where each value came from so we can surface confusing
    # situations (e.g. file says X but git says Y).
    sources = set()
    def _resolve_with_src(env_key, file_key, repo):
        if (v := os.environ.get(env_key)):
            sources.add("env")
            return v
        if (v := _read_provenance_file().get(file_key)):
            sources.add("file")
            return v
        if repo and (v := _git_rev(repo)):
            sources.add("git")
            return v
        sources.add("unknown")
        return "<unknown>"

    kit_sha = _resolve_with_src("KCD_PROVENANCE_KIT_SHA", "kit_sha", kit_root)
    deimv2_sha = _resolve_with_src("KCD_PROVENANCE_DEIMV2_SHA", "deimv2_sha", deimv2)
    ogdino_sha = _resolve_with_src("KCD_PROVENANCE_OGDINO_SHA", "opengroundingdino_sha", ogdino)

    if len(sources - {"unknown"}) > 1:
        src = "mixed"
    elif sources:
        src = next(iter(sources))
    else:
        src = "unknown"

    kit_dirty = _git_dirty(kit_root) if kit_root else None
    deimv2_dirty = _git_dirty(deimv2) if deimv2 else None
    ogdino_dirty = _git_dirty(ogdino) if ogdino else None

    return {
        "kit_sha": kit_sha,
        "kit_dirty": kit_dirty,
        "deimv2_sha": deimv2_sha,
        "deimv2_dirty": deimv2_dirty,
        "opengroundingdino_sha": ogdino_sha,
        "opengroundingdino_dirty": ogdino_dirty,
        "source": src,
    }


def stamp_into(d: Dict[str, Any], *, key: str = "provenance") -> Dict[str, Any]:
    """Mutate `d` in place to add a provenance entry; also return it."""
    d[key] = provenance_dict()
    return d
