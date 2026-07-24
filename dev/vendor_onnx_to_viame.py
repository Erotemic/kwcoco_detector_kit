#!/usr/bin/env python3
"""
Vendor the canonical OnnxPredictor into the VIAME plugin tree.

Why
---
The VIAME ``kwcoco_detector_kit_detector`` plugin runs inference through
``OnnxPredictor``. Rather than make VIAME depend on ``kwcoco_detector_kit``
being installed (the kit evolves fast and may break ONNX-package compatibility
on purpose), we VENDOR a copy of the predictor straight into VIAME's plugin
package. This script is the single source of truth for that copy: it stamps
provenance (kit git SHA + a content hash of the source) into a header so drift
is detectable and a resync is one command.

The CANONICAL predictor is ``kwcoco_detector_kit/predictors/onnx.py`` in THIS
repo. Edit it here; never edit the vendored copy by hand.

Usage
-----
    # (re)vendor into a VIAME checkout
    python dev/vendor_onnx_to_viame.py --viame-root ~/code/VIAME

    # CI / pre-commit: fail if the vendored copy is stale vs the canonical source
    python dev/vendor_onnx_to_viame.py --viame-root ~/code/VIAME --check
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import subprocess
import sys
from pathlib import Path

# Canonical source (relative to the kit root) and vendored destination
# (relative to the VIAME root).
SOURCE_REL = Path("kwcoco_detector_kit/predictors/onnx.py")
DEST_REL = Path("plugins/pytorch/kwcoco_detector_kit_onnx_predictor.py")


def _kit_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git(kit_root: Path, *args: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(kit_root), *args],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


def _source_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _header_comment() -> str:
    """A PURE-COMMENT banner (no statements) safe to put at the very top of the
    file -- crucially before the source's own ``from __future__`` import."""
    return f'''# ============================================================================
# VENDORED FILE -- DO NOT EDIT HERE.
#
# This is a vendored copy of the canonical OnnxPredictor from the
# kwcoco_detector_kit repository. VIAME inference uses this copy so it does NOT
# depend on kwcoco_detector_kit being importable -- the kit evolves quickly and
# may intentionally break ONNX-package compatibility, so VIAME pins/resyncs on
# purpose rather than tracking the kit live.
#
# The CANONICAL source lives in the kit at:
#     {SOURCE_REL}
# Edit it THERE, then resync this copy with the kit's vendoring tool:
#     # in the kwcoco_detector_kit repo:
#     python dev/vendor_onnx_to_viame.py --viame-root /path/to/VIAME
#
# Provenance is recorded in the importable ``__vendored_provenance__`` dict
# below. ``source_sha256`` is the hash of the canonical file's body; ``--check``
# recomputes it to detect drift.
# ============================================================================
'''


def _provenance_block(*, kit_root: Path, source_text: str, when: str) -> str:
    sha = _git(kit_root, "rev-parse", "HEAD") or "unknown"
    dirty = bool(_git(kit_root, "status", "--porcelain", str(SOURCE_REL)))
    src_hash = _source_sha256(source_text)
    return f'''
__vendored_provenance__ = {{
    "source_repo": "kwcoco_detector_kit",
    "source_path": "{SOURCE_REL}",
    "kit_git_sha": "{sha}",
    "kit_git_dirty": {dirty!r},
    "source_sha256": "{src_hash}",
    "vendored_at": "{when}",
    "vendor_tool": "dev/vendor_onnx_to_viame.py",
}}
'''


def _assemble(*, header: str, prov: str, source_text: str) -> str:
    """Place the comment banner first, keep the source's docstring + any
    ``from __future__`` import in their required leading position, then inject
    the provenance dict immediately after the last ``from __future__`` import
    (or after the leading import block if there is none)."""
    lines = source_text.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("from __future__"):
            insert_at = i + 1
    if insert_at == 0:
        # No __future__ import: insert after a leading module docstring if any,
        # else at the very top of the body.
        body = source_text.lstrip()
        if body.startswith(('"""', "'''")):
            quote = body[:3]
            end = source_text.find(quote, source_text.find(quote) + 3) + 3
            return header + source_text[:end] + "\n" + prov + source_text[end:]
        return header + prov + source_text
    return header + "".join(lines[:insert_at]) + prov + "".join(lines[insert_at:])


def _stamped_sha_of(dest_text: str) -> str | None:
    for line in dest_text.splitlines():
        s = line.strip()
        if s.startswith('"source_sha256":'):
            return s.split('"')[3]
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viame-root", required=True,
                    help="path to the VIAME checkout to vendor into")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the vendored copy is stale (no write)")
    args = ap.parse_args(argv)

    kit_root = _kit_root()
    source = kit_root / SOURCE_REL
    dest = Path(args.viame_root).expanduser() / DEST_REL

    if not source.is_file():
        print(f"ERROR: canonical source not found: {source}", file=sys.stderr)
        return 2

    source_text = source.read_text()
    src_hash = _source_sha256(source_text)

    if args.check:
        if not dest.is_file():
            print(f"STALE: vendored copy missing: {dest}", file=sys.stderr)
            return 1
        stamped = _stamped_sha_of(dest.read_text())
        if stamped != src_hash:
            print(
                f"STALE: vendored copy is out of date.\n"
                f"  canonical sha256: {src_hash}\n"
                f"  vendored  sha256: {stamped}\n"
                f"  resync: python dev/vendor_onnx_to_viame.py --viame-root {args.viame_root}",
                file=sys.stderr,
            )
            return 1
        print(f"OK: vendored copy is current ({dest})")
        return 0

    when = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = _header_comment()
    prov = _provenance_block(kit_root=kit_root, source_text=source_text, when=when)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(_assemble(header=header, prov=prov, source_text=source_text))
    print(f"vendored {SOURCE_REL} -> {dest}")
    print(f"  kit_git_sha   : {_git(kit_root, 'rev-parse', '--short', 'HEAD')}")
    print(f"  source_sha256 : {src_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
