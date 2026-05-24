#!/usr/bin/env python3
"""
Apply a scheme's class-collapse to a single kwcoco file.

Mirror of build_scheme_kwcoco.py but operating on one arbitrary input/
output pair instead of a fixed train/vali/test triple. Used by
_launch_train.sh to convert a scheme-agnostic tile bundle (tiled once
from training_ready_v1/train.kwcoco.zip, with `source_category`
preserved on every annotation) into a per-scheme bundle the sweep
consumes.

Usage:

    python3 scripts/apply_scheme_to_kwcoco.py \\
        --src  /path/to/universal_tiles.kwcoco.zip \\
        --dst  /path/to/per_scheme_tiles.kwcoco.zip \\
        --scheme pup_vs_nonpup

The scheme YAML defaults to docs/class_schemes.yaml in the project.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the canonical scheme-loading + remap implementation. Keeps the
# scheme semantics in exactly one place; this script is the thin CLI
# adapter for the universal-tile pipeline.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_scheme_kwcoco import (  # noqa: E402
    load_scheme,
    remap_split,
)

DEFAULT_SCHEMES_FILE = (
    Path(__file__).resolve().parent.parent / "docs" / "class_schemes.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src", type=Path, required=True,
                        help="input kwcoco file (must have source_category on each annotation)")
    parser.add_argument("--dst", type=Path, required=True,
                        help="output kwcoco file (per-scheme)")
    parser.add_argument("--scheme", required=True,
                        help="scheme name from docs/class_schemes.yaml")
    parser.add_argument("--schemes-file", type=Path, default=DEFAULT_SCHEMES_FILE,
                        help=f"path to schemes YAML (default: {DEFAULT_SCHEMES_FILE})")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scheme = load_scheme(args.schemes_file, args.scheme)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    stats = remap_split(args.src, args.dst, scheme, dry_run=args.dry_run)
    print(json.dumps(stats, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
