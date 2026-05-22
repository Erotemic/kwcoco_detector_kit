#!/usr/bin/env python3
"""Convert already-unpacked redacted sea lion sources to raw/norm kwcoco."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sealion_pipeline import build_combined, build_per_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--unpacked', type=Path, default=Path('unpacked'))
    parser.add_argument('--force-convert', action='store_true')
    parser.add_argument('--seed', type=int, default=20260514)
    args = parser.parse_args()

    repo = args.repo.resolve()
    unpacked = args.unpacked if args.unpacked.is_absolute() else repo / args.unpacked
    report = {
        'per_source': build_per_source(repo, unpacked, force_unpack=False, force_convert=args.force_convert),
        'combined': build_combined(unpacked, seed=args.seed),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
