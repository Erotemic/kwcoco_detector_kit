#!/usr/bin/env python3
"""Unpack configured redacted sea lion source zips into unpacked/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sealion_pipeline import REDACTED_SOURCES, unpack_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--unpacked', type=Path, default=Path('unpacked'))
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    repo = args.repo.resolve()
    unpacked = args.unpacked if args.unpacked.is_absolute() else repo / args.unpacked
    reports = []
    for spec in REDACTED_SOURCES:
        reports.append(unpack_source(repo, unpacked, spec, force=args.force))
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
