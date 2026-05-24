#!/usr/bin/env python3
"""
Spot-check one image name across local CSV and kwcoco annotation files.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def find_csv_rows(repo: Path, image_name: str) -> list[tuple[Path, dict]]:
    rows = []
    for root in ['IncludesNewAnnotations', 'Redundant']:
        for fpath in sorted((repo / root).glob('*.csv')):
            with fpath.open(newline='') as file:
                reader = csv.DictReader(file)
                if 'IMAGE' not in (reader.fieldnames or []):
                    continue
                for row in reader:
                    if row.get('IMAGE') == image_name:
                        rows.append((fpath, row))
    return rows


def candidate_image_names(image_name: str) -> list[str]:
    name = Path(image_name).name
    candidates = [name]
    stripped = re.sub(r'^\d+_[^_]+_', '', name)
    if stripped != name:
        candidates.append(stripped)
    stripped = re.sub(r'\.view_(ann|img)\.jpg$', '', stripped)
    if stripped not in candidates:
        candidates.append(stripped)
    return candidates


def print_csv_rows(rows: list[tuple[Path, dict]]) -> None:
    print('CSV rows:')
    if not rows:
        print('  none')
        return
    for fpath, row in rows:
        tl_x = float(row['TL_X'])
        tl_y = float(row['TL_Y'])
        br_x = float(row['BR_X'])
        br_y = float(row['BR_Y'])
        xywh = [tl_x, tl_y, br_x - tl_x, br_y - tl_y]
        attr = row.get('ATTRIBUTE', '')
        print(
            f'  {fpath}: id={row.get("ID")} class={row.get("CLASS")} '
            f'xyxy={[tl_x, tl_y, br_x, br_y]} xywh={xywh} attr={attr!r}'
        )


def print_image_files(repo: Path, image_name: str) -> None:
    print('Image files:')
    matches = [
        path for path in repo.rglob(image_name)
        if path.is_file() and '.venv' not in path.parts
    ]
    if not matches:
        print('  none')
    for path in matches:
        print(f'  {path} ({path.stat().st_size} bytes)')


def print_kwcoco_rows(repo: Path, image_name: str, kwcoco_paths: list[Path]) -> None:
    try:
        import kwcoco
    except ImportError:
        print('kwcoco rows: skipped, kwcoco is not importable')
        return

    print('kwcoco rows:')
    any_rows = False
    for fpath in kwcoco_paths:
        if not fpath.exists():
            continue
        dset = kwcoco.CocoDataset(fpath)
        gids = [
            gid for gid, img in dset.index.imgs.items()
            if img.get('name') == image_name or Path(img.get('file_name', '')).name == image_name
        ]
        if not gids:
            continue
        any_rows = True
        print(f'  dataset={fpath}')
        for gid in gids:
            img = dset.index.imgs[gid]
            print(f'    image gid={gid} file={img.get("file_name")} size={img.get("width")}x{img.get("height")}')
            for aid in dset.index.gid_to_aids[gid]:
                ann = dset.index.anns[aid]
                cat = dset.index.cats[ann['category_id']]['name']
                attr = ann.get('viame', {}).get('attribute', '')
                print(f'      aid={aid} class={cat} bbox={ann.get("bbox")} attr={attr!r}')
    if not any_rows:
        print('  none')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image_name')
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--kwcoco', type=Path, nargs='*', default=[
        Path('sealions_2021_2024.kwcoco.zip'),
        Path('sealions_2021_2024_sample40.kwcoco.zip'),
        Path('training_ready_v1/all_collapsed.kwcoco.zip'),
    ])
    args = parser.parse_args()

    repo = args.repo.resolve()
    image_names = candidate_image_names(args.image_name)
    kwcoco_paths = [path if path.is_absolute() else repo / path for path in args.kwcoco]

    print(f'input_image_name={Path(args.image_name).name}')
    print(f'candidate_image_names={image_names}')
    for image_name in image_names:
        print(f'\n== {image_name} ==')
        print_image_files(repo, image_name)
        print_csv_rows(find_csv_rows(repo, image_name))
        print_kwcoco_rows(repo, image_name, kwcoco_paths)


if __name__ == '__main__':
    main()
