#!/usr/bin/env python3
"""
Summarize local VIAME sea lion source data coverage.

This is intended to answer provenance/setup questions before building ML-ready
kwcoco datasets. It reads local CSVs and zip manifests without extracting large
imagery archives.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


def summarize_zip(zpath: Path) -> dict:
    with ZipFile(zpath) as zfile:
        infos = [info for info in zfile.infolist() if not info.is_dir()]
    exts = Counter(Path(info.filename).suffix.lower() or '<none>' for info in infos)
    image_count = sum(count for ext, count in exts.items() if ext in IMAGE_EXTS)
    return {
        'path': str(zpath),
        'files': len(infos),
        'images': image_count,
        'zip_bytes': zpath.stat().st_size,
        'uncompressed_bytes': sum(info.file_size for info in infos),
        'exts': dict(sorted(exts.items())),
    }


def summarize_headered_csv(fpath: Path) -> dict:
    with fpath.open(newline='') as file:
        reader = csv.DictReader(file)
        rows = 0
        images = set()
        classes = Counter()
        for row in reader:
            rows += 1
            images.add(row.get('IMAGE', ''))
            classes[row.get('CLASS', '')] += 1
    return {
        'path': str(fpath),
        'rows': rows,
        'images': len(images),
        'classes': dict(classes.most_common()),
    }


def summarize_viame_csv_from_zip(zpath: Path) -> list[dict]:
    summaries = []
    with ZipFile(zpath) as zfile:
        names = sorted(name for name in zfile.namelist() if name.lower().endswith('.csv'))
        for name in names:
            rows = 0
            images = set()
            classes = Counter()
            bad_rows = 0
            with zfile.open(name) as raw:
                text = io.TextIOWrapper(raw, newline='')
                reader = csv.reader(text)
                for row in reader:
                    if not row or row[0].lstrip().startswith('#'):
                        continue
                    if len(row) < 10:
                        bad_rows += 1
                        continue
                    rows += 1
                    images.add(row[1])
                    classes[row[9]] += 1
            summaries.append({
                'path': f'{zpath}:{name}',
                'name': Path(name).name,
                'rows': rows,
                'images': len(images),
                'bad_rows': bad_rows,
                'classes': dict(classes.most_common()),
            })
    return summaries


def summarize_extracted_images(root: Path) -> dict[str, int]:
    if not root.exists():
        return {}
    year_to_images = {}
    for dpath in sorted(path for path in root.iterdir() if path.is_dir()):
        count = sum(
            1
            for fpath in dpath.rglob('*')
            if fpath.is_file() and fpath.suffix.lower() in IMAGE_EXTS
        )
        if count:
            year_to_images[dpath.name] = count
    return year_to_images


def build_inventory(repo_dpath: Path) -> dict:
    redacted_zip_root = repo_dpath / 'burlynb/Public/Redacted_Imagery'
    raw_zip_root = repo_dpath / 'burlynb/Public/RAW_Imagery'
    girder_detection_zip = redacted_zip_root / '2007-2024_detections.zip'

    inventory = {
        'repo': str(repo_dpath),
        'redacted_zips': [],
        'raw_zips': [],
        'girder_detection_csvs': [],
        'email_new_annotation_csvs': [],
        'email_redundant_annotation_csvs': [],
        'extracted_redacted_images': summarize_extracted_images(repo_dpath / 'Redacted_Imagery'),
        'extracted_raw_images': summarize_extracted_images(raw_zip_root),
    }

    if redacted_zip_root.exists():
        for zpath in sorted(redacted_zip_root.glob('*.zip')):
            if zpath.name != '2007-2024_detections.zip':
                inventory['redacted_zips'].append(summarize_zip(zpath))

    if raw_zip_root.exists():
        for zpath in sorted(raw_zip_root.rglob('*.zip')):
            inventory['raw_zips'].append(summarize_zip(zpath))

    if girder_detection_zip.exists():
        inventory['girder_detection_csvs'] = summarize_viame_csv_from_zip(girder_detection_zip)

    for rel in ['IncludesNewAnnotations', 'Redundant']:
        csvs = []
        for fpath in sorted((repo_dpath / rel).glob('*.csv')):
            csvs.append(summarize_headered_csv(fpath))
        key = 'email_new_annotation_csvs' if rel == 'IncludesNewAnnotations' else 'email_redundant_annotation_csvs'
        inventory[key] = csvs

    return inventory


def year_from_name(path_or_name: str) -> str:
    name = Path(path_or_name).name
    if name.startswith('2024_ForDetections'):
        return '2024'
    if name[:5] == '2008W':
        return '2008W'
    if name[:4].isdigit():
        return name[:4]
    return ''


def markdown_table(rows: list[list[object]], headers: list[str]) -> str:
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(str(item) for item in row) + ' |')
    return '\n'.join(lines)


def render_markdown(inventory: dict) -> str:
    redacted_by_year = {
        year_from_name(item['path']): item
        for item in inventory['redacted_zips']
        if year_from_name(item['path'])
    }
    edited_csvs = {}
    incomplete_csvs = {}
    for item in inventory['girder_detection_csvs']:
        year = year_from_name(item['name'])
        if 'incomplete' in item['name']:
            incomplete_csvs[year] = item
        else:
            edited_csvs[year] = item

    email_csvs = {}
    for group in ['email_new_annotation_csvs', 'email_redundant_annotation_csvs']:
        for item in inventory[group]:
            email_csvs[Path(item['path']).name] = item

    years = sorted(set(redacted_by_year) | set(edited_csvs) | set(incomplete_csvs))
    rows = []
    extracted = inventory['extracted_redacted_images']
    for year in years:
        zinfo = redacted_by_year.get(year, {})
        edited = edited_csvs.get(year, {})
        incomplete = incomplete_csvs.get(year, {})
        rows.append([
            year,
            zinfo.get('images', 0),
            extracted.get('2024_ForDetections' if year == '2024' else year, 0),
            edited.get('images', ''),
            edited.get('rows', ''),
            incomplete.get('images', ''),
            incomplete.get('rows', ''),
        ])

    email_rows = []
    for group, label in [
        ('email_new_annotation_csvs', 'IncludesNewAnnotations'),
        ('email_redundant_annotation_csvs', 'Redundant'),
    ]:
        for item in inventory[group]:
            email_rows.append([
                label,
                Path(item['path']).name,
                item['images'],
                item['rows'],
            ])

    raw_rows = []
    for item in inventory['raw_zips']:
        raw_rows.append([
            item['path'].replace(inventory['repo'] + '/', ''),
            item['images'],
            item['files'],
            item['zip_bytes'],
        ])

    parts = [
        '# Sea Lion Data Inventory',
        '',
        '## Redacted imagery and Girder detections',
        '',
        markdown_table(
            rows,
            ['year', 'redacted zip images', 'extracted images', 'edited csv images', 'edited rows', 'incomplete csv images', 'incomplete rows'],
        ),
        '',
        '## Email annotation CSVs',
        '',
        markdown_table(email_rows, ['bucket', 'csv', 'images', 'rows']),
        '',
        '## RAW imagery zips',
        '',
        markdown_table(raw_rows, ['zip', 'images', 'files', 'zip bytes']),
        '',
    ]
    return '\n'.join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--format', choices=['markdown', 'json'], default='markdown')
    args = parser.parse_args()

    inventory = build_inventory(args.repo.resolve())
    if args.format == 'json':
        print(json.dumps(inventory, indent=2, sort_keys=True))
    else:
        print(render_markdown(inventory))


if __name__ == '__main__':
    main()
