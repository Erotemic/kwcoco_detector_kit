#!/usr/bin/env python
"""
Convert the 2021-2024 sea lion annotation CSV files into kwcoco.

This handles the headered CSV format in IncludesNewAnnotations / Redundant,
which is different from the older VIAME alternating class/score CSV format used
by bioharn.io.viame_csv.
"""

import argparse
import collections
import csv
import json
import pathlib
import re


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


DEFAULT_YEAR_TO_CSV = {
    2021: pathlib.Path('Redundant/2021_annotations.csv'),
    2022: pathlib.Path('IncludesNewAnnotations/2022_annotations.csv'),
    2023: pathlib.Path('IncludesNewAnnotations/2023_annotations.csv'),
    2024: pathlib.Path('IncludesNewAnnotations/2024_annotations.csv'),
}


def normalize_name(name):
    return re.sub('[^A-Z0-9]', '', pathlib.Path(name).stem.upper())


def build_image_index(image_roots):
    by_name = collections.defaultdict(list)
    by_normalized = collections.defaultdict(list)
    for root in image_roots:
        root = pathlib.Path(root)
        if not root.exists():
            continue
        for path in root.rglob('*'):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                by_name[path.name].append(path)
                by_normalized[normalize_name(path.name)].append(path)
    return by_name, by_normalized


def choose_image_path(csv_name, by_name, by_normalized):
    matches = by_name.get(csv_name, [])
    if len(matches) == 1:
        return matches[0], 'exact'
    if len(matches) > 1:
        raise AssertionError(f'Ambiguous exact image name {csv_name!r}: {matches}')

    norm_matches = by_normalized.get(normalize_name(csv_name), [])
    if len(norm_matches) == 1:
        return norm_matches[0], 'normalized'
    if len(norm_matches) > 1:
        raise AssertionError(f'Ambiguous normalized image name {csv_name!r}: {norm_matches}')
    return None, 'missing'


def coerce_int(text):
    if text in {None, ''}:
        return None
    return int(float(text))


def coerce_float(text):
    if text in {None, ''}:
        return None
    return float(text)


def image_size(path):
    from PIL import Image
    with Image.open(path) as img:
        return img.size


def convert(year_to_csv, image_roots, dst, strict=False, workers=0):
    import kwcoco
    import ubelt as ub

    cwd = pathlib.Path.cwd()
    by_name, by_normalized = build_image_index(image_roots)

    dset = kwcoco.CocoDataset()
    dset.dataset['info'] = [{
        'description': 'VIAME sea lion annotations converted from headered CSV',
        'years': sorted(year_to_csv),
        'source_csvs': {str(k): str(v) for k, v in year_to_csv.items()},
    }]

    # Use the source class tokens directly until a confirmed semantic mapping is
    # available. This avoids accidentally baking in a wrong biological label.
    image_cache = {}
    stats = collections.Counter()
    per_year = collections.defaultdict(collections.Counter)
    class_hist = collections.Counter()
    unmatched_images = collections.defaultdict(set)
    normalized_matches = {}
    bbox_warnings = collections.Counter()

    rows_by_image = collections.defaultdict(list)
    for year, csv_fpath in year_to_csv.items():
        csv_fpath = pathlib.Path(csv_fpath)
        with csv_fpath.open(newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                row['_year'] = year
                row['_csv_fpath'] = str(csv_fpath)
                rows_by_image[row['IMAGE']].append(row)

    for csv_name, rows in ub.ProgIter(rows_by_image.items(), desc='add images'):
        path, match_type = choose_image_path(csv_name, by_name, by_normalized)
        year = rows[0]['_year']
        if path is None:
            stats['missing_images'] += 1
            per_year[year]['missing_images'] += 1
            unmatched_images[year].add(csv_name)
            if strict:
                raise FileNotFoundError(csv_name)
            continue

        rel_path = path.relative_to(cwd) if path.is_relative_to(cwd) else path
        if path not in image_cache:
            width, height = image_size(path)
            image_cache[path] = (width, height)
        else:
            width, height = image_cache[path]

        img = {
            'file_name': str(rel_path),
            'name': csv_name,
            'width': width,
            'height': height,
            'year': year,
        }
        if match_type == 'normalized':
            img['viame_csv_image_name'] = csv_name
            normalized_matches[csv_name] = path.name
            per_year[year]['normalized_image_names'] += 1

        gid = dset.add_image(**img)
        stats['images'] += 1
        per_year[year]['images'] += 1

        for row in rows:
            tl_x = coerce_float(row['TL_X'])
            tl_y = coerce_float(row['TL_Y'])
            br_x = coerce_float(row['BR_X'])
            br_y = coerce_float(row['BR_Y'])
            w = br_x - tl_x
            h = br_y - tl_y
            if w <= 0 or h <= 0:
                bbox_warnings['nonpositive'] += 1
                if strict:
                    raise ValueError(f'Nonpositive bbox in {row}')
                continue
            if tl_x < 0 or tl_y < 0 or br_x > width or br_y > height:
                bbox_warnings['outside_image'] += 1

            catname = row['CLASS'].strip() or 'unknown'
            cid = dset.ensure_category(name=catname)
            class_hist[catname] += 1
            ann = {
                'image_id': gid,
                'category_id': cid,
                'bbox': [tl_x, tl_y, w, h],
                'score': coerce_float(row['REVIEW_D']),
                'target_len': coerce_float(row['TARGET_LENGTH']),
                'viame': {
                    'id': coerce_int(row['ID']),
                    'frame': coerce_int(row['FRAME']),
                    'review_d': coerce_float(row.get('REVIEW_D')),
                    'review_c': coerce_float(row.get('REVIEW_C')),
                    'attribute': row.get('ATTRIBUTE', ''),
                    'csv_image': row['IMAGE'],
                    'csv_fpath': row['_csv_fpath'],
                    'year': row['_year'],
                },
            }
            dset.add_annotation(**ann)
            stats['annotations'] += 1
            per_year[year]['annotations'] += 1

    dset.fpath = str(dst)
    dset.dump(dset.fpath, newlines=True)

    report = {
        'dst': str(dst),
        'stats': dict(stats),
        'per_year': {str(k): dict(v) for k, v in sorted(per_year.items())},
        'classes': dict(sorted(class_hist.items())),
        'bbox_warnings': dict(bbox_warnings),
        'normalized_matches': normalized_matches,
        'unmatched_images': {str(k): sorted(v) for k, v in sorted(unmatched_images.items())},
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dst', default='sealions_2021_2024.kwcoco.zip')
    parser.add_argument('--years', nargs='+', type=int, default=[2021, 2022, 2023, 2024])
    parser.add_argument('--image-roots', nargs='+', default=[
        'Redacted_Imagery',
        'RAW_Imagery',
        'burlynb/Public/Redacted_Imagery',
    ])
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()

    year_to_csv = {year: DEFAULT_YEAR_TO_CSV[year] for year in args.years}
    report = convert(
        year_to_csv=year_to_csv,
        image_roots=[pathlib.Path(p) for p in args.image_roots],
        dst=pathlib.Path(args.dst),
        strict=args.strict,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
