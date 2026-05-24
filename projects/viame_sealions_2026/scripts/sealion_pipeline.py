#!/usr/bin/env python3
"""
Helpers for unpacking and converting VIAME sea lion source data.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    year: str
    zip_rel: str
    image_dir_rel: str
    csv_rels: tuple[str, ...]


REDACTED_SOURCES = [
    SourceSpec('redacted_2007', '2007', 'burlynb/Public/Redacted_Imagery/2007.zip', 'burlynb/Public/Redacted_Imagery/2007', ('Redundant/2007_annotations.csv',)),
    SourceSpec('redacted_2008', '2008', 'burlynb/Public/Redacted_Imagery/2008.zip', 'burlynb/Public/Redacted_Imagery/2008', ('IncludesNewAnnotations/2008_annotations.csv',)),
    SourceSpec('redacted_2008W', '2008W', 'burlynb/Public/Redacted_Imagery/2008W.zip', 'burlynb/Public/Redacted_Imagery/2008W', ('Redundant/2008W_annotations.csv',)),
    SourceSpec('redacted_2009', '2009', 'burlynb/Public/Redacted_Imagery/2009.zip', 'burlynb/Public/Redacted_Imagery/2009', ('IncludesNewAnnotations/2009_annotations.csv',)),
    SourceSpec('redacted_2010', '2010', 'burlynb/Public/Redacted_Imagery/2010.zip', 'burlynb/Public/Redacted_Imagery/2010', ('Redundant/2010_annotations.csv',)),
    SourceSpec('redacted_2011', '2011', 'burlynb/Public/Redacted_Imagery/2011.zip', 'burlynb/Public/Redacted_Imagery/2011', ('IncludesNewAnnotations/2011_annotations.csv',)),
    SourceSpec('redacted_2012', '2012', 'burlynb/Public/Redacted_Imagery/2012.zip', 'burlynb/Public/Redacted_Imagery/2012', ('Redundant/2012_annotations.csv',)),
    SourceSpec('redacted_2013', '2013', 'burlynb/Public/Redacted_Imagery/2013.zip', 'burlynb/Public/Redacted_Imagery/2013', ('Redundant/2013_detections_incomplete.csv',)),
    SourceSpec('redacted_2014', '2014', 'burlynb/Public/Redacted_Imagery/2014.zip', 'burlynb/Public/Redacted_Imagery/2014', ('Redundant/2014_annotations.csv',)),
    SourceSpec('redacted_2015', '2015', 'burlynb/Public/Redacted_Imagery/2015.zip', 'burlynb/Public/Redacted_Imagery/2015', ('Redundant/2015_annotations.csv',)),
    SourceSpec('redacted_2016', '2016', 'burlynb/Public/Redacted_Imagery/2016.zip', 'burlynb/Public/Redacted_Imagery/2016', ('Redundant/2016_annotations.csv',)),
    SourceSpec('redacted_2017', '2017', 'burlynb/Public/Redacted_Imagery/2017.zip', 'burlynb/Public/Redacted_Imagery/2017', ('Redundant/2017_detections_incomplete.csv',)),
    SourceSpec('redacted_2018', '2018', 'burlynb/Public/Redacted_Imagery/2018.zip', 'burlynb/Public/Redacted_Imagery/2018', ('Redundant/2018_detections_incomplete.csv',)),
    SourceSpec('redacted_2019', '2019', 'burlynb/Public/Redacted_Imagery/2019.zip', 'burlynb/Public/Redacted_Imagery/2019', ('Redundant/2019_annotations.csv',)),
    SourceSpec('redacted_2021', '2021', 'burlynb/Public/Redacted_Imagery/2021.zip', 'burlynb/Public/Redacted_Imagery/2021', ('Redundant/2021_annotations.csv', 'Redundant/2021_detections_incomplete.csv')),
    SourceSpec('redacted_2022', '2022', 'burlynb/Public/Redacted_Imagery/2022.zip', 'burlynb/Public/Redacted_Imagery/2022', ('IncludesNewAnnotations/2022_annotations.csv',)),
    SourceSpec('redacted_2023', '2023', 'burlynb/Public/Redacted_Imagery/2023.zip', 'burlynb/Public/Redacted_Imagery/2023', ('IncludesNewAnnotations/2023_annotations.csv',)),
    SourceSpec('redacted_2024', '2024', 'burlynb/Public/Redacted_Imagery/2024.zip', 'burlynb/Public/Redacted_Imagery/2024_ForDetections', ('IncludesNewAnnotations/2024_annotations.csv',)),
]


CLASS_ALIASES = {
    'b': 'bull',
    'bull': 'bull',
    's': 'subadult_male',
    'sam': 'subadult_male',
    'f': 'female',
    'fem': 'female',
    'female': 'female',
    'j': 'juvenile',
    'juv': 'juvenile',
    'juvenile': 'juvenile',
    'p': 'pup',
    'pup': 'pup',
    'dn': 'dead_nonpup',
    'deadnp': 'dead_nonpup',
    'dead np': 'dead_nonpup',
    'dead non pup': 'dead_nonpup',
    'dead non-pup': 'dead_nonpup',
    'dp': 'dead_pup',
    'deadpup': 'dead_pup',
    'dead pup': 'dead_pup',
    'dead-pup': 'dead_pup',
    'nfs': 'northern_fur_seal',
    'furseal': 'northern_fur_seal',
    'fur seal': 'northern_fur_seal',
    'o': 'negative',
    'background': 'negative',
    'unknown': 'negative',
    '': 'negative',
    'age_sex': 'negative',
}


POSITIVE_CLASSES = {
    'bull',
    'subadult_male',
    'female',
    'juvenile',
    'pup',
    'dead_nonpup',
    'dead_pup',
}


def normalize_image_key(name: str) -> str:
    return re.sub('[^A-Z0-9]', '', Path(name).stem.upper())


def coerce_int(text):
    if text in {None, ''}:
        return None
    return int(float(text))


def coerce_float(text):
    if text in {None, ''}:
        return None
    return float(text)


def json_dump(data, fpath: Path) -> None:
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def zip_manifest_hash(zpath: Path) -> str:
    hasher = hashlib.sha256()
    with ZipFile(zpath) as zfile:
        for info in sorted(zfile.infolist(), key=lambda item: item.filename):
            hasher.update(info.filename.encode('utf8', errors='surrogateescape'))
            hasher.update(str(info.file_size).encode())
            hasher.update(str(info.CRC).encode())
    return hasher.hexdigest()


def source_zip_info(zpath: Path) -> dict:
    return {
        'path': str(zpath),
        'size': zpath.stat().st_size,
        'mtime_ns': zpath.stat().st_mtime_ns,
        'zip_manifest_sha256': zip_manifest_hash(zpath),
    }


def safe_extract(zpath: Path, dst: Path) -> None:
    dst = dst.resolve()
    with ZipFile(zpath) as zfile:
        for info in zfile.infolist():
            target = (dst / info.filename).resolve()
            if not str(target).startswith(str(dst) + os.sep):
                raise ValueError(f'Unsafe zip member: {info.filename!r}')
        zfile.extractall(dst)


def unpack_source(repo: Path, unpacked: Path, spec: SourceSpec, force: bool = False) -> dict:
    zpath = repo / spec.zip_rel
    final_dpath = unpacked / spec.image_dir_rel
    manifest_fpath = final_dpath / 'source_manifest.json'
    zinfo = source_zip_info(zpath)
    expected = {
        'source_name': spec.name,
        'year': spec.year,
        'source_zip': zinfo,
        'image_dir': str(final_dpath),
    }
    if not force and manifest_fpath.exists():
        old = json.loads(manifest_fpath.read_text())
        if old.get('source_zip', {}).get('zip_manifest_sha256') == zinfo['zip_manifest_sha256']:
            expected['status'] = 'skipped_existing'
            return expected

    final_dpath.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'unpack_{spec.name}_', dir=str(final_dpath.parent)) as tmp:
        tmp_dpath = Path(tmp)
        safe_extract(zpath, tmp_dpath)
        entries = [p for p in tmp_dpath.iterdir()]
        if len(entries) != 1 or not entries[0].is_dir():
            raise AssertionError(f'Expected one top-level folder in {zpath}, got {entries}')
        top_dpath = entries[0]
        if final_dpath.exists():
            shutil.rmtree(final_dpath)
        shutil.move(str(top_dpath), str(final_dpath))

    image_count = len(list(iter_image_files(final_dpath)))
    expected['image_count'] = image_count
    expected['status'] = 'unpacked'
    json_dump(expected, manifest_fpath)
    return expected


def iter_image_files(dpath: Path):
    yield from sorted(
        p for p in dpath.rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as img:
        return img.size


def read_headered_csv(csv_fpath: Path, year: str) -> list[dict]:
    rows = []
    with csv_fpath.open(newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            row['_year'] = year
            row['_csv_fpath'] = str(csv_fpath)
            rows.append(row)
    return rows


def build_image_indexes(image_dpath: Path):
    by_name = collections.defaultdict(list)
    by_norm = collections.defaultdict(list)
    images = list(iter_image_files(image_dpath))
    for path in images:
        rel = path.relative_to(image_dpath)
        by_name[path.name].append(rel)
        by_norm[normalize_image_key(path.name)].append(rel)
    return images, by_name, by_norm


def choose_image_rel(csv_name: str, by_name, by_norm):
    exact = by_name.get(csv_name, [])
    if len(exact) == 1:
        return exact[0], 'exact'
    if len(exact) > 1:
        return None, 'ambiguous_exact'
    norm = by_norm.get(normalize_image_key(csv_name), [])
    if len(norm) == 1:
        return norm[0], 'normalized'
    if len(norm) > 1:
        return None, 'ambiguous_normalized'
    return None, 'missing'


def convert_csv_to_raw_kwcoco(repo: Path, image_dpath: Path, csv_fpath: Path, dst: Path, year: str, source_zip: Path) -> dict:
    import kwcoco

    image_paths, by_name, by_norm = build_image_indexes(image_dpath)
    rows = read_headered_csv(csv_fpath, year)
    rows_by_rel = collections.defaultdict(list)
    unmatched = []
    match_hist = collections.Counter()
    for row in rows:
        rel, status = choose_image_rel(row['IMAGE'], by_name, by_norm)
        match_hist[status] += 1
        if rel is None:
            unmatched.append({'image': row['IMAGE'], 'status': status, 'id': row.get('ID')})
        else:
            rows_by_rel[rel].append(row)

    dset = kwcoco.CocoDataset()
    dset.dataset['info'] = [{
        'description': 'Raw VIAME sea lion CSV conversion',
        'stage': 'raw',
        'year': year,
        'source_csv': str(csv_fpath),
        'source_zip': str(source_zip),
        'image_root': str(image_dpath),
    }]

    stats = collections.Counter()
    class_hist = collections.Counter()
    bbox_warnings = collections.Counter()
    bad_images = []

    for path in image_paths:
        rel = path.relative_to(image_dpath)
        try:
            width, height = image_size(path)
        except Exception as ex:
            width, height = None, None
            bad_images.append({
                'file_name': str(rel),
                'error': f'{type(ex).__name__}: {ex}',
            })
            bbox_warnings['unreadable_image'] += 1
        gid = dset.add_image(
            file_name=str(rel),
            name=path.name,
            width=width,
            height=height,
            year=year,
            source_zip=str(source_zip),
            source_csv=str(csv_fpath),
        )
        stats['images'] += 1
        for row in rows_by_rel.get(rel, []):
            tl_x = coerce_float(row['TL_X'])
            tl_y = coerce_float(row['TL_Y'])
            br_x = coerce_float(row['BR_X'])
            br_y = coerce_float(row['BR_Y'])
            w = br_x - tl_x
            h = br_y - tl_y
            if w <= 0 or h <= 0:
                bbox_warnings['nonpositive'] += 1
                continue
            if width is not None and height is not None and (tl_x < 0 or tl_y < 0 or br_x > width or br_y > height):
                bbox_warnings['outside_image'] += 1
            catname = row['CLASS'].strip()
            if not catname:
                catname = 'unknown'
            cid = dset.ensure_category(name=catname)
            class_hist[catname] += 1
            ann = {
                'image_id': gid,
                'category_id': cid,
                'bbox': [tl_x, tl_y, w, h],
                'score': coerce_float(row.get('REVIEW_D')),
                'target_len': coerce_float(row.get('TARGET_LENGTH')),
                'viame': {
                    'id': coerce_int(row.get('ID')),
                    'frame': coerce_int(row.get('FRAME')),
                    'review_d': coerce_float(row.get('REVIEW_D')),
                    'review_c': coerce_float(row.get('REVIEW_C')),
                    'attribute': row.get('ATTRIBUTE', ''),
                    'csv_image': row.get('IMAGE'),
                    'csv_fpath': row.get('_csv_fpath'),
                    'year': row.get('_year'),
                    'raw_class': row.get('CLASS', ''),
                    'raw_row': {k: v for k, v in row.items() if not k.startswith('_')},
                },
            }
            dset.add_annotation(**ann)
            stats['annotations'] += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    dset.fpath = str(dst)
    dset.dump(str(dst), newlines=True)
    report = {
        'dst': str(dst),
        'stage': 'raw',
        'year': year,
        'source_csv': str(csv_fpath),
        'source_zip': str(source_zip),
        'image_root': str(image_dpath),
        'stats': dict(stats),
        'csv_rows': len(rows),
        'match_hist': dict(match_hist),
        'unmatched_rows': unmatched,
        'bad_images': bad_images,
        'classes': dict(sorted(class_hist.items())),
        'bbox_warnings': dict(bbox_warnings),
    }
    json_dump(report, dst.with_suffix(dst.suffix + '.report.json'))
    return report


def normalize_class(raw: str, attr: str) -> tuple[str, dict]:
    token = re.sub(r'\s+', ' ', (raw or '').strip()).lower()
    token_no_space = token.replace(' ', '')
    cname = CLASS_ALIASES.get(token, CLASS_ALIASES.get(token_no_space, 'negative'))
    role = 'positive' if cname in POSITIVE_CLASSES else 'negative'
    negative_kind = None
    if role == 'negative':
        if 'NOTE WATER' in (attr or '').upper():
            negative_kind = 'water_region'
        elif cname == 'northern_fur_seal':
            negative_kind = 'northern_fur_seal'
        elif token in {'o', 'background', ''}:
            negative_kind = 'background'
        else:
            negative_kind = 'other'
    return cname, {'role': role, 'negative_kind': negative_kind}


def normalize_kwcoco(raw_fpath: Path, norm_fpath: Path) -> dict:
    import kwcoco

    raw = kwcoco.CocoDataset(raw_fpath)
    norm = kwcoco.CocoDataset()
    norm.dataset['info'] = list(raw.dataset.get('info', [])) + [{
        'description': 'Normalized VIAME sea lion annotations',
        'stage': 'normalized',
        'source_raw_kwcoco': str(raw_fpath),
        'class_policy': {
            'positive_classes': sorted(POSITIVE_CLASSES),
            'negative_class': 'negative',
            'water_regions': 'negative',
        },
    }]

    gid_map = {}
    for gid, img in raw.index.imgs.items():
        new = dict(img)
        old_id = new.pop('id', None)
        gid_map[old_id] = norm.add_image(**new)

    class_hist = collections.Counter()
    for aid, ann in raw.index.anns.items():
        raw_cat = raw.index.cats[ann['category_id']]['name']
        attr = ann.get('viame', {}).get('attribute', '')
        cname, norm_meta = normalize_class(raw_cat, attr)
        cid = norm.ensure_category(name=cname)
        new_ann = dict(ann)
        new_ann.pop('id', None)
        new_ann['image_id'] = gid_map[ann['image_id']]
        new_ann['category_id'] = cid
        new_ann['normalized'] = {
            'class': cname,
            'source_class': raw_cat,
            **norm_meta,
        }
        class_hist[cname] += 1
        norm.add_annotation(**new_ann)

    norm_fpath.parent.mkdir(parents=True, exist_ok=True)
    norm.fpath = str(norm_fpath)
    norm.dump(str(norm_fpath), newlines=True)
    report = {
        'dst': str(norm_fpath),
        'stage': 'normalized',
        'source_raw_kwcoco': str(raw_fpath),
        'n_images': norm.n_images,
        'n_annotations': norm.n_annots,
        'classes': dict(sorted(class_hist.items())),
    }
    json_dump(report, norm_fpath.with_suffix(norm_fpath.suffix + '.report.json'))
    return report


def source_csv_stem(csv_rel: str) -> str:
    return Path(csv_rel).stem


def is_incomplete_csv(csv_rel: str) -> bool:
    return 'detections_incomplete' in Path(csv_rel).name


def build_per_source(repo: Path, unpacked: Path, force_unpack: bool = False, force_convert: bool = False) -> dict:
    reports = {'unpack': [], 'raw': [], 'norm': []}
    for spec in REDACTED_SOURCES:
        reports['unpack'].append(unpack_source(repo, unpacked, spec, force=force_unpack))
        image_dpath = unpacked / spec.image_dir_rel
        source_zip = repo / spec.zip_rel
        manifest_updates = {}
        for csv_rel in spec.csv_rels:
            csv_fpath = repo / csv_rel
            stem = source_csv_stem(csv_rel)
            raw_fpath = image_dpath / f'{stem}.raw.kwcoco.zip'
            norm_fpath = image_dpath / f'{stem}.norm.kwcoco.zip'
            if force_convert or not raw_fpath.exists():
                raw_report = convert_csv_to_raw_kwcoco(repo, image_dpath, csv_fpath, raw_fpath, spec.year, source_zip)
            else:
                raw_report = {'dst': str(raw_fpath), 'stage': 'raw', 'status': 'skipped_existing'}
            if force_convert or not norm_fpath.exists():
                norm_report = normalize_kwcoco(raw_fpath, norm_fpath)
            else:
                norm_report = {'dst': str(norm_fpath), 'stage': 'normalized', 'status': 'skipped_existing'}
            reports['raw'].append(raw_report)
            reports['norm'].append(norm_report)
            manifest_updates[stem] = {
                'source_csv': str(csv_fpath),
                'raw_kwcoco': str(raw_fpath),
                'norm_kwcoco': str(norm_fpath),
                'incomplete': is_incomplete_csv(csv_rel),
            }
        manifest_fpath = image_dpath / 'source_manifest.json'
        manifest = json.loads(manifest_fpath.read_text())
        manifest['annotations'] = manifest_updates
        json_dump(manifest, manifest_fpath)
    return reports


def copy_dataset_into(dst, src_fpath: Path, rel_prefix: Path, split_lookup: dict | None = None, desired_split: str | None = None):
    import kwcoco

    src = kwcoco.CocoDataset(src_fpath)
    gid_map = {}
    for gid, img in src.index.imgs.items():
        split = None if split_lookup is None else split_lookup.get((str(src_fpath), img['name']))
        if desired_split is not None and split != desired_split:
            continue
        new_img = dict(img)
        old_id = new_img.pop('id', None)
        new_img['file_name'] = str(rel_prefix / img['file_name'])
        new_img['source_kwcoco'] = str(src_fpath)
        if split is not None:
            new_img['split'] = split
        gid_map[old_id] = dst.add_image(**new_img)
    for aid, ann in src.index.anns.items():
        if ann['image_id'] not in gid_map:
            continue
        new_ann = dict(ann)
        new_ann.pop('id', None)
        cname = src.index.cats[ann['category_id']]['name']
        new_ann['category_id'] = dst.ensure_category(name=cname)
        new_ann['image_id'] = gid_map[ann['image_id']]
        dst.add_annotation(**new_ann)


def build_split_lookup(norm_sources: list[tuple[Path, Path, str]], seed: int = 20260514, test_years=('2009', '2019', '2024'), test_frac: float = 0.25):
    import kwcoco

    rng = random.Random(seed)
    split_lookup = {}
    manifest = {
        'seed': seed,
        'test_years': list(test_years),
        'test_frac': test_frac,
        'sources': [],
        'assignments': [],
    }
    for norm_fpath, rel_prefix, year in norm_sources:
        dset = kwcoco.CocoDataset(norm_fpath)
        image_names = sorted(img['name'] for img in dset.index.imgs.values())
        test_names = set()
        chunk = None
        if year in test_years and image_names:
            n_test = max(1, int(math.ceil(len(image_names) * test_frac)))
            max_start = max(0, len(image_names) - n_test)
            start = rng.randint(0, max_start)
            test_names = set(image_names[start:start + n_test])
            chunk = {'start': start, 'size': n_test, 'total': len(image_names)}
        source_info = {
            'norm_kwcoco': str(norm_fpath),
            'year': year,
            'image_count': len(image_names),
            'test_chunk': chunk,
        }
        manifest['sources'].append(source_info)
        for name in image_names:
            split = 'test' if name in test_names else 'learn'
            split_lookup[(str(norm_fpath), name)] = split
            manifest['assignments'].append({
                'source_kwcoco': str(norm_fpath),
                'image_name': name,
                'year': year,
                'split': split,
            })
    return split_lookup, manifest


def collect_complete_norm_sources(unpacked: Path):
    sources = []
    for spec in REDACTED_SOURCES:
        image_dpath = unpacked / spec.image_dir_rel
        for csv_rel in spec.csv_rels:
            if is_incomplete_csv(csv_rel):
                continue
            stem = source_csv_stem(csv_rel)
            norm_fpath = image_dpath / f'{stem}.norm.kwcoco.zip'
            if norm_fpath.exists():
                rel_prefix = image_dpath.relative_to(unpacked)
                sources.append((norm_fpath, rel_prefix, spec.year))
    return sources


def build_combined(unpacked: Path, seed: int = 20260514) -> dict:
    import kwcoco

    norm_sources = collect_complete_norm_sources(unpacked)
    split_lookup, split_manifest = build_split_lookup(norm_sources, seed=seed)
    json_dump(split_manifest, unpacked / 'splits_v1.json')

    outputs = {}
    for name, desired_split in [('all_norm', None), ('learn_norm', 'learn'), ('train_norm', 'train'), ('vali_norm', 'vali'), ('test_norm', 'test')]:
        pass

    # Convert learn into train/vali with a seeded random split.
    rng = random.Random(seed + 1)
    learn_keys = sorted([key for key, split in split_lookup.items() if split == 'learn'])
    rng.shuffle(learn_keys)
    n_vali = max(1, int(round(len(learn_keys) * 0.15))) if learn_keys else 0
    vali_keys = set(learn_keys[:n_vali])
    for key in learn_keys:
        split_lookup[key] = 'vali' if key in vali_keys else 'train'
    for assignment in split_manifest['assignments']:
        key = (assignment['source_kwcoco'], assignment['image_name'])
        if assignment['split'] == 'learn':
            assignment['train_vali_split'] = split_lookup[key]
    json_dump(split_manifest, unpacked / 'splits_v1.json')

    output_specs = [
        ('all_norm', None),
        ('learn_norm', {'train', 'vali'}),
        ('train_norm', {'train'}),
        ('vali_norm', {'vali'}),
        ('test_norm', {'test'}),
    ]
    for out_stem, allowed_splits in output_specs:
        dst = kwcoco.CocoDataset()
        dst.dataset['info'] = [{
            'description': f'Combined normalized sea lion dataset: {out_stem}',
            'stage': 'combined_normalized',
            'split_seed': seed,
            'sources': [str(item[0]) for item in norm_sources],
        }]
        for norm_fpath, rel_prefix, year in norm_sources:
            if allowed_splits is None:
                local_lookup = split_lookup
                desired = None
            elif len(allowed_splits) == 1:
                local_lookup = split_lookup
                desired = next(iter(allowed_splits))
            else:
                # Copy all images whose final split is in the allowed set.
                import kwcoco as _kwcoco
                src = _kwcoco.CocoDataset(norm_fpath)
                for gid, img in src.index.imgs.items():
                    split = split_lookup.get((str(norm_fpath), img['name']))
                    if split not in allowed_splits:
                        continue
                    new_img = dict(img)
                    old_id = new_img.pop('id', None)
                    new_img['file_name'] = str(rel_prefix / img['file_name'])
                    new_img['source_kwcoco'] = str(norm_fpath)
                    new_img['split'] = 'learn'
                    new_gid = dst.add_image(**new_img)
                    for aid in src.index.gid_to_aids[old_id]:
                        ann = dict(src.index.anns[aid])
                        ann.pop('id', None)
                        cname = src.index.cats[ann['category_id']]['name']
                        ann['category_id'] = dst.ensure_category(name=cname)
                        ann['image_id'] = new_gid
                        dst.add_annotation(**ann)
                continue
            copy_dataset_into(dst, norm_fpath, rel_prefix, split_lookup, desired_split=desired)
        out_fpath = unpacked / f'{out_stem}.kwcoco.zip'
        dst.fpath = str(out_fpath)
        dst.dump(str(out_fpath), newlines=True)
        outputs[out_stem] = {
            'path': str(out_fpath),
            'n_images': dst.n_images,
            'n_annotations': dst.n_annots,
            'categories': sorted(cat['name'] for cat in dst.index.cats.values()),
        }
    json_dump(outputs, unpacked / 'combined_report.json')
    return outputs


def run_pipeline(repo: Path, unpacked: Path, force_unpack: bool = False, force_convert: bool = False, seed: int = 20260514) -> dict:
    unpacked.mkdir(parents=True, exist_ok=True)
    report = {
        'repo': str(repo),
        'unpacked': str(unpacked),
        'per_source': build_per_source(repo, unpacked, force_unpack=force_unpack, force_convert=force_convert),
        'combined': build_combined(unpacked, seed=seed),
    }
    json_dump(report, unpacked / 'pipeline_report.json')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--unpacked', type=Path, default=Path('unpacked'))
    parser.add_argument('--force-unpack', action='store_true')
    parser.add_argument('--force-convert', action='store_true')
    parser.add_argument('--seed', type=int, default=20260514)
    args = parser.parse_args()

    repo = args.repo.resolve()
    unpacked = args.unpacked
    if not unpacked.is_absolute():
        unpacked = repo / unpacked
    report = run_pipeline(
        repo=repo,
        unpacked=unpacked,
        force_unpack=args.force_unpack,
        force_convert=args.force_convert,
        seed=args.seed,
    )
    print(json.dumps({
        'unpacked': report['unpacked'],
        'combined': report['combined'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
