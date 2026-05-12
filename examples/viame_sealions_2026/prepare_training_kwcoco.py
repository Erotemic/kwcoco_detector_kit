#!/usr/bin/env python
"""
Prepare the VIAME sea-lion kwcoco bundle for first-pass detector tuning.

The source annotations currently use short class codes (B, F, J, ...).  For the
first detector pass this script collapses all selected classes into one
``sealion`` category, clips boxes to image bounds, and writes image-level
train/vali/test splits stratified by year.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random


def _split_items(items, *, vali_frac, test_frac, rng):
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_test = max(1, round(n * test_frac)) if n else 0
    n_vali = max(1, round(n * vali_frac)) if n else 0
    test = items[:n_test]
    vali = items[n_test:n_test + n_vali]
    train = items[n_test + n_vali:]
    return train, vali, test


def prepare(
    src,
    dst_dpath,
    *,
    category_name='sealion',
    include_cats=None,
    vali_frac=0.10,
    test_frac=0.10,
    seed=0,
):
    import kwcoco

    src = pathlib.Path(src)
    dst_dpath = pathlib.Path(dst_dpath)
    dst_dpath.mkdir(parents=True, exist_ok=True)
    include_cats = set(include_cats or [])

    src_dset = kwcoco.CocoDataset(src)
    out = kwcoco.CocoDataset()
    out.fpath = str(dst_dpath / 'all_collapsed.kwcoco.zip')
    new_cid = out.add_category(name=category_name)

    gid_mapping = {}
    cat_hist = collections.Counter()
    clipped = 0
    dropped = 0

    for img in src_dset.images().objs:
        new_img = {
            k: v for k, v in img.items()
            if k not in {'id', 'channels', 'auxiliary', 'assets'}
        }
        new_img['file_name'] = str(pathlib.Path(src_dset.get_image_fpath(img['id'])).resolve())
        gid_mapping[img['id']] = out.add_image(**new_img)

    for ann in src_dset.annots().objs:
        old_cat = src_dset.index.cats[ann['category_id']]['name']
        if include_cats and old_cat not in include_cats:
            continue
        img = src_dset.index.imgs[ann['image_id']]
        width = float(img['width'])
        height = float(img['height'])
        x, y, w, h = [float(v) for v in ann['bbox']]
        x1 = max(0.0, min(width, x))
        y1 = max(0.0, min(height, y))
        x2 = max(0.0, min(width, x + w))
        y2 = max(0.0, min(height, y + h))
        nw = x2 - x1
        nh = y2 - y1
        if nw <= 1 or nh <= 1:
            dropped += 1
            continue
        if [x1, y1, nw, nh] != [x, y, w, h]:
            clipped += 1
        new_ann = {
            'image_id': gid_mapping[ann['image_id']],
            'category_id': new_cid,
            'bbox': [x1, y1, nw, nh],
            'area': float(nw * nh),
            'iscrowd': int(ann.get('iscrowd', 0)),
            'source_category': old_cat,
        }
        if ann.get('viame') is not None:
            new_ann['viame'] = ann['viame']
        out.add_annotation(**new_ann)
        cat_hist[old_cat] += 1

    out.dataset['info'] = [{
        'description': 'VIAME sea-lion annotations collapsed for first-pass detection training',
        'source_kwcoco': str(src),
        'category_name': category_name,
        'source_category_histogram': dict(sorted(cat_hist.items())),
        'clipped_boxes': clipped,
        'dropped_boxes': dropped,
    }]
    out.dump(out.fpath, newlines=True)

    rng = random.Random(seed)
    gids_by_year = collections.defaultdict(list)
    for gid, img in out.index.imgs.items():
        gids_by_year[img.get('year', 'unknown')].append(gid)

    split_to_gids = collections.defaultdict(list)
    for year, gids in sorted(gids_by_year.items()):
        train, vali, test = _split_items(gids, vali_frac=vali_frac, test_frac=test_frac, rng=rng)
        split_to_gids['train'].extend(train)
        split_to_gids['vali'].extend(vali)
        split_to_gids['test'].extend(test)

    split_paths = {}
    for split, gids in split_to_gids.items():
        sub = out.subset(sorted(gids), copy=True)
        sub.fpath = str(dst_dpath / f'{split}.kwcoco.zip')
        sub.dump(sub.fpath, newlines=True)
        split_paths[split] = sub.fpath

    report = {
        'all': out.fpath,
        'splits': split_paths,
        'stats': {
            'images': out.n_images,
            'annotations': out.n_annots,
            'source_category_histogram': dict(sorted(cat_hist.items())),
            'clipped_boxes': clipped,
            'dropped_boxes': dropped,
        },
        'split_stats': {
            split: {
                'images': kwcoco.CocoDataset(fpath).n_images,
                'annotations': kwcoco.CocoDataset(fpath).n_annots,
            }
            for split, fpath in split_paths.items()
        },
    }
    report_fpath = dst_dpath / 'prepare_report.json'
    report_fpath.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', required=True, help='source kwcoco bundle')
    parser.add_argument('--dst_dpath', required=True, help='output directory')
    parser.add_argument('--category_name', default='sealion')
    parser.add_argument('--include_cats', nargs='*', default=None)
    parser.add_argument('--vali_frac', type=float, default=0.10)
    parser.add_argument('--test_frac', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    report = prepare(**vars(args))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
