#!/usr/bin/env python3
"""
Project normalized sea lion kwcoco files into detector-ready categories.

The default projection is:
    sealion_adult: bull, subadult_male, female, juvenile, dead_nonpup
    sealion_pup: pup, dead_pup
    negative: negative
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ADULT_CLASSES = {
    'bull',
    'subadult_male',
    'female',
    'juvenile',
    'dead_nonpup',
}

PUP_CLASSES = {
    'pup',
    'dead_pup',
}

NEGATIVE_CLASSES = {
    'negative',
    'northern_fur_seal',
}


def project_category(cname: str) -> str:
    if cname in ADULT_CLASSES:
        return 'sealion_adult'
    if cname in PUP_CLASSES:
        return 'sealion_pup'
    if cname in NEGATIVE_CLASSES:
        return 'negative'
    raise KeyError(f'No detector projection for category {cname!r}')


def project_kwcoco(src_fpath: Path, dst_fpath: Path) -> dict:
    import kwcoco

    src = kwcoco.CocoDataset(src_fpath)
    dst = kwcoco.CocoDataset()
    dst.dataset['info'] = list(src.dataset.get('info', [])) + [{
        'description': 'Detector-ready sea lion adult / pup / negative projection',
        'stage': 'detector_projection',
        'source_kwcoco': str(src_fpath),
        'category_policy': {
            'sealion_adult': sorted(ADULT_CLASSES),
            'sealion_pup': sorted(PUP_CLASSES),
            'negative': sorted(NEGATIVE_CLASSES),
        },
    }]

    gid_map = {}
    for gid, img in src.index.imgs.items():
        new_img = dict(img)
        old_id = new_img.pop('id', None)
        gid_map[old_id] = dst.add_image(**new_img)

    hist = {}
    source_hist = {}
    for aid, ann in src.index.anns.items():
        src_cname = src.index.cats[ann['category_id']]['name']
        dst_cname = project_category(src_cname)
        cid = dst.ensure_category(name=dst_cname)
        new_ann = dict(ann)
        new_ann.pop('id', None)
        new_ann['image_id'] = gid_map[ann['image_id']]
        new_ann['category_id'] = cid
        projection = dict(new_ann.get('detector_projection', {}))
        projection.update({
            'class': dst_cname,
            'source_class': src_cname,
        })
        new_ann['detector_projection'] = projection
        dst.add_annotation(**new_ann)
        hist[dst_cname] = hist.get(dst_cname, 0) + 1
        source_hist[src_cname] = source_hist.get(src_cname, 0) + 1

    dst_fpath.parent.mkdir(parents=True, exist_ok=True)
    dst.fpath = str(dst_fpath)
    dst.dump(str(dst_fpath), newlines=True)

    report = {
        'src': str(src_fpath),
        'dst': str(dst_fpath),
        'n_images': dst.n_images,
        'n_annotations': dst.n_annots,
        'categories': dict(sorted(hist.items())),
        'source_categories': dict(sorted(source_hist.items())),
    }
    return report


def default_inputs(unpacked: Path) -> list[tuple[Path, str]]:
    return [
        (unpacked / 'all_norm.kwcoco.zip', 'all_detection_v1.kwcoco.zip'),
        (unpacked / 'learn_norm.kwcoco.zip', 'learn_detection_v1.kwcoco.zip'),
        (unpacked / 'train_norm.kwcoco.zip', 'train_detection_v1.kwcoco.zip'),
        (unpacked / 'vali_norm.kwcoco.zip', 'vali_detection_v1.kwcoco.zip'),
        (unpacked / 'test_norm.kwcoco.zip', 'test_detection_v1.kwcoco.zip'),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unpacked', type=Path, default=Path('unpacked'))
    parser.add_argument('--dst', type=Path, default=None)
    args = parser.parse_args()

    unpacked = args.unpacked.resolve()
    dst_dpath = (args.dst.resolve() if args.dst is not None else unpacked)
    reports = {}
    for src_fpath, dst_name in default_inputs(unpacked):
        if not src_fpath.exists():
            continue
        report = project_kwcoco(src_fpath, dst_dpath / dst_name)
        reports[dst_name] = report

    reports['policy'] = {
        'sealion_adult': sorted(ADULT_CLASSES),
        'sealion_pup': sorted(PUP_CLASSES),
        'negative': sorted(NEGATIVE_CLASSES),
    }
    report_fpath = dst_dpath / 'prepare_report.json'
    report_fpath.write_text(json.dumps(reports, indent=2, sort_keys=True) + '\n')
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
