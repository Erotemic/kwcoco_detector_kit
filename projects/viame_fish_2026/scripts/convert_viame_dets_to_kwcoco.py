#!/usr/bin/env python3
"""
Convert VIAME detection CSV output into a kwcoco predictions bundle.

Why this is separate from convert_viame_to_kwcoco.py
----------------------------------------------------
That script reads GROUND TRUTH: it discards confidences, folds species through
labels.txt, and builds images from a corpus directory. This one reads a
DETECTOR'S OUTPUT, where the confidence is the whole point -- average precision
is computed over ranked detections, so dropping the score would silently turn
an AP into something much worse and no error would be raised.

The two also index images differently. Ground truth is addressed by sequence +
frame index; detector output is addressed by the image path VIAME was handed.
This script therefore aligns predictions to an EXISTING kwcoco bundle by file
path, so that predicted and true annotations refer to the same `image_id`.
Without that alignment `kwcoco eval` silently scores against the wrong images.

Fairness
--------
The output is only comparable to another model's predictions if both were run
over the same images with the same ground truth. This script enforces the first
half: every prediction must land on an image that exists in `--like`, and it
reports how many of that bundle's images received no detections at all (which
is legitimate -- a detector may find nothing -- but a *large* count usually
means the inference run covered less than it should have).

Usage
-----
    python3 projects/viame_fish_2026/scripts/convert_viame_dets_to_kwcoco.py \
        --csv     computed_detections.csv \
        --like    $HOME/ssd-data/fish_kcd/bundle/test.kwcoco.json \
        --out     rfdetr_test_preds.kwcoco.json \
        --category fish
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys

VIAME_MIN_COLUMNS = 9
ATTRIBUTE_PREFIXES = ('(', '+')


def parse_detection_row(row):
    """(image_key, bbox_xywh, category, score) from a VIAME detection row.

    Column 2 holds the image VIAME was reading. Column 8 is the detection
    confidence; columns 9+ are (species, confidence) pairs. We take the first
    pair's confidence as the class score when present and fall back to column
    8, because different VIAME writers populate these differently and a
    detection with no score at all cannot participate in an AP computation.
    """
    if len(row) < VIAME_MIN_COLUMNS:
        return None
    try:
        tl_x, tl_y, br_x, br_y = (float(v) for v in row[3:7])
    except ValueError:
        return None
    width, height = br_x - tl_x, br_y - tl_y
    if width <= 0 or height <= 0:
        return None

    try:
        det_conf = float(row[7])
    except (ValueError, IndexError):
        det_conf = None

    # Scan forward ONE token at a time rather than stepping in pairs.
    #
    # The (species, confidence) pairs are not at a fixed parity: attribute
    # tokens are interleaved, and an ODD number of them before the first
    # species shifts every subsequent pair. Stepping by 2 then lands on a
    # confidence and returns it as the class name -- e.g. `('0.77', 0.9)`
    # instead of `('fish', 0.77)`. Scanning singly and taking the immediately
    # following token as the score is correct wherever the attributes sit.
    category, pair_conf = None, None
    tail = row[VIAME_MIN_COLUMNS:]
    for index, token in enumerate(tail):
        name = token.strip()
        if not name or name.startswith(ATTRIBUTE_PREFIXES):
            continue
        category = name
        if index + 1 < len(tail):
            try:
                pair_conf = float(tail[index + 1])
            except ValueError:
                pair_conf = None
        break

    score = pair_conf if pair_conf is not None else det_conf
    if score is None:
        return None
    return row[1].strip(), [tl_x, tl_y, width, height], category, score


def build_image_index(like_dset):
    """Map several spellings of an image path to its kwcoco image id.

    VIAME echoes back whatever path it was given, which may be absolute, or
    relative, or just a basename depending on how the list was written. Index
    all three so alignment does not depend on that choice.
    """
    index = {}
    for image in like_dset['images']:
        file_name = image['file_name']
        path = pathlib.Path(file_name)
        for key in (file_name, str(path), path.name):
            index.setdefault(key, image['id'])
        # sequence/frame.jpg — distinguishes identically-named frames that
        # belong to different sequences.
        index.setdefault('/'.join(path.parts[-2:]), image['id'])
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', required=True, help='VIAME detection CSV')
    parser.add_argument('--like', required=True,
                        help='kwcoco bundle whose image ids the predictions must align to')
    parser.add_argument('--out', required=True, help='output predictions .kwcoco.json')
    parser.add_argument('--category', default=None,
                        help='force every detection to this category name '
                             '(use for single-class models whose CSV may carry '
                             'species labels)')
    parser.add_argument('--min-score', type=float, default=0.0,
                        help='drop detections below this score. Leave at 0 for '
                             'AP: truncating the ranking lowers it.')
    args = parser.parse_args(argv)

    like = json.loads(pathlib.Path(args.like).read_text())
    index = build_image_index(like)
    categories = ([{'id': 1, 'name': args.category}] if args.category
                  else list(like['categories']))
    category_ids = {c['name']: c['id'] for c in categories}

    annotations = []
    stats = collections.Counter()
    unmatched_samples = set()
    hit_images = set()

    with open(args.csv, 'r', errors='replace') as file:
        for row in csv.reader(file):
            if not row or row[0].lstrip().startswith('#'):
                continue
            parsed = parse_detection_row(row)
            if parsed is None:
                stats['rows_unparsed'] += 1
                continue
            image_key, bbox, category, score = parsed
            if score < args.min_score:
                stats['below_min_score'] += 1
                continue

            path = pathlib.Path(image_key)
            image_id = None
            for key in (image_key, path.name, '/'.join(path.parts[-2:])):
                image_id = index.get(key)
                if image_id is not None:
                    break
            if image_id is None:
                stats['unmatched_image'] += 1
                if len(unmatched_samples) < 5:
                    unmatched_samples.add(image_key)
                continue

            name = args.category or category
            if name not in category_ids:
                stats['unknown_category'] += 1
                continue

            hit_images.add(image_id)
            annotations.append({
                'id': len(annotations) + 1,
                'image_id': image_id,
                'category_id': category_ids[name],
                'bbox': [round(v, 2) for v in bbox],
                'score': score,
            })

    out = {
        'images': like['images'],
        'annotations': annotations,
        'categories': categories,
        'videos': like.get('videos', []),
    }
    out_fpath = pathlib.Path(args.out)
    out_fpath.parent.mkdir(parents=True, exist_ok=True)
    out_fpath.write_text(json.dumps(out))

    n_images = len(like['images'])
    print('predictions: {:,}'.format(len(annotations)))
    print('images in bundle:      {:,}'.format(n_images))
    print('images with >=1 det:   {:,} ({:.1%})'.format(
        len(hit_images), len(hit_images) / max(n_images, 1)))
    print('images with no det:    {:,}'.format(n_images - len(hit_images)))
    if annotations:
        scores = sorted(a['score'] for a in annotations)
        print('score min/median/max:  {:.4f} / {:.4f} / {:.4f}'.format(
            scores[0], scores[len(scores) // 2], scores[-1]))
        print('detections per image:  {:.1f}'.format(len(annotations) / max(n_images, 1)))
    for key in sorted(stats):
        print('  {:<24} {:,}'.format(key, stats[key]))
    if unmatched_samples:
        print('  unmatched examples:', sorted(unmatched_samples))
    print('wrote {}'.format(out_fpath))

    # An inference run that covered far fewer images than the bundle is the
    # failure mode that quietly ruins a comparison, so make it loud.
    if stats['unmatched_image']:
        print('\nWARNING: {:,} detections referenced images not in --like. The two '
              'models were not run over the same set.'.format(stats['unmatched_image']),
              file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
