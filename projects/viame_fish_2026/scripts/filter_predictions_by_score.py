#!/usr/bin/env python3
"""
Filter a kwcoco predictions bundle to a minimum detection score.

Why this exists
---------------
Two detectors can only be compared on AP if their detections were kept down to
the same score floor. AP integrates precision over the whole recall curve, so a
model whose output was truncated at a high threshold cannot reach the recall a
model kept down to ~0 can, and it loses AP for a reason that has nothing to do
with how good it is.

That happened here. DEIMv2 was scored with `score_thresh=0.001` and emitted 300
detections per image; RF-DETR's VIAME plugin applies its own default threshold
of 0.5 (`viame/pytorch/rf_detr_detector.py:39`) before anything reaches the CSV,
so it emitted 2.0 per image with nothing below 0.5000. Scored as-is, DEIMv2
would win partly on that artifact.

The rigorous fix is to re-run RF-DETR at a low threshold. The cheap fix, which
is what this implements, is to truncate the OTHER model to the same floor: both
curves are then cut at the same place, so the comparison is honest even though
it is no longer full-curve AP. It costs minutes instead of hours of GPU time.

What it is and is not:

  IS   -- a like-for-like comparison of the two models over the detections each
          would actually emit at a 0.5 operating point.
  NOT  -- an estimate of either model's full AP. Both numbers are lower than
          the untruncated value would be. Do not quote them as "AP" without
          saying what floor they were computed at.

Usage
-----
    python3 projects/viame_fish_2026/scripts/filter_predictions_by_score.py \
        --in  pred_boxes.kwcoco.zip \
        --out pred_boxes_min05.kwcoco.json \
        --min-score 0.5
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import zipfile


def load_kwcoco(fpath):
    """Read a .kwcoco.json or a .kwcoco.zip without needing kwcoco itself."""
    fpath = pathlib.Path(fpath)
    if fpath.suffix == '.zip' or zipfile.is_zipfile(fpath):
        with zipfile.ZipFile(fpath) as archive:
            name = next(n for n in archive.namelist() if n.endswith('.json'))
            with archive.open(name) as handle:
                return json.load(handle)
    return json.loads(fpath.read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--in', dest='in_fpath', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--min-score', type=float, required=True)
    args = parser.parse_args(argv)

    dset = load_kwcoco(args.in_fpath)
    before = dset['annotations']
    after = [a for a in before if a.get('score', 1.0) >= args.min_score]
    dset['annotations'] = after

    n_images = len(dset['images'])
    hit_before = len({a['image_id'] for a in before})
    hit_after = len({a['image_id'] for a in after})

    out_fpath = pathlib.Path(args.out)
    out_fpath.parent.mkdir(parents=True, exist_ok=True)
    out_fpath.write_text(json.dumps(dset))

    print('min score:            {}'.format(args.min_score))
    print('predictions:          {:,} -> {:,} ({:.2%} kept)'.format(
        len(before), len(after), len(after) / max(len(before), 1)))
    print('dets per image:       {:.1f} -> {:.1f}'.format(
        len(before) / max(n_images, 1), len(after) / max(n_images, 1)))
    print('images with >=1 det:  {:,} -> {:,} of {:,}'.format(
        hit_before, hit_after, n_images))
    if after:
        scores = sorted(a['score'] for a in after)
        print('score min/med/max:    {:.4f} / {:.4f} / {:.4f}'.format(
            scores[0], scores[len(scores) // 2], scores[-1]))
    print('wrote {}'.format(out_fpath))
    return 0


if __name__ == '__main__':
    sys.exit(main())
