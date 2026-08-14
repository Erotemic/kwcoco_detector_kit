#!/usr/bin/env python3
"""
Partition the converted Train/ kwcoco into sequence-disjoint train and vali.

Why not a random split
----------------------
This is tracked video. 665,228 boxes span 16,867 tracks, so a track carries ~39
annotated frames of the same individual fish. A frame-level random split puts
adjacent frames of one track on both sides of the boundary -- the validation
images are then near-duplicates of training images, the metric is inflated, and
checkpoint selection rewards memorization. That is exactly what happened to the
RF-DETR baseline, whose `valid/` and `test/` annotation files are byte-identical
(same md5) and were carved by the trainer out of its own input directory.

So we split on whole sequences, never frames.

Deployment grouping: the subtler leak
-------------------------------------
Sequence-disjoint is necessary but not sufficient here, because several
sequences are *the same scene*. `SEFSC-SeaMap-761901231-Cam2` and `-Cam3` are
two cameras on one deployment, looking at one baited site at one time.
`CDFW-LakeCam-April-Tules1/2/4` are repeat drops at one site in one month.
Splitting those across train and vali leaks background, lighting, and often the
same physical fish.

So sequences are grouped into *deployments* first -- trailing camera and
segment indices stripped -- and a whole deployment goes to one side. This is
deliberately conservative: over-grouping costs a little split granularity,
under-grouping silently inflates the validation score.

Balancing
---------
Within each collection (CDFW-LakeCam, SEFSC-SeaMap, PIFSC-MOUSS,
IFREMER-DropCam, ...) deployments are shuffled under a fixed seed and assigned
to vali until that collection's share of ANNOTATIONS reaches the target
fraction. Balancing on annotation count rather than sequence count keeps the
validation set representative -- sequence lengths here span 3 orders of
magnitude.

The held-out test set is NOT produced here. It comes from the corpus's own
`Test/` directory, converted separately, which the RF-DETR run provably never
saw.

Usage
-----
    python3 projects/viame_fish_2026/scripts/build_splits.py \
        --in-kwcoco  $HOME/ssd-data/fish_kcd/bundle/train_all.kwcoco.json \
        --out-train  $HOME/ssd-data/fish_kcd/bundle/train.kwcoco.json \
        --out-vali   $HOME/ssd-data/fish_kcd/bundle/vali.kwcoco.json \
        --vali-fraction 0.12 --seed 0 --stride 1
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import re
import sys

# A trailing `-Cam<N>`, optionally followed by segment indices (`-Cam3-4`).
#
# Scoped deliberately narrowly. The correlation that actually matters is
# multiple cameras on ONE deployment: SEFSC-SeaMap-761901231-Cam2 and -Cam3 are
# simultaneous views of one baited station, frequently showing the same
# individual fish, so they must not straddle the split. Same-site-different-day
# sequences (CDFW-LakeCam-April-Tules1 vs -Tules2) are left as separate
# deployments -- they differ in water, light and fish, and grouping them is not
# worth the granularity.
#
# An earlier, greedier version of this pattern stripped every trailing numeric
# token and collapsed 295 of the 378 SEFSC-SeaMap sequences into a single group,
# which makes a balanced split impossible. Keep it anchored on `Cam`.
DEPLOYMENT_STRIP_RE = re.compile(r'[-_](?:Cam|CAM|cam)\d+(?:[-_]\d+)*$')


def deployment_key(sequence_name):
    """Collapse `...-761901231-Cam2` and `...-761901231-Cam3` to one key."""
    stripped = DEPLOYMENT_STRIP_RE.sub('', sequence_name)
    return stripped or sequence_name


def collection_key(sequence_name):
    """Coarse source grouping, e.g. `SEFSC-SeaMap`, used for stratification."""
    parts = sequence_name.split('-')
    return '-'.join(parts[:2]) if len(parts) >= 2 else sequence_name


def assign_splits(video_stats, vali_fraction, seed):
    """Choose which deployments go to vali. Returns {video_name: 'train'|'vali'}.

    `video_stats` maps sequence name -> annotation count.
    """
    by_collection = collections.defaultdict(lambda: collections.defaultdict(list))
    for name in video_stats:
        by_collection[collection_key(name)][deployment_key(name)].append(name)

    assignment = {}
    rng = random.Random(seed)

    for collection in sorted(by_collection):
        deployments = by_collection[collection]
        total = sum(video_stats[n] for names in deployments.values() for n in names)
        target = total * vali_fraction

        order = sorted(deployments)
        rng.shuffle(order)

        chosen = 0
        for deployment in order:
            names = deployments[deployment]
            weight = sum(video_stats[n] for n in names)
            # Take the deployment if we are still short of target. The check is
            # against the running total, so we stop at the first deployment that
            # reaches it rather than overshooting by a whole large sequence.
            if chosen < target:
                for name in names:
                    assignment[name] = 'vali'
                chosen += weight
            else:
                for name in names:
                    assignment[name] = 'train'
    return assignment


def subset(dset, keep_video_ids, stride):
    """Extract a split, optionally keeping every `stride`-th annotated frame.

    Subsampling happens per sequence over frames sorted by index, so a stride of
    3 keeps temporally spread frames rather than a contiguous third.
    """
    images_by_video = collections.defaultdict(list)
    for image in dset['images']:
        if image['video_id'] in keep_video_ids:
            images_by_video[image['video_id']].append(image)

    keep_image_ids = set()
    images = []
    for video_id in sorted(images_by_video):
        ordered = sorted(images_by_video[video_id], key=lambda im: im['frame_index'])
        for position, image in enumerate(ordered):
            if stride > 1 and position % stride:
                continue
            images.append(image)
            keep_image_ids.add(image['id'])

    annotations = [a for a in dset['annotations'] if a['image_id'] in keep_image_ids]
    videos = [v for v in dset['videos'] if v['id'] in keep_video_ids]
    return {
        'images': sorted(images, key=lambda im: im['id']),
        'annotations': annotations,
        'categories': dset['categories'],
        'videos': videos,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--in-kwcoco', required=True)
    parser.add_argument('--out-train', required=True)
    parser.add_argument('--out-vali', required=True)
    parser.add_argument('--vali-fraction', type=float, default=0.12)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1,
                        help='keep every Nth annotated frame per sequence '
                             '(1 = all, matching what RF-DETR consumed)')
    parser.add_argument('--manifest', default=None,
                        help='where to record the sequence assignment '
                             '(default: alongside --out-train)')
    args = parser.parse_args(argv)

    dset = json.loads(pathlib.Path(args.in_kwcoco).read_text())

    video_name = {v['id']: v['name'] for v in dset['videos']}
    counts = collections.Counter()
    for annotation in dset['annotations']:
        counts[annotation['image_id']] += 1
    per_video = collections.Counter()
    for image in dset['images']:
        per_video[video_name[image['video_id']]] += counts.get(image['id'], 0)

    assignment = assign_splits(per_video, args.vali_fraction, args.seed)

    train_ids = {vid for vid, name in video_name.items()
                 if assignment.get(name) == 'train'}
    vali_ids = {vid for vid, name in video_name.items()
                if assignment.get(name) == 'vali'}

    train = subset(dset, train_ids, args.stride)
    vali = subset(dset, vali_ids, args.stride)

    # The invariant this whole script exists to guarantee.
    overlap = {v['name'] for v in train['videos']} & {v['name'] for v in vali['videos']}
    assert not overlap, 'sequence appears in both splits: {}'.format(overlap)
    deploy_overlap = ({deployment_key(v['name']) for v in train['videos']} &
                      {deployment_key(v['name']) for v in vali['videos']})
    assert not deploy_overlap, 'deployment straddles the split: {}'.format(deploy_overlap)

    for path, split in ((args.out_train, train), (args.out_vali, vali)):
        out_fpath = pathlib.Path(path)
        out_fpath.parent.mkdir(parents=True, exist_ok=True)
        out_fpath.write_text(json.dumps(split))

    manifest_fpath = pathlib.Path(
        args.manifest or (pathlib.Path(args.out_train).parent / 'split_manifest.json'))
    manifest_fpath.write_text(json.dumps({
        'in_kwcoco': str(args.in_kwcoco),
        'vali_fraction': args.vali_fraction,
        'seed': args.seed,
        'stride': args.stride,
        'assignment': assignment,
        'counts': {
            'train': {'sequences': len(train['videos']),
                      'images': len(train['images']),
                      'annotations': len(train['annotations'])},
            'vali': {'sequences': len(vali['videos']),
                     'images': len(vali['images']),
                     'annotations': len(vali['annotations'])},
        },
    }, indent=2, sort_keys=True))

    for label, split in (('train', train), ('vali', vali)):
        print('{:<6} sequences={:<5} images={:<9,} annotations={:,}'.format(
            label, len(split['videos']), len(split['images']),
            len(split['annotations'])))
    total_ann = len(train['annotations']) + len(vali['annotations'])
    if total_ann:
        print('vali annotation share: {:.1%} (target {:.1%})'.format(
            len(vali['annotations']) / total_ann, args.vali_fraction))
    print('manifest: {}'.format(manifest_fpath))
    return 0


if __name__ == '__main__':
    sys.exit(main())
