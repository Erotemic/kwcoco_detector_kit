#!/usr/bin/env python3
"""
Draw annotations onto sampled frames so a human can confirm they land on fish.

Why this exists
---------------
Frame-index alignment has been wrong twice in this pipeline, and both times the
code was self-consistent enough to look fine:

  1. Video frames were selected by NATIVE frame number when a VIAME index
     actually counts annotation frames (29.97 fps container, 10 Hz
     annotation) -- every box ~3x too deep into the video.
  2. Before that, the annotation rate was read from a column that parses as a
     timestamp but is not a clock.

Unit tests catch structure, not semantics. The only check that closes the loop
is looking at a box on a frame and seeing a fish inside it. This makes that
cheap: sample across sequences, burn the boxes in, write JPEGs to a directory.

Uses ffmpeg's `drawbox`, which the extraction step already depends on, so this
adds no new requirement.

A misalignment is obvious at a glance: boxes land on open water, on the
substrate, or clipped to frame edges, and they will not track the animal across
consecutive frames of one track. `--track` samples consecutive frames of a
single track, which is the most sensitive view -- a correct extraction shows
the box following one fish smoothly.

Usage
-----
    python3 projects/viame_fish_2026/scripts/spotcheck_annotations.py \
        --kwcoco $HOME/ssd-data/fish_kcd/bundle/train.kwcoco.json \
        --out-dir $HOME/ssd-data/fish_kcd/spotcheck \
        --count 40

    # follow one track across consecutive frames (strongest signal)
    python3 .../spotcheck_annotations.py --kwcoco ... --out-dir ... --track

Then look at them:  eog <out-dir>    or   scp them somewhere with a screen.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys

BOX_COLORS = ['lime', 'red', 'cyan', 'yellow', 'magenta', 'orange']


def build_drawbox_chain(annotations, thickness=3):
    """One `drawbox` filter per annotation, chained.

    Colors cycle by track so that in a multi-fish frame it is visually obvious
    which box belongs to which animal across successive frames.
    """
    parts = []
    for index, annotation in enumerate(annotations):
        x, y, width, height = annotation['bbox']
        color = BOX_COLORS[hash(str(annotation.get('track_id', index))) % len(BOX_COLORS)]
        parts.append(
            'drawbox=x={:.0f}:y={:.0f}:w={:.0f}:h={:.0f}:color={}:t={}'.format(
                x, y, width, height, color, thickness))
    return ','.join(parts) if parts else 'null'


def render(image, annotations, out_fpath, ffmpeg='ffmpeg', thickness=3):
    """Burn boxes into one frame. Returns True on success."""
    chain = build_drawbox_chain(annotations, thickness)
    cmd = [
        ffmpeg, '-nostdin', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', image['file_name'],
        '-vf', chain,
        '-q:v', '3',
        str(out_fpath),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print('  FAILED {}: {}'.format(
            image['file_name'], proc.stderr.strip().splitlines()[-1:]), file=sys.stderr)
        return False
    return True


def load(kwcoco_fpath):
    dset = json.loads(pathlib.Path(kwcoco_fpath).read_text())
    by_image = collections.defaultdict(list)
    for annotation in dset['annotations']:
        by_image[annotation['image_id']].append(annotation)
    images = {image['id']: image for image in dset['images']}
    return dset, images, by_image


def pick_spread(images, by_image, count, rng):
    """Sample annotated frames spread across sequences, not clustered in one."""
    by_sequence = collections.defaultdict(list)
    for image_id, annotations in by_image.items():
        if annotations:
            by_sequence[images[image_id].get('sequence', '?')].append(image_id)

    sequences = sorted(by_sequence)
    rng.shuffle(sequences)
    picked = []
    # Round-robin over sequences so a 10k-frame video cannot crowd out the rest.
    while len(picked) < count and sequences:
        for sequence in list(sequences):
            pool = by_sequence[sequence]
            if not pool:
                sequences.remove(sequence)
                continue
            picked.append(pool.pop(rng.randrange(len(pool))))
            if len(picked) >= count:
                break
    return picked


def pick_track(images, by_image, count, rng):
    """Consecutive frames of one track -- the most sensitive alignment view."""
    frames_of_track = collections.defaultdict(list)
    for image_id, annotations in by_image.items():
        for annotation in annotations:
            frames_of_track[annotation.get('track_id')].append(image_id)

    long_tracks = [track for track, frames in frames_of_track.items()
                   if len(frames) >= count]
    if not long_tracks:
        return []
    track = rng.choice(sorted(long_tracks, key=str))
    ordered = sorted(set(frames_of_track[track]),
                     key=lambda image_id: images[image_id]['frame_index'])
    print('following track {} ({} annotated frames)'.format(track, len(ordered)))
    return ordered[:count]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kwcoco', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--count', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--track', action='store_true',
                        help='sample consecutive frames of one track instead of '
                             'spreading across sequences')
    parser.add_argument('--thickness', type=int, default=3)
    parser.add_argument('--ffmpeg', default=os.environ.get('VF_FFMPEG') or 'ffmpeg')
    args = parser.parse_args(argv)

    if shutil.which(args.ffmpeg) is None and not os.path.isfile(args.ffmpeg):
        parser.error('ffmpeg not found at {!r}'.format(args.ffmpeg))

    dset, images, by_image = load(args.kwcoco)
    print('bundle:      {}'.format(args.kwcoco))
    print('images:      {:,}  annotations: {:,}'.format(
        len(dset['images']), len(dset['annotations'])))

    rng = random.Random(args.seed)
    chosen = (pick_track(images, by_image, args.count, rng) if args.track
              else pick_spread(images, by_image, args.count, rng))
    if not chosen:
        print('nothing to sample', file=sys.stderr)
        return 1

    out_dpath = pathlib.Path(args.out_dir)
    out_dpath.mkdir(parents=True, exist_ok=True)

    written = 0
    for order, image_id in enumerate(chosen):
        image = images[image_id]
        annotations = by_image[image_id]
        # Encode the provenance in the filename: which sequence, which frame
        # index, how many boxes. That is what you need to chase a bad one.
        name = '{:03d}_{}_frame{:06d}_n{}.jpg'.format(
            order, image.get('sequence', 'seq'), image['frame_index'], len(annotations))
        if render(image, annotations, out_dpath / name, args.ffmpeg, args.thickness):
            written += 1

    print()
    print('wrote {} annotated frames to {}'.format(written, out_dpath))
    print()
    print('What you are checking: every box should contain a fish, and boxes')
    print('should sit on animals rather than open water or substrate. With')
    print('--track, one box should follow a single fish smoothly across frames.')
    print('Boxes drifting steadily off-target across a track means the frame')
    print('index is misaligned.')
    return 0 if written else 1


if __name__ == '__main__':
    sys.exit(main())
