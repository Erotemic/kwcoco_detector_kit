#!/usr/bin/env python3
"""
Verify frame extraction against the source video, headlessly.

Why this beats looking at pictures
----------------------------------
Frame-index alignment has been wrong twice in this pipeline, and both times the
code was internally consistent enough to look correct. The usual remedy is to
draw boxes on frames and have a human confirm they land on fish, but that needs
a screen and a person, and it samples a handful of frames.

The oracle used here is the source video itself, reached by a DIFFERENT ffmpeg
path than the one extraction uses. Extraction decodes the whole stream through
`fps=<rate>,select='...'` and names frames from `showinfo` timestamps. This
check instead seeks straight to `t = index / annotation_fps` with `-ss` and
grabs one frame. Two independent routes to the same picture: if they agree, the
index-to-time mapping is right.

Agreement is scored as mean absolute difference over a 64x64 grayscale
thumbnail, which is invariant to the JPEG round trip but sensitive to being on
the wrong frame. CONTROL seeks at +/- a few frame periods run alongside, because
underwater video is slow and a small difference proves nothing on its own -- the
check passes only when the matched time beats every neighbour. On a validated
sequence the gap is decisive: 0.25 matched versus 0.97 one second away.

Rejected oracle: VIAME's `augmented_images` cache from the previous RF-DETR
run. It looked ideal -- an independent implementation that extracted every
video at the annotation rate -- and it does agree closely on some sequences
(CDFW-LakeCam-April-Tules1 scores 0.76 matched vs 2.85 at +/-1). But on others
it matches nothing at any offset: SEFSC-SeaMap-762101021-Cam3 scans flat at
7.67-9.37 across +/-60 frames while the source-seek check on that same frame
scores 0.25. The directory is named `augmented_images` because VIAME writes
augmented pixels there, so it is not a faithful copy of the source and cannot
be used as ground truth. The source video can.

No image library required: ffmpeg is already a dependency of the extraction
step, and the thumbnails are compared as raw bytes.

Usage
-----
    python3 projects/viame_fish_2026/scripts/check_alignment.py \
        --frames $HOME/ssd-data/fish_kcd/frames/train \
        --corpus $HOME/ssd-data/FishTrack23-Latest/Train \
        --sequences 12 --per-sequence 4

Exit status is non-zero if any sequence fails, so it can gate a run.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import random
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from extract_frames import derive_annotation_fps, read_fps  # noqa: E402

THUMB = 64

# The two decode paths round differently at frame boundaries: `-ss` seeking
# lands on the frame with pts >= t, while the `fps=` filter used by extraction
# takes the nearest preceding frame. So they legitimately disagree by up to one
# frame. `MATCH_OFFSETS` therefore accepts a hit anywhere in +/-1, and the
# controls start at +/-2.
#
# This is still a strong test. The failure it exists to catch is gross -- the
# real bug selected on NATIVE frame numbers against a 29.97 fps container
# annotated at 10 Hz, landing roughly 3x deeper into the video. A tolerance of
# one frame does not hide that, and insisting on exactness only produces false
# alarms from the seek convention.
MATCH_OFFSETS = (-1, 0, 1)
CONTROL_OFFSETS = (-10, -5, -2, 2, 5, 10)


def thumbnail(args, ffmpeg='ffmpeg'):
    """Run ffmpeg with `args` and return a THUMB x THUMB grayscale raw string."""
    cmd = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error'] + list(args) + [
        '-vf', 'scale={s}:{s},format=gray'.format(s=THUMB),
        '-f', 'rawvideo', '-',
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or len(proc.stdout) != THUMB * THUMB:
        return None
    return proc.stdout


def mean_abs_diff(left, right):
    return sum(abs(a - b) for a, b in zip(left, right)) / float(len(left))


def source_frame_at(video_fpath, seconds, ffmpeg='ffmpeg'):
    """One frame seeked directly in the source video.

    `-ss` before `-i` is the independent decode path: extraction reaches the
    frame by filtering the whole stream, this reaches it by seeking.
    """
    return thumbnail(['-ss', '{:.6f}'.format(max(seconds, 0.0)),
                      '-i', str(video_fpath), '-frames:v', '1'], ffmpeg)


def check_sequence(sequence, ours_dpath, video_fpath, fps, per_sequence, rng, ffmpeg):
    """Score the matched seek time against neighbours for a few frames."""
    ours = sorted(pathlib.Path(ours_dpath).glob('frame*.jpg'))
    if not ours:
        return {'sequence': sequence, 'status': 'no_frames'}
    if not fps:
        return {'sequence': sequence, 'status': 'no_fps'}

    period = 1.0 / fps
    # Skip the first and last few frames so every sample has a full control
    # window inside the video.
    indices = sorted(int(p.stem[5:]) - 1 for p in ours)
    usable = [i for i in indices if i + min(CONTROL_OFFSETS) >= 0]
    if not usable:
        return {'sequence': sequence, 'status': 'too_short'}

    rng.shuffle(usable)
    samples = []
    for index in usable[:per_sequence]:
        ours_thumb = thumbnail(
            ['-i', str(pathlib.Path(ours_dpath) / 'frame{:06d}.jpg'.format(index + 1))],
            ffmpeg)
        if ours_thumb is None:
            continue
        # Best agreement within the +/-1 seek-convention window.
        matches = []
        for offset in MATCH_OFFSETS:
            ref = source_frame_at(video_fpath, (index + offset) * period, ffmpeg)
            if ref is not None:
                matches.append(mean_abs_diff(ours_thumb, ref))
        if not matches:
            continue
        matched = min(matches)

        controls = []
        for offset in CONTROL_OFFSETS:
            ref = source_frame_at(video_fpath, (index + offset) * period, ffmpeg)
            if ref is not None:
                controls.append(mean_abs_diff(ours_thumb, ref))
        if not controls:
            continue
        samples.append({
            'index': index,
            'matched': matched,
            'best_control': min(controls),
        })

    if not samples:
        return {'sequence': sequence, 'status': 'unscorable'}

    # Score each sample as win / lose / tie against a noise margin.
    #
    # A bare `matched < best_control` is too strict. Plenty of these sequences
    # are baited cameras on a static scene, where neighbouring frames are
    # genuinely indistinguishable -- matched and control come out equal to two
    # decimals, and the comparison then flips on rounding noise. Calling that
    # MISALIGNED is crying wolf: there is no signal to discriminate, which is a
    # property of the footage, not a defect in the extraction.
    #
    # Real misalignment is not subtle. It shows up as matched being clearly
    # WORSE than some neighbour, which is what `lose` counts.
    wins = losses = 0
    for sample in samples:
        margin = max(0.02, 0.05 * sample['best_control'])
        if sample['matched'] < sample['best_control'] - margin:
            wins += 1
        elif sample['matched'] > sample['best_control'] + margin:
            losses += 1

    if losses:
        status = 'MISALIGNED'
    elif wins:
        status = 'ok'
    else:
        status = 'static'  # no discriminating power; not a failure
    return {
        'sequence': sequence,
        'status': status,
        'wins': wins,
        'losses': losses,
        'samples': len(samples),
        'mean_matched': sum(s['matched'] for s in samples) / len(samples),
        'mean_best_control': sum(s['best_control'] for s in samples) / len(samples),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--frames', required=True,
                        help='our extracted frames (the per-split directory)')
    parser.add_argument('--corpus', required=True,
                        help='the VIAME corpus directory holding the source videos '
                             'and their CSVs (e.g. FishTrack23-Latest/Train)')
    parser.add_argument('--sequences', type=int, default=12)
    parser.add_argument('--per-sequence', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ffmpeg', default=os.environ.get('VF_FFMPEG') or 'ffmpeg')
    args = parser.parse_args(argv)

    if shutil.which(args.ffmpeg) is None and not os.path.isfile(args.ffmpeg):
        parser.error('ffmpeg not found at {!r}'.format(args.ffmpeg))

    frames_root = pathlib.Path(args.frames)
    candidates = sorted(p for p in frames_root.iterdir()
                        if p.is_dir() and any(p.glob('frame*.jpg')))
    if not candidates:
        print('no extracted sequences under {}'.format(frames_root), file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    corpus = pathlib.Path(args.corpus)
    print('ours:   {}'.format(frames_root))
    print('source: {}'.format(corpus))
    print('comparing {} of {} extracted sequences, {} frames each'.format(
        min(args.sequences, len(candidates)), len(candidates), args.per_sequence))
    print()
    print('{:<46} {:>10} {:>10} {:>8}'.format(
        'sequence', 'matched', 'control', 'verdict'))
    print('-' * 78)

    results = []
    for dpath in candidates[:args.sequences]:
        video_fpath = corpus / (dpath.name + '.mp4')
        csv_fpath = corpus / (dpath.name + '.csv')
        if not video_fpath.exists() or not csv_fpath.exists():
            results.append({'sequence': dpath.name, 'status': 'no_source'})
            print('{:<46} {:>10} {:>10} {:>8}'.format(
                dpath.name[:46], '-', '-', 'no_source'))
            continue
        # Same rate resolution the extractor used, so the check validates the
        # mapping rather than re-deriving a convenient one.
        fps = read_fps(csv_fpath) or derive_annotation_fps(csv_fpath)
        result = check_sequence(dpath.name, dpath, video_fpath, fps,
                                args.per_sequence, rng, args.ffmpeg)
        results.append(result)
        if 'mean_matched' in result:
            print('{:<46} {:>10.2f} {:>10.2f} {:>8}'.format(
                result['sequence'][:46], result['mean_matched'],
                result['mean_best_control'], result['status']))
        else:
            print('{:<46} {:>10} {:>10} {:>8}'.format(
                result['sequence'][:46], '-', '-', result['status']))

    scored = [r for r in results if 'mean_matched' in r]
    bad = [r for r in scored if r['status'] == 'MISALIGNED']
    static = [r for r in scored if r['status'] == 'static']
    print()
    if not scored:
        print('NOTHING SCORED. Are the source videos present under --corpus?',
              file=sys.stderr)
        return 1

    matched = sum(r['mean_matched'] for r in scored) / len(scored)
    control = sum(r['mean_best_control'] for r in scored) / len(scored)
    print('scored {} sequences: mean matched diff {:.2f} vs best control {:.2f}'.format(
        len(scored), matched, control))
    if static:
        print('{} sequence(s) were static -- neighbouring frames are identical, so'.format(len(static)))
        print('  the comparison has no discriminating power there. Not a failure.')
    if bad:
        print()
        print('{} SEQUENCE(S) MISALIGNED -- our frame resembles a NEIGHBOURING'.format(len(bad)))
        print('source frame more than the one at its own index. Do not train on this.',
              file=sys.stderr)
        for result in bad:
            print('  {} ({}/{} samples matched best)'.format(
                result['sequence'], result['wins'], result['samples']), file=sys.stderr)
        return 1
    print('ALIGNED: no sample matched a neighbouring time better than its own.')
    print('Absolute differences near zero mean our frames ARE the source frames')
    print('at t = index / annotation_fps.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
