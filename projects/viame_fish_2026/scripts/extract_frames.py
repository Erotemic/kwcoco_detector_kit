#!/usr/bin/env python3
"""
Extract the annotated frames of a VIAME-style corpus to JPEG on fast storage.

Why only the annotated frames
-----------------------------
FishTrack23's 420 training videos hold roughly 4.4M frames, of which 250,753
carry an annotation -- about 6%. The other 94% are not merely wasted disk: an
unannotated frame that still contains a fish is a false negative, and training
on it actively teaches the detector to suppress real targets. VIAME's own
augmentation cache extracted every frame as PNG (771 GB, measured) and then
relied on the chipper to sort it out. We take the annotated subset directly.

At JPEG q95 the annotated subset is ~75 GB, which is what makes it possible to
put the whole training corpus on aiq-gpu's NVMe (506 GB free) instead of the
md0 RAID array where the previous run's frames and chips lived.

Frame-index alignment is the whole ballgame
-------------------------------------------
A VIAME CSV addresses frames by integer index. If our extracted filenames are
off by even one relative to that index, every box in the run is attached to the
wrong image and training silently produces garbage -- there is no error, just a
bad model. So this script does not *assume* an alignment, it *recovers* one:

  * The `select` filter picks the annotated frames out of the stream.
  * `showinfo`, placed after `select`, reports the ORIGINAL presentation
    timestamp of each frame that survived.
  * Each emitted file is then named from its own `pts_time`, not from its
    position in the output sequence.

That last point is what makes a mid-stream decoder hiccup harmless: a dropped
frame shortens the output, but every remaining file still carries its true
index, rather than shifting all its successors by one. The mapping is verified
against the CSV's own timestamp column, which we confirmed satisfies
`timestamp == index / fps` exactly across the corpus.

Frames are written as `frame%06d.jpg` with a 1-BASED counter, matching VIAME's
extract_frames convention (CSV index 0 -> frame000001.jpg), so an index means
the same thing here as in a VIAME run.

Usage
-----
    python3 projects/viame_fish_2026/scripts/extract_frames.py \
        --input  $HOME/ssd-data/FishTrack23-Latest/Train \
        --out-dir $HOME/ssd-data/fish_kcd/frames/train \
        --jobs 32

Re-running is cheap: a sequence whose expected frame count is already on disk
is skipped, so an interrupted extraction resumes where it stopped.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

VIDEO_SUFFIXES = ('.mp4', '.avi', '.mov', '.mkv')
IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
VIAME_MIN_COLUMNS = 9

# showinfo writes one line per frame that reached it. We want pts_time, which
# is the frame's position in the ORIGINAL stream even though showinfo sits
# downstream of select.
SHOWINFO_RE = re.compile(r'pts_time:\s*([0-9]+\.?[0-9]*)')
FPS_RE = re.compile(r'fps:\s*([0-9]+\.?[0-9]*)')


def read_fps(csv_fpath):
    """Frame rate from the VIAME CSV metadata comment, or None if absent."""
    with open(csv_fpath, 'r', errors='replace') as file:
        for line in file:
            if not line.startswith('#'):
                break
            match = FPS_RE.search(line)
            if match:
                return float(match.group(1))
    return None


def read_annotated_indices(csv_fpath):
    """The set of frame indices carrying at least one box."""
    indices = set()
    with open(csv_fpath, 'r', errors='replace') as file:
        for row in csv.reader(file):
            if not row or row[0].lstrip().startswith('#'):
                continue
            if len(row) < VIAME_MIN_COLUMNS:
                continue
            try:
                indices.add(int(row[2]))
            except (ValueError, IndexError):
                continue
    return indices


def probe_fps(video_fpath, ffprobe='ffprobe'):
    """Fall back to the container's own frame rate when the CSV omits it.

    Needed for the ~20 Train videos whose CSV carries no `fps:` comment.
    """
    cmd = [
        ffprobe, '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0',
        str(video_fpath),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    if '/' in out:
        num, den = out.split('/')
        return float(num) / float(den)
    return float(out)


def compress_to_ranges(indices):
    """[1,2,3,7,8] -> [(1,3),(7,8)]. Keeps the select expression compact."""
    ranges = []
    for index in sorted(indices):
        if ranges and index == ranges[-1][1] + 1:
            ranges[-1][1] = index
        else:
            ranges.append([index, index])
    return [(lo, hi) for lo, hi in ranges]


def build_select_expr(ranges):
    """A `select` expression matching exactly the requested frame numbers.

    `between` is inclusive on both ends. Summing the terms is the filtergraph
    idiom for OR: select keeps the frame when the expression is non-zero.
    """
    terms = [
        'eq(n\\,{})'.format(lo) if lo == hi else 'between(n\\,{}\\,{})'.format(lo, hi)
        for lo, hi in ranges
    ]
    return '+'.join(terms)


def extract_video(video_fpath, csv_fpath, out_dpath, quality=2, force=False,
                  ffmpeg='ffmpeg', ffprobe='ffprobe'):
    """Extract one video's annotated frames. Returns a result dict."""
    video_fpath = pathlib.Path(video_fpath)
    out_dpath = pathlib.Path(out_dpath)
    name = video_fpath.stem

    wanted = read_annotated_indices(csv_fpath)
    result = {
        'sequence': name,
        'kind': 'video',
        'video': str(video_fpath),
        'n_requested': len(wanted),
    }
    if not wanted:
        result.update(status='empty', n_written=0, missing=[])
        return result

    fps = read_fps(csv_fpath)
    if fps is None:
        try:
            fps = probe_fps(video_fpath, ffprobe)
            result['fps_source'] = 'ffprobe'
        except Exception as ex:
            result.update(status='error', error='fps unavailable: {}'.format(ex))
            return result
    else:
        result['fps_source'] = 'csv'
    result['fps'] = fps

    out_dpath.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in out_dpath.glob('frame*.jpg')}
    expected = {'frame{:06d}.jpg'.format(index + 1) for index in wanted}
    if not force and expected.issubset(existing):
        result.update(status='cached', n_written=len(expected), missing=[])
        return result

    ranges = compress_to_ranges(wanted)
    select_expr = build_select_expr(ranges)

    # The filtergraph goes in a file: a corpus-scale select expression can run
    # to tens of kilobytes, past what is comfortable on a command line.
    staging = pathlib.Path(tempfile.mkdtemp(prefix='kcdfish_', dir=str(out_dpath.parent)))
    filter_fpath = staging / 'filter.txt'
    filter_fpath.write_text("select='{}',showinfo".format(select_expr))

    cmd = [
        ffmpeg, '-nostdin', '-y',
        '-loglevel', 'info',
        '-i', str(video_fpath),
        '-filter_script:v', str(filter_fpath),
        # passthrough == the old `-vsync 0`: emit exactly the frames that
        # survived the filter, with no duplication or dropping of its own.
        '-fps_mode', 'passthrough',
        '-q:v', str(quality),
        str(staging / '%08d.jpg'),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            result.update(status='error', error=proc.stderr[-2000:])
            return result

        # Emission order pairs 1:1 with the showinfo lines; the INDEX comes
        # from each frame's own pts_time rather than from its ordinal, so a
        # dropped frame cannot shift the frames that follow it.
        times = [float(m) for m in SHOWINFO_RE.findall(proc.stderr)]
        produced = sorted(staging.glob('*.jpg'))
        if len(times) != len(produced):
            result.update(
                status='error',
                error='showinfo reported {} frames but {} files were written'.format(
                    len(times), len(produced)),
            )
            return result

        written = set()
        for pts_time, src in zip(times, produced):
            index = int(round(pts_time * fps))
            if index not in wanted:
                # Timestamp did not land on a requested frame: the video is
                # not the constant-rate stream the CSV assumes. Refuse rather
                # than write a misaligned frame.
                result.update(
                    status='error',
                    error='frame at pts_time={} maps to index {}, which was not requested '
                          '(variable frame rate?)'.format(pts_time, index),
                )
                return result
            dst = out_dpath / 'frame{:06d}.jpg'.format(index + 1)
            shutil.move(str(src), str(dst))
            written.add(index)

        missing = sorted(wanted - written)
        result.update(
            status='ok' if not missing else 'partial',
            n_written=len(written),
            missing=missing[:20],
            n_missing=len(missing),
        )
        return result
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def discover(input_dpath):
    """Find every sequence under a VIAME training directory.

    Two layouts coexist in FishTrack23 and they address frames differently:

      video    <name>.mp4 beside <name>.csv, CSV column 2 is a timestamp
      imagedir <name>/ containing frames and <name>.csv, column 2 is empty
               and the frame index is a position in the sorted file listing

    Only videos need extraction; image directories are already frames on disk
    and are reported here so the converter can see the full sequence list.
    """
    input_dpath = pathlib.Path(input_dpath)
    videos, imagedirs = [], []

    for csv_fpath in sorted(input_dpath.glob('*.csv')):
        stem = csv_fpath.stem
        for suffix in VIDEO_SUFFIXES:
            candidate = input_dpath / (stem + suffix)
            if candidate.exists():
                videos.append((candidate, csv_fpath))
                break

    for sub in sorted(p for p in input_dpath.iterdir() if p.is_dir()):
        csvs = sorted(sub.glob('*.csv'))
        if not csvs:
            continue
        has_images = any(
            p.suffix.lower() in IMAGE_SUFFIXES for p in sub.iterdir() if p.is_file()
        )
        if has_images:
            imagedirs.append((sub, csvs[0]))

    return videos, imagedirs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True,
                        help='VIAME training directory (e.g. FishTrack23-Latest/Train)')
    parser.add_argument('--out-dir', required=True,
                        help='destination for extracted frames (put this on the NVMe)')
    parser.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 8) // 4),
                        help='concurrent ffmpeg processes')
    parser.add_argument('--quality', type=int, default=2,
                        help='JPEG quality for -q:v (2 is visually lossless, 31 worst)')
    parser.add_argument('--limit', type=int, default=0,
                        help='extract only the first N videos (smoke test)')
    parser.add_argument('--force', action='store_true',
                        help='re-extract sequences that are already complete')
    parser.add_argument('--manifest', default=None,
                        help='where to write the JSON manifest '
                             '(default: <out-dir>/extraction_manifest.json)')
    parser.add_argument('--ffmpeg', default=os.environ.get('VF_FFMPEG') or 'ffmpeg',
                        help='ffmpeg binary')
    parser.add_argument('--ffprobe', default=os.environ.get('VF_FFPROBE') or 'ffprobe',
                        help='ffprobe binary, needed for videos whose CSV omits fps')
    args = parser.parse_args(argv)

    for label, binary in (('ffmpeg', args.ffmpeg), ('ffprobe', args.ffprobe)):
        if shutil.which(binary) is None and not os.path.isfile(binary):
            parser.error(
                '{} not found at {!r}. Install both with `apt install ffmpeg`. '
                'This script must run on the training host.'.format(label, binary))

    input_dpath = pathlib.Path(args.input).resolve()
    out_dpath = pathlib.Path(args.out_dir).resolve()
    videos, imagedirs = discover(input_dpath)
    if args.limit:
        videos = videos[:args.limit]

    print('input:      {}'.format(input_dpath))
    print('out-dir:    {}'.format(out_dpath))
    print('videos:     {} (need frame extraction)'.format(len(videos)))
    print('imagedirs:  {} (already frames on disk)'.format(len(imagedirs)))
    print('jobs:       {}'.format(args.jobs))
    print('ffmpeg:     {}'.format(args.ffmpeg))
    print('ffprobe:    {}'.format(args.ffprobe))
    print()

    out_dpath.mkdir(parents=True, exist_ok=True)
    results = []
    counts = collections.Counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(extract_video, video, csv_fpath,
                        out_dpath / video.stem, args.quality, args.force,
                        args.ffmpeg, args.ffprobe): video
            for video, csv_fpath in videos
        }
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            video = futures[future]
            try:
                result = future.result()
            except Exception as ex:
                result = {'sequence': video.stem, 'kind': 'video',
                          'status': 'error', 'error': repr(ex)}
            results.append(result)
            counts[result['status']] += 1
            flag = '' if result['status'] in ('ok', 'cached', 'empty') else '  <-- '
            print('[{:>4}/{}] {:<10} {}{}'.format(
                done, len(videos), result['status'], result['sequence'], flag), flush=True)
            if result['status'] == 'error':
                print('           {}'.format(str(result.get('error'))[:400]), flush=True)

    for dpath, csv_fpath in imagedirs:
        n_images = sum(1 for p in dpath.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        results.append({
            'sequence': dpath.name, 'kind': 'imagedir', 'status': 'in_place',
            'source_dpath': str(dpath), 'n_images': n_images,
            'n_requested': len(read_annotated_indices(csv_fpath)),
        })
        counts['in_place'] += 1

    manifest_fpath = pathlib.Path(
        args.manifest or (out_dpath / 'extraction_manifest.json'))
    manifest_fpath.parent.mkdir(parents=True, exist_ok=True)
    manifest_fpath.write_text(json.dumps({
        'input_dpath': str(input_dpath),
        'out_dpath': str(out_dpath),
        'jpeg_quality': args.quality,
        'status_counts': dict(counts),
        'sequences': sorted(results, key=lambda r: r['sequence']),
    }, indent=2))

    total_written = sum(r.get('n_written', 0) for r in results)
    print()
    print('status counts:  {}'.format(dict(counts)))
    print('frames written: {:,}'.format(total_written))
    print('manifest:       {}'.format(manifest_fpath))

    n_bad = counts['error']
    if n_bad:
        print('\n{} sequence(s) FAILED -- see the manifest. Nothing downstream '
              'should run until these are resolved or explicitly excluded.'.format(n_bad),
              file=sys.stderr)
    return 1 if n_bad else 0


if __name__ == '__main__':
    sys.exit(main())
