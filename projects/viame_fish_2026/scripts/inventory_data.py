#!/usr/bin/env python3
"""
Inventory a VIAME-style training directory (e.g. FishTrack23-Latest/Train) and
emit a compact JSON + markdown summary that is safe to transfer off the host.

Why this exists
---------------
Detector hyperparameters that actually matter -- input resolution, chip size and
step, class balance, whether a run needs tiling at all -- are determined by the
*shape* of the corpus: image dimensions, object size distribution, category
frequency, and how much of the corpus is video vs still imagery. None of that is
recorded anywhere in this repo for FishTrack23, so we measure it once and commit
the summary.

The output is deliberately small (a few hundred KB at most): no imagery, only
counts, percentiles, and short file-header excerpts. That makes it cheap to
rsync from the training host back to a workstation for analysis.

Dependencies
------------
Standard library only, on purpose. This has to run on whatever interpreter the
training host happens to have -- system python3, or VIAME's bundled python --
without installing anything into either.

Usage
-----
    python3 projects/viame_fish_2026/scripts/inventory_data.py \
        --input /home/$USER/ssd-data/FishTrack23-Latest/Train \
        --out-dir /data/users/$USER/fish/inventory

Reads image dimensions from file headers (no full decode). Sampled per
directory by default; pass --dim-sample 0 to measure every image.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import pathlib
import struct
import sys

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv', '.mpg', '.mpeg', '.wmv'}
ANNOT_SUFFIXES = {'.csv', '.json', '.txt', '.kw18', '.xml'}

# VIAME CSV column layout:
#   0 detection/track id, 1 video-or-image identifier, 2 frame index,
#   3-6 bbox TL_x TL_y BR_x BR_y, 7 confidence, 8 target length,
#   9.. repeated (species, confidence) pairs and +attributes.
VIAME_MIN_COLUMNS = 9


def percentiles(values, points=(1, 5, 25, 50, 75, 95, 99)):
    """Nearest-rank percentiles of a numeric sequence."""
    if not values:
        return {}
    ordered = sorted(values)
    out = {}
    for point in points:
        index = int(round((point / 100.0) * (len(ordered) - 1)))
        out[f'p{point}'] = round(ordered[index], 2)
    return out


def read_image_size(path):
    """Return (width, height) from an image header without decoding pixels."""
    try:
        with open(path, 'rb') as file:
            head = file.read(32)
            if head[:8] == b'\x89PNG\r\n\x1a\n':
                width, height = struct.unpack('>II', head[16:24])
                return int(width), int(height)
            if head[:2] == b'\xff\xd8':
                return _read_jpeg_size(file)
            if head[:2] in (b'II', b'MM'):
                return _read_tiff_size(file, head)
    except Exception:
        return None
    return None


def _read_jpeg_size(file):
    file.seek(2)
    while True:
        marker = file.read(2)
        if len(marker) < 2 or marker[0] != 0xFF:
            return None
        # SOF0..SOF15, excluding the non-frame markers DHT/JPG/DAC.
        if 0xC0 <= marker[1] <= 0xCF and marker[1] not in (0xC4, 0xC8, 0xCC):
            file.read(3)  # segment length + sample precision
            height, width = struct.unpack('>HH', file.read(4))
            return int(width), int(height)
        (length,) = struct.unpack('>H', file.read(2))
        file.seek(length - 2, 1)


def _read_tiff_size(file, head):
    endian = '<' if head[:2] == b'II' else '>'
    (offset,) = struct.unpack(endian + 'I', head[4:8])
    file.seek(offset)
    (count,) = struct.unpack(endian + 'H', file.read(2))
    width = height = None
    for _ in range(count):
        entry = file.read(12)
        tag, dtype = struct.unpack(endian + 'HH', entry[:4])
        if tag in (256, 257):
            fmt = 'H' if dtype == 3 else 'I'
            (value,) = struct.unpack(endian + fmt, entry[8:8 + struct.calcsize(fmt)])
            if tag == 256:
                width = int(value)
            else:
                height = int(value)
    if width and height:
        return width, height
    return None


def looks_like_viame_csv(path):
    """Cheap sniff: a VIAME CSV has a '# 1: Detection' banner or 9+ columns."""
    try:
        with open(path, 'r', errors='replace') as file:
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith('#'):
                    if 'Detection or Track' in stripped or 'Track-id' in stripped:
                        return True
                    continue
                return len(stripped.split(',')) >= VIAME_MIN_COLUMNS
    except Exception:
        return False
    return False


def parse_viame_csv(path):
    """Extract per-category counts and box geometry from one VIAME CSV."""
    categories = collections.Counter()
    widths = []
    heights = []
    areas = []
    frames = set()
    sources = set()
    malformed = 0
    track_ids = set()

    with open(path, 'r', errors='replace') as file:
        for row in csv.reader(file):
            if not row or row[0].lstrip().startswith('#'):
                continue
            if len(row) < VIAME_MIN_COLUMNS:
                malformed += 1
                continue
            try:
                tl_x, tl_y, br_x, br_y = (float(value) for value in row[3:7])
            except ValueError:
                malformed += 1
                continue
            width = br_x - tl_x
            height = br_y - tl_y
            widths.append(width)
            heights.append(height)
            areas.append(max(width, 0.0) * max(height, 0.0))
            track_ids.add(row[0].strip())
            sources.add(row[1].strip())
            frames.add(row[2].strip())

            # Species/confidence pairs start at column 9. Attribute tokens begin
            # with '(' or '+' and are not class names.
            tail = row[VIAME_MIN_COLUMNS:]
            named = False
            for index in range(0, len(tail) - 1, 2):
                name = tail[index].strip()
                if not name or name.startswith(('(', '+')):
                    continue
                categories[name] += 1
                named = True
                break
            if not named:
                categories['<unlabeled>'] += 1

    return {
        'path': str(path),
        'n_boxes': len(widths),
        'n_malformed_rows': malformed,
        'n_frames_referenced': len(frames),
        'n_track_ids': len(track_ids),
        'n_source_identifiers': len(sources),
        'source_identifiers_sample': sorted(sources)[:5],
        'categories': dict(categories.most_common()),
        'box_width_px': percentiles(widths),
        'box_height_px': percentiles(heights),
        'box_area_px': percentiles(areas),
    }


def head_lines(path, count=5, max_chars=400):
    lines = []
    try:
        with open(path, 'r', errors='replace') as file:
            for line in file:
                lines.append(line.rstrip('\n')[:max_chars])
                if len(lines) >= count:
                    break
    except Exception as ex:
        lines.append(f'<unreadable: {ex}>')
    return lines


def inventory(input_dpath, dim_sample):
    input_dpath = pathlib.Path(input_dpath).resolve()
    per_dir = {}
    annot_files = []
    ext_counts = collections.Counter()
    total_bytes = 0

    for dpath, _dnames, fnames in os.walk(input_dpath):
        dpath = pathlib.Path(dpath)
        images = []
        videos = []
        annots = []
        for fname in fnames:
            fpath = dpath / fname
            suffix = fpath.suffix.lower()
            ext_counts[suffix] += 1
            try:
                total_bytes += fpath.stat().st_size
            except OSError:
                pass
            if suffix in IMAGE_SUFFIXES:
                images.append(fpath)
            elif suffix in VIDEO_SUFFIXES:
                videos.append(fpath)
            elif suffix in ANNOT_SUFFIXES:
                annots.append(fpath)

        if not (images or videos or annots):
            continue

        sizes = collections.Counter()
        sample = images if dim_sample <= 0 else images[:dim_sample]
        for fpath in sample:
            size = read_image_size(fpath)
            if size is not None:
                sizes[size] += 1

        relpath = str(dpath.relative_to(input_dpath)) or '.'
        per_dir[relpath] = {
            'n_images': len(images),
            'n_videos': len(videos),
            'n_annotation_files': len(annots),
            'image_sizes_sampled': {f'{w}x{h}': n for (w, h), n in sizes.most_common(8)},
            'video_names_sample': sorted(p.name for p in videos)[:5],
        }

        for fpath in annots:
            entry = {
                'relpath': str(fpath.relative_to(input_dpath)),
                'bytes': fpath.stat().st_size if fpath.exists() else None,
                'is_viame_csv': False,
                'head': head_lines(fpath),
            }
            if fpath.suffix.lower() == '.csv' and looks_like_viame_csv(fpath):
                entry['is_viame_csv'] = True
                try:
                    entry['viame'] = parse_viame_csv(fpath)
                    entry['viame'].pop('path', None)
                except Exception as ex:
                    entry['viame_parse_error'] = repr(ex)
            annot_files.append(entry)

    return {
        'input_dpath': str(input_dpath),
        'total_bytes': total_bytes,
        'extension_counts': dict(ext_counts.most_common()),
        'n_directories_with_content': len(per_dir),
        'directories': per_dir,
        'annotation_files': annot_files,
    }


def summarize(report):
    """Roll the per-file VIAME stats up into corpus-level totals."""
    categories = collections.Counter()
    n_boxes = 0
    n_viame_csv = 0
    widths = collections.Counter()
    heights = collections.Counter()
    for entry in report['annotation_files']:
        viame = entry.get('viame')
        if not viame:
            continue
        n_viame_csv += 1
        n_boxes += viame['n_boxes']
        categories.update(viame['categories'])
        for key, value in viame['box_width_px'].items():
            widths[key] += value
        for key, value in viame['box_height_px'].items():
            heights[key] += value

    image_sizes = collections.Counter()
    n_images = n_videos = 0
    for info in report['directories'].values():
        n_images += info['n_images']
        n_videos += info['n_videos']
        for key, count in info['image_sizes_sampled'].items():
            image_sizes[key] += count

    summary = {
        'n_images': n_images,
        'n_videos': n_videos,
        'n_viame_csv_files': n_viame_csv,
        'n_boxes': n_boxes,
        'n_categories': len(categories),
        'categories': dict(categories.most_common()),
        'image_sizes_sampled': dict(image_sizes.most_common(12)),
        # Mean-of-per-file percentiles: a rough shape indicator, not an exact
        # corpus percentile. Use the per-file numbers for anything precise.
        'mean_box_width_percentiles': {
            key: round(value / n_viame_csv, 2) for key, value in widths.items()
        } if n_viame_csv else {},
        'mean_box_height_percentiles': {
            key: round(value / n_viame_csv, 2) for key, value in heights.items()
        } if n_viame_csv else {},
    }
    report['summary'] = summary
    return summary


def render_markdown(report):
    summary = report['summary']
    lines = [
        '# FishTrack23 data inventory',
        '',
        f"Input: `{report['input_dpath']}`",
        f"Total bytes: {report['total_bytes'] / 1e9:.2f} GB",
        '',
        '## Corpus shape',
        '',
        '| quantity | value |',
        '|---|---|',
        f"| images | {summary['n_images']} |",
        f"| videos | {summary['n_videos']} |",
        f"| VIAME CSV files | {summary['n_viame_csv_files']} |",
        f"| boxes | {summary['n_boxes']} |",
        f"| categories | {summary['n_categories']} |",
        '',
        '## Categories',
        '',
        '| category | boxes |',
        '|---|---|',
    ]
    for name, count in summary['categories'].items():
        lines.append(f'| {name} | {count} |')

    lines += ['', '## Image sizes (sampled)', '', '| size | files |', '|---|---|']
    for size, count in summary['image_sizes_sampled'].items():
        lines.append(f'| {size} | {count} |')

    lines += [
        '',
        '## Box size (mean of per-file percentiles, pixels)',
        '',
        '| percentile | width | height |',
        '|---|---|---|',
    ]
    widths = summary['mean_box_width_percentiles']
    heights = summary['mean_box_height_percentiles']
    for key in widths:
        lines.append(f"| {key} | {widths[key]} | {heights.get(key, '')} |")

    lines += ['', '## Extensions', '', '| suffix | files |', '|---|---|']
    for suffix, count in report['extension_counts'].items():
        lines.append(f'| `{suffix or "<none>"}` | {count} |')

    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True,
                        help='training input directory (VF_INPUT_DPATH)')
    parser.add_argument('--out-dir', required=True,
                        help='where to write inventory.json / inventory.md')
    parser.add_argument('--dim-sample', type=int, default=25,
                        help='images per directory to header-probe (0 = all)')
    args = parser.parse_args(argv)

    report = inventory(args.input, args.dim_sample)
    summarize(report)

    out_dpath = pathlib.Path(args.out_dir)
    out_dpath.mkdir(parents=True, exist_ok=True)
    json_fpath = out_dpath / 'inventory.json'
    md_fpath = out_dpath / 'inventory.md'
    json_fpath.write_text(json.dumps(report, indent=2, sort_keys=False))
    md_fpath.write_text(render_markdown(report))

    print(render_markdown(report))
    print(f'wrote {json_fpath}')
    print(f'wrote {md_fpath}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
