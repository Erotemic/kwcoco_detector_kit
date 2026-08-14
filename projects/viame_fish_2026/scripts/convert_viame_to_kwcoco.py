#!/usr/bin/env python3
"""
Convert a FishTrack23 / VIAME training directory into a kwcoco dataset.

Why this exists
---------------
The kit had no reader for the VIAME *alternating class/score* CSV format.
`projects/viame_sealions_2026/scripts/convert_sealions_csv_to_kwcoco.py` handles
a different, headered format and says so in its own docstring. The row-level
logic here follows `scripts/inventory_data.py:parse_viame_csv`, which is already
under test and has parsed this exact corpus with zero malformed rows.

Stdlib only, like the inventory tool, so the prep step runs under whatever
python3 the training host has without installing into it.

The two layouts, and why frame indexing differs between them
------------------------------------------------------------
FishTrack23 ships sequences in two shapes, and a frame index means something
different in each:

  video     `<name>.mp4` beside `<name>.csv`. CSV column 2 is a timestamp and
            column 3 the frame index; the two satisfy `timestamp == index / fps`
            exactly across this corpus. Frames are extracted ahead of this
            script by `extract_frames.py` into `frame%06d.jpg`, 1-based, so
            index `i` is `frame{i+1:06d}.jpg`.

  imagedir  `<name>/` holding frames plus `<name>.csv`. Column 2 is EMPTY, and
            the frame index is a position in the sorted file listing. Verified
            against PIFSC-MOUSS-Onaga1: 541 images, indices 0..540, exact.

Getting this wrong attaches every box to the wrong image and yields a silently
bad model, so both mappings are covered by unit tests.

Class handling
--------------
Single-class by default, via the corpus's own `Train/labels.txt`. That file is
a single line -- the output class `fish` followed by 321 aliases -- and it is
the same file the existing RF-DETR model was trained through (its
`rf_detr_mgpu_params.json` records `class_names: ["fish"]`). Reusing it is what
makes the two models label-compatible.

Annotations whose species is absent from labels.txt are DROPPED, which is what
VIAME did (it logged them to TRAINING_DATA_WARNINGS.txt and excluded them). In
this corpus that is the four `non_fish_*` categories, ~25.4k boxes. They are
not exhaustively annotated across the release, per the dataset readme, so
dropping them is also the defensible choice on its own merits -- but the reason
it is done *here* is comparability with the baseline.

Usage
-----
    python3 projects/viame_fish_2026/scripts/convert_viame_to_kwcoco.py \
        --input   $HOME/ssd-data/FishTrack23-Latest/Train \
        --frames  $HOME/ssd-data/fish_kcd/frames/train \
        --labels  $HOME/ssd-data/FishTrack23-Latest/Train/labels.txt \
        --out     $HOME/ssd-data/fish_kcd/bundle/train_all.kwcoco.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from inventory_data import read_image_size  # noqa: E402

IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
VIDEO_SUFFIXES = ('.mp4', '.avi', '.mov', '.mkv')
VIAME_MIN_COLUMNS = 9

# Everything from column 9 on is either a (species, confidence) pair or an
# attribute token. Attributes are what VIAME writes for polygons, keypoints,
# and free-form notes; they are never class names.
ATTRIBUTE_PREFIXES = ('(', '+')


def load_label_map(labels_fpath):
    """Map every alias in a VIAME labels.txt to its output class.

    Each line is `<output_class> <alias> <alias> ...`; the output class is also
    a valid alias for itself. Returns None when no labels file is given, which
    means "keep species names as-is".
    """
    if labels_fpath is None:
        return None
    mapping = {}
    with open(labels_fpath, 'r', errors='replace') as file:
        for line in file:
            tokens = line.split()
            if not tokens:
                continue
            output_class = tokens[0]
            for token in tokens:
                mapping[token] = output_class
    return mapping


def parse_row_category(row):
    """First (species, confidence) pair of a VIAME row, or None.

    VIAME writes the pairs in descending confidence, so the first non-attribute
    token is the assigned class.
    """
    tail = row[VIAME_MIN_COLUMNS:]
    for index in range(0, len(tail), 2):
        name = tail[index].strip()
        if not name or name.startswith(ATTRIBUTE_PREFIXES):
            continue
        return name
    return None


def parse_viame_csv(csv_fpath):
    """Rows of one VIAME CSV as dicts. Box-only: polygons/keypoints ignored."""
    records = []
    malformed = 0
    with open(csv_fpath, 'r', errors='replace') as file:
        for row in csv.reader(file):
            if not row or row[0].lstrip().startswith('#'):
                continue
            if len(row) < VIAME_MIN_COLUMNS:
                malformed += 1
                continue
            try:
                frame_index = int(row[2])
                tl_x, tl_y, br_x, br_y = (float(value) for value in row[3:7])
            except (ValueError, IndexError):
                malformed += 1
                continue
            width = br_x - tl_x
            height = br_y - tl_y
            if width <= 0 or height <= 0:
                malformed += 1
                continue
            records.append({
                'track_id': row[0].strip(),
                'frame_index': frame_index,
                'bbox': [tl_x, tl_y, width, height],
                'category': parse_row_category(row),
            })
    return records, malformed


def clip_bbox(bbox, width, height):
    """Clip [x, y, w, h] to the image. None when nothing survives.

    Measured on PIFSC-MOUSS-Onaga1: 7 of 2,285 boxes overhang by 1-13 px,
    which is a fish leaving the frame with the annotator still tracking it.
    """
    x, y, box_width, box_height = bbox
    x0 = min(max(x, 0.0), width)
    y0 = min(max(y, 0.0), height)
    x1 = min(max(x + box_width, 0.0), width)
    y1 = min(max(y + box_height, 0.0), height)
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def discover_sequences(input_dpath):
    """Every annotated sequence under a VIAME training directory.

    Yields dicts with `name`, `kind` ('video' | 'imagedir') and `csv_fpath`.
    """
    input_dpath = pathlib.Path(input_dpath)
    sequences = []

    for csv_fpath in sorted(input_dpath.glob('*.csv')):
        stem = csv_fpath.stem
        for suffix in VIDEO_SUFFIXES:
            if (input_dpath / (stem + suffix)).exists():
                sequences.append({'name': stem, 'kind': 'video',
                                  'csv_fpath': csv_fpath})
                break

    for sub in sorted(p for p in input_dpath.iterdir() if p.is_dir()):
        csvs = sorted(sub.glob('*.csv'))
        images = sorted(p for p in sub.iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if csvs and images:
            sequences.append({'name': sub.name, 'kind': 'imagedir',
                              'csv_fpath': csvs[0], 'image_fpaths': images})

    return sequences


def resolve_frame_paths(sequence, frames_dpath):
    """Map frame index -> image path for one sequence.

    Returns a dict rather than a list because video extractions are sparse:
    only annotated frames exist on disk.
    """
    if sequence['kind'] == 'imagedir':
        # Index is a position in the sorted listing.
        return {index: path for index, path in enumerate(sequence['image_fpaths'])}

    seq_dpath = pathlib.Path(frames_dpath) / sequence['name']
    if not seq_dpath.is_dir():
        return {}
    mapping = {}
    for path in seq_dpath.glob('frame*.jpg'):
        try:
            # frame%06d.jpg is 1-based; the CSV index is 0-based.
            mapping[int(path.stem[5:]) - 1] = path
        except ValueError:
            continue
    return mapping


def convert(input_dpath, frames_dpath, labels_fpath, verify_dims=False):
    """Build a COCO/kwcoco dict from a VIAME training directory."""
    label_map = load_label_map(labels_fpath)
    sequences = discover_sequences(input_dpath)

    dset = {'images': [], 'annotations': [], 'categories': [], 'videos': []}
    category_ids = {}
    stats = collections.Counter()
    dropped_categories = collections.Counter()

    image_id = annotation_id = 0

    for video_id, sequence in enumerate(sequences, start=1):
        records, malformed = parse_viame_csv(sequence['csv_fpath'])
        stats['malformed_rows'] += malformed
        if not records:
            stats['sequences_empty'] += 1
            continue

        frame_paths = resolve_frame_paths(sequence, frames_dpath)
        if not frame_paths:
            stats['sequences_missing_frames'] += 1
            print('  WARNING: no frames on disk for {} ({})'.format(
                sequence['name'], sequence['kind']), file=sys.stderr)
            continue

        dset['videos'].append({
            'id': video_id,
            'name': sequence['name'],
            'kind': sequence['kind'],
        })

        # All frames of a sequence share dimensions, so probe once. --verify-dims
        # reads every header instead, at ~250k extra file opens.
        seq_size = None
        if not verify_dims:
            for index in sorted(frame_paths):
                seq_size = read_image_size(frame_paths[index])
                if seq_size:
                    break
            if seq_size is None:
                stats['sequences_unreadable'] += 1
                print('  WARNING: could not read image size for {}'.format(
                    sequence['name']), file=sys.stderr)
                continue

        by_frame = collections.defaultdict(list)
        for record in records:
            by_frame[record['frame_index']].append(record)

        for frame_index in sorted(by_frame):
            path = frame_paths.get(frame_index)
            if path is None:
                # Annotation references a frame that does not exist. Seen at the
                # tail of a few videos where the CSV runs one frame past what the
                # container actually decodes.
                stats['annotations_missing_frame'] += len(by_frame[frame_index])
                stats['frames_missing'] += 1
                continue

            size = read_image_size(path) if verify_dims else seq_size
            if size is None:
                stats['frames_unreadable'] += 1
                continue

            image_id += 1
            dset['images'].append({
                'id': image_id,
                'file_name': str(path),
                'width': size[0],
                'height': size[1],
                'video_id': video_id,
                'frame_index': frame_index,
                'sequence': sequence['name'],
            })
            stats['images'] += 1

            for record in by_frame[frame_index]:
                name = record['category']
                if name is None:
                    stats['annotations_unlabeled'] += 1
                    continue
                if label_map is not None:
                    mapped = label_map.get(name)
                    if mapped is None:
                        # Not in labels.txt -- VIAME excluded these too.
                        dropped_categories[name] += 1
                        stats['annotations_dropped_unknown_class'] += 1
                        continue
                    name = mapped

                if name not in category_ids:
                    category_ids[name] = len(category_ids) + 1
                    dset['categories'].append({'id': category_ids[name], 'name': name})

                # Annotators track a fish as it swims out of frame, so boxes
                # can overhang the image edge by a few pixels. Only the visible
                # extent is learnable, and an out-of-bounds box upsets COCO
                # evaluators and augmentation pipelines downstream.
                bbox = clip_bbox(record['bbox'], size[0], size[1])
                if bbox is None:
                    stats['annotations_outside_image'] += 1
                    continue
                if bbox != record['bbox']:
                    stats['annotations_clipped'] += 1

                annotation_id += 1
                dset['annotations'].append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': category_ids[name],
                    'bbox': [round(value, 2) for value in bbox],
                    # VIAME track ids restart per sequence, so namespace them.
                    'track_id': '{}_{}'.format(sequence['name'], record['track_id']),
                })
                stats['annotations'] += 1

    stats['sequences'] = len(dset['videos'])
    return dset, stats, dropped_categories


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True,
                        help='VIAME training directory (Train/ or Test/)')
    parser.add_argument('--frames', required=True,
                        help='directory of extracted video frames for this input')
    parser.add_argument('--labels', default=None,
                        help='VIAME labels.txt used to fold species into output '
                             'classes. Omit to keep raw species names.')
    parser.add_argument('--out', required=True, help='output .kwcoco.json path')
    parser.add_argument('--verify-dims', action='store_true',
                        help='read every image header instead of one per sequence')
    args = parser.parse_args(argv)

    print('input:  {}'.format(args.input))
    print('frames: {}'.format(args.frames))
    print('labels: {}'.format(args.labels))
    print()

    dset, stats, dropped = convert(args.input, args.frames, args.labels,
                                   verify_dims=args.verify_dims)

    out_fpath = pathlib.Path(args.out)
    out_fpath.parent.mkdir(parents=True, exist_ok=True)
    out_fpath.write_text(json.dumps(dset))

    print()
    print('sequences:   {:,}'.format(stats['sequences']))
    print('images:      {:,}'.format(stats['images']))
    print('annotations: {:,}'.format(stats['annotations']))
    print('categories:  {}'.format([c['name'] for c in dset['categories']][:10]))
    print()
    for key in sorted(stats):
        if key.startswith(('annotations_', 'frames_', 'sequences_', 'malformed')):
            print('  {:<36} {:,}'.format(key, stats[key]))
    if dropped:
        print()
        print('  dropped classes absent from labels.txt:')
        for name, count in dropped.most_common(10):
            print('    {:<32} {:,}'.format(name, count))
    print()
    print('wrote {}'.format(out_fpath))
    return 0


if __name__ == '__main__':
    sys.exit(main())
