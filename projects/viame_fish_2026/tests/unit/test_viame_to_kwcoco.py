"""
Unit tests for the VIAME -> kwcoco conversion path.

The thing being defended here is frame-index alignment. A VIAME CSV addresses
frames by integer index, and FishTrack23 uses two different conventions for
what that index means (1-based extracted `frame%06d` files for video sequences;
a position in the sorted listing for image directories). If either mapping is
off by one, every box attaches to the wrong image and training produces a
silently bad model -- no exception, no warning, just a worse detector. So the
mappings are pinned by tests rather than trusted.

Both conventions were verified against the real corpus before being encoded:
PIFSC-MOUSS-Onaga1 has 541 images with CSV indices 0..540, and the video CSVs
satisfy `timestamp == index / fps` exactly.
"""
import json
import struct

import pytest

from build_splits import assign_splits, collection_key, deployment_key, subset
from convert_viame_to_kwcoco import (
    clip_bbox,
    convert,
    discover_sequences,
    load_label_map,
    parse_row_category,
    parse_viame_csv,
    resolve_frame_paths,
)
import extract_frames
from extract_frames import (
    build_select_expr,
    compress_to_ranges,
    derive_annotation_fps,
    parse_timestamp,
    passthrough_flag,
    read_annotated_indices,
    read_fps,
)

# A video-style CSV: column 2 is a timestamp, and fps comes from the metadata
# comment. Trailing (poly)/(kp) tokens are attributes, not class names.
VIDEO_CSV = '''\
# 1: Detection or Track-id,2: Video or Image Identifier,3: Unique Frame Identifier,4-7: Img-bbox(TL_x,TL_y,BR_x,BR_y),8: Detection or Length Confidence,9: Target Length,10-11+: Repeated Species,Confidence Pairs or Attributes
# metadata,fps: 10,"exported_by: ""dive:python"""
0,00:00:00.000000,0,10,20,110,90,1.0,-1,micropterus_salmoides,1.0,(poly) 1 2 3 4,(kp) head 5 6
0,00:00:00.100000,1,12,22,112,92,1.0,-1,micropterus_salmoides,1.0
1,00:00:00.100000,1,200,200,260,280,0.9,-1,lutjanus_campechanus,0.9
2,00:00:00.200000,2,5,5,25,35,1.0,-1,non_fish_bait,1.0
'''

# An imagedir-style CSV: column 2 is EMPTY and the index is a position in the
# sorted file listing.
IMAGEDIR_CSV = '''\
1,,0,298,359,324,387,1,0,etelis_coruscans,1.0,(poly) 314 360 313 361
1,,1,300,359,326,386,1,0,etelis_coruscans,1.0
1,,2,301,359,328,386,1,0,etelis_coruscans,1.0
'''

LABELS_TXT = 'fish micropterus_salmoides lutjanus_campechanus etelis_coruscans\n'


def write_jpeg(fpath, width, height):
    """Minimal JPEG with a parseable SOF0, enough for read_image_size."""
    fpath.write_bytes(
        b'\xff\xd8'
        + b'\xff\xc0' + struct.pack('>H', 17) + b'\x08'
        + struct.pack('>HH', height, width)
        + b'\x03' + b'\x00' * 9
        + b'\xff\xd9'
    )


# --------------------------------------------------------------- row parsing

def test_parse_row_category_skips_attribute_tokens():
    row = ['0', '00:00:00.0', '0', '1', '2', '3', '4', '1.0', '-1',
           'micropterus_salmoides', '1.0', '(poly) 1 2', '(kp) head 1 2']
    assert parse_row_category(row) == 'micropterus_salmoides'


def test_parse_row_category_when_attributes_come_first():
    row = ['0', '', '0', '1', '2', '3', '4', '1.0', '-1',
           '(poly) 1 2', '+note', 'etelis_coruscans', '0.9']
    assert parse_row_category(row) == 'etelis_coruscans'


def test_parse_row_category_absent():
    row = ['0', '', '0', '1', '2', '3', '4', '1.0', '-1']
    assert parse_row_category(row) is None


def test_parse_viame_csv_converts_corners_to_width_height(tmp_path):
    fpath = tmp_path / 'seq.csv'
    fpath.write_text(VIDEO_CSV)
    records, malformed = parse_viame_csv(fpath)
    assert malformed == 0
    assert len(records) == 4
    # VIAME stores TL/BR corners; kwcoco wants [x, y, w, h].
    assert records[0]['bbox'] == [10.0, 20.0, 100.0, 70.0]
    assert records[0]['frame_index'] == 0
    assert records[0]['track_id'] == '0'


def test_parse_viame_csv_rejects_degenerate_boxes(tmp_path):
    fpath = tmp_path / 'seq.csv'
    fpath.write_text('0,,0,10,20,10,90,1.0,-1,fish,1.0\n')  # zero width
    records, malformed = parse_viame_csv(fpath)
    assert records == []
    assert malformed == 1


# ------------------------------------------------------------- label folding

def test_load_label_map_folds_every_alias_and_the_class_itself(tmp_path):
    fpath = tmp_path / 'labels.txt'
    fpath.write_text(LABELS_TXT)
    mapping = load_label_map(fpath)
    assert mapping['micropterus_salmoides'] == 'fish'
    assert mapping['lutjanus_campechanus'] == 'fish'
    assert mapping['fish'] == 'fish'
    # Absent species must not silently map to anything.
    assert 'non_fish_bait' not in mapping


# ---------------------------------------------------- frame index resolution

def test_resolve_frame_paths_video_is_one_based(tmp_path):
    """CSV index i lives in frame{i+1:06d}.jpg -- VIAME's own convention."""
    frames = tmp_path / 'frames' / 'SEQ'
    frames.mkdir(parents=True)
    for number in (1, 2, 3):
        write_jpeg(frames / 'frame{:06d}.jpg'.format(number), 1920, 1200)

    mapping = resolve_frame_paths({'name': 'SEQ', 'kind': 'video'}, tmp_path / 'frames')
    assert set(mapping) == {0, 1, 2}
    assert mapping[0].name == 'frame000001.jpg'
    assert mapping[2].name == 'frame000003.jpg'


def test_resolve_frame_paths_video_is_sparse(tmp_path):
    """Only annotated frames are extracted, so gaps are normal, not an error."""
    frames = tmp_path / 'frames' / 'SEQ'
    frames.mkdir(parents=True)
    for number in (1, 500, 900):
        write_jpeg(frames / 'frame{:06d}.jpg'.format(number), 640, 480)

    mapping = resolve_frame_paths({'name': 'SEQ', 'kind': 'video'}, tmp_path / 'frames')
    assert set(mapping) == {0, 499, 899}


def test_resolve_frame_paths_imagedir_is_sorted_position(tmp_path):
    """For image directories the index is a position in the sorted listing."""
    seq = tmp_path / 'SEQ'
    seq.mkdir()
    names = ['20170310.231230.141.008460.png',
             '20170310.231230.225.008461.png',
             '20170310.231230.308.008462.png']
    for name in names:
        write_jpeg(seq / name, 968, 728)
    image_fpaths = sorted(seq.glob('*.png'))

    mapping = resolve_frame_paths(
        {'name': 'SEQ', 'kind': 'imagedir', 'image_fpaths': image_fpaths}, tmp_path)
    assert [mapping[i].name for i in (0, 1, 2)] == names


# ------------------------------------------------------------- end to end

@pytest.fixture
def video_corpus(tmp_path):
    """A one-video VIAME directory with its frames already extracted."""
    corpus = tmp_path / 'Train'
    corpus.mkdir()
    (corpus / 'SEQ.csv').write_text(VIDEO_CSV)
    (corpus / 'SEQ.mp4').write_bytes(b'\x00')
    (corpus / 'labels.txt').write_text(LABELS_TXT)

    frames = tmp_path / 'frames' / 'SEQ'
    frames.mkdir(parents=True)
    for number in (1, 2, 3):
        write_jpeg(frames / 'frame{:06d}.jpg'.format(number), 1920, 1200)
    return corpus, tmp_path / 'frames'


def test_convert_folds_to_single_class(video_corpus):
    corpus, frames = video_corpus
    dset, stats, dropped = convert(corpus, frames, corpus / 'labels.txt')

    assert [c['name'] for c in dset['categories']] == ['fish']
    # 4 rows in, 1 dropped for being absent from labels.txt.
    assert stats['annotations'] == 3
    assert dropped == {'non_fish_bait': 1}
    assert stats['annotations_dropped_unknown_class'] == 1


def test_convert_attaches_boxes_to_the_right_frame(video_corpus):
    corpus, frames = video_corpus
    dset, _, _ = convert(corpus, frames, corpus / 'labels.txt')

    by_id = {image['id']: image for image in dset['images']}
    for annotation in dset['annotations']:
        image = by_id[annotation['image_id']]
        # The box at frame 0 is the only one with x == 10.
        if annotation['bbox'][0] == 10.0:
            assert image['frame_index'] == 0
        if annotation['bbox'][0] == 200.0:
            assert image['frame_index'] == 1
    assert {im['width'] for im in dset['images']} == {1920}
    assert {im['height'] for im in dset['images']} == {1200}


def test_convert_namespaces_track_ids_across_sequences(tmp_path):
    """VIAME track ids restart per sequence, so they must be namespaced."""
    corpus = tmp_path / 'Train'
    corpus.mkdir()
    (corpus / 'labels.txt').write_text(LABELS_TXT)
    frames_root = tmp_path / 'frames'
    for name in ('A', 'B'):
        (corpus / '{}.csv'.format(name)).write_text(VIDEO_CSV)
        (corpus / '{}.mp4'.format(name)).write_bytes(b'\x00')
        seq_frames = frames_root / name
        seq_frames.mkdir(parents=True)
        for number in (1, 2, 3):
            write_jpeg(seq_frames / 'frame{:06d}.jpg'.format(number), 640, 480)

    dset, _, _ = convert(corpus, frames_root, corpus / 'labels.txt')
    track_ids = {a['track_id'] for a in dset['annotations']}
    assert 'A_0' in track_ids and 'B_0' in track_ids
    assert len({t.split('_')[0] for t in track_ids}) == 2


def test_convert_reports_annotations_whose_frame_is_missing(tmp_path):
    """A CSV that runs past the end of the decoded video must not crash.

    Seen for real: CDFW-LakeCam-April-SpiderBlocks1's CSV references frame
    10581 while the container decodes 10581 frames (max index 10580).
    """
    corpus = tmp_path / 'Train'
    corpus.mkdir()
    (corpus / 'SEQ.csv').write_text(VIDEO_CSV)
    (corpus / 'SEQ.mp4').write_bytes(b'\x00')
    (corpus / 'labels.txt').write_text(LABELS_TXT)
    frames_root = tmp_path / 'frames'
    (frames_root / 'SEQ').mkdir(parents=True)
    # Only index 0 was extracted; the CSV also references frames 1 and 2.
    write_jpeg(frames_root / 'SEQ' / 'frame000001.jpg', 640, 480)

    dset, stats, _ = convert(corpus, frames_root, corpus / 'labels.txt')
    assert stats['images'] == 1
    assert stats['annotations_missing_frame'] == 3
    assert len(dset['annotations']) == 1


def test_clip_bbox_trims_overhang():
    """Fish leaving the frame: annotators track past the edge."""
    assert clip_bbox([889.0, 469.0, 80.0, 47.0], 968, 728) == [889.0, 469.0, 79.0, 47.0]
    assert clip_bbox([-5.0, -5.0, 20.0, 20.0], 100, 100) == [0.0, 0.0, 15.0, 15.0]


def test_clip_bbox_leaves_interior_boxes_untouched():
    bbox = [10.0, 20.0, 30.0, 40.0]
    assert clip_bbox(bbox, 100, 100) == bbox


def test_clip_bbox_rejects_fully_outside():
    assert clip_bbox([200.0, 200.0, 10.0, 10.0], 100, 100) is None


def test_convert_clips_and_counts_overhanging_boxes(tmp_path):
    corpus = tmp_path / 'Train'
    corpus.mkdir()
    # Box runs 60 px past the right edge of a 100x100 frame.
    (corpus / 'SEQ.csv').write_text('0,,0,50,10,160,90,1.0,-1,etelis_coruscans,1.0\n')
    (corpus / 'SEQ.mp4').write_bytes(b'\x00')
    (corpus / 'labels.txt').write_text(LABELS_TXT)
    frames = tmp_path / 'frames' / 'SEQ'
    frames.mkdir(parents=True)
    write_jpeg(frames / 'frame000001.jpg', 100, 100)

    dset, stats, _ = convert(corpus, tmp_path / 'frames', corpus / 'labels.txt')
    assert stats['annotations_clipped'] == 1
    assert dset['annotations'][0]['bbox'] == [50.0, 10.0, 50.0, 80.0]


def test_discover_sequences_finds_both_layouts(tmp_path):
    corpus = tmp_path / 'Train'
    (corpus / 'IMGDIR').mkdir(parents=True)
    (corpus / 'VID.csv').write_text(VIDEO_CSV)
    (corpus / 'VID.mp4').write_bytes(b'\x00')
    (corpus / 'IMGDIR' / 'IMGDIR.csv').write_text(IMAGEDIR_CSV)
    write_jpeg(corpus / 'IMGDIR' / 'a.png', 10, 10)

    kinds = {s['name']: s['kind'] for s in discover_sequences(corpus)}
    assert kinds == {'VID': 'video', 'IMGDIR': 'imagedir'}


def test_discover_sequences_ignores_csv_without_media(tmp_path):
    """labels.txt-adjacent CSVs with no matching video are not sequences."""
    corpus = tmp_path / 'Train'
    corpus.mkdir()
    (corpus / 'orphan.csv').write_text(VIDEO_CSV)
    assert discover_sequences(corpus) == []


# ------------------------------------------------------- extraction helpers

def test_read_fps_from_metadata_comment(tmp_path):
    fpath = tmp_path / 'seq.csv'
    fpath.write_text(VIDEO_CSV)
    assert read_fps(fpath) == 10.0


def test_read_fps_absent_returns_none(tmp_path):
    fpath = tmp_path / 'seq.csv'
    fpath.write_text(IMAGEDIR_CSV)
    assert read_fps(fpath) is None


def test_parse_timestamp():
    assert parse_timestamp('00:00:00.100000') == pytest.approx(0.1)
    assert parse_timestamp('00:17:37.900000') == pytest.approx(1057.9)
    # Image-directory CSVs leave column 2 empty.
    assert parse_timestamp('') is None
    assert parse_timestamp('frame_0001.png') is None


def test_derive_annotation_fps_recovers_the_rate_from_timestamps(tmp_path):
    """The annotation rate must come from the CSV, not the container.

    The CDFW videos are 29.97 fps native but annotated at 10 Hz. Trusting the
    container's rate resamples wrong and attaches every box to the wrong frame.
    """
    rows = ['0,00:00:{:09.6f},{},10,20,60,80,1.0,-1,fish,1.0'.format(i / 10.0, i)
            for i in range(12)]
    fpath = tmp_path / 'seq.csv'
    fpath.write_text('\n'.join(rows) + '\n')
    assert derive_annotation_fps(fpath) == 10.0


def test_derive_annotation_fps_needs_enough_evidence(tmp_path):
    """Too few timestamped rows to distinguish a clock from coincidence."""
    fpath = tmp_path / 'seq.csv'
    fpath.write_text(VIDEO_CSV)  # 4 annotation rows
    assert derive_annotation_fps(fpath) is None


def test_derive_annotation_fps_handles_five_hertz(tmp_path):
    """349 of the 401 annotated Train videos are 5 Hz."""
    rows = ['0,00:00:{:09.6f},{},10,20,60,80,1.0,-1,fish,1.0'.format(i / 5.0, i)
            for i in range(1, 12)]
    fpath = tmp_path / 'seq.csv'
    fpath.write_text('\n'.join(rows) + '\n')
    assert derive_annotation_fps(fpath) == 5.0


def test_derive_annotation_fps_none_without_timestamps(tmp_path):
    """Image-directory CSVs have no timestamp column to derive from."""
    fpath = tmp_path / 'seq.csv'
    fpath.write_text(IMAGEDIR_CSV)
    assert derive_annotation_fps(fpath) is None


def test_derive_annotation_fps_rejects_a_near_constant_column(tmp_path):
    """SEFSC-SeaMap column 2 parses as a time but is not a clock.

    Rows look like `0:1:0.000`, `0:1:0.001`, `0:1:0.045` while the frame index
    runs into the thousands. Fitting a rate to that yielded anything from 0.2
    to 107 Hz across the corpus, and resampling at such a rate would attach
    every box to the wrong frame. It must be rejected, not fitted.
    """
    rows = ['4,0:1:0.{:03d},{},90,817,159,1006,1.0,-1,seriola_zonata,1.0'.format(
        i // 20, i + 3) for i in range(40)]
    fpath = tmp_path / 'seq.csv'
    fpath.write_text('\n'.join(rows) + '\n')
    assert derive_annotation_fps(fpath) is None


def test_derive_annotation_fps_rejects_implausible_rates(tmp_path):
    """A tiny time span over a huge index span implies an absurd rate."""
    rows = ['0,00:00:00.{:03d},{},10,20,60,80,1.0,-1,fish,1.0'.format(i, i * 100)
            for i in range(10)]
    fpath = tmp_path / 'seq.csv'
    fpath.write_text('\n'.join(rows) + '\n')
    assert derive_annotation_fps(fpath) is None


def test_read_fps_accepts_both_header_syntaxes(tmp_path):
    """`fps: 10` and `fps=5` both occur, and both are authoritative.

    Matching only the colon form dropped 19 videos (11% of annotated frames)
    into a native-rate guess.
    """
    colon = tmp_path / 'a.csv'
    colon.write_text('# metadata,fps: 10,"exported_by: ""dive:python"""\n')
    assert read_fps(colon) == 10.0

    equals = tmp_path / 'b.csv'
    equals.write_text('# 1: Detection or Track-id,2: Video\n#meta fps=5\n')
    assert read_fps(equals) == 5.0


def test_read_annotated_indices(tmp_path):
    fpath = tmp_path / 'seq.csv'
    fpath.write_text(VIDEO_CSV)
    assert read_annotated_indices(fpath) == {0, 1, 2}


def test_compress_to_ranges():
    assert compress_to_ranges([1, 2, 3, 7, 8, 20]) == [(1, 3), (7, 8), (20, 20)]
    assert compress_to_ranges([5]) == [(5, 5)]
    assert compress_to_ranges([]) == []


class _FakeRun:
    """Stand-in for subprocess.run returning a canned `ffmpeg -version`."""

    def __init__(self, stdout):
        self.stdout = stdout

    def __call__(self, *args, **kwargs):
        return self


def test_passthrough_flag_uses_vsync_before_ffmpeg_5(monkeypatch):
    """Ubuntu 22.04 ships 4.4.2, which fails hard on -fps_mode."""
    banner = ('ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 '
              'the FFmpeg developers\nbuilt with gcc 11\n')
    monkeypatch.setattr(extract_frames.subprocess, 'run', _FakeRun(banner))
    assert passthrough_flag('ffmpeg') == ['-vsync', '0']


def test_passthrough_flag_uses_fps_mode_from_ffmpeg_5(monkeypatch):
    banner = 'ffmpeg version 7.0.2-static https://johnvansickle.com/ffmpeg/\n'
    monkeypatch.setattr(extract_frames.subprocess, 'run', _FakeRun(banner))
    assert passthrough_flag('ffmpeg') == ['-fps_mode', 'passthrough']


def test_passthrough_flag_handles_the_n_prefixed_form(monkeypatch):
    """Some distro builds report `ffmpeg version n6.0`."""
    monkeypatch.setattr(extract_frames.subprocess, 'run',
                        _FakeRun('ffmpeg version n6.0\n'))
    assert passthrough_flag('ffmpeg') == ['-fps_mode', 'passthrough']


def test_passthrough_flag_falls_back_when_version_is_unreadable(monkeypatch):
    """Git-snapshot builds report a date, not a number. Prefer the wider option."""
    monkeypatch.setattr(extract_frames.subprocess, 'run',
                        _FakeRun('ffmpeg version 2023-05-01-git-abc123\n'))
    assert passthrough_flag('ffmpeg') == ['-vsync', '0']


def test_build_select_expr_uses_eq_for_singletons():
    expr = build_select_expr([(1, 3), (20, 20)])
    # Commas are escaped for the filtergraph parser; '+' is its OR.
    assert expr == r'between(n\,1\,3)+eq(n\,20)'


# --------------------------------------------------------------- splitting

def test_deployment_key_groups_cameras_on_one_station():
    """The leak that matters: simultaneous cameras on one baited station."""
    assert deployment_key('SEFSC-SeaMap-761901231-Cam2') == 'SEFSC-SeaMap-761901231'
    assert deployment_key('SEFSC-SeaMap-761901231-Cam3') == 'SEFSC-SeaMap-761901231'
    assert deployment_key('SEFSC-SeaMap-761901329-Cam3-4') == 'SEFSC-SeaMap-761901329'


def test_deployment_key_does_not_over_group():
    """A greedier earlier version collapsed 295 of 378 SEFSC sequences into one.

    Trailing digits that are not a camera index denote genuinely different
    deployments and must stay distinct, or a balanced split is impossible.
    """
    assert deployment_key('CDFW-LakeCam-April-Tules1') == 'CDFW-LakeCam-April-Tules1'
    assert deployment_key('SEFSC-SeaMap-JRS-28') == 'SEFSC-SeaMap-JRS-28'
    assert deployment_key('PIFSC-MOUSS-Onaga1') == 'PIFSC-MOUSS-Onaga1'
    assert deployment_key('IFREMER-DropCam-Fish-7') == 'IFREMER-DropCam-Fish-7'


def test_collection_key():
    assert collection_key('SEFSC-SeaMap-761901231-Cam2') == 'SEFSC-SeaMap'
    assert collection_key('CDFW-LakeCam-April-Tules1') == 'CDFW-LakeCam'


def test_assign_splits_never_straddles_a_deployment():
    stats = {'SEFSC-SeaMap-{}-Cam{}'.format(station, cam): 100
             for station in range(30) for cam in range(3)}
    assignment = assign_splits(stats, vali_fraction=0.2, seed=0)

    train = {deployment_key(n) for n, s in assignment.items() if s == 'train'}
    vali = {deployment_key(n) for n, s in assignment.items() if s == 'vali'}
    assert not (train & vali)
    assert vali and train


def test_assign_splits_covers_every_collection():
    stats = {}
    for collection in ('SEFSC-SeaMap', 'CDFW-LakeCam', 'PIFSC-MOUSS'):
        for index in range(10):
            stats['{}-Site{}'.format(collection, index)] = 100
    assignment = assign_splits(stats, vali_fraction=0.2, seed=0)
    vali_collections = {collection_key(n) for n, s in assignment.items() if s == 'vali'}
    assert vali_collections == {'SEFSC-SeaMap', 'CDFW-LakeCam', 'PIFSC-MOUSS'}


def test_assign_splits_does_not_overshoot_on_a_coarse_population():
    """Few, large deployments must not blow past the target fraction.

    Taking a deployment whenever `chosen < target` overshoots by a whole
    deployment. On an 89-sequence smoke run that put vali at 30.4% of a 12%
    target. Choosing whichever side lands closer keeps it near the mark even
    when the population is lumpy.
    """
    # All one collection: names must share their first two hyphen tokens,
    # or collection_key puts each in a collection of its own.
    stats = {'SEFSC-SeaMap-Site{}'.format(i): weight
             for i, weight in enumerate([5000, 4000, 3000, 900, 400, 200, 100])}
    assignment = assign_splits(stats, vali_fraction=0.12, seed=0)
    total = sum(stats.values())
    vali = sum(stats[n] for n, s in assignment.items() if s == 'vali')
    assert 0.04 < vali / total < 0.22, 'vali share {:.1%} is far off 12%'.format(
        vali / total)


def test_assign_splits_skips_a_too_large_deployment_and_keeps_looking():
    """One huge deployment must not end the search for a usable smaller one."""
    stats = {'SEFSC-SeaMap-Huge': 10000, 'SEFSC-SeaMap-Small1': 600,
             'SEFSC-SeaMap-Small2': 500, 'SEFSC-SeaMap-Small3': 400}
    assignment = assign_splits(stats, vali_fraction=0.10, seed=0)
    assert assignment['SEFSC-SeaMap-Huge'] == 'train'
    assert any(s == 'vali' for s in assignment.values())


def test_assign_splits_is_deterministic_for_a_seed():
    stats = {'C-Site{}'.format(i): 10 for i in range(20)}
    assert (assign_splits(stats, 0.2, 7) == assign_splits(stats, 0.2, 7))


def test_subset_stride_spreads_frames_over_time():
    """Stride must sample across the sequence, not take a contiguous chunk."""
    dset = {
        'images': [{'id': i, 'video_id': 1, 'frame_index': i} for i in range(9)],
        'annotations': [{'id': i, 'image_id': i} for i in range(9)],
        'categories': [{'id': 1, 'name': 'fish'}],
        'videos': [{'id': 1, 'name': 'SEQ'}],
    }
    result = subset(dset, {1}, stride=3)
    assert [im['frame_index'] for im in result['images']] == [0, 3, 6]
    assert {a['image_id'] for a in result['annotations']} == {0, 3, 6}


def test_subset_drops_annotations_of_excluded_videos():
    dset = {
        'images': [{'id': 1, 'video_id': 1, 'frame_index': 0},
                   {'id': 2, 'video_id': 2, 'frame_index': 0}],
        'annotations': [{'id': 1, 'image_id': 1}, {'id': 2, 'image_id': 2}],
        'categories': [{'id': 1, 'name': 'fish'}],
        'videos': [{'id': 1, 'name': 'A'}, {'id': 2, 'name': 'B'}],
    }
    result = subset(dset, {1}, stride=1)
    assert [im['id'] for im in result['images']] == [1]
    assert [a['id'] for a in result['annotations']] == [1]
    assert [v['name'] for v in result['videos']] == ['A']
