"""
Unit tests for the FishTrack23 inventory tool.

The tool runs unattended on a training host against a corpus we cannot see from
here, so its parsers (VIAME CSV, image headers) get exercised against synthetic
fixtures rather than trusted on first contact with real data.
"""
import json
import struct
import subprocess
import sys

import pytest

from inventory_data import (
    inventory,
    looks_like_viame_csv,
    parse_viame_csv,
    percentiles,
    read_image_size,
    summarize,
)

VIAME_CSV_TEXT = '''\
# 1: Detection or Track-id, 2: Video or Image Identifier, 3: Unique Frame Identifier, 4-7: Img-bbox(TL_x,TL_y,BR_x,BR_y), 8: Detection or Length Confidence, 9: Target Length (0 or -1 if invalid), 10-11+: Repeated Species, Confidence Pairs or Attributes
# metadata line that should be ignored
1,frame_0001.png,0,10,20,50,80,0.9,-1,rockfish,0.9
2,frame_0001.png,0,100,100,140,160,0.8,-1,flatfish,0.8,+kp head 1 2
3,frame_0002.png,1,0,0,20,30,1.0,-1,rockfish,1.0
'''


def write_png(fpath, width, height):
    payload = struct.pack('>II', width, height) + b'\x08\x02\x00\x00\x00'
    fpath.write_bytes(
        b'\x89PNG\r\n\x1a\n'
        + struct.pack('>I', len(payload))
        + b'IHDR'
        + payload
        + b'\x00' * 4
    )


def write_jpeg(fpath, width, height):
    """A header-only JPEG: enough bytes for the SOF0 probe, not decodable."""
    app0 = b'\xff\xe0' + struct.pack('>H', 16) + b'JFIF\x00' + b'\x00' * 9
    sof0 = (
        b'\xff\xc0'
        + struct.pack('>H', 17)
        + b'\x08'
        + struct.pack('>HH', height, width)
        + b'\x03'
        + b'\x00' * 9
    )
    fpath.write_bytes(b'\xff\xd8' + app0 + sof0 + b'\xff\xd9')


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / 'Train'
    seq = root / 'sequence_a'
    seq.mkdir(parents=True)
    (seq / 'annotations.csv').write_text(VIAME_CSV_TEXT)
    write_png(seq / 'frame_0001.png', 4000, 3000)
    write_png(seq / 'frame_0002.png', 4000, 3000)

    other = root / 'sequence_b'
    other.mkdir()
    write_jpeg(other / 'frame_0001.jpg', 1920, 1080)
    (other / 'notes.txt').write_text('not an annotation file\n')
    return root


def test_read_png_size(tmp_path):
    fpath = tmp_path / 'a.png'
    write_png(fpath, 640, 480)
    assert read_image_size(fpath) == (640, 480)


def test_read_jpeg_size(tmp_path):
    fpath = tmp_path / 'a.jpg'
    write_jpeg(fpath, 1920, 1080)
    assert read_image_size(fpath) == (1920, 1080)


def test_read_image_size_rejects_non_image(tmp_path):
    fpath = tmp_path / 'a.txt'
    fpath.write_text('hello')
    assert read_image_size(fpath) is None


def test_looks_like_viame_csv(tmp_path):
    good = tmp_path / 'good.csv'
    good.write_text(VIAME_CSV_TEXT)
    bad = tmp_path / 'bad.csv'
    bad.write_text('name,value\nfoo,1\n')
    assert looks_like_viame_csv(good)
    assert not looks_like_viame_csv(bad)


def test_parse_viame_csv_counts_and_geometry(tmp_path):
    fpath = tmp_path / 'annotations.csv'
    fpath.write_text(VIAME_CSV_TEXT)
    stats = parse_viame_csv(fpath)
    assert stats['n_boxes'] == 3
    assert stats['n_malformed_rows'] == 0
    assert stats['categories'] == {'rockfish': 2, 'flatfish': 1}
    assert stats['n_frames_referenced'] == 2
    assert stats['n_track_ids'] == 3
    # Boxes are 40x60, 40x60, 20x30 -> median width 40.
    assert stats['box_width_px']['p50'] == 40.0


def test_parse_viame_csv_skips_attribute_tokens(tmp_path):
    """A '+'-prefixed keypoint token must not be mistaken for a species name."""
    fpath = tmp_path / 'attrs.csv'
    fpath.write_text('9,f.png,0,0,0,10,10,1,-1,+kp head 1 2,0,tuna,0.7\n')
    stats = parse_viame_csv(fpath)
    assert stats['categories'] == {'tuna': 1}


def test_parse_viame_csv_flags_malformed_rows(tmp_path):
    fpath = tmp_path / 'broken.csv'
    fpath.write_text('1,f.png,0,10,20\n2,f.png,0,a,b,c,d,1,-1,tuna,1\n')
    stats = parse_viame_csv(fpath)
    assert stats['n_boxes'] == 0
    assert stats['n_malformed_rows'] == 2


def test_percentiles_are_nearest_rank():
    assert percentiles([1, 2, 3, 4, 5], points=(50,)) == {'p50': 3}
    assert percentiles([]) == {}


def test_inventory_walks_tree(corpus):
    report = inventory(corpus, dim_sample=25)
    summary = summarize(report)
    assert summary['n_images'] == 3
    assert summary['n_videos'] == 0
    assert summary['n_boxes'] == 3
    assert summary['categories'] == {'rockfish': 2, 'flatfish': 1}
    assert summary['image_sizes_sampled'] == {'4000x3000': 2, '1920x1080': 1}
    assert set(report['directories']) == {'sequence_a', 'sequence_b'}
    # notes.txt is picked up as a candidate annotation file but not parsed.
    annots = {entry['relpath']: entry for entry in report['annotation_files']}
    assert annots['sequence_a/annotations.csv']['is_viame_csv']
    assert not annots['sequence_b/notes.txt']['is_viame_csv']


def test_cli_writes_reports(corpus, tmp_path):
    """The tool is invoked as a script on the training host, so test that path."""
    out_dpath = tmp_path / 'inventory'
    script = str(pytest.importorskip('inventory_data').__file__)
    subprocess.run(
        [sys.executable, script, '--input', str(corpus), '--out-dir', str(out_dpath)],
        check=True, capture_output=True,
    )
    report = json.loads((out_dpath / 'inventory.json').read_text())
    assert report['summary']['n_boxes'] == 3
    assert 'rockfish' in (out_dpath / 'inventory.md').read_text()
