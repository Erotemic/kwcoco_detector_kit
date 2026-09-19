from __future__ import annotations

import json
import subprocess
import sys

import cv2
import numpy as np
import pytest


def _identities(source_digest="a" * 64, quality=91):
    from kwcoco_detector_kit.data.tile_cache import (
        make_materialization_identity,
        make_tile_identity,
    )
    tile = make_tile_identity(
        dataset_fingerprint="dataset-v1",
        source_asset_digest=source_digest,
        source_image_id=7,
        source_asset_name="images/example.jpg",
        extent_xyxy=[10, 20, 74, 84],
        scale=1.0,
    )
    material = make_materialization_identity(
        tile_id=tile["tile_id"], output_width=64, output_height=64,
        interpolation="area", padding="none", orientation="normalized",
        color_space="rgb", codec="jpg", quality=quality, writer_version=3,
    )
    return tile, material


def _jpeg_bytes():
    image = np.full((64, 64, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 91])
    assert ok
    return encoded.tobytes()


def test_source_bytes_and_codec_settings_change_separate_identities():
    tile1, mat1 = _identities(source_digest="a" * 64, quality=91)
    tile2, mat2 = _identities(source_digest="b" * 64, quality=91)
    tile3, mat3 = _identities(source_digest="a" * 64, quality=80)
    assert tile1["tile_id"] != tile2["tile_id"]
    assert mat1["materialization_id"] != mat2["materialization_id"]
    assert tile1["tile_id"] == tile3["tile_id"]
    assert mat1["materialization_id"] != mat3["materialization_id"]


def test_cache_publish_reuse_and_encoded_digest(tmp_path):
    from kwcoco_detector_kit.data.tile_cache import TileMaterializationCache, sha256_file
    _, material = _identities()
    cache = TileMaterializationCache(tmp_path)
    path1, created1 = cache.publish_bytes(material, _jpeg_bytes(), suffix="jpg")
    path2, created2 = cache.publish_bytes(material, _jpeg_bytes(), suffix="jpg")
    assert created1 is True
    assert created2 is False
    assert path1 == path2
    sidecar = json.loads(path1.with_suffix(".jpg.json").read_text())
    assert sidecar["encoded_sha256"] == sha256_file(path1)


def test_cache_detects_corruption(tmp_path):
    from kwcoco_detector_kit.data.tile_cache import CacheCorruptionError, TileMaterializationCache
    _, material = _identities()
    cache = TileMaterializationCache(tmp_path)
    path, _ = cache.publish_bytes(material, _jpeg_bytes(), suffix="jpg")
    path.write_bytes(b"broken")
    with pytest.raises(CacheCorruptionError, match="digest mismatch"):
        cache.validate(material, suffix="jpg")


@pytest.mark.parametrize("missing", ["image", "sidecar"])
def test_cache_repairs_partial_canonical_pair(tmp_path, missing):
    from kwcoco_detector_kit.data.tile_cache import TileMaterializationCache
    _, material = _identities()
    cache = TileMaterializationCache(tmp_path)
    path, _ = cache.publish_bytes(material, _jpeg_bytes(), suffix="jpg")
    sidecar = path.with_suffix(".jpg.json")
    (path if missing == "image" else sidecar).unlink()
    repaired, created = cache.publish_bytes(material, _jpeg_bytes(), suffix="jpg")
    assert created is True
    assert cache.validate(material, suffix="jpg") == repaired
    assert list(path.parent.joinpath("quarantine").glob("*.corrupt"))


def test_cache_quarantines_corruption_and_rebuilds(tmp_path):
    from kwcoco_detector_kit.data.tile_cache import TileMaterializationCache
    _, material = _identities()
    cache = TileMaterializationCache(tmp_path)
    path, _ = cache.publish_bytes(material, _jpeg_bytes(), suffix="jpg")
    path.write_bytes(b"broken")
    repaired, created = cache.publish_bytes(material, _jpeg_bytes(), suffix="jpg")
    assert created is True
    assert cache.validate(material, suffix="jpg") == repaired
    assert list(path.parent.joinpath("quarantine").glob("*.corrupt"))


def test_unpublished_temp_file_does_not_poison_cache(tmp_path):
    from kwcoco_detector_kit.data.tile_cache import TileMaterializationCache
    _, material = _identities()
    cache = TileMaterializationCache(tmp_path)
    final, _ = cache.paths(material["materialization_id"], "jpg")
    final.parent.mkdir(parents=True)
    final.with_name(final.name + ".interrupted.tmp").write_bytes(b"partial")
    path, created = cache.publish_bytes(material, _jpeg_bytes(), suffix="jpg")
    assert created is True
    assert cache.validate(material, suffix="jpg") == path


def test_concurrent_duplicate_publish_has_one_winner(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from kwcoco_detector_kit.data.tile_cache import TileMaterializationCache
    _, material = _identities()
    cache = TileMaterializationCache(tmp_path)
    payload = _jpeg_bytes()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda _: cache.publish_bytes(material, payload, suffix="jpg"),
            range(16),
        ))
    assert sum(created for _path, created in results) == 1
    assert len({path for path, _created in results}) == 1


def test_killed_lock_holder_does_not_wedge_cache(tmp_path):
    from kwcoco_detector_kit.data.tile_cache import TileMaterializationCache
    _, material = _identities()
    cache = TileMaterializationCache(tmp_path, lock_timeout=2)
    image_path, _ = cache.paths(material["materialization_id"], "jpg")
    image_path.parent.mkdir(parents=True)
    lock_path = image_path.with_suffix(image_path.suffix + ".lock")
    code = (
        "from filelock import FileLock; import sys,time; "
        "lock=FileLock(sys.argv[1]); lock.acquire(); "
        "print('locked', flush=True); time.sleep(60)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(lock_path)],
        stdout=subprocess.PIPE, text=True,
    )
    assert proc.stdout.readline().strip() == "locked"
    proc.kill()
    proc.wait(timeout=5)
    path, created = cache.publish_bytes(material, _jpeg_bytes(), suffix="jpg")
    assert created is True
    assert cache.validate(material, suffix="jpg") == path


def test_concurrent_subprocess_publish_converges(tmp_path):
    from pathlib import Path
    _, material = _identities()
    material_path = tmp_path / "material.json"
    payload_path = tmp_path / "payload.jpg"
    material_path.write_text(json.dumps(material))
    payload_path.write_bytes(_jpeg_bytes())
    helper = Path(__file__).parents[1] / "helpers" / "cache_publish_worker.py"
    procs = [subprocess.Popen(
        [sys.executable, str(helper), str(tmp_path / "cache"),
         str(material_path), str(payload_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) for _ in range(4)]
    results = []
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=20)
        assert proc.returncode == 0, stderr
        results.append(json.loads(stdout))
    assert sum(row["created"] for row in results) == 1
    assert len({row["path"] for row in results}) == 1
