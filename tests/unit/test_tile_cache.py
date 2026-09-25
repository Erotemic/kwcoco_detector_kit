from __future__ import annotations

import json
import subprocess
import sys

import cv2
import numpy as np
import pytest


def _identities(
    source_digest="a" * 64,
    quality=91,
    dataset_fingerprint="dataset-v1",
    extent_xyxy=(10, 20, 74, 84),
    actual_scale=1.0,
    channels="r|g|b",
):
    from kwcoco_detector_kit.data.tile_cache import (
        make_materialization_identity,
        make_raster_identity,
        make_tile_identity,
    )
    tile = make_tile_identity(
        dataset_fingerprint=dataset_fingerprint,
        source_asset_digest=source_digest,
        source_image_id=7,
        source_asset_name="images/example.jpg",
        extent_xyxy=extent_xyxy,
        scale=actual_scale,
        channels=channels,
    )
    raster = make_raster_identity(
        source_asset_digest=source_digest,
        extent_xyxy=extent_xyxy,
        actual_scale=actual_scale,
        channels=channels,
        realization="resize_source_then_crop",
    )
    material = make_materialization_identity(
        raster_id=raster["raster_id"], output_width=64, output_height=64,
        interpolation="area", padding="none", orientation="normalized",
        color_space="rgb", codec="jpg", quality=quality, writer_version=4,
    )
    return tile, raster, material


def _jpeg_bytes():
    image = np.full((64, 64, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 91])
    assert ok
    return encoded.tobytes()


def test_truth_identity_is_separate_from_raster_and_materialization_identity():
    tile1, raster1, mat1 = _identities(dataset_fingerprint="truth-v1")
    tile2, raster2, mat2 = _identities(dataset_fingerprint="truth-v2")
    assert tile1["tile_id"] != tile2["tile_id"]
    assert raster1["raster_id"] == raster2["raster_id"]
    assert mat1["materialization_id"] == mat2["materialization_id"]


def test_source_bytes_and_codec_settings_change_separate_identities():
    tile1, raster1, mat1 = _identities(source_digest="a" * 64, quality=91)
    tile2, raster2, mat2 = _identities(source_digest="b" * 64, quality=91)
    tile3, raster3, mat3 = _identities(source_digest="a" * 64, quality=80)
    assert tile1["tile_id"] != tile2["tile_id"]
    assert raster1["raster_id"] != raster2["raster_id"]
    assert mat1["materialization_id"] != mat2["materialization_id"]
    assert tile1["tile_id"] == tile3["tile_id"]
    assert raster1["raster_id"] == raster3["raster_id"]
    assert mat1["materialization_id"] != mat3["materialization_id"]


def test_raster_geometry_scale_and_channels_invalidate_identity():
    _, base, _ = _identities()
    _, moved, _ = _identities(extent_xyxy=(11, 20, 75, 84))
    _, scaled, _ = _identities(actual_scale=0.5)
    _, channels, _ = _identities(channels="gray")
    ids = {
        base["raster_id"], moved["raster_id"], scaled["raster_id"],
        channels["raster_id"],
    }
    assert len(ids) == 4


def test_cache_publish_reuse_and_encoded_digest(tmp_path):
    from kwcoco_detector_kit.data.tile_cache import TileMaterializationCache, sha256_file
    _, _, material = _identities()
    cache = TileMaterializationCache(tmp_path)
    path1, created1 = cache.publish_bytes(material, _jpeg_bytes(), suffix="jpg")
    path2, created2 = cache.publish_bytes(material, _jpeg_bytes(), suffix="jpg")
    assert created1 is True
    assert created2 is False
    assert path1 == path2
    sidecar = json.loads(path1.with_suffix(".jpg.json").read_text())
    assert sidecar["encoded_sha256"] == sha256_file(path1)
    assert cache.lookup(material, suffix="jpg") == path1


def test_cache_lookup_verifies_bytes_without_redecoding(tmp_path, monkeypatch):
    from kwcoco_detector_kit.data.tile_cache import TileMaterializationCache

    _, _, material = _identities()
    cache = TileMaterializationCache(tmp_path)
    path, _ = cache.publish_bytes(material, _jpeg_bytes(), suffix="jpg")

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("warm lookup must not decode previously validated bytes")

    monkeypatch.setattr(cv2, "imread", forbidden_decode)
    assert cache.lookup(material, suffix="jpg") == path


def test_cache_adopts_previously_validated_bytes_without_reencoding(tmp_path):
    from kwcoco_detector_kit.data.tile_cache import TileMaterializationCache

    _, _, material = _identities()
    legacy = tmp_path / "legacy.jpg"
    legacy.write_bytes(_jpeg_bytes())
    import hashlib
    digest = hashlib.sha256(legacy.read_bytes()).hexdigest()
    cache = TileMaterializationCache(tmp_path / "cache")
    path, created = cache.adopt_validated_file(
        material, legacy, suffix="jpg", encoded_sha256=digest,
    )
    assert created is True
    assert cache.lookup(material, suffix="jpg") == path
    assert path.read_bytes() == legacy.read_bytes()


def test_cache_detects_corruption(tmp_path):
    from kwcoco_detector_kit.data.tile_cache import CacheCorruptionError, TileMaterializationCache
    _, _, material = _identities()
    cache = TileMaterializationCache(tmp_path)
    path, _ = cache.publish_bytes(material, _jpeg_bytes(), suffix="jpg")
    path.write_bytes(b"broken")
    with pytest.raises(CacheCorruptionError, match="digest mismatch"):
        cache.validate(material, suffix="jpg")
    assert cache.lookup(material, suffix="jpg") is None


@pytest.mark.parametrize("missing", ["image", "sidecar"])
def test_cache_repairs_partial_canonical_pair(tmp_path, missing):
    from kwcoco_detector_kit.data.tile_cache import TileMaterializationCache
    _, _, material = _identities()
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
    _, _, material = _identities()
    cache = TileMaterializationCache(tmp_path)
    path, _ = cache.publish_bytes(material, _jpeg_bytes(), suffix="jpg")
    path.write_bytes(b"broken")
    repaired, created = cache.publish_bytes(material, _jpeg_bytes(), suffix="jpg")
    assert created is True
    assert cache.validate(material, suffix="jpg") == repaired
    assert list(path.parent.joinpath("quarantine").glob("*.corrupt"))


def test_unpublished_temp_file_does_not_poison_cache(tmp_path):
    from kwcoco_detector_kit.data.tile_cache import TileMaterializationCache
    _, _, material = _identities()
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
    _, _, material = _identities()
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
    _, _, material = _identities()
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
    _, _, material = _identities()
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
