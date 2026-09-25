from __future__ import annotations


def test_legacy_multiscale_manifest_plan_uses_raster_identity():
    from kwcoco_detector_kit.data.tile_cache_migrate import _legacy_plan

    image = {
        "file_name": "/cache/legacy.jpg",
        "width": 64,
        "height": 64,
        "name": "gid00000001_s10_x00032_y00048_positive",
        "tile_source_asset_sha256": "a" * 64,
        "tile_extent_xyxy_in_source": [32, 48, 96, 112],
        "tile_actual_scale_xy": [1.0, 1.0],
    }
    sidecar = {
        "materialization": {
            "schema_version": 1,
            "materialization_id": "b" * 64,
            "output_width": 64,
            "output_height": 64,
            "interpolation": "area",
            "padding": "none",
            "orientation": "normalized",
            "color_space": "rgb",
            "codec": "jpg",
            "quality": 92,
            "writer_version": "3",
        }
    }
    raster, material, suffix = _legacy_plan(image, sidecar)
    assert raster["scaled_extent_xyxy"] == [32.0, 48.0, 96.0, 112.0]
    assert raster["source_extent_xyxy"] == [32.0, 48.0, 96.0, 112.0]
    assert material["raster_id"] == raster["raster_id"]
    assert suffix == "jpg"


def test_legacy_candidate_plan_uses_explicit_scaled_extent():
    from kwcoco_detector_kit.data.tile_cache_migrate import _legacy_plan

    image = {
        "file_name": "/cache/legacy.jpg",
        "width": 64,
        "height": 64,
        "source_asset_digest": "a" * 64,
        "source_extent_xyxy": [16, 24, 80, 88],
        "actual_scale_xy": [0.5, 0.5],
        "scaled_extent_xyxy": [8, 12, 72, 76],
    }
    sidecar = {
        "materialization": {
            "schema_version": 1,
            "materialization_id": "c" * 64,
            "output_width": 64,
            "output_height": 64,
            "interpolation": "area",
            "padding": "zero_bottom_right",
            "orientation": "normalized",
            "color_space": "rgb",
            "codec": "jpg",
            "quality": 92,
            "writer_version": "3",
        }
    }
    raster, material, suffix = _legacy_plan(image, sidecar)
    assert raster["scaled_extent_xyxy"] == [8.0, 12.0, 72.0, 76.0]
    assert material["padding"] == "zero_bottom_right"
    assert suffix == "jpg"


def test_fast_resume_hit_recognizes_completed_hardlink(tmp_path):
    import hashlib
    import json
    import os

    from kwcoco_detector_kit.data.tile_cache import TileMaterializationCache
    from kwcoco_detector_kit.data.tile_cache_migrate import _fast_resume_hit

    source = tmp_path / "legacy.jpg"
    payload = b"already-validated-legacy-jpeg-bytes"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    material = {
        "schema_version": 2,
        "raster_id": "a" * 64,
        "output_width": 64,
        "output_height": 64,
        "interpolation": "area",
        "padding": "none",
        "crop_rounding": "python_round_then_int",
        "orientation": "normalized",
        "color_space": "rgb",
        "codec": "jpg",
        "quality": 92,
        "writer_version": "4",
        "materialization_id": "b" * 64,
    }
    cache = TileMaterializationCache(tmp_path / "cache")
    image_fpath, sidecar_fpath = cache.paths(material["materialization_id"], "jpg")
    image_fpath.parent.mkdir(parents=True)
    os.link(source, image_fpath)
    sidecar_fpath.write_text(json.dumps({
        "materialization": material,
        "encoded_sha256": digest,
        "encoded_num_bytes": len(payload),
    }))

    assert _fast_resume_hit(cache, material, "jpg", source, digest)


def test_fast_resume_hit_rejects_incomplete_or_distinct_copy(tmp_path):
    import hashlib
    import json
    import os

    from kwcoco_detector_kit.data.tile_cache import TileMaterializationCache
    from kwcoco_detector_kit.data.tile_cache_migrate import _fast_resume_hit

    source = tmp_path / "legacy.jpg"
    payload = b"legacy-jpeg-bytes"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    material = {
        "materialization_id": "c" * 64,
    }
    cache = TileMaterializationCache(tmp_path / "cache")
    image_fpath, sidecar_fpath = cache.paths(material["materialization_id"], "jpg")
    image_fpath.parent.mkdir(parents=True)

    # A killed migration can leave the image published before its sidecar.
    os.link(source, image_fpath)
    assert not _fast_resume_hit(cache, material, "jpg", source, digest)

    sidecar_fpath.write_text(json.dumps({
        "materialization": material,
        "encoded_sha256": digest,
        "encoded_num_bytes": len(payload),
    }))
    assert _fast_resume_hit(cache, material, "jpg", source, digest)

    image_fpath.unlink()
    image_fpath.write_bytes(payload)
    assert not os.path.samefile(source, image_fpath)
    assert not _fast_resume_hit(cache, material, "jpg", source, digest)


def test_progress_format_contains_resume_and_counts():
    from kwcoco_detector_kit.data.tile_cache_migrate import _format_progress

    stats = {
        "considered": 250,
        "adopted": 100,
        "resumed_fast": 120,
        "already_current": 20,
        "missing": 3,
        "unsupported": 4,
        "corrupt": 3,
    }
    text = _format_progress(stats, total=1000, elapsed=10.0)
    assert "250/1,000" in text
    assert "25.0%" in text
    assert "adopted=100" in text
    assert "resumed=120" in text
    assert "current=20" in text


def test_adoption_repairs_interrupted_canonical_pair(tmp_path):
    import hashlib
    import os

    from kwcoco_detector_kit.data.tile_cache import TileMaterializationCache

    source = tmp_path / "legacy.jpg"
    payload = b"legacy-bytes-that-were-validated-before-migration"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    material = {
        "materialization_id": "d" * 64,
        "output_width": 1,
        "output_height": 1,
    }
    cache = TileMaterializationCache(tmp_path / "cache")
    image_fpath, sidecar_fpath = cache.paths(material["materialization_id"], "jpg")
    image_fpath.parent.mkdir(parents=True)

    # Simulate SIGKILL after the image has become canonical but before the
    # sidecar publication.  A restart must quarantine the incomplete pair and
    # recreate a valid canonical entry.
    os.link(source, image_fpath)
    stale_image = image_fpath.with_name(image_fpath.name + ".dead.tmp")
    stale_sidecar = sidecar_fpath.with_name(sidecar_fpath.name + ".dead.tmp")
    stale_image.write_bytes(b"partial")
    stale_sidecar.write_text("partial")
    assert image_fpath.is_file()
    assert not sidecar_fpath.exists()

    path, created = cache.adopt_validated_file(
        material,
        source,
        suffix="jpg",
        encoded_sha256=digest,
    )
    assert created
    assert path == image_fpath
    assert sidecar_fpath.is_file()
    assert not stale_image.exists()
    assert not stale_sidecar.exists()
    assert cache.validate(material, suffix="jpg", decode=False) == image_fpath
    quarantine = image_fpath.parent / "quarantine"
    assert list(quarantine.glob("*"))
