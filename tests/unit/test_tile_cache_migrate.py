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
