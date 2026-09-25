"""Migrate legacy KDK tile-cache entries into raster-addressed cache keys.

This is a one-time bridge for caches written before raster identity was split
from truth/control-plane tile identity.  It reads an existing tiled KWCoco
manifest, reconstructs the truth-independent raster identity for each cached
multiscale tile, and adopts the already encoded bytes under the new cache key.
On the same filesystem adoption uses hard links, so JPEG bytes are not copied
or re-encoded.

Example:
    python -m kwcoco_detector_kit.data.tile_cache_migrate \
        --src=path/to/train_positive_tiles.kwcoco.zip \
        --cache_dpath=path/to/shared_tile_cache
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import kwconf


class TileCacheMigrateConfig(kwconf.Config):
    src = kwconf.Value(
        None,
        required=True,
        help="existing KDK tiled KWCoco manifest containing legacy cache paths",
    )
    cache_dpath = kwconf.Value(
        None,
        required=True,
        help="shared KDK tile cache root to seed with raster-addressed entries",
    )

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        run(config)


_XY_PATTERN = re.compile(r"(?:^|_)x(-?\d+)_y(-?\d+)(?:_|$)")


def _coerce_scaled_extent(image):
    scaled_extent = (
        image.get("tile_scaled_extent_xyxy")
        or image.get("scaled_extent_xyxy")
    )
    if scaled_extent is not None:
        return [float(v) for v in scaled_extent]
    name = str(image.get("name", ""))
    match = _XY_PATTERN.search(name)
    if match is None:
        return None
    x0, y0 = map(int, match.groups())
    return [
        x0,
        y0,
        x0 + int(image["width"]),
        y0 + int(image["height"]),
    ]


def _legacy_plan(image, sidecar):
    """Return ``(raster, materialization, suffix)`` or ``None`` if unsupported."""
    from kwcoco_detector_kit.data.tile import _TILE_WRITER_VERSION
    from kwcoco_detector_kit.data.tile_cache import (
        make_materialization_identity,
        make_raster_identity,
    )

    old_material = sidecar.get("materialization")
    if not isinstance(old_material, dict):
        return None
    if int(old_material.get("schema_version", -1)) != 1:
        return None
    # This bridge is specifically for the immediately preceding KDK cache
    # writer.  Adopting bytes from an unknown writer version would assert
    # equivalence that we have not established.
    if str(old_material.get("writer_version")) != "3":
        return None

    source_digest = (
        image.get("tile_source_asset_sha256")
        or image.get("source_asset_digest")
    )
    source_extent = (
        image.get("tile_extent_xyxy_in_source")
        or image.get("source_extent_xyxy")
    )
    actual_scale = (
        image.get("tile_actual_scale_xy")
        or image.get("actual_scale_xy")
    )
    scaled_extent = _coerce_scaled_extent(image)
    if any(value is None for value in (
        source_digest, source_extent, actual_scale, scaled_extent,
    )):
        return None

    raster = make_raster_identity(
        source_asset_digest=source_digest,
        extent_xyxy=source_extent,
        actual_scale=actual_scale,
        scaled_extent_xyxy=scaled_extent,
        channels=image.get("channels", "r|g|b"),
        realization="resize_source_then_crop",
    )
    codec = str(old_material.get("codec", "jpg")).lower().lstrip(".")
    codec = "jpg" if codec == "jpeg" else codec
    material = make_materialization_identity(
        raster_id=raster["raster_id"],
        output_width=int(old_material["output_width"]),
        output_height=int(old_material["output_height"]),
        interpolation=old_material.get("interpolation", "area"),
        padding=old_material.get("padding", "none"),
        crop_rounding=old_material.get("crop_rounding", "python_round_then_int"),
        orientation=old_material.get("orientation", "normalized"),
        color_space=old_material.get("color_space", "rgb"),
        codec=codec,
        quality=old_material.get("quality"),
        writer_version=_TILE_WRITER_VERSION,
    )
    return raster, material, codec


def run(config):
    import kwcoco

    from kwcoco_detector_kit.data.tile_cache import (
        CacheCorruptionError,
        TileMaterializationCache,
    )

    src = Path(str(config.src)).expanduser().resolve()
    cache = TileMaterializationCache(config.cache_dpath)
    dset = kwcoco.CocoDataset.coerce(str(src))
    stats = {
        "considered": 0,
        "adopted": 0,
        "already_current": 0,
        "unsupported": 0,
        "missing": 0,
        "corrupt": 0,
    }

    for image in dset.images().objs:
        stats["considered"] += 1
        old_path = Path(str(image.get("file_name", ""))).expanduser()
        if not old_path.is_absolute():
            old_path = (src.parent / old_path).resolve()
        if not old_path.is_file():
            stats["missing"] += 1
            continue
        old_sidecar_path = old_path.with_suffix(old_path.suffix + ".json")
        if not old_sidecar_path.is_file():
            stats["unsupported"] += 1
            continue
        try:
            old_sidecar = json.loads(old_sidecar_path.read_text())
        except Exception:
            stats["corrupt"] += 1
            continue
        old_material = old_sidecar.get("materialization", {})
        expected_old_id = image.get("tile_materialization_id") or image.get(
            "materialization_identity"
        )
        if (
            expected_old_id is not None
            and old_material.get("materialization_id") != expected_old_id
        ):
            stats["corrupt"] += 1
            continue
        plan = _legacy_plan(image, old_sidecar)
        if plan is None:
            stats["unsupported"] += 1
            continue
        _raster, material, suffix = plan
        if cache.lookup(material, suffix=suffix) is not None:
            stats["already_current"] += 1
            continue
        encoded_sha256 = old_sidecar.get("encoded_sha256")
        if not encoded_sha256:
            stats["corrupt"] += 1
            continue
        try:
            _path, created = cache.adopt_validated_file(
                material,
                old_path,
                suffix=suffix,
                encoded_sha256=encoded_sha256,
            )
        except (CacheCorruptionError, FileNotFoundError):
            stats["corrupt"] += 1
            continue
        if created:
            stats["adopted"] += 1
        else:
            stats["already_current"] += 1

    print(json.dumps(stats, sort_keys=True, indent=2))
    return stats


__cli__ = TileCacheMigrateConfig


if __name__ == "__main__":
    TileCacheMigrateConfig.main()
