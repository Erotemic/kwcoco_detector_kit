"""Migrate legacy KDK tile-cache entries into raster-addressed cache keys.

This is a one-time bridge for caches written before raster identity was split
from truth/control-plane tile identity.  It reads an existing tiled KWCoco
manifest, reconstructs the truth-independent raster identity for each cached
multiscale tile, and adopts the already encoded bytes under the new cache key.
On the same filesystem adoption uses hard links, so JPEG bytes are not copied
or re-encoded.

The migration is intentionally restart-safe.  Canonical cache publication is
atomic at the entry level, and a rerun recognizes completed hard-link
adoptions without rereading and hashing their JPEG bytes.  An interrupted
entry is simply repaired on the next pass.

Progress is written to stderr so stdout remains a machine-readable final JSON
summary.

Example:
    python -m kwcoco_detector_kit.data.tile_cache_migrate \
        --src=path/to/train_positive_tiles.kwcoco.zip \
        --cache_dpath=path/to/shared_tile_cache
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
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
    progress_every = kwconf.Value(
        250,
        help="emit progress after this many considered images; 0 disables item cadence",
    )
    progress_interval = kwconf.Value(
        5.0,
        help="emit progress after this many seconds even before progress_every; 0 disables time cadence",
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


def _fast_resume_hit(cache, material, suffix, old_path, encoded_sha256):
    """Recognize a completed same-filesystem adoption without hashing bytes.

    The first successful migration hashes the legacy source before publishing.
    On a later invocation, if the new cache image is the *same inode* as that
    legacy source and its canonical sidecar exactly records the expected new
    materialization and the legacy encoded digest/size, there are no distinct
    destination bytes to re-verify.  This makes interrupted migrations cheap
    to resume while retaining the byte verification on first adoption.

    Copied cross-filesystem entries deliberately do not take this shortcut;
    they fall back to normal cache validation.
    """
    key = str(material["materialization_id"])
    image_fpath, sidecar_fpath = cache.paths(key, suffix)
    if not image_fpath.is_file() or not sidecar_fpath.is_file():
        return False
    try:
        if not os.path.samefile(old_path, image_fpath):
            return False
        sidecar = json.loads(sidecar_fpath.read_text())
        if sidecar.get("materialization") != dict(material):
            return False
        if sidecar.get("encoded_sha256") != str(encoded_sha256):
            return False
        expected_size = int(old_path.stat().st_size)
        if int(sidecar.get("encoded_num_bytes", -1)) != expected_size:
            return False
        if int(image_fpath.stat().st_size) != expected_size:
            return False
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _format_progress(stats, total, elapsed):
    considered = int(stats["considered"])
    rate = considered / elapsed if elapsed > 0 else 0.0
    remaining = max(0, int(total) - considered)
    eta = remaining / rate if rate > 0 else None
    percent = (100.0 * considered / total) if total else 100.0
    eta_text = "?" if eta is None else f"{eta / 60.0:.1f}m"
    return (
        f"tile-cache-migrate {considered:,}/{int(total):,} ({percent:5.1f}%) "
        f"rate={rate:,.1f}/s eta={eta_text} "
        f"adopted={stats['adopted']:,} "
        f"resumed={stats['resumed_fast']:,} "
        f"current={stats['already_current']:,} "
        f"missing={stats['missing']:,} "
        f"unsupported={stats['unsupported']:,} "
        f"corrupt={stats['corrupt']:,}"
    )


def run(config):
    import kwcoco

    from kwcoco_detector_kit.data.tile_cache import (
        CacheCorruptionError,
        TileMaterializationCache,
    )

    src = Path(str(config.src)).expanduser().resolve()
    cache = TileMaterializationCache(config.cache_dpath)
    dset = kwcoco.CocoDataset.coerce(str(src))
    images = dset.images().objs
    total = len(images)
    stats = {
        "considered": 0,
        "total": total,
        "adopted": 0,
        "resumed_fast": 0,
        "already_current": 0,
        "unsupported": 0,
        "missing": 0,
        "corrupt": 0,
        "interrupted": False,
    }

    progress_every = max(0, int(config.progress_every))
    progress_interval = max(0.0, float(config.progress_interval))
    started = time.monotonic()
    last_report_time = started
    last_report_count = 0

    def report(*, force=False):
        nonlocal last_report_time, last_report_count
        now = time.monotonic()
        item_due = (
            progress_every > 0
            and stats["considered"] - last_report_count >= progress_every
        )
        time_due = (
            progress_interval > 0
            and now - last_report_time >= progress_interval
        )
        if force or item_due or time_due:
            print(
                _format_progress(stats, total, now - started),
                file=sys.stderr,
                flush=True,
            )
            last_report_time = now
            last_report_count = stats["considered"]

    report(force=True)
    try:
        for image in images:
            stats["considered"] += 1
            old_path = Path(str(image.get("file_name", ""))).expanduser()
            if not old_path.is_absolute():
                old_path = (src.parent / old_path).resolve()
            if not old_path.is_file():
                stats["missing"] += 1
                report()
                continue
            old_sidecar_path = old_path.with_suffix(old_path.suffix + ".json")
            if not old_sidecar_path.is_file():
                stats["unsupported"] += 1
                report()
                continue
            try:
                old_sidecar = json.loads(old_sidecar_path.read_text())
            except Exception:
                stats["corrupt"] += 1
                report()
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
                report()
                continue
            plan = _legacy_plan(image, old_sidecar)
            if plan is None:
                stats["unsupported"] += 1
                report()
                continue
            _raster, material, suffix = plan
            encoded_sha256 = old_sidecar.get("encoded_sha256")
            if not encoded_sha256:
                stats["corrupt"] += 1
                report()
                continue

            # Fast restart path: successful same-filesystem migrations are
            # hard links.  Their new sidecar plus inode identity proves that
            # this exact legacy payload was already adopted, without another
            # multi-GB pass hashing the same JPEGs.
            if _fast_resume_hit(
                cache, material, suffix, old_path, encoded_sha256,
            ):
                stats["resumed_fast"] += 1
                report()
                continue

            # This also handles valid entries produced elsewhere.  It hashes
            # the destination bytes, so only the same-inode restart path above
            # bypasses revalidation.
            if cache.lookup(material, suffix=suffix) is not None:
                stats["already_current"] += 1
                report()
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
                report()
                continue
            if created:
                stats["adopted"] += 1
            else:
                stats["already_current"] += 1
            report()
    except KeyboardInterrupt:
        stats["interrupted"] = True
        report(force=True)
        print(json.dumps(stats, sort_keys=True, indent=2), file=sys.stderr, flush=True)
        raise

    report(force=True)
    print(json.dumps(stats, sort_keys=True, indent=2))
    return stats


__cli__ = TileCacheMigrateConfig


if __name__ == "__main__":
    TileCacheMigrateConfig.main()
