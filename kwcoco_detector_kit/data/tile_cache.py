"""Deterministic, validated materialization cache for training tiles.

The semantic identity of a tile and the identity of its encoded bytes are
deliberately separate.  A source-image byte digest is part of the former, so
replacing an asset in-place cannot silently reuse stale crops.  Codec and
writer details are part of the latter, because they can change bytes without
changing which source-space window a tile represents.

This module is intentionally small and filesystem-only.  Round manifests may
point at cache entries, but the cache is not a workflow database and does not
record whether an experiment stage is complete.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping


IDENTITY_SCHEMA_VERSION = 2
MATERIALIZATION_SCHEMA_VERSION = 1


class CacheCorruptionError(RuntimeError):
    """A published cache entry disagrees with its sidecar or cannot decode."""


def _canonical_json(data: Any) -> bytes:
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf8")


def canonical_digest(data: Any) -> str:
    """SHA-256 of a stable JSON representation."""
    return hashlib.sha256(_canonical_json(data)).hexdigest()


def sha256_file(fpath: str | os.PathLike, chunk_size: int = 1024 * 1024) -> str:
    """Digest file bytes without loading the whole asset into memory."""
    hasher = hashlib.sha256()
    with open(fpath, "rb") as file:
        while chunk := file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def make_tile_identity(
    *,
    dataset_fingerprint: str,
    source_asset_digest: str,
    source_image_id: str | int,
    source_asset_name: str,
    extent_xyxy: list[float] | tuple[float, float, float, float],
    scale: float | None = None,
    requested_scale: float | None = None,
    actual_scale: float | tuple[float, float] | list[float] | None = None,
    scaled_extent_xyxy: list[float] | tuple[float, float, float, float] | None = None,
    channels: str = "r|g|b",
) -> dict[str, Any]:
    """Return the complete semantic identity record for a source window."""
    realized = actual_scale if actual_scale is not None else scale
    if isinstance(realized, (tuple, list)):
        actual_scale_xy = [float(realized[0]), float(realized[1])]
    else:
        actual_scale_xy = [float(realized), float(realized)]
    record = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "dataset_fingerprint": str(dataset_fingerprint),
        "source_asset_digest": str(source_asset_digest),
        "source_image_id": source_image_id,
        "source_asset_name": str(source_asset_name),
        "source_extent_xyxy": [float(v) for v in extent_xyxy],
        "scaled_extent_xyxy": (
            [float(v) for v in scaled_extent_xyxy]
            if scaled_extent_xyxy is not None else None
        ),
        "requested_scale": float(requested_scale if requested_scale is not None else scale),
        "actual_scale_xy": actual_scale_xy,
        "channels": str(channels),
    }
    record["tile_id"] = canonical_digest(record)
    return record


def make_materialization_identity(
    *,
    tile_id: str,
    output_width: int,
    output_height: int,
    interpolation: str,
    padding: str,
    crop_rounding: str = "python_round_then_int",
    orientation: str,
    color_space: str,
    codec: str,
    quality: int | None,
    writer_version: str | int,
) -> dict[str, Any]:
    """Return the identity record for one concrete encoding of a tile."""
    record = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "tile_id": str(tile_id),
        "output_width": int(output_width),
        "output_height": int(output_height),
        "interpolation": str(interpolation),
        "padding": str(padding),
        "crop_rounding": str(crop_rounding),
        "orientation": str(orientation),
        "color_space": str(color_space),
        "codec": str(codec).lower().lstrip("."),
        "quality": None if quality is None else int(quality),
        "writer_version": str(writer_version),
    }
    record["materialization_id"] = canonical_digest(record)
    return record


class TileMaterializationCache:
    """Hash-prefix filesystem cache with validated, concurrency-safe publish."""

    def __init__(self, root: str | os.PathLike, lock_timeout: float = 60.0):
        self.root = Path(root).expanduser().resolve()
        self.lock_timeout = float(lock_timeout)

    def paths(self, materialization_id: str, suffix: str) -> tuple[Path, Path]:
        key = str(materialization_id)
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError(f"invalid materialization SHA-256: {key!r}")
        suffix = "." + str(suffix).lower().lstrip(".")
        dpath = self.root / key[:2] / key[2:4]
        image_fpath = dpath / (key + suffix)
        return image_fpath, image_fpath.with_suffix(image_fpath.suffix + ".json")

    def _lock(self, image_fpath: Path):
        from filelock import FileLock

        lock_fpath = image_fpath.with_suffix(image_fpath.suffix + ".lock")
        return FileLock(str(lock_fpath), timeout=self.lock_timeout)

    def _quarantine(self, image_fpath: Path, sidecar_fpath: Path) -> list[Path]:
        """Preserve incomplete/corrupt canonical entries for inspection."""
        moved = []
        quarantine = image_fpath.parent / "quarantine"
        quarantine.mkdir(parents=True, exist_ok=True)
        token = f"{os.getpid()}-{uuid.uuid4().hex}"
        for path in (image_fpath, sidecar_fpath):
            if path.exists():
                dst = quarantine / f"{path.name}.{token}.corrupt"
                os.replace(path, dst)
                moved.append(dst)
        return moved

    def validate(
        self,
        materialization: Mapping[str, Any],
        *,
        suffix: str,
    ) -> Path:
        """Validate digest, decodability, dimensions, and identity sidecar."""
        import cv2

        key = str(materialization["materialization_id"])
        image_fpath, sidecar_fpath = self.paths(key, suffix)
        if not image_fpath.is_file() or not sidecar_fpath.is_file():
            raise FileNotFoundError(image_fpath)
        try:
            sidecar = json.loads(sidecar_fpath.read_text())
        except Exception as ex:
            raise CacheCorruptionError(f"invalid sidecar {sidecar_fpath}: {ex}") from ex
        if sidecar.get("materialization") != dict(materialization):
            raise CacheCorruptionError(f"identity mismatch in {sidecar_fpath}")
        actual_digest = sha256_file(image_fpath)
        if sidecar.get("encoded_sha256") != actual_digest:
            raise CacheCorruptionError(f"encoded byte digest mismatch: {image_fpath}")
        decoded = cv2.imread(str(image_fpath), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise CacheCorruptionError(f"cached image is not decodable: {image_fpath}")
        height, width = decoded.shape[:2]
        expected = (
            int(materialization["output_height"]),
            int(materialization["output_width"]),
        )
        if (height, width) != expected:
            raise CacheCorruptionError(
                f"cached dimensions {(height, width)} != {expected}: {image_fpath}"
            )
        return image_fpath

    def publish_bytes(
        self,
        materialization: Mapping[str, Any],
        encoded: bytes,
        *,
        suffix: str,
    ) -> tuple[Path, bool]:
        """Publish bytes once; return ``(path, created)``.

        The lock and temporary files live beside the final artifact.  Thus the
        two ``os.replace`` calls stay on one filesystem, and readers only trust
        an entry after both the image and its identity/digest sidecar validate.
        """
        key = str(materialization["materialization_id"])
        image_fpath, sidecar_fpath = self.paths(key, suffix)
        image_fpath.parent.mkdir(parents=True, exist_ok=True)
        with self._lock(image_fpath):
            if image_fpath.exists() or sidecar_fpath.exists():
                try:
                    return self.validate(materialization, suffix=suffix), False
                except (FileNotFoundError, CacheCorruptionError):
                    self._quarantine(image_fpath, sidecar_fpath)
            token = f"{os.getpid()}-{uuid.uuid4().hex}"
            tmp_image = image_fpath.with_name(image_fpath.name + f".{token}.tmp")
            tmp_sidecar = sidecar_fpath.with_name(sidecar_fpath.name + f".{token}.tmp")
            try:
                with open(tmp_image, "xb") as file:
                    file.write(encoded)
                    file.flush()
                    os.fsync(file.fileno())
                sidecar = {
                    "materialization": dict(materialization),
                    "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
                    "encoded_num_bytes": len(encoded),
                }
                with open(tmp_sidecar, "x", encoding="utf8") as file:
                    json.dump(sidecar, file, sort_keys=True, indent=2)
                    file.write("\n")
                    file.flush()
                    os.fsync(file.fileno())
                # Validate the temporary payload before it becomes canonical.
                import cv2
                import numpy as np
                decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                expected = (
                    int(materialization["output_height"]),
                    int(materialization["output_width"]),
                )
                if decoded is None or decoded.shape[:2] != expected:
                    got = None if decoded is None else decoded.shape[:2]
                    raise ValueError(f"encoded image dimensions {got} != {expected}")
                os.replace(tmp_image, image_fpath)
                os.replace(tmp_sidecar, sidecar_fpath)
            finally:
                tmp_image.unlink(missing_ok=True)
                tmp_sidecar.unlink(missing_ok=True)
            return self.validate(materialization, suffix=suffix), True
