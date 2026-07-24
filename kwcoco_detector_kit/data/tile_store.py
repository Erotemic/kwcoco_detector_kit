"""
TileStore — backend-agnostic read/write API for tile bundles.

The kit's Phase 1 default emits one JPEG per tile + a kwcoco manifest
(see ``data.tile``). Phase 3 introduces the ``TileStore`` Protocol so
alternative on-disk layouts can plug in without changing the trainer
plugins. The two backends in v1.0:

  ``KwcocoJpegStore``    one-JPEG-per-tile + kwcoco.zip manifest.
                         The Phase 1 default; what every existing
                         data/tile.py output is already in.

  ``WebdatasetStore``    tar shards with per-tile (image, anns, metadata)
                         records — sequential-read friendly on spinning
                         disks and slow network mounts. Writes via the
                         ``webdataset`` package; reads via ``wds.WebDataset``.

Both produce ``TileRecord`` instances on iteration, so the trainer-side
loader (``data.tile_loader.TileLoader``) doesn't care which backend
materialised the bundle.

Future backends (parquet rows, mmap'd numpy shards, LMDB) plug in by
implementing the Protocol.

Phase 2 path still works: trainer plugins keep consuming kwcoco bundles
directly. The TileStore abstraction is opt-in via the dataloader.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Record + Protocol
# ---------------------------------------------------------------------------


@dataclass
class TileRecord:
    """One tile, decoded into memory.

    Attributes:
        image_np: HxWxC uint8 (typically HxWx3 RGB; multispectral C may differ).
        bboxes_xywh: Mx4 float — MSCOCO-style ``[x, y, w, h]`` in *on-disk*
            pixel coords. A ``TileLoader`` applies load-time crop / resize
            to model-input coords.
        category_ids: M int — kit-internal category ids (0-indexed).
        metadata: per-tile metadata copied from the source bundle's
            image record (``tile_role``, ``tile_scale_name``,
            ``tile_model_input_size``, ``tile_oversize_factor``,
            ``tile_extent_xyxy_in_source``, ...).
    """
    image_np: np.ndarray
    bboxes_xywh: np.ndarray
    category_ids: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TileStore(Protocol):
    """Read API every TileStore backend implements."""

    @property
    def num_tiles(self) -> int: ...

    @property
    def metadata(self) -> Dict[str, Any]:
        """Bundle-level metadata — categories, channels, oversize_factor, etc."""

    def __iter__(self) -> Iterator[TileRecord]: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bundle_metadata(dset) -> Dict[str, Any]:
    """Pull bundle-level info from a kwcoco dataset."""
    info_blocks = dset.dataset.get("info") or []
    config = {}
    if info_blocks and isinstance(info_blocks[0], dict):
        config = info_blocks[0].get("config", {}) or {}
    cats = [
        {"id": c["id"], "name": c["name"]}
        for c in dset.dataset.get("categories", [])
    ]
    return {
        "categories": cats,
        "channels": str(config.get("channels", "r|g|b")),
        "oversize_factor": float(config.get("oversize_factor", 1.0)),
        "mode": str(config.get("mode", "")),
        "tile_size": int(config.get("tile_size", 0)) or None,
        "source_kwcoco": str((info_blocks[0] or {}).get("src", "")) if info_blocks else "",
    }


def _img_to_anns_index(dset) -> Dict[int, List[dict]]:
    """gid -> list of annotation dicts."""
    out: Dict[int, List[dict]] = {}
    for ann in dset.dataset.get("annotations", []):
        out.setdefault(int(ann.get("image_id")), []).append(ann)
    return out


# ---------------------------------------------------------------------------
# KwcocoJpegStore — Phase 1 default backend
# ---------------------------------------------------------------------------


class KwcocoJpegStore:
    """Wraps a kwcoco tile bundle (the output of ``data.tile.run``).

    No conversion needed — points at an existing ``.kwcoco.zip`` produced
    by the kit's tile module and iterates its images directly.
    """

    def __init__(self, src):
        import kwcoco

        self._src_fpath = Path(str(src)).expanduser().resolve()
        self._dset = kwcoco.CocoDataset.coerce(str(self._src_fpath))
        self._anns_by_gid = _img_to_anns_index(self._dset)
        self._metadata = _bundle_metadata(self._dset)

    # ---- TileStore Protocol ----

    @property
    def num_tiles(self) -> int:
        return self._dset.n_images

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    def __iter__(self) -> Iterator[TileRecord]:
        for img in self._dset.images().objs:
            gid = int(img["id"])
            anns = self._anns_by_gid.get(gid, [])
            try:
                coco_img = self._dset.coco_image(gid)
                image_np = coco_img.imdelay().finalize()
            except Exception as ex:
                # Skip unreadable tiles rather than aborting iteration.
                print(f"  warn: failed to read gid {gid}: {ex}")
                continue
            if image_np.ndim == 2:
                image_np = np.repeat(image_np[..., None], 3, axis=-1)
            # Strip per-tile keys that point at this bundle (file_name etc.)
            tile_meta = {
                k: v for k, v in img.items()
                if k.startswith("tile_") or k in ("name", "width", "height")
            }
            yield TileRecord(
                image_np=np.ascontiguousarray(image_np),
                bboxes_xywh=np.array(
                    [list(ann.get("bbox", [0, 0, 0, 0])) for ann in anns],
                    dtype=np.float32,
                ).reshape(-1, 4),
                category_ids=np.array(
                    [int(ann.get("category_id", 0)) for ann in anns],
                    dtype=np.int64,
                ),
                metadata=tile_meta,
            )


# ---------------------------------------------------------------------------
# WebdatasetStore — sequential-read tar shards
# ---------------------------------------------------------------------------


def _encode_jpeg(image_np: np.ndarray, jpeg_quality: int = 90) -> bytes:
    import cv2

    bgr = image_np[..., ::-1] if image_np.shape[-1] == 3 else image_np
    ok, buf = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    import cv2

    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode failed")
    return np.ascontiguousarray(bgr[..., ::-1])  # BGR -> RGB


def _shard_pattern(out_dpath: Path) -> str:
    """``wds.ShardWriter`` pattern: <dpath>/shard-%06d.tar."""
    return str(Path(out_dpath) / "shard-%06d.tar")


class WebdatasetStore:
    """tar-shard tile bundle. Writes via ``wds.ShardWriter``; reads via ``wds.WebDataset``.

    Each tile becomes one record with:

      ``__key__``      e.g. ``tile00000123``
      ``image.jpg``    encoded JPEG bytes
      ``boxes.json``   list of {bbox_xywh, category_id} dicts
      ``meta.json``    bundle + tile metadata
    """

    SHARD_PATTERN = "shard-%06d.tar"
    META_FNAME = "_bundle_meta.json"

    def __init__(self, dpath):
        self._dpath = Path(str(dpath)).expanduser().resolve()
        meta_fpath = self._dpath / self.META_FNAME
        if not meta_fpath.exists():
            raise FileNotFoundError(
                f"{meta_fpath} not found — is this a WebdatasetStore directory?"
            )
        self._metadata = json.loads(meta_fpath.read_text())

    # ---- write ----

    @classmethod
    def from_kwcoco(
        cls,
        src,
        dst,
        *,
        maxcount: int = 1024,
        maxsize: float = 256 * 1024 * 1024,  # 256 MB
        jpeg_quality: int = 90,
    ) -> "WebdatasetStore":
        """Convert a kwcoco tile bundle to a WebdatasetStore on disk."""
        import webdataset as wds

        src_store = KwcocoJpegStore(src)
        dst = Path(str(dst)).expanduser().resolve()
        dst.mkdir(parents=True, exist_ok=True)

        pattern = str(dst / cls.SHARD_PATTERN)
        # ShardWriter rolls a new tar whenever maxcount OR maxsize crosses.
        n_written = 0
        with wds.ShardWriter(pattern, maxcount=int(maxcount), maxsize=int(maxsize)) as sink:
            for i, record in enumerate(src_store):
                payload = {
                    "__key__": f"tile{i:08d}",
                    "image.jpg": _encode_jpeg(record.image_np, jpeg_quality=jpeg_quality),
                    "boxes.json": json.dumps([
                        {
                            "bbox_xywh": [float(v) for v in bbox],
                            "category_id": int(cid),
                        }
                        for bbox, cid in zip(record.bboxes_xywh.tolist(),
                                              record.category_ids.tolist())
                    ]).encode("utf-8"),
                    "meta.json": json.dumps(record.metadata).encode("utf-8"),
                }
                sink.write(payload)
                n_written += 1

        bundle_meta = dict(src_store.metadata)
        bundle_meta["num_tiles"] = n_written
        bundle_meta["shard_pattern"] = cls.SHARD_PATTERN
        (dst / cls.META_FNAME).write_text(json.dumps(bundle_meta, indent=2))
        return cls(dst)

    # ---- read (TileStore Protocol) ----

    @property
    def num_tiles(self) -> int:
        return int(self._metadata.get("num_tiles", 0))

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    def __iter__(self) -> Iterator[TileRecord]:
        import webdataset as wds

        shards = sorted(self._dpath.glob("shard-*.tar"))
        if not shards:
            return
        url = "{" + ",".join(str(s) for s in shards) + "}" if len(shards) > 1 else str(shards[0])
        ds = wds.WebDataset(url, shardshuffle=False, empty_check=False)
        for sample in ds:
            image_np = _decode_jpeg(sample["image.jpg"])
            anns = json.loads(sample["boxes.json"].decode("utf-8"))
            meta = json.loads(sample["meta.json"].decode("utf-8"))
            bboxes = np.array(
                [a["bbox_xywh"] for a in anns], dtype=np.float32,
            ).reshape(-1, 4)
            cids = np.array(
                [int(a["category_id"]) for a in anns], dtype=np.int64,
            )
            yield TileRecord(
                image_np=image_np,
                bboxes_xywh=bboxes,
                category_ids=cids,
                metadata=meta,
            )


# ---------------------------------------------------------------------------
# Factory + kwconf CLI for `python -m kwcoco_detector_kit convert-store`
# ---------------------------------------------------------------------------


def open_store(fpath) -> TileStore:
    """Auto-detect the backend by inspecting ``fpath``.

    - a ``.kwcoco.zip`` / ``.kwcoco.json`` -> KwcocoJpegStore
    - a directory containing ``_bundle_meta.json`` + ``shard-*.tar`` -> WebdatasetStore
    """
    p = Path(str(fpath)).expanduser().resolve()
    if p.is_dir() and (p / WebdatasetStore.META_FNAME).exists():
        return WebdatasetStore(p)
    return KwcocoJpegStore(p)


import kwconf


class ConvertStoreConfig(kwconf.Config):
    """Convert a kwcoco tile bundle to an alternative TileStore backend."""

    src = kwconf.Value(None, position=1, required=True,
                     help="input kwcoco tile bundle (a .kwcoco.zip)")
    dst = kwconf.Value(None, position=2, required=True,
                     help="output directory (will hold shard-*.tar + _bundle_meta.json)")
    backend = kwconf.Value("webdataset", choices=["webdataset"],
                        help="target backend")
    maxcount = kwconf.Value(1024, help="webdataset: max tiles per shard")
    maxsize_mb = kwconf.Value(256, help="webdataset: max shard size in MB")
    jpeg_quality = kwconf.Value(90)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        if str(config.backend) == "webdataset":
            store = WebdatasetStore.from_kwcoco(
                config.src, config.dst,
                maxcount=int(config.maxcount),
                maxsize=int(config.maxsize_mb) * 1024 * 1024,
                jpeg_quality=int(config.jpeg_quality),
            )
            print(
                f"wrote {store.num_tiles} tiles to {store._dpath} "
                f"({len(list(store._dpath.glob('shard-*.tar')))} shards)"
            )
        else:
            raise ValueError(f"unknown backend: {config.backend}")


__cli__ = ConvertStoreConfig
