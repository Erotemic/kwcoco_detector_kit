"""
kwcoco tile augmentation — three modes:

  ``full_only``    Emit each source image resized to a long-side cap.
                   Useful when the source bundle is already at a sensible
                   resolution and you only want category filtering + an
                   asset directory layout.

  ``quadrant``     Emit an NxN grid of overlapping tiles cut from the
                   full-resolution source image. Each tile is resized to
                   a long-side cap. Optionally also emit a downsized
                   full-frame view. (Ported from the prior project's
                   ``tile_kwcoco.py``.)

  ``multiscale``   Emit fixed-size square tiles cut from N downscaled
                   copies of each source image. The same physical
                   object appears in multiple tiles at different
                   apparent sizes — the data-time mirror of multi-
                   resolution inference. (Ported from the prior
                   project's ``v5_tile.py``.)

Oversized tiles for crop augmentation
-------------------------------------
By default the on-disk tile is exactly the size the model trains at. The
``oversize_factor`` knob lets you cut larger tiles than the eventual
model input — e.g. ``tile_size=320`` + ``oversize_factor=1.4`` cuts
448×448 tiles on disk and records ``tile_model_input_size=[320, 320]``
in each tile image record. A future trainer-side load-time crop can
then jitter scale + position within the oversized tile without bleeding
into zero-padded borders. ``oversize_factor=1.0`` (default) matches
the v4/v5 behaviour exactly.

Output kwcoco
-------------
Every emitted tile image carries these metadata fields:

  ``tile_source_gid``              source image's gid in --src
  ``tile_role``                    ``"positive"``, ``"negative"``, or ``"full"``
  ``tile_num_kept_anns``           number of GT anns surviving in this tile
  ``tile_model_input_size``        [H, W] — the size the model will see
                                    after load-time crop / resize
  ``tile_extent_xyxy_in_source``   (x0, y0, x1, y1) in source-pixel coords
  ``tile_oversize_factor``         the configured oversize_factor

Multi-scale tiles additionally carry::

  ``tile_scale_name``              e.g. "s10", "s07", "s04", "s02"
  ``tile_scale_factor``            float, e.g. 1.0, 0.66, 0.40, 0.25
  ``tile_actual_scale_xy``         [sx, sy] — realized resize after integer rounding

Quadrant tiles additionally carry::

  ``tile_grid``                    NxN
  ``tile_resize_scale``            scale used for the long-side resize after crop
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import kwconf
import ubelt as ub


DEFAULT_SOURCE_SCALES = "1.0,0.66,0.40,0.25"


class TileConfig(kwconf.Config):
    """Tile-augment a kwcoco bundle. Output is a new kwcoco bundle on disk."""

    src = kwconf.Value(None, position=1, help="input kwcoco path")
    dst = kwconf.Value(None, position=2, help="output kwcoco bundle path")

    mode = kwconf.Value(
        "multiscale",
        choices=["full_only", "quadrant", "multiscale"],
        help=(
            "tile extraction strategy. full_only: resize full image; "
            "quadrant: NxN overlapping tiles from full-res; "
            "multiscale: fixed-size tiles from N source scales."
        ),
    )

    category_names = kwconf.Value(
        "widget",
        help=(
            "comma-separated positive detector categories. Order is preserved "
            "and assigned to output category_id 1, 2, ..."
        ),
    )
    ignore_categories = kwconf.Value(
        "",
        help=(
            "comma-separated uncertain source categories. Windows intersecting "
            "these annotations are dropped rather than trained as background"
        ),
    )
    uncategorized_annotation_policy = kwconf.Value(
        "ignore", choices=["background", "ignore", "error"],
        help="treatment of source annotations whose category_id is missing/None",
    )
    default_non_target_policy = kwconf.Value(
        "background", choices=["background", "ignore", "error"],
        help="treatment of declared non-target categories not listed in --ignore-categories",
    )
    unclassified_category_policy = kwconf.Value(
        "ignore", choices=["background", "ignore", "error"],
        help="treatment of source annotations referencing undeclared categories",
    )
    output_ext = kwconf.Value(".jpg", help="asset extension")
    jpeg_quality = kwconf.Value(90, help="JPEG quality if output_ext is .jpg")
    cache_dpath = kwconf.Value(
        None,
        help=(
            "optional deterministic materialization cache. When set, tile "
            "manifests point directly at validated hash-addressed assets so "
            "identical windows are reused across immutable training rounds"
        ),
    )
    source_dataset_fingerprint = kwconf.Value(
        None,
        help=(
            "optional stable identity of the logical source dataset. Use this "
            "when src is a regenerated selection manifest; the immediate src "
            "manifest hash is still recorded separately for stale checks"
        ),
    )
    progress = kwconf.Value(True, help="show ubelt.ProgIter progress")
    seed = kwconf.Value(0, help="RNG seed (used by sampled modes)")
    oversize_factor = kwconf.Value(
        1.0,
        help=(
            "emit tiles oversize_factor larger than the model input so a "
            "load-time crop augmentation has margin. 1.0 (default) matches "
            "v4/v5 behaviour; ~1.4 is a sensible margin for random "
            "scale+position jitter."
        ),
    )
    min_keep_fraction = kwconf.Value(
        0.30,
        help=(
            "annotations whose visible fraction in a tile falls below this "
            "are dropped. Common to all tile-cutting modes."
        ),
    )

    # ---- mode=full_only / quadrant ----
    full_dim = kwconf.Value(1280, help="long-side cap for the kept full-frame image")
    keep_full = kwconf.Value(True, help="quadrant mode: also emit the resized full image")

    # ---- mode=quadrant only ----
    tile_grid = kwconf.Value(2, help="NxN grid (quadrant mode)")
    tile_overlap = kwconf.Value(0.20, help="fractional overlap between adjacent tiles (quadrant mode)")
    tile_output_dim = kwconf.Value(640, help="long-side cap on each tile after resize (quadrant mode)")

    # ---- mode=multiscale only ----
    tile_size = kwconf.Value(320, help="fixed output tile size — the model's eventual input size (multiscale)")
    source_scales = kwconf.Value(DEFAULT_SOURCE_SCALES, help="comma-separated source-scale factors (multiscale)")
    stride_frac = kwconf.Value(0.5, help="sliding-window stride as a fraction of disk tile size (multiscale)")
    min_gt_area_frac = kwconf.Value(
        0.005,
        help="multiscale: tile is positive iff total surviving GT area / tile_area >= this",
    )
    negative_safety_margin = kwconf.Value(
        0,
        help=(
            "multiscale: scaled-image pixels around a crop that must also be "
            "free of target geometry before the crop can be a training "
            "negative. A target in this margin makes the window ignored."
        ),
    )
    min_source_scale_long_side = kwconf.Value(
        64,
        help=(
            "multiscale: skip a source scale whose downscaled long side is "
            "below this; protects against downscaling into uselessness."
        ),
    )
    keep_negative = kwconf.Value(True, help="multiscale: also emit negative tiles for hard-neg mining")
    negative_keep_fraction = kwconf.Value(
        1.0,
        help=(
            "multiscale: deterministic fraction of legal negative windows to "
            "materialize. Applied before image encoding; 1.0 keeps all and "
            "0.0 keeps none"
        ),
    )

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        run(config)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resize_with_long_side(image, max_dim: int):
    """Downsize so the long side ≤ max_dim; preserve aspect; returns (image, scale).

    Always passes ``dsize=`` to kwimage.imresize — never positional 2 (failure
    #2: positional 2 is ``scale=``, not ``dsize=``). Falls back to ``'linear'``
    interpolation if the skimage backend rejects ``'area'`` (failure #5).
    """
    import kwimage

    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_dim:
        return image, 1.0
    scale = max_dim / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    try:
        resized = kwimage.imresize(image, dsize=(new_w, new_h), interpolation="area")
    except NotImplementedError:
        resized = kwimage.imresize(image, dsize=(new_w, new_h), interpolation="linear")
    return resized, scale


def _keep_negative_window(*, fraction, seed, source_gid, scale_name, x0, y0):
    """Deterministically sample a negative before paying its encode/write cost."""
    fraction = float(fraction)
    if fraction <= 0:
        return False
    if fraction >= 1:
        return True
    key = f"{int(seed)}:{source_gid}:{scale_name}:{int(x0)}:{int(y0)}".encode()
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return value < int(fraction * (1 << 64))


def _resize_image_to_dsize(image, dsize):
    """Shared eager/virtual area-resize primitive."""
    import kwimage

    new_w, new_h = map(int, dsize)
    h, w = image.shape[:2]
    if new_w == w and new_h == h:
        return image
    try:
        resized = kwimage.imresize(image, dsize=(new_w, new_h), interpolation="area")
    except NotImplementedError:
        resized = kwimage.imresize(image, dsize=(new_w, new_h), interpolation="linear")
    return resized


def _resize_image_to_scale(image, scale: float):
    """Resize image by ``scale``; returns ``(resized, (sx, sy))``."""
    h, w = image.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = _resize_image_to_dsize(image, (new_w, new_h))
    return resized, (new_w / float(w), new_h / float(h))


def _imwrite(fpath: Path, image, ext: str, jpeg_quality: int):
    """Write image with cv2's flat params= form (failure #3 — imwrite_params is not a kwarg)."""
    import kwimage
    import numpy as np

    # cv2's writer rejects non-contiguous arrays. Tiles are frequently numpy
    # slice views of a larger image (multiscale cuts fixed-size tiles straight
    # from the scaled source with no intervening imresize), so force a
    # contiguous copy here — covers every caller/mode.
    image = np.ascontiguousarray(image)

    try:
        if ext.lower() in (".jpg", ".jpeg"):
            import cv2

            kwimage.imwrite(
                str(fpath), image,
                params=[int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
        else:
            kwimage.imwrite(str(fpath), image)
    except Exception as ex:
        import numpy as np
        shp = getattr(image, "shape", None)
        dt = getattr(image, "dtype", None)
        contig = getattr(getattr(image, "flags", None), "c_contiguous", None)
        raise IOError(
            f"tile write failed for {fpath}: {type(ex).__name__}: {ex}; "
            f"image shape={shp} dtype={dt} c_contiguous={contig} "
            f"(expected contiguous uint8 HxWx3 — see _read_image_rgb)"
        ) from ex


class _TileWriter:
    """Write legacy bundle assets or publish them to the shared tile cache."""

    def __init__(self, config, src_dset, src_fpath, dst_fpath, asset_dpath):
        from kwcoco_detector_kit.data.tile_cache import sha256_file

        self.config = config
        self.src_dset = src_dset
        self.dst_fpath = dst_fpath
        self.asset_dpath = asset_dpath
        self.source_manifest_sha256 = sha256_file(src_fpath)
        self.dataset_fingerprint = (
            str(config.source_dataset_fingerprint)
            if config.source_dataset_fingerprint
            else self.source_manifest_sha256
        )
        self._source_digests = {}
        cache_dpath = getattr(config, "cache_dpath", None)
        if cache_dpath:
            from kwcoco_detector_kit.data.tile_cache import TileMaterializationCache
            self.cache = TileMaterializationCache(cache_dpath)
        else:
            self.cache = None

    def _source_identity(self, coco_img):
        from kwcoco_detector_kit.data.tile_cache import sha256_file

        gid = coco_img.img["id"]
        source_fpath = Path(self.src_dset.get_image_fpath(gid)).resolve()
        digest = self._source_digests.get(source_fpath)
        if digest is None:
            digest = sha256_file(source_fpath)
            self._source_digests[source_fpath] = digest
        return source_fpath, digest

    def write(self, image, *, coco_img, stem, extent_xyxy, scale,
              requested_scale=None, scaled_extent_xyxy=None,
              interpolation="area", padding="none"):
        """Materialize one image and return ``(file_name, identity_metadata)``."""
        import numpy as np

        ext = str(self.config.output_ext)
        if self.cache is None:
            asset_fpath = self.asset_dpath / (stem + ext)
            _imwrite(asset_fpath, image, ext, int(self.config.jpeg_quality))
            return str(asset_fpath.relative_to(self.dst_fpath.parent)), {}

        import cv2
        from kwcoco_detector_kit.data.tile_cache import (
            make_materialization_identity,
            make_tile_identity,
        )

        source_fpath, source_digest = self._source_identity(coco_img)
        tile = make_tile_identity(
            dataset_fingerprint=self.dataset_fingerprint,
            source_asset_digest=source_digest,
            source_image_id=coco_img.img["id"],
            source_asset_name=str(coco_img.img.get("file_name", source_fpath.name)),
            extent_xyxy=extent_xyxy,
            requested_scale=scale if requested_scale is None else requested_scale,
            actual_scale=scale,
            scaled_extent_xyxy=scaled_extent_xyxy,
        )
        h, w = image.shape[:2]
        codec = ext.lower().lstrip(".")
        codec = "jpg" if codec == "jpeg" else codec
        material = make_materialization_identity(
            tile_id=tile["tile_id"], output_width=w, output_height=h,
            interpolation=interpolation, padding=padding,
            orientation="normalized", color_space="rgb", codec=codec,
            quality=int(self.config.jpeg_quality) if codec == "jpg" else None,
            writer_version=_TILE_WRITER_VERSION,
        )
        rgb = np.ascontiguousarray(image)
        encoded_input = rgb[..., ::-1] if rgb.ndim == 3 and rgb.shape[2] == 3 else rgb
        params = []
        if codec == "jpg":
            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.jpeg_quality)]
        ok, encoded = cv2.imencode("." + codec, encoded_input, params)
        if not ok:
            raise IOError(f"failed to encode cached tile {stem!r} as {codec}")
        cache_fpath, _created = self.cache.publish_bytes(
            material, encoded.tobytes(), suffix=codec,
        )
        identity_meta = {
            "tile_id": tile["tile_id"],
            "tile_identity": tile["tile_id"],
            "tile_materialization_id": material["materialization_id"],
            "materialization_identity": material["materialization_id"],
            "tile_source_asset_sha256": source_digest,
        }
        return str(cache_fpath), identity_meta


def _clip_bbox_xywh(bbox, x0, y0, x1, y1, min_keep_fraction):
    """Clip an xywh bbox to a tile; return (new_xywh_in_tile_coords, keep_fraction) or None."""
    bx, by, bw, bh = [float(v) for v in bbox]
    if bw <= 0 or bh <= 0:
        return None
    src_area = bw * bh
    nx0 = max(bx, x0)
    ny0 = max(by, y0)
    nx1 = min(bx + bw, x1)
    ny1 = min(by + bh, y1)
    new_w = nx1 - nx0
    new_h = ny1 - ny0
    if new_w <= 1 or new_h <= 1:
        return None
    keep = (new_w * new_h) / src_area
    if keep < min_keep_fraction:
        return None
    return [nx0 - x0, ny0 - y0, new_w, new_h], keep


def _parse_scales(scales) -> List[Tuple[str, float]]:
    """'1.0,0.66,0.4' OR [1.0, 0.66, 0.4] -> [('s10', 1.0), ('s07', 0.66), ('s04', 0.4)].

    Accepts either form: kwconf hands us the literal comma-separated string
    (which we split here), while a programmatic caller may pass a pre-split
    list of strings.
    """
    if isinstance(scales, (list, tuple)):
        items = [str(s) for s in scales]
    else:
        items = str(scales).strip("[]").split(",")
    out: List[Tuple[str, float]] = []
    for tok in items:
        tok = tok.strip().strip("'\"")
        if not tok:
            continue
        s = float(tok)
        if s <= 0 or s > 4.0:
            raise ValueError(f"scale {s} out of plausible range (0, 4]")
        out.append((f"s{int(round(s * 10)):02d}", s))
    if not out:
        raise ValueError("no scales parsed")
    return out


def _grid_positions(extent: int, tile: int, stride: int) -> List[int]:
    """Sliding-window starts covering an extent, delegated to kwarray."""
    import kwarray

    padded_extent = max(int(extent), int(tile))
    windows = kwarray.SlidingWindow(
        shape=(padded_extent,), window=(int(tile),), stride=(max(int(stride), 1),),
        keepbound=True, allow_overshoot=True,
    )
    return [int(window[0].start) for window in windows]


def _tile_extents_quadrant(width: int, height: int, grid: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    """Yield (x0, y0, x1, y1) extents for an NxN grid with fractional overlap.

    For grid=2, overlap=0.2: tile_size = extent / (N - overlap*(N-1));
    second tile shifted to overlap by `overlap` of its own width.
    """
    grid = max(int(grid), 1)
    overlap = max(min(float(overlap), 0.5), 0.0)
    if grid == 1:
        return [(0, 0, int(width), int(height))]

    def _axis(extent: int) -> List[Tuple[int, int]]:
        denom = grid - overlap * (grid - 1)
        tile = extent / denom
        starts = [int(round(i * tile * (1 - overlap))) for i in range(grid)]
        starts[-1] = max(starts[-1], int(extent - tile))
        ends = [min(int(round(s + tile)), int(extent)) for s in starts]
        starts = [max(0, s) for s in starts]
        return list(zip(starts, ends))

    xs = _axis(int(width))
    ys = _axis(int(height))
    return [(x0, y0, x1, y1) for (x0, x1) in xs for (y0, y1) in ys]


# Annotation fields preserved as-is from source to tile output. The
# core fields (id, image_id, category_id, bbox, area, iscrowd) are
# computed by tile.py; everything in this whitelist passes through so
# downstream pipelines (e.g. scheme-aware MSCOCO export) can collapse
# / filter using metadata that survived the tiling step.
#
# Extending: add the field name; do NOT enable wildcard passthrough
# (some sources carry multi-MB caption fields that would bloat tiles).
_PASSTHROUGH_ANN_FIELDS = (
    "source_category",   # raw class label before any scheme collapse
    "track_id",          # cross-frame instance tracking
    "caption",           # short text caption per annotation
    "score",             # confidence (e.g. weak-labeler output)
)


def _passthrough_fields(src_ann: dict, src_dset=None) -> dict:
    out = {k: src_ann[k] for k in _PASSTHROUGH_ANN_FIELDS if k in src_ann}
    if "source_category" not in out and src_dset is not None:
        # Stamp source_category from the source dataset's category lookup
        # when the input is raw (untiled) data that carries class info via
        # category_id only. Without this, downstream scheme-collapse has
        # no way to recover the original class name.
        cid = src_ann.get("category_id")
        if cid is not None:
            cat = src_dset.cats.get(cid) if hasattr(src_dset, "cats") else None
            if cat is not None and "name" in cat:
                out["source_category"] = cat["name"]
    return out


def _clip_annotation_geometry(
    ann,
    *,
    source_dims,
    scale,
    crop_xyxy,
    output_dims,
):
    """Transform one annotation through source scaling and tile cropping."""
    from kwcoco_detector_kit.data.tile_geometry import (
        clip_bbox_geometry,
        clip_segmentation,
    )

    segmentation = ann.get("segmentation")
    if segmentation is not None:
        return clip_segmentation(
            segmentation,
            scale=scale,
            crop_xyxy=crop_xyxy,
            source_dims=source_dims,
            output_dims=output_dims,
        )
    bbox = ann.get("bbox")
    if bbox is None:
        return None
    return clip_bbox_geometry(bbox, scale=scale, crop_xyxy=crop_xyxy)


def _annotation_from_geometry(ann, geom, *, image_id, category_id, ann_id, src_dset):
    new_ann = {
        **_passthrough_fields(ann, src_dset),
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": list(geom.bbox_xywh),
        "area": float(geom.area),
        "iscrowd": int(ann.get("iscrowd", 0)),
        "tile_keep_fraction": float(geom.visible_fraction),
    }
    if geom.segmentation is not None:
        new_ann["segmentation"] = geom.segmentation
    if ann.get("id") is not None:
        new_ann["src_ann_id"] = ann["id"]
    return new_ann


# Bump this when changing the tile-writer's annotation/image emit semantics
# in a way that downstream consumers can detect (e.g. new passthrough field,
# new stamping logic). Mixed into the universal-tile cache fingerprint so
# the launcher gets a fresh hash and rebuilds the bundle.
_TILE_WRITER_VERSION = 3


def _normalize_image_rgb(arr):
    """Normalize a finalized image to contiguous uint8 RGB."""
    import numpy as np

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    # Coerce to 8-bit so the JPEG/PNG writer never gets a float/uint16 array
    # (the no-gdal `finalize()` fallback can return float), and make it
    # C-contiguous so cv2's writer never rejects a slice view. This keeps the
    # docstring's "uint8 ndarray" promise that the multiscale full-scale path
    # (which writes the array without an intervening imresize) relies on.
    if arr.dtype != np.uint8:
        import kwimage
        arr = kwimage.ensure_uint255(arr)
    return np.ascontiguousarray(arr)


def _read_image_rgb(coco_img):
    """Read a kwcoco coco_image as a (H, W, 3) uint8 ndarray; gracefully fallback."""
    return _normalize_image_rgb(coco_img.imdelay().finalize())


def _dump_kwcoco(out: dict, dst_fpath: Path):
    """Materialise `out` dict as a kwcoco bundle at dst_fpath."""
    import kwcoco

    dst_fpath.parent.mkdir(parents=True, exist_ok=True)
    if dst_fpath.suffix == ".zip":
        json_fpath = dst_fpath.with_suffix(".json")
        json_fpath.write_text(json.dumps(out))
        dset = kwcoco.CocoDataset.coerce(json_fpath)
        dset.fpath = str(dst_fpath)
        dset.dump()
        json_fpath.unlink(missing_ok=True)
    else:
        dst_fpath.write_text(json.dumps(out))


# ---------------------------------------------------------------------------
# Mode entry points
# ---------------------------------------------------------------------------


def _semantic_parts(config, src_dset, gid):
    from kwcoco_detector_kit.data.truth_semantics import TruthSemantics

    semantics = TruthSemantics.from_config(config)
    return semantics, semantics.partition_annotations(
        src_dset, list(src_dset.annots(gid=gid).objs)
    )


def _run_full_only(config, src_dset, dst_fpath, asset_dpath, target_cat_names, src_cid_to_new_cid, writer):
    """Resize each source image to ``full_dim``; warp annotations through the scale."""
    import ubelt as ub

    full_dim = int(config.full_dim)
    out = _init_out(config, target_cat_names, "full_only")
    out["info"][0]["source_manifest_sha256"] = writer.source_manifest_sha256
    out["info"][0]["source_dataset_fingerprint"] = writer.dataset_fingerprint
    next_gid = 1
    next_ann_id = 1

    coco_imgs = list(src_dset.images().coco_images)
    iterator = ub.ProgIter(coco_imgs, desc="tile.full_only", enabled=bool(config.progress))
    for coco_img in iterator:
        try:
            image = _read_image_rgb(coco_img)
        except Exception as ex:
            print(f"  warn: failed to read {coco_img.img.get('file_name')}: {ex}")
            continue
        h, w = image.shape[:2]
        gid = coco_img.img["id"]
        _semantics, parts = _semantic_parts(config, src_dset, gid)
        anns = [
            ann for ann in parts["target"]
            if ann.get("bbox") is not None or ann.get("segmentation") is not None
        ]
        # full_only has no region-ignore representation. Conservatively omit
        # a whole-image sample if any uncertain annotation is present.
        if parts["ignore"]:
            continue
        resized, scale = _resize_with_long_side(image, full_dim)
        stem = f"gid{gid:08d}_full"
        out_h, out_w = resized.shape[:2]
        file_name, identity_meta = writer.write(
            resized, coco_img=coco_img, stem=stem,
            extent_xyxy=(0, 0, w, h), scale=scale,
        )
        out["images"].append({
            "id": next_gid,
            "file_name": file_name,
            "width": int(out_w),
            "height": int(out_h),
            "name": stem,
            "tile_role": "full",
            "tile_source_gid": int(gid),
            "tile_resize_scale": float(scale),
            "tile_model_input_size": [int(out_h), int(out_w)],
            "tile_oversize_factor": float(config.oversize_factor),
            **identity_meta,
        })
        kept_count = 0
        for ann in anns:
            geom = _clip_annotation_geometry(
                ann,
                source_dims=(h, w),
                scale=scale,
                crop_xyxy=(0, 0, out_w, out_h),
                output_dims=(out_h, out_w),
            )
            if geom is None:
                continue
            out["annotations"].append(_annotation_from_geometry(
                ann,
                geom,
                image_id=next_gid,
                category_id=src_cid_to_new_cid[ann["category_id"]],
                ann_id=next_ann_id,
                src_dset=src_dset,
            ))
            next_ann_id += 1
            kept_count += 1
        out["images"][-1]["tile_num_kept_anns"] = kept_count
        next_gid += 1

    _dump_kwcoco(out, dst_fpath)
    n_imgs = len(out["images"])
    n_anns = len(out["annotations"])
    print(f"tile.full_only: wrote {n_imgs} images, {n_anns} annotations to {dst_fpath}")


def _run_quadrant(config, src_dset, dst_fpath, asset_dpath, target_cat_names, src_cid_to_new_cid, writer):
    """NxN overlapping tiles cut from full-resolution source images + optional resized full-frame."""
    import ubelt as ub

    out = _init_out(config, target_cat_names, "quadrant")
    out["info"][0]["source_manifest_sha256"] = writer.source_manifest_sha256
    out["info"][0]["source_dataset_fingerprint"] = writer.dataset_fingerprint
    next_gid = 1
    next_ann_id = 1
    full_dim = int(config.full_dim)
    grid = int(config.tile_grid)
    overlap = float(config.tile_overlap)
    base_tile_dim = int(config.tile_output_dim)
    over = float(config.oversize_factor)
    disk_tile_dim = max(1, int(round(base_tile_dim * over)))
    min_keep = float(config.min_keep_fraction)

    coco_imgs = list(src_dset.images().coco_images)
    iterator = ub.ProgIter(coco_imgs, desc=f"tile.quadrant g{grid}", enabled=bool(config.progress))
    for coco_img in iterator:
        try:
            image = _read_image_rgb(coco_img)
        except Exception as ex:
            print(f"  warn: failed to read {coco_img.img.get('file_name')}: {ex}")
            continue
        h, w = image.shape[:2]
        gid = coco_img.img["id"]
        semantics, parts = _semantic_parts(config, src_dset, gid)
        anns = [
            ann for ann in parts["target"]
            if ann.get("bbox") is not None or ann.get("segmentation") is not None
        ]
        ignored = semantics.partition_ignored_annotations(src_dset, parts["ignore"])
        ignore_anns = ignored["region"]
        has_global_ignore = bool(ignored["image"])

        # full frame
        if bool(config.keep_full) and not parts["ignore"]:
            full_resized, scale = _resize_with_long_side(image, full_dim)
            stem = f"gid{gid:08d}_full"
            out_h, out_w = full_resized.shape[:2]
            file_name, identity_meta = writer.write(
                full_resized, coco_img=coco_img, stem=stem,
                extent_xyxy=(0, 0, w, h), scale=scale,
            )
            out["images"].append({
                "id": next_gid,
                "file_name": file_name,
                "width": int(out_w),
                "height": int(out_h),
                "name": stem,
                "tile_role": "full",
                "tile_source_gid": int(gid),
                "tile_resize_scale": float(scale),
                "tile_model_input_size": [int(out_h), int(out_w)],
                "tile_oversize_factor": float(over),
                **identity_meta,
            })
            full_kept = 0
            for ann in anns:
                geom = _clip_annotation_geometry(
                    ann,
                    source_dims=(h, w),
                    scale=scale,
                    crop_xyxy=(0, 0, out_w, out_h),
                    output_dims=(out_h, out_w),
                )
                if geom is None:
                    continue
                out["annotations"].append(_annotation_from_geometry(
                    ann,
                    geom,
                    image_id=next_gid,
                    category_id=src_cid_to_new_cid[ann["category_id"]],
                    ann_id=next_ann_id,
                    src_dset=src_dset,
                ))
                next_ann_id += 1
                full_kept += 1
            out["images"][-1]["tile_num_kept_anns"] = full_kept
            next_gid += 1

        # NxN tiles
        extents = _tile_extents_quadrant(w, h, grid, overlap)
        for tile_idx, (x0, y0, x1, y1) in enumerate(extents):
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            tile_image = image[y0:y1, x0:x1]
            tile_resized, scale = _resize_with_long_side(tile_image, disk_tile_dim)
            if has_global_ignore:
                continue
            blocked = False
            for ann in ignore_anns:
                geom = _clip_annotation_geometry(
                    ann,
                    source_dims=(h, w),
                    scale=scale,
                    crop_xyxy=(x0 * scale, y0 * scale, x1 * scale, y1 * scale),
                    output_dims=tile_resized.shape[:2],
                )
                if geom is not None:
                    blocked = True
                    break
            if blocked:
                continue
            stem = f"gid{gid:08d}_tile{tile_idx:02d}_g{grid}"
            out_h, out_w = tile_resized.shape[:2]
            file_name, identity_meta = writer.write(
                tile_resized, coco_img=coco_img, stem=stem,
                extent_xyxy=(x0, y0, x1, y1), scale=scale,
            )
            out["images"].append({
                "id": next_gid,
                "file_name": file_name,
                "width": int(out_w),
                "height": int(out_h),
                "name": stem,
                "tile_role": "tile",
                "tile_source_gid": int(gid),
                "tile_extent_xyxy_in_source": [int(x0), int(y0), int(x1), int(y1)],
                "tile_resize_scale": float(scale),
                "tile_grid": int(grid),
                "tile_model_input_size": [int(base_tile_dim), int(base_tile_dim)],
                "tile_oversize_factor": float(over),
                **identity_meta,
            })
            kept = 0
            for ann in anns:
                # Crop in source space, then apply the tile resize. Expressing
                # the crop in scaled coordinates lets the shared geometry
                # helper perform both operations in one transform.
                geom = _clip_annotation_geometry(
                    ann,
                    source_dims=(h, w),
                    scale=scale,
                    crop_xyxy=(x0 * scale, y0 * scale, x1 * scale, y1 * scale),
                    output_dims=(out_h, out_w),
                )
                if geom is None or geom.visible_fraction < min_keep:
                    continue
                out["annotations"].append(_annotation_from_geometry(
                    ann,
                    geom,
                    image_id=next_gid,
                    category_id=src_cid_to_new_cid[ann["category_id"]],
                    ann_id=next_ann_id,
                    src_dset=src_dset,
                ))
                next_ann_id += 1
                kept += 1
            out["images"][-1]["tile_num_kept_anns"] = kept
            next_gid += 1

    _dump_kwcoco(out, dst_fpath)
    n_imgs = len(out["images"])
    n_anns = len(out["annotations"])
    print(f"tile.quadrant: wrote {n_imgs} images, {n_anns} annotations to {dst_fpath}")


def _run_multiscale(config, src_dset, dst_fpath, asset_dpath, target_cat_names, src_cid_to_new_cid, writer):
    """Fixed-size square tiles from N pre-downscaled copies of each source image."""
    import numpy as np
    import ubelt as ub

    scales = _parse_scales(str(config.source_scales))
    base_tile_size = int(config.tile_size)
    over = float(config.oversize_factor)
    disk_tile_size = max(1, int(round(base_tile_size * over)))
    stride = max(1, int(round(disk_tile_size * float(config.stride_frac))))
    min_long_side = int(config.min_source_scale_long_side)
    min_gt_area_abs = float(config.min_gt_area_frac) * (base_tile_size * base_tile_size)
    min_keep = float(config.min_keep_fraction)
    safety_margin = max(0, int(config.negative_safety_margin))
    negative_keep_fraction = float(config.negative_keep_fraction)
    if not 0 <= negative_keep_fraction <= 1:
        raise ValueError("negative_keep_fraction must be between 0 and 1")

    out = _init_out(config, target_cat_names, "multiscale")
    out["info"][0]["source_manifest_sha256"] = writer.source_manifest_sha256
    out["info"][0]["source_dataset_fingerprint"] = writer.dataset_fingerprint
    next_gid = 1
    next_ann_id = 1
    n_pos = 0
    n_neg_kept = 0
    n_neg_dropped = 0
    n_ignored = 0

    coco_imgs = list(src_dset.images().coco_images)
    iterator = ub.ProgIter(coco_imgs, desc="tile.multiscale", enabled=bool(config.progress))
    for coco_img in iterator:
        try:
            image_full = _read_image_rgb(coco_img)
        except Exception as ex:
            print(f"  warn: failed to read {coco_img.img.get('file_name')}: {ex}")
            continue
        H, W = image_full.shape[:2]
        gid = coco_img.img["id"]
        all_anns_src = list(src_dset.annots(gid=gid).objs)
        semantics, parts = _semantic_parts(config, src_dset, gid)
        anns_src = [
            ann for ann in parts["target"]
            if ann.get("bbox") is not None or ann.get("segmentation") is not None
        ]
        ignored = semantics.partition_ignored_annotations(src_dset, parts["ignore"])
        ignore_anns = ignored["region"]
        # Uncategorized / undeclared ignored truth, and any ignored annotation
        # without geometry, blocks the whole image from detector supervision.
        if ignored["image"]:
            n_ignored += 1
            continue

        for scale_name, scale_factor in scales:
            scaled_long = max(int(round(W * scale_factor)), int(round(H * scale_factor)))
            if scaled_long < min_long_side:
                continue
            scaled_img, actual_scale_xy = _resize_image_to_scale(image_full, scale_factor)
            actual_scale = tuple(actual_scale_xy)
            sH, sW = scaled_img.shape[:2]

            xs = _grid_positions(sW, disk_tile_size, stride)
            ys = _grid_positions(sH, disk_tile_size, stride)

            for x0 in xs:
                for y0 in ys:
                    x1 = min(x0 + disk_tile_size, sW)
                    y1 = min(y0 + disk_tile_size, sH)
                    crop = scaled_img[y0:y1, x0:x1]
                    was_padded = (
                        crop.shape[0] < disk_tile_size or crop.shape[1] < disk_tile_size
                    )
                    if crop.shape[0] < disk_tile_size or crop.shape[1] < disk_tile_size:
                        pad = np.zeros((disk_tile_size, disk_tile_size, 3), dtype=crop.dtype)
                        pad[:crop.shape[0], :crop.shape[1]] = crop
                        crop = pad

                    kept_anns = []
                    total_kept_area = 0.0
                    num_intersecting = 0
                    has_unsafe_target = False
                    crop_xyxy = (
                        x0, y0, x0 + disk_tile_size, y0 + disk_tile_size,
                    )
                    has_uncertain_truth = False
                    for ann in ignore_anns:
                        geom = _clip_annotation_geometry(
                            ann,
                            source_dims=(H, W),
                            scale=actual_scale,
                            crop_xyxy=crop_xyxy,
                            output_dims=(disk_tile_size, disk_tile_size),
                        )
                        if geom is not None:
                            has_uncertain_truth = True
                            break
                    if has_uncertain_truth:
                        n_ignored += 1
                        continue
                    for ann in anns_src:
                        geom = _clip_annotation_geometry(
                            ann,
                            source_dims=(H, W),
                            scale=actual_scale,
                            crop_xyxy=crop_xyxy,
                            output_dims=(disk_tile_size, disk_tile_size),
                        )
                        if geom is None:
                            if safety_margin:
                                margin_geom = _clip_annotation_geometry(
                                    ann,
                                    source_dims=(H, W),
                                    scale=actual_scale,
                                    crop_xyxy=(
                                        x0 - safety_margin,
                                        y0 - safety_margin,
                                        x0 + disk_tile_size + safety_margin,
                                        y0 + disk_tile_size + safety_margin,
                                    ),
                                    output_dims=(
                                        disk_tile_size + (2 * safety_margin),
                                        disk_tile_size + (2 * safety_margin),
                                    ),
                                )
                                if margin_geom is not None:
                                    has_unsafe_target = True
                            continue
                        num_intersecting += 1
                        if geom.visible_fraction < min_keep:
                            has_unsafe_target = True
                            continue
                        kept_anns.append((ann, geom))
                        total_kept_area += geom.area

                    if num_intersecting:
                        is_positive = (
                            not has_unsafe_target
                            and len(kept_anns) == num_intersecting
                            and total_kept_area >= min_gt_area_abs
                        )
                        if not is_positive:
                            # Known target content is present but cannot be
                            # represented as valid positive supervision. It is
                            # never legal to turn this into background.
                            n_ignored += 1
                            continue
                        role = "positive"
                    elif has_unsafe_target:
                        n_ignored += 1
                        continue
                    else:
                        role = "negative"

                    if role == "negative":
                        keep_this_negative = bool(config.keep_negative) and _keep_negative_window(
                            fraction=negative_keep_fraction,
                            seed=config.seed,
                            source_gid=gid,
                            scale_name=scale_name,
                            x0=x0,
                            y0=y0,
                        )
                        if not keep_this_negative:
                            n_neg_dropped += 1
                            continue

                    import kwimage
                    source_from_scaled = kwimage.Affine.scale(actual_scale_xy).inv()
                    source_box = kwimage.Boxes(
                        [[x0, y0, x0 + disk_tile_size, y0 + disk_tile_size]], "ltrb"
                    ).warp(source_from_scaled).to_ltrb().data[0]
                    src_x0, src_y0, src_x1, src_y1 = map(lambda v: int(round(v)), source_box)

                    stem = (f"gid{gid:08d}_{scale_name}"
                            f"_x{x0:05d}_y{y0:05d}_{role}")
                    file_name, identity_meta = writer.write(
                        crop, coco_img=coco_img, stem=stem,
                        extent_xyxy=(src_x0, src_y0, src_x1, src_y1),
                        scale=actual_scale,
                        requested_scale=scale_factor,
                        scaled_extent_xyxy=(x0, y0, x0 + disk_tile_size, y0 + disk_tile_size),
                        interpolation="area",
                        padding="zero_bottom_right" if was_padded else "none",
                    )

                    out["images"].append({
                        "id": next_gid,
                        "file_name": file_name,
                        "width": int(disk_tile_size),
                        "height": int(disk_tile_size),
                        "name": stem,
                        "tile_source_gid": int(gid),
                        "tile_scale_name": scale_name,
                        "tile_scale_factor": float(scale_factor),
                        "tile_actual_scale_xy": [float(v) for v in actual_scale_xy],
                        "tile_extent_xyxy_in_source": [src_x0, src_y0, src_x1, src_y1],
                        "tile_role": role,
                        "tile_num_kept_anns": len(kept_anns),
                        "tile_num_intersecting_anns": int(num_intersecting),
                        "tile_model_input_size": [int(base_tile_size), int(base_tile_size)],
                        "tile_oversize_factor": float(over),
                        **identity_meta,
                        **({
                            "negative_origin": (
                                "zero_annotation_source" if not all_anns_src
                                else "safe_background_window"
                            ),
                        } if role == "negative" else {}),
                    })
                    for ann, geom in kept_anns:
                        out["annotations"].append(_annotation_from_geometry(
                            ann,
                            geom,
                            image_id=next_gid,
                            category_id=src_cid_to_new_cid[ann["category_id"]],
                            ann_id=next_ann_id,
                            src_dset=src_dset,
                        ))
                        next_ann_id += 1
                    next_gid += 1
                    if role == "positive":
                        n_pos += 1
                    else:
                        n_neg_kept += 1

    out["info"][0]["tile_role_counts"] = {
        "positive": n_pos,
        "negative": n_neg_kept,
        "ignore": n_ignored,
        "dropped_negative": n_neg_dropped,
    }
    _dump_kwcoco(out, dst_fpath)
    print(
        f"tile.multiscale: wrote {len(out['images'])} tiles "
        f"(pos={n_pos}, neg={n_neg_kept}, ignore={n_ignored}, "
        f"dropped_neg={n_neg_dropped})"
    )
    print(f"  annotations: {len(out['annotations'])}")
    print(f"  scales: " + ", ".join(f"{n}={s}" for n, s in scales))
    print(f"  -> {dst_fpath}")


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def _init_out(config, target_cat_names, mode_label):
    """target_cat_names is a list of (name, new_cid) pairs ordered as
    they appear in --category_names; new_cid starts at 1 and increments."""
    return {
        "info": [{
            "name": "kwcoco_detector_kit.data.tile",
            "mode": mode_label,
            "src": str(config.src),
            "config": {k: getattr(config, k) for k in [
                "mode", "category_names", "ignore_categories",
                "uncategorized_annotation_policy", "default_non_target_policy",
                "unclassified_category_policy", "output_ext", "jpeg_quality",
                "cache_dpath", "source_dataset_fingerprint",
                "oversize_factor", "min_keep_fraction",
                "full_dim", "keep_full",
                "tile_grid", "tile_overlap", "tile_output_dim",
                "tile_size", "source_scales", "stride_frac",
                "min_gt_area_frac", "min_source_scale_long_side",
                "negative_safety_margin", "keep_negative",
                "negative_keep_fraction", "seed",
            ]},
        }],
        "categories": [
            {"id": new_cid, "name": name, "supercategory": name}
            for name, new_cid in target_cat_names
        ],
        "images": [],
        "annotations": [],
    }


def run(config):
    """Entry point — dispatches to the per-mode runner."""
    import os
    import kwcoco

    # Group-writable outputs so a teammate (or a future docker-as-root
    # run) can clean / mutate the tile cache. umask 0o002 → files 0664,
    # dirs 0775. Affects every mkdir/imwrite/dump in this run.
    os.umask(0o002)

    src_fpath = Path(str(config.src)).expanduser().resolve()
    dst_fpath = Path(str(config.dst)).expanduser().resolve()
    if not src_fpath.exists():
        raise FileNotFoundError(src_fpath)

    src_dset = kwcoco.CocoDataset.coerce(src_fpath)
    asset_dname = dst_fpath.stem.replace(".kwcoco", "") + "_assets"
    asset_dpath = dst_fpath.parent / asset_dname
    asset_dpath.mkdir(parents=True, exist_ok=True)

    raw = config.category_names
    if isinstance(raw, (list, tuple)):
        cat_name_list = [str(n).strip() for n in raw if str(n).strip()]
    else:
        cat_name_list = [s.strip() for s in str(raw).split(",") if s.strip()]
    if not cat_name_list:
        raise RuntimeError("--category_names must contain at least one name")

    src_cats_by_id = {c["id"]: c for c in src_dset.dataset.get("categories", [])}
    src_name_to_cid = {cat["name"]: cid for cid, cat in src_cats_by_id.items()}
    missing = [n for n in cat_name_list if n not in src_name_to_cid]
    if missing:
        raise RuntimeError(
            f"Categories {missing!r} not present in {src_fpath}; "
            f"available: {sorted(src_name_to_cid)}"
        )

    # Build (name, new_cid) pairs in CLI order and the src_cid -> new_cid map.
    target_cat_names = [(name, i + 1) for i, name in enumerate(cat_name_list)]
    src_cid_to_new_cid = {
        src_name_to_cid[name]: new_cid for name, new_cid in target_cat_names
    }

    mode = str(config.mode)
    writer = _TileWriter(config, src_dset, src_fpath, dst_fpath, asset_dpath)
    args = (
        config, src_dset, dst_fpath, asset_dpath, target_cat_names,
        src_cid_to_new_cid, writer,
    )
    if mode == "full_only":
        _run_full_only(*args)
    elif mode == "quadrant":
        _run_quadrant(*args)
    elif mode == "multiscale":
        _run_multiscale(*args)
    else:
        raise ValueError(f"Unknown tile mode: {mode!r}")


__cli__ = TileConfig


if __name__ == "__main__":
    __cli__.main()
