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
  ``tile_actual_scale``            float — may differ from requested due to int rounding

Quadrant tiles additionally carry::

  ``tile_grid``                    NxN
  ``tile_resize_scale``            scale used for the long-side resize after crop
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import scriptconfig as scfg
import ubelt as ub


DEFAULT_SOURCE_SCALES = "1.0,0.66,0.40,0.25"


class TileConfig(scfg.DataConfig):
    """Tile-augment a kwcoco bundle. Output is a new kwcoco bundle on disk."""

    src = scfg.Value(None, position=1, help="input kwcoco path")
    dst = scfg.Value(None, position=2, help="output kwcoco bundle path")

    mode = scfg.Value(
        "multiscale",
        choices=["full_only", "quadrant", "multiscale"],
        help=(
            "tile extraction strategy. full_only: resize full image; "
            "quadrant: NxN overlapping tiles from full-res; "
            "multiscale: fixed-size tiles from N source scales."
        ),
    )

    category_names = scfg.Value(
        "widget",
        help=(
            "comma-separated category names to keep (others dropped). Order "
            "is preserved and assigned to output category_id 1, 2, ... so "
            "downstream MSCOCO export can map class indices consistently."
        ),
    )
    output_ext = scfg.Value(".jpg", help="asset extension")
    jpeg_quality = scfg.Value(90, help="JPEG quality if output_ext is .jpg")
    progress = scfg.Value(True, help="show ubelt.ProgIter progress")
    seed = scfg.Value(0, help="RNG seed (used by sampled modes)")
    oversize_factor = scfg.Value(
        1.0,
        help=(
            "emit tiles oversize_factor larger than the model input so a "
            "load-time crop augmentation has margin. 1.0 (default) matches "
            "v4/v5 behaviour; ~1.4 is a sensible margin for random "
            "scale+position jitter."
        ),
    )
    min_keep_fraction = scfg.Value(
        0.30,
        help=(
            "annotations whose visible fraction in a tile falls below this "
            "are dropped. Common to all tile-cutting modes."
        ),
    )

    # ---- mode=full_only / quadrant ----
    full_dim = scfg.Value(1280, help="long-side cap for the kept full-frame image")
    keep_full = scfg.Value(True, help="quadrant mode: also emit the resized full image")

    # ---- mode=quadrant only ----
    tile_grid = scfg.Value(2, help="NxN grid (quadrant mode)")
    tile_overlap = scfg.Value(0.20, help="fractional overlap between adjacent tiles (quadrant mode)")
    tile_output_dim = scfg.Value(640, help="long-side cap on each tile after resize (quadrant mode)")

    # ---- mode=multiscale only ----
    tile_size = scfg.Value(320, help="fixed output tile size — the model's eventual input size (multiscale)")
    source_scales = scfg.Value(DEFAULT_SOURCE_SCALES, help="comma-separated source-scale factors (multiscale)")
    stride_frac = scfg.Value(0.5, help="sliding-window stride as a fraction of disk tile size (multiscale)")
    min_gt_area_frac = scfg.Value(
        0.005,
        help="multiscale: tile is positive iff total surviving GT area / tile_area >= this",
    )
    min_source_scale_long_side = scfg.Value(
        64,
        help=(
            "multiscale: skip a source scale whose downscaled long side is "
            "below this; protects against downscaling into uselessness."
        ),
    )
    keep_negative = scfg.Value(True, help="multiscale: also emit negative tiles for hard-neg mining")

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


def _resize_image_to_scale(image, scale: float):
    """Resize image by ``scale``; returns (resized, actual_scale)."""
    import kwimage

    h, w = image.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w == w and new_h == h:
        return image, 1.0
    try:
        resized = kwimage.imresize(image, dsize=(new_w, new_h), interpolation="area")
    except NotImplementedError:
        resized = kwimage.imresize(image, dsize=(new_w, new_h), interpolation="linear")
    return resized, new_w / float(w)


def _imwrite(fpath: Path, image, ext: str, jpeg_quality: int):
    """Write image with cv2's flat params= form (failure #3 — imwrite_params is not a kwarg)."""
    import kwimage

    if ext.lower() in (".jpg", ".jpeg"):
        import cv2

        kwimage.imwrite(
            str(fpath), image,
            params=[int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
    else:
        kwimage.imwrite(str(fpath), image)


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

    Tolerates the scriptconfig auto-comma-split (which can hand us a list
    of strings instead of the literal comma-separated string we expected).
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
    """Sliding-window start positions covering [0, extent), inclusive of edges."""
    if extent <= tile:
        return [0]
    positions = list(range(0, extent - tile + 1, max(stride, 1)))
    last = extent - tile
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


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


# Bump this when changing the tile-writer's annotation/image emit semantics
# in a way that downstream consumers can detect (e.g. new passthrough field,
# new stamping logic). Mixed into the universal-tile cache fingerprint so
# the launcher gets a fresh hash and rebuilds the bundle.
_TILE_WRITER_VERSION = 2


def _read_image_rgb(coco_img):
    """Read a kwcoco coco_image as a (H, W, 3) uint8 ndarray; gracefully fallback."""
    import numpy as np

    arr = coco_img.imdelay().finalize()
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    return arr


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


def _run_full_only(config, src_dset, dst_fpath, asset_dpath, target_cat_names, src_cid_to_new_cid):
    """Resize each source image to ``full_dim``; warp annotations through the scale."""
    import ubelt as ub

    full_dim = int(config.full_dim)
    out = _init_out(config, target_cat_names, "full_only")
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
        anns = [
            ann for ann in src_dset.annots(gid=gid).objs
            if ann.get("category_id") in src_cid_to_new_cid and ann.get("bbox") is not None
        ]
        resized, scale = _resize_with_long_side(image, full_dim)
        stem = f"gid{gid:08d}_full"
        asset_fname = stem + str(config.output_ext)
        asset_fpath = asset_dpath / asset_fname
        _imwrite(asset_fpath, resized, str(config.output_ext), int(config.jpeg_quality))
        out_h, out_w = resized.shape[:2]
        out["images"].append({
            "id": next_gid,
            "file_name": str(asset_fpath.relative_to(dst_fpath.parent)),
            "width": int(out_w),
            "height": int(out_h),
            "name": stem,
            "tile_role": "full",
            "tile_source_gid": int(gid),
            "tile_resize_scale": float(scale),
            "tile_model_input_size": [int(out_h), int(out_w)],
            "tile_oversize_factor": float(config.oversize_factor),
        })
        kept_count = 0
        for ann in anns:
            bx, by, bw, bh = ann["bbox"]
            new_bbox = [bx * scale, by * scale, bw * scale, bh * scale]
            out["annotations"].append({
                **_passthrough_fields(ann, src_dset),
                "id": next_ann_id,
                "image_id": next_gid,
                "category_id": src_cid_to_new_cid[ann["category_id"]],
                "bbox": new_bbox,
                "area": float(new_bbox[2] * new_bbox[3]),
                "iscrowd": int(ann.get("iscrowd", 0)),
            })
            next_ann_id += 1
            kept_count += 1
        out["images"][-1]["tile_num_kept_anns"] = kept_count
        next_gid += 1

    _dump_kwcoco(out, dst_fpath)
    n_imgs = len(out["images"])
    n_anns = len(out["annotations"])
    print(f"tile.full_only: wrote {n_imgs} images, {n_anns} annotations to {dst_fpath}")


def _run_quadrant(config, src_dset, dst_fpath, asset_dpath, target_cat_names, src_cid_to_new_cid):
    """NxN overlapping tiles cut from full-resolution source images + optional resized full-frame."""
    import ubelt as ub

    out = _init_out(config, target_cat_names, "quadrant")
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
        anns = [
            ann for ann in src_dset.annots(gid=gid).objs
            if ann.get("category_id") in src_cid_to_new_cid and ann.get("bbox") is not None
        ]

        # full frame
        if bool(config.keep_full):
            full_resized, scale = _resize_with_long_side(image, full_dim)
            stem = f"gid{gid:08d}_full"
            asset_fname = stem + str(config.output_ext)
            asset_fpath = asset_dpath / asset_fname
            _imwrite(asset_fpath, full_resized, str(config.output_ext), int(config.jpeg_quality))
            out_h, out_w = full_resized.shape[:2]
            out["images"].append({
                "id": next_gid,
                "file_name": str(asset_fpath.relative_to(dst_fpath.parent)),
                "width": int(out_w),
                "height": int(out_h),
                "name": stem,
                "tile_role": "full",
                "tile_source_gid": int(gid),
                "tile_resize_scale": float(scale),
                "tile_model_input_size": [int(out_h), int(out_w)],
                "tile_oversize_factor": float(over),
            })
            for ann in anns:
                bx, by, bw, bh = ann["bbox"]
                new_bbox = [bx * scale, by * scale, bw * scale, bh * scale]
                out["annotations"].append({
                    **_passthrough_fields(ann, src_dset),
                    "id": next_ann_id,
                    "image_id": next_gid,
                    "category_id": src_cid_to_new_cid[ann["category_id"]],
                    "bbox": new_bbox,
                    "area": float(new_bbox[2] * new_bbox[3]),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                })
                next_ann_id += 1
            out["images"][-1]["tile_num_kept_anns"] = len(anns)
            next_gid += 1

        # NxN tiles
        extents = _tile_extents_quadrant(w, h, grid, overlap)
        for tile_idx, (x0, y0, x1, y1) in enumerate(extents):
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            tile_image = image[y0:y1, x0:x1]
            tile_resized, scale = _resize_with_long_side(tile_image, disk_tile_dim)
            stem = f"gid{gid:08d}_tile{tile_idx:02d}_g{grid}"
            asset_fname = stem + str(config.output_ext)
            asset_fpath = asset_dpath / asset_fname
            _imwrite(asset_fpath, tile_resized, str(config.output_ext), int(config.jpeg_quality))
            out_h, out_w = tile_resized.shape[:2]
            out["images"].append({
                "id": next_gid,
                "file_name": str(asset_fpath.relative_to(dst_fpath.parent)),
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
            })
            kept = 0
            for ann in anns:
                clipped = _clip_bbox_xywh(ann["bbox"], x0, y0, x1, y1, min_keep)
                if clipped is None:
                    continue
                new_bbox, keep = clipped
                new_bbox = [v * scale for v in new_bbox]
                out["annotations"].append({
                    **_passthrough_fields(ann, src_dset),
                    "id": next_ann_id,
                    "image_id": next_gid,
                    "category_id": src_cid_to_new_cid[ann["category_id"]],
                    "bbox": new_bbox,
                    "area": float(new_bbox[2] * new_bbox[3]),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                    "tile_keep_fraction": float(keep),
                })
                next_ann_id += 1
                kept += 1
            out["images"][-1]["tile_num_kept_anns"] = kept
            next_gid += 1

    _dump_kwcoco(out, dst_fpath)
    n_imgs = len(out["images"])
    n_anns = len(out["annotations"])
    print(f"tile.quadrant: wrote {n_imgs} images, {n_anns} annotations to {dst_fpath}")


def _run_multiscale(config, src_dset, dst_fpath, asset_dpath, target_cat_names, src_cid_to_new_cid):
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

    out = _init_out(config, target_cat_names, "multiscale")
    next_gid = 1
    next_ann_id = 1
    n_pos = 0
    n_neg_kept = 0
    n_neg_dropped = 0

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
        anns_src = [
            ann for ann in src_dset.annots(gid=gid).objs
            if ann.get("category_id") in src_cid_to_new_cid and ann.get("bbox") is not None
        ]

        for scale_name, scale_factor in scales:
            scaled_long = max(int(round(W * scale_factor)), int(round(H * scale_factor)))
            if scaled_long < min_long_side:
                continue
            scaled_img, actual_scale = _resize_image_to_scale(image_full, scale_factor)
            sH, sW = scaled_img.shape[:2]

            xs = _grid_positions(sW, disk_tile_size, stride)
            ys = _grid_positions(sH, disk_tile_size, stride)

            anns_scaled = []
            for ann in anns_src:
                bx, by, bw, bh = ann["bbox"]
                anns_scaled.append({
                    "bbox": [bx * actual_scale, by * actual_scale,
                             bw * actual_scale, bh * actual_scale],
                    "iscrowd": int(ann.get("iscrowd", 0)),
                    "src_ann_id": ann.get("id"),
                    "new_cid": src_cid_to_new_cid[ann["category_id"]],
                    "passthrough": _passthrough_fields(ann, src_dset),
                })

            for x0 in xs:
                for y0 in ys:
                    x1 = min(x0 + disk_tile_size, sW)
                    y1 = min(y0 + disk_tile_size, sH)
                    crop = scaled_img[y0:y1, x0:x1]
                    if crop.shape[0] < disk_tile_size or crop.shape[1] < disk_tile_size:
                        pad = np.zeros((disk_tile_size, disk_tile_size, 3), dtype=crop.dtype)
                        pad[:crop.shape[0], :crop.shape[1]] = crop
                        crop = pad

                    kept_anns = []
                    total_kept_area = 0.0
                    for ann in anns_scaled:
                        clipped = _clip_bbox_xywh(
                            ann["bbox"], x0, y0, x0 + disk_tile_size, y0 + disk_tile_size,
                            min_keep,
                        )
                        if clipped is None:
                            continue
                        new_bbox, _keep = clipped
                        kept_anns.append({
                            "bbox": new_bbox,
                            "iscrowd": ann["iscrowd"],
                            "src_ann_id": ann["src_ann_id"],
                            "new_cid": ann["new_cid"],
                            "passthrough": ann.get("passthrough", {}),
                        })
                        total_kept_area += new_bbox[2] * new_bbox[3]

                    is_positive = total_kept_area >= min_gt_area_abs
                    if not is_positive and not bool(config.keep_negative):
                        n_neg_dropped += 1
                        continue

                    role = "positive" if is_positive else "negative"

                    src_x0 = int(round(x0 / max(actual_scale, 1e-6)))
                    src_y0 = int(round(y0 / max(actual_scale, 1e-6)))
                    src_x1 = int(round((x0 + disk_tile_size) / max(actual_scale, 1e-6)))
                    src_y1 = int(round((y0 + disk_tile_size) / max(actual_scale, 1e-6)))

                    stem = (f"gid{gid:08d}_{scale_name}"
                            f"_x{x0:05d}_y{y0:05d}_{role}")
                    asset_fname = stem + str(config.output_ext)
                    asset_fpath = asset_dpath / asset_fname
                    _imwrite(asset_fpath, crop, str(config.output_ext), int(config.jpeg_quality))

                    out["images"].append({
                        "id": next_gid,
                        "file_name": str(asset_fpath.relative_to(dst_fpath.parent)),
                        "width": int(disk_tile_size),
                        "height": int(disk_tile_size),
                        "name": stem,
                        "tile_source_gid": int(gid),
                        "tile_scale_name": scale_name,
                        "tile_scale_factor": float(scale_factor),
                        "tile_actual_scale": float(actual_scale),
                        "tile_extent_xyxy_in_source": [src_x0, src_y0, src_x1, src_y1],
                        "tile_role": role,
                        "tile_num_kept_anns": len(kept_anns),
                        "tile_model_input_size": [int(base_tile_size), int(base_tile_size)],
                        "tile_oversize_factor": float(over),
                    })
                    for ann in kept_anns:
                        out["annotations"].append({
                            **ann.get("passthrough", {}),
                            "id": next_ann_id,
                            "image_id": next_gid,
                            "category_id": ann["new_cid"],
                            "bbox": ann["bbox"],
                            "area": float(ann["bbox"][2] * ann["bbox"][3]),
                            "iscrowd": ann["iscrowd"],
                            "src_ann_id": ann["src_ann_id"],
                        })
                        next_ann_id += 1
                    next_gid += 1
                    if is_positive:
                        n_pos += 1
                    else:
                        n_neg_kept += 1

    _dump_kwcoco(out, dst_fpath)
    print(
        f"tile.multiscale: wrote {len(out['images'])} tiles "
        f"(pos={n_pos}, neg={n_neg_kept}, dropped_neg={n_neg_dropped})"
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
                "mode", "category_names", "output_ext", "jpeg_quality",
                "oversize_factor", "min_keep_fraction",
                "full_dim", "keep_full",
                "tile_grid", "tile_overlap", "tile_output_dim",
                "tile_size", "source_scales", "stride_frac",
                "min_gt_area_frac", "min_source_scale_long_side",
                "keep_negative",
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
    import kwcoco

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
    args = (config, src_dset, dst_fpath, asset_dpath, target_cat_names, src_cid_to_new_cid)
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
