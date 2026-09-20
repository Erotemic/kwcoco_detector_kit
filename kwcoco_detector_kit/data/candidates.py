"""Virtual, mask-safe negative-window candidate indexes.

Candidate records contain geometry and provenance, never an encoded tile.
Selected records can later be materialized into the normal KDK tile cache.
"""
from __future__ import annotations

import json
import os
import tempfile
from itertools import groupby, islice
from pathlib import Path

import kwconf


def _atomic_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as file:
        # Candidate universes can contain millions of rows. Keep the durable
        # representation human-readable JSON but avoid indentation overhead.
        json.dump(data, file, sort_keys=True, separators=(",", ":"))
        file.write("\n")
        tmp = Path(file.name)
    os.replace(tmp, path)


class CandidateConfig(kwconf.Config):
    src = kwconf.Value(None, required=True)
    dst = kwconf.Value(None, required=True)
    category_names = kwconf.Value("widget")
    tile_size = kwconf.Value(320)
    oversize_factor = kwconf.Value(1.0)
    source_scales = kwconf.Value("1.0,0.66,0.4,0.25")
    stride_frac = kwconf.Value(0.5)
    min_keep_fraction = kwconf.Value(0.3)
    min_gt_area_frac = kwconf.Value(0.005)
    negative_safety_margin = kwconf.Value(0)
    min_source_scale_long_side = kwconf.Value(64)
    source_dataset_fingerprint = kwconf.Value(None)
    context_fields = kwconf.Value("video_id,date_captured,sensor_coarse,cohort,context")
    rows_per_shard = kwconf.Value(10000)


class _CandidateIndexWriter:
    """Bounded-memory JSONL shard writer; manifest publication is the commit."""

    def __init__(self, root, base_manifest, rows_per_shard):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_manifest = base_manifest
        self.rows_per_shard = max(1, int(rows_per_shard))
        self.shards = []
        self.total = 0
        self._file = None
        self._hasher = None
        self._count = 0
        self._tmp = None

    def _open(self):
        index = len(self.shards)
        name = f"candidates-{index:05d}.jsonl"
        self._final = self.root / name
        self._tmp = self.root / f".{name}.{os.getpid()}.tmp"
        self._file = open(self._tmp, "xb")
        import hashlib
        self._hasher = hashlib.sha256()
        self._count = 0

    def write(self, row):
        if self._file is None:
            self._open()
        payload = json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        self._file.write(payload)
        self._hasher.update(payload)
        self._count += 1
        self.total += 1
        if self._count >= self.rows_per_shard:
            self._close_shard()

    def _close_shard(self):
        if self._file is None:
            return
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        os.replace(self._tmp, self._final)
        self.shards.append({
            "name": self._final.name, "num_candidates": self._count,
            "sha256": self._hasher.hexdigest(),
        })
        self._file = self._tmp = self._hasher = None
        self._count = 0

    def close(self):
        self._close_shard()
        manifest = {
            **self.base_manifest,
            "num_candidates": self.total,
            "candidate_shards": self.shards,
            "candidate_content_digest": canonical_candidate_shard_digest(self.shards),
        }
        _atomic_json(manifest, self.root / "manifest.json")
        return manifest


def canonical_candidate_shard_digest(shards):
    from kwcoco_detector_kit.data.tile_cache import canonical_digest
    return canonical_digest([
        {"sha256": row["sha256"], "num_candidates": row["num_candidates"]}
        for row in shards
    ])


def load_candidate_index(path):
    """Load only the small index manifest, never all candidate rows."""
    root = Path(path)
    manifest_path = root / "manifest.json" if root.is_dir() else root
    doc = json.loads(manifest_path.read_text())
    if doc.get("candidate_content_digest") != canonical_candidate_shard_digest(
        doc.get("candidate_shards", [])
    ):
        raise ValueError("candidate manifest shard digest mismatch")
    doc["index_dpath"] = str(manifest_path.parent.resolve())
    return doc


def iter_candidate_records(path, *, file_shard_indices=None, validate=True):
    """Stream records in deterministic generation order with digest checks."""
    import hashlib

    manifest = load_candidate_index(path) if not isinstance(path, dict) else path
    root = Path(manifest["index_dpath"])
    wanted = None if file_shard_indices is None else set(map(int, file_shard_indices))
    total = 0
    for index, shard in enumerate(manifest["candidate_shards"]):
        if wanted is not None and index not in wanted:
            continue
        hasher = hashlib.sha256()
        count = 0
        with open(root / shard["name"], "rb") as file:
            for line_number, line in enumerate(file, 1):
                hasher.update(line)
                try:
                    row = json.loads(line)
                except Exception as ex:
                    raise ValueError(
                        f"invalid candidate JSONL {shard['name']}:{line_number}: {ex}"
                    ) from ex
                count += 1
                total += 1
                yield row
        if validate and (
            count != shard["num_candidates"] or hasher.hexdigest() != shard["sha256"]
        ):
            raise ValueError(f"candidate shard validation failed: {shard['name']}")
    if validate and wanted is None and total != manifest["num_candidates"]:
        raise ValueError("candidate index count mismatch")


def iter_candidate_records_for_shard(path, shard_index):
    """Read one physical deterministic file shard and no unrelated files."""
    yield from iter_candidate_records(path, file_shard_indices=[shard_index])


def materialize_candidate_records(path, limit=None):
    """Explicit small-workflow compatibility helper."""
    rows = []
    for row in iter_candidate_records(path):
        rows.append(row)
        if limit is not None and len(rows) >= int(limit):
            break
    return rows


def candidate_source_scale_key(item):
    """Return the semantic source/scale stratum for a virtual candidate."""
    if "tile_source_gid" not in item:
        raise KeyError("candidate is missing required tile_source_gid metadata")
    scale = item.get("tile_scale_name")
    if scale is None:
        scale = tuple(item.get("tile_actual_scale_xy", []))
    if scale in (None, ()):
        raise KeyError("candidate is missing explicit scale metadata")
    return item["tile_source_gid"], scale


def selected_candidate_record_factory(index_or_path, max_candidates, seed=0,
                                      strategy="stratified_by_image"):
    """Build a repeatable bounded candidate selection.

    The returned callable yields the same candidate records on every call.
    Full-universe selections stay streaming. Finite selections retain only
    ``O(number of source/scale groups + max_candidates)`` state and are
    returned in source/scale locality order so materialization can reuse each
    decoded/scaled source.
    """
    import hashlib
    import heapq
    import itertools

    index = (
        load_candidate_index(index_or_path)
        if not isinstance(index_or_path, dict)
        else index_or_path
    )
    max_candidates = int(max_candidates or 0)
    if not max_candidates or max_candidates >= int(index["num_candidates"]):
        return lambda: iter_candidate_records(index)

    strategy = str(strategy)
    if strategy == "first":
        selected = list(itertools.islice(iter_candidate_records(index), max_candidates))
    elif strategy == "random":
        heap = []
        for row in iter_candidate_records(index):
            priority = int(hashlib.sha256(
                f"{int(seed)}:{row['tile_id']}".encode()
            ).hexdigest(), 16)
            item = (-priority, row["tile_id"], row)
            if len(heap) < max_candidates:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        selected = [item[2] for item in heap]
    elif strategy == "stratified_by_image":
        group_counts = {}
        for row in iter_candidate_records(index):
            key = candidate_source_scale_key(row)
            group_counts[key] = group_counts.get(key, 0) + 1

        def _group_priority(key):
            payload = json.dumps(key, sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(f"{int(seed)}:{payload}".encode()).hexdigest()

        active = sorted(group_counts, key=lambda key: (_group_priority(key), repr(key)))
        quotas = {key: 0 for key in active}
        remaining = max_candidates
        while remaining and active:
            next_active = []
            for key in active:
                if quotas[key] < group_counts[key]:
                    quotas[key] += 1
                    remaining -= 1
                    if remaining == 0:
                        break
                if quotas[key] < group_counts[key]:
                    next_active.append(key)
            active = next_active

        group_heaps = {}
        for row in iter_candidate_records(index):
            key = candidate_source_scale_key(row)
            quota = quotas.get(key, 0)
            if not quota:
                continue
            heap = group_heaps.setdefault(key, [])
            priority = int(hashlib.sha256(
                f"{int(seed)}:{row['tile_id']}".encode()
            ).hexdigest(), 16)
            item = (-priority, row["tile_id"], row)
            if len(heap) < quota:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        selected = [
            item[2]
            for heap in group_heaps.values()
            for item in heap
        ]
    else:
        raise ValueError(f"unknown candidate_strategy: {strategy!r}")

    selected.sort(key=lambda row: (
        row["tile_source_gid"], tuple(row["tile_actual_scale_xy"]),
        row["tile_scaled_extent_xyxy"], row["tile_id"],
    ))
    return lambda: iter(selected)


def enumerate_candidates(config):
    """Enumerate every safe negative without decoding or encoding imagery."""
    import kwcoco
    import kwimage

    from kwcoco_detector_kit.data.tile import (
        _clip_annotation_geometry, _grid_positions, _parse_scales,
    )
    from kwcoco_detector_kit.data.tile_cache import (
        canonical_digest, make_tile_identity, sha256_file,
    )

    src = Path(config.src).expanduser().resolve()
    dset = kwcoco.CocoDataset.coerce(str(src))
    names = config.category_names
    names = names if isinstance(names, (list, tuple)) else str(names).split(",")
    names = {str(name).strip() for name in names if str(name).strip()}
    target_cids = {cat["id"] for cat in dset.cats.values() if cat["name"] in names}
    base_tile = int(config.tile_size)
    disk_tile = max(1, int(round(base_tile * float(config.oversize_factor))))
    stride = max(1, int(round(disk_tile * float(config.stride_frac))))
    margin = max(0, int(config.negative_safety_margin))
    min_keep = float(config.min_keep_fraction)
    min_area = float(config.min_gt_area_frac) * base_tile * base_tile
    source_fingerprint = str(config.source_dataset_fingerprint or sha256_file(src))
    policy = {
        "category_names": sorted(names), "tile_size": base_tile,
        "oversize_factor": float(config.oversize_factor),
        "source_scales": list(_parse_scales(config.source_scales)),
        "stride_frac": float(config.stride_frac),
        "min_keep_fraction": min_keep, "min_gt_area_frac": float(config.min_gt_area_frac),
        "negative_safety_margin": margin,
        "min_source_scale_long_side": int(config.min_source_scale_long_side),
    }
    policy_fingerprint = canonical_digest(policy)
    context_fields = [p.strip() for p in str(config.context_fields).split(",") if p.strip()]
    source_digests = {}
    dst = Path(config.dst)
    base_manifest = {
        "schema_version": 2, "source_kwcoco": str(src),
        "source_dataset_fingerprint": source_fingerprint,
        "policy": policy, "policy_fingerprint": policy_fingerprint,
        "tile_identity_schema_version": 2,
    }
    writer = _CandidateIndexWriter(dst, base_manifest, config.rows_per_shard)

    for image in dset.images().objs:
        gid = image["id"]
        width, height = int(image["width"]), int(image["height"])
        anns = [
            ann for ann in dset.annots(gid=gid).objs
            if ann.get("category_id") in target_cids
            and (ann.get("bbox") is not None or ann.get("segmentation") is not None)
        ]
        source_path = Path(dset.get_image_fpath(gid)).resolve()
        source_digest = source_digests.setdefault(str(source_path), sha256_file(source_path))
        for scale_name, requested_scale in _parse_scales(config.source_scales):
            scaled_w = max(1, int(round(width * requested_scale)))
            scaled_h = max(1, int(round(height * requested_scale)))
            if max(scaled_w, scaled_h) < int(config.min_source_scale_long_side):
                continue
            scale_xy = (scaled_w / width, scaled_h / height)
            source_from_scaled = kwimage.Affine.scale(scale_xy).inv()
            for y0 in _grid_positions(scaled_h, disk_tile, stride):
                for x0 in _grid_positions(scaled_w, disk_tile, stride):
                    crop = (x0, y0, x0 + disk_tile, y0 + disk_tile)
                    unsafe = False
                    intersecting = kept = 0
                    kept_area = 0.0
                    for ann in anns:
                        geom = _clip_annotation_geometry(
                            ann, source_dims=(height, width), scale=scale_xy,
                            crop_xyxy=crop, output_dims=(disk_tile, disk_tile),
                        )
                        if geom is None:
                            if margin:
                                margin_geom = _clip_annotation_geometry(
                                    ann, source_dims=(height, width), scale=scale_xy,
                                    crop_xyxy=(x0 - margin, y0 - margin,
                                               x0 + disk_tile + margin,
                                               y0 + disk_tile + margin),
                                    output_dims=(disk_tile + 2 * margin,
                                                 disk_tile + 2 * margin),
                                )
                                unsafe |= margin_geom is not None
                            continue
                        intersecting += 1
                        if geom.visible_fraction < min_keep:
                            unsafe = True
                        else:
                            kept += 1
                            kept_area += geom.area
                    if intersecting or unsafe:
                        # Positives and ignores belong to eager supervised tiling.
                        continue
                    source_box = kwimage.Boxes([crop], "ltrb").warp(
                        source_from_scaled
                    ).to_ltrb().data[0]
                    source_extent = [int(round(v)) for v in source_box]
                    tile = make_tile_identity(
                        dataset_fingerprint=source_fingerprint,
                        source_asset_digest=source_digest,
                        source_image_id=gid,
                        source_asset_name=str(image.get("file_name", source_path.name)),
                        extent_xyxy=source_extent,
                        requested_scale=requested_scale,
                        actual_scale=scale_xy,
                        scaled_extent_xyxy=crop,
                    )
                    record = {
                        **tile,
                        "tile_identity": tile["tile_id"],
                        "tile_source_gid": gid,
                        "tile_scale_name": scale_name,
                        "tile_scale_factor": float(requested_scale),
                        "tile_actual_scale_xy": [float(v) for v in scale_xy],
                        "tile_scaled_extent_xyxy": list(crop),
                        "tile_extent_xyxy_in_source": source_extent,
                        "output_width": disk_tile, "output_height": disk_tile,
                        "padding": (
                            "zero_bottom_right"
                            if x0 + disk_tile > scaled_w or y0 + disk_tile > scaled_h
                            else "none"
                        ),
                        "tile_model_input_size": [base_tile, base_tile],
                        "tile_role": "negative",
                        "negative_origin": (
                            "zero_annotation_source" if not anns
                            else "safe_background_window"
                        ),
                        "policy_fingerprint": policy_fingerprint,
                        "context": {key: image[key] for key in context_fields if key in image},
                    }
                    writer.write(record)
    writer.close()
    return dst


def _candidate_locality_key(row):
    return row["tile_source_gid"], tuple(row["tile_actual_scale_xy"])


def _realize_scaled_source(dset, gid, scale_xy):
    """Decode and resize one source/scale realization."""
    from kwcoco_detector_kit.data.tile import (
        _normalize_image_rgb, _read_image_rgb, _resize_image_to_dsize,
    )

    image = dset.imgs[gid]
    dsize = (
        max(1, int(round(int(image["width"]) * scale_xy[0]))),
        max(1, int(round(int(image["height"]) * scale_xy[1]))),
    )
    source = _read_image_rgb(dset.coco_image(gid))
    scaled = _resize_image_to_dsize(source, dsize)
    return _normalize_image_rgb(scaled)


def iter_realized_candidate_batches(index, records, batch_size, realization_hook=None):
    """Yield bounded crop batches while realizing each source/scale only once.

    ``records`` must be in deterministic source/scale locality order.  A failed
    source realization yields bounded batches with ``arrays=None`` and the
    exception so callers can durably record terminal read failures.
    """
    import kwcoco
    import numpy as np

    dset = kwcoco.CocoDataset.coerce(index["source_kwcoco"])
    batch_size = max(1, int(batch_size))
    for (gid, scale_xy), group in groupby(records, key=_candidate_locality_key):
        if realization_hook is not None:
            realization_hook(gid, scale_xy)
        try:
            scaled = _realize_scaled_source(dset, gid, scale_xy)
            error = None
        except Exception as ex:
            scaled = None
            error = ex
        group = iter(group)
        while True:
            batch = list(islice(group, batch_size))
            if not batch:
                break
            if error is not None:
                yield batch, None, error
                continue
            arrays = []
            for row in batch:
                x0, y0, x1, y1 = map(int, row["tile_scaled_extent_xyxy"])
                crop = scaled[max(0, y0):min(y1, scaled.shape[0]),
                              max(0, x0):min(x1, scaled.shape[1])]
                output = np.zeros(
                    (row["output_height"], row["output_width"], 3),
                    dtype=scaled.dtype,
                )
                output[:crop.shape[0], :crop.shape[1]] = crop
                arrays.append(output)
            yield batch, arrays, None


def realize_candidate_arrays(index, records):
    """Small-workflow helper returning all realized arrays by candidate ID."""
    result = {}
    for batch, arrays, error in iter_realized_candidate_batches(
        index, records, batch_size=64,
    ):
        if error is not None:
            raise error
        for row, output in zip(batch, arrays):
            result[row["tile_id"]] = output
    return result


def materialize_candidates(index_path, records, *, cache_dpath, jpeg_quality=90,
                           batch_size=16):
    """Stream selected candidates into the existing cache with bounded crops."""
    import cv2
    import itertools
    import kwcoco
    import numpy as np

    from kwcoco_detector_kit.data.tile_cache import (
        TileMaterializationCache, make_materialization_identity,
    )
    from kwcoco_detector_kit.data.tile import _TILE_WRITER_VERSION

    index = load_candidate_index(index_path) if not isinstance(index_path, dict) else index_path
    records = iter(records)
    first = next(records, None)
    if first is None:
        records = iter(())
    else:
        records = itertools.chain([first], records)
        if isinstance(first, str):
            wanted = set(records)
            records = (
                row for row in iter_candidate_records(index)
                if row["tile_id"] in wanted
            )
    cache = TileMaterializationCache(cache_dpath)
    out = kwcoco.CocoDataset()
    out.add_category(name="background_candidate")
    for rows, arrays, error in iter_realized_candidate_batches(
        index, records, batch_size=batch_size,
    ):
        if error is not None:
            raise error
        for row, array in zip(rows, arrays):
            arr = np.ascontiguousarray(array)
            material = make_materialization_identity(
                tile_id=row["tile_id"], output_width=row["output_width"],
                output_height=row["output_height"], interpolation="area",
                padding=row["padding"], orientation="normalized",
                color_space="rgb", codec="jpg", quality=jpeg_quality,
                writer_version=_TILE_WRITER_VERSION,
            )
            ok, encoded = cv2.imencode(
                ".jpg", arr[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
            )
            if not ok:
                raise IOError(f"failed to encode candidate {row['tile_id']}")
            path, _ = cache.publish_bytes(material, encoded.tobytes(), suffix="jpg")
            out.add_image(
                file_name=str(path), width=row["output_width"], height=row["output_height"],
                tile_materialization_id=material["materialization_id"],
                materialization_identity=material["materialization_id"], **row,
            )
    return out


__cli__ = CandidateConfig


if __name__ == "__main__":
    enumerate_candidates(CandidateConfig.cli(strict=True))
