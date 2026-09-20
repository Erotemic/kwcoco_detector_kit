"""Build a human-review queue from hard-negative mining scores.

The mining ledger is intentionally lossless enough to support truth QA: each
scored tile records its stable tile identity plus the highest-scoring predicted
box.  This module joins those records back to the virtual candidate index and
source KWCoco dataset, maps the prediction into source-image coordinates, and
writes review artifacts that are easy to inspect without mutating truth.

Outputs under ``dst_dpath``:

* ``review_queue.json``: complete ranked provenance records.
* ``review_queue.tsv``: spreadsheet-friendly queue with blank review columns.
* ``review.kwcoco.zip``: source-image subset with existing truth plus a dedicated
  review-proposal category in source coordinates.
* ``previews/*.jpg`` and ``index.html``: optional static browser review UI.

The review KWCoco is diagnostic only.  Corrections belong in the canonical
source annotation system identified by ``source_kwcoco`` / ``source_fpath``.
"""
from __future__ import annotations

import csv
import heapq
import html
import json
import math
from pathlib import Path

import kwconf


_REVIEW_CATEGORY = "__hard_negative_review__"


class ReviewMineConfig(kwconf.Config):
    """Turn mining ledgers into a ranked, source-linked hard-negative review queue."""

    candidate_index = kwconf.Value(None, required=True, help="virtual negative candidate index")
    ledgers = kwconf.Value(None, required=True, nargs="+", help="completed mining shard ledgers")
    selected_candidates = kwconf.Value(
        None,
        help=(
            "optional mine-finalize *.selected_candidates.json sidecar. When set, "
            "review is restricted to globally selected hard negatives; otherwise "
            "the highest scoring ledger records are reviewed."
        ),
    )
    dst_dpath = kwconf.Value(None, required=True, help="review output directory")
    top_n = kwconf.Value(200, help="maximum review items after per-source capping")
    per_source = kwconf.Value(3, help="maximum reviewed tiles per source image; 0 disables cap")
    min_score = kwconf.Value(0.0, help="drop predictions below this score")
    make_previews = kwconf.Value(True, help="render static preview JPEGs and index.html")
    preview_max_dim = kwconf.Value(1000, help="maximum preview width/height")
    context_margin = kwconf.Value(0.20, help="fractional margin around the tile for previews")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        return run(config)


def _iter_ledger_records(ledger_paths):
    """Stream terminal mining records from schema-1/2 ledger representations."""
    for ledger_path in map(Path, ledger_paths):
        doc = json.loads(ledger_path.read_text())
        if not doc.get("scan_complete") and not doc.get("complete"):
            raise RuntimeError(f"incomplete mining ledger: {ledger_path}")
        records_path = doc.get("records_path")
        if records_path:
            with open(records_path, encoding="utf8") as file:
                for line in file:
                    if line.strip():
                        yield json.loads(line)
        else:
            yield from doc.get("records", [])


def _load_selected_ids(path):
    if not path:
        return None
    doc = json.loads(Path(path).read_text())
    rows = doc.get("selected")
    if not isinstance(rows, list):
        raise ValueError(f"selected-candidate sidecar has no selected list: {path}")
    return {
        str(row["tile_id"]): float(row.get("max_score", 0.0))
        for row in rows
    }


def _rank_score_records(ledger_paths, *, selected_scores, top_n, per_source, min_score):
    """Retain a bounded deterministic set of high-score terminal records."""
    min_score = float(min_score)
    if selected_scores is not None:
        wanted = set(selected_scores)
        found = {}
        for record in _iter_ledger_records(ledger_paths):
            tile_id = str(record.get("tile_id", ""))
            if tile_id in wanted and record.get("status") == "ok":
                rec = dict(record)
                rec["max_score"] = float(selected_scores[tile_id])
                found[tile_id] = rec
        missing = wanted - set(found)
        if missing:
            raise RuntimeError(
                f"{len(missing)} selected hard negatives are absent from mining ledgers; "
                f"examples={sorted(missing)[:3]}"
            )
        ranked = sorted(found.values(), key=lambda r: (-float(r["max_score"]), str(r["tile_id"])))
        return [r for r in ranked if float(r["max_score"]) >= min_score]

    # Without a finalizer sidecar, keep a bounded oversample so per-source
    # diversity can still be applied after candidate provenance is joined.
    top_n = max(1, int(top_n))
    cap = max(top_n, top_n * max(8, int(per_source or 1) * 4))
    heap = []
    serial = 0
    for record in _iter_ledger_records(ledger_paths):
        if record.get("status") != "ok":
            continue
        score = float(record.get("max_score", 0.0))
        if score < min_score:
            continue
        tile_id = str(record["tile_id"])
        # serial prevents Python from comparing dicts when score/id tie.
        item = (score, tile_id, serial, dict(record))
        serial += 1
        if len(heap) < cap:
            heapq.heappush(heap, item)
        elif (score, tile_id) > (heap[0][0], heap[0][1]):
            heapq.heapreplace(heap, item)
    return [item[3] for item in sorted(heap, key=lambda x: (-x[0], x[1]))]


def _bbox_tile_to_source(row, bbox_xyxy):
    """Map a detector box in realized tile pixels back to source-image pixels."""
    if bbox_xyxy is None:
        return None
    sx, sy = map(float, row["tile_actual_scale_xy"])
    x0, y0, _x1, _y1 = map(float, row["tile_scaled_extent_xyxy"])
    bx0, by0, bx1, by1 = map(float, bbox_xyxy)
    return [
        (x0 + bx0) / sx,
        (y0 + by0) / sy,
        (x0 + bx1) / sx,
        (y0 + by1) / sy,
    ]


def _ltrb_to_xywh(box):
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def _intersection_area_ltrb(a, b):
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _ann_ltrb(ann):
    bbox = ann.get("bbox")
    if bbox is None:
        return None
    x, y, w, h = map(float, bbox)
    return [x, y, x + w, y + h]


def _normalize_rgb_uint8(arr):
    import numpy as np

    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(float)
        if arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _draw_box_cv2(canvas, box, *, origin, scale, color, thickness=2):
    import cv2

    if box is None:
        return
    ox, oy = origin
    x0, y0, x1, y1 = box
    p1 = (int(round((x0 - ox) * scale)), int(round((y0 - oy) * scale)))
    p2 = (int(round((x1 - ox) * scale)), int(round((y1 - oy) * scale)))
    cv2.rectangle(canvas, p1, p2, color, thickness, lineType=cv2.LINE_AA)


def _render_preview(source_dset, item, preview_fpath, *, preview_max_dim, context_margin, target_cids):
    import cv2

    gid = item["source_gid"]
    image = source_dset.imgs[gid]
    arr = _normalize_rgb_uint8(source_dset.coco_image(gid).imdelay().finalize())
    h, w = arr.shape[:2]
    tile_box = list(map(float, item["tile_extent_xyxy_in_source"]))
    tx0, ty0, tx1, ty1 = tile_box
    margin_x = max(16.0, (tx1 - tx0) * float(context_margin))
    margin_y = max(16.0, (ty1 - ty0) * float(context_margin))
    cx0 = max(0, int(math.floor(tx0 - margin_x)))
    cy0 = max(0, int(math.floor(ty0 - margin_y)))
    cx1 = min(w, int(math.ceil(tx1 + margin_x)))
    cy1 = min(h, int(math.ceil(ty1 + margin_y)))
    if cx1 <= cx0 or cy1 <= cy0:
        cx0, cy0, cx1, cy1 = 0, 0, w, h
    crop = arr[cy0:cy1, cx0:cx1].copy()
    max_dim = max(crop.shape[:2])
    scale = min(1.0, float(preview_max_dim) / max(1, max_dim))
    if scale != 1.0:
        crop = cv2.resize(
            crop,
            dsize=(max(1, int(round(crop.shape[1] * scale))),
                   max(1, int(round(crop.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    # RGB tuples: candidate window blue, existing target truth green, mined prediction red.
    _draw_box_cv2(crop, tile_box, origin=(cx0, cy0), scale=scale, color=(50, 140, 255), thickness=2)
    for ann in source_dset.annots(gid=gid).objs:
        if ann.get("category_id") not in target_cids:
            continue
        _draw_box_cv2(
            crop, _ann_ltrb(ann), origin=(cx0, cy0), scale=scale,
            color=(70, 220, 70), thickness=2,
        )
    _draw_box_cv2(
        crop, item.get("top_bbox_xyxy_in_source"), origin=(cx0, cy0), scale=scale,
        color=(255, 70, 70), thickness=3,
    )
    label = f"rank={item['rank']} score={item['max_score']:.4f} gid={gid}"
    cv2.putText(
        crop, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    preview_fpath.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(preview_fpath), crop[..., ::-1])
    if not ok:
        raise IOError(f"failed to write preview: {preview_fpath}")


def _write_html(queue, dst_dpath):
    cards = []
    for item in queue:
        preview_rel = item.get("preview_relpath")
        image_html = (
            f'<img loading="lazy" src="{html.escape(preview_rel)}" alt="review preview">'
            if preview_rel else ""
        )
        sidecar = item.get("adjacent_json_sidecar") or ""
        cards.append(f"""
<article class="card">
  {image_html}
  <div class="meta">
    <b>#{item['rank']} score={item['max_score']:.5f}</b><br>
    source gid: <code>{item['source_gid']}</code><br>
    image: <code>{html.escape(str(item['source_fpath']))}</code><br>
    adjacent json: <code>{html.escape(str(sidecar))}</code>
    ({'exists' if item.get('adjacent_json_sidecar_exists') else 'missing'})<br>
    tile: <code>{html.escape(str(item['tile_id']))}</code><br>
    source box: <code>{html.escape(json.dumps(item.get('top_bbox_xyxy_in_source')))}</code><br>
    current target truth on source: {item['num_target_truth_on_source']};
    overlapping prediction bbox: {item['num_target_truth_overlapping_prediction']}
  </div>
</article>""")
    page = """<!doctype html>
<meta charset="utf-8">
<title>Hard-negative truth review</title>
<style>
body { font-family: sans-serif; margin: 1rem; background: #111; color: #eee; }
.notice { max-width: 80rem; padding: 1rem; background: #252525; margin-bottom: 1rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fill,minmax(420px,1fr)); gap: 1rem; }
.card { background: #1c1c1c; padding: .7rem; border: 1px solid #444; }
.card img { width: 100%; height: auto; display: block; margin-bottom: .6rem; }
code { overflow-wrap: anywhere; }
</style>
<div class="notice">
<h2>Hard-negative truth review</h2>
<p>Sorted hardest first. Red = mined model prediction; blue = mined tile extent;
green = existing target-truth bounding boxes. This is a diagnostic review bundle:
edit the canonical source annotations, not <code>review.kwcoco.zip</code>.</p>
<p>Record decisions in <code>review_queue.tsv</code> using the
<code>review_status</code> / <code>review_note</code> columns.</p>
</div>
<div class="grid">
""" + "\n".join(cards) + "\n</div>\n"
    (dst_dpath / "index.html").write_text(page)


def run(config):
    import kwcoco

    from kwcoco_detector_kit.data.candidates import iter_candidate_records, load_candidate_index

    candidate_index = Path(str(config.candidate_index)).expanduser().resolve()
    ledger_paths = [Path(str(p)).expanduser().resolve() for p in config.ledgers]
    dst_dpath = Path(str(config.dst_dpath)).expanduser().resolve()
    dst_dpath.mkdir(parents=True, exist_ok=True)

    selected_scores = _load_selected_ids(config.selected_candidates)
    ranked_records = _rank_score_records(
        ledger_paths,
        selected_scores=selected_scores,
        top_n=int(config.top_n),
        per_source=int(config.per_source),
        min_score=float(config.min_score),
    )
    ranked_by_id = {str(r["tile_id"]): r for r in ranked_records}

    index = load_candidate_index(candidate_index)
    candidate_by_id = {}
    for row in iter_candidate_records(index):
        tile_id = str(row["tile_id"])
        if tile_id in ranked_by_id:
            candidate_by_id[tile_id] = row
            if len(candidate_by_id) == len(ranked_by_id):
                break
    missing = set(ranked_by_id) - set(candidate_by_id)
    if missing:
        raise RuntimeError(f"mined tile IDs missing from candidate index: {sorted(missing)[:3]}")

    source_kwcoco = Path(index["source_kwcoco"]).expanduser().resolve()
    from kwcoco_detector_kit.data.tile_cache import sha256_file
    current_source_fingerprint = sha256_file(source_kwcoco)
    indexed_source_fingerprint = index.get("source_dataset_fingerprint")
    source_fingerprint_matches = (
        indexed_source_fingerprint is None
        or str(indexed_source_fingerprint) == str(current_source_fingerprint)
    )
    if not source_fingerprint_matches:
        print(
            "WARNING: source KWCoco fingerprint changed since candidate enumeration; "
            "review is showing current truth against historical mined candidates. "
            "Do not reuse the old candidate index for new training."
        )
    source_dset = kwcoco.CocoDataset.coerce(str(source_kwcoco))
    target_names = set(index.get("policy", {}).get("category_names", []))
    target_cids = {
        cat["id"] for cat in source_dset.dataset.get("categories", [])
        if cat.get("name") in target_names
    }

    per_source = max(0, int(config.per_source))
    source_counts = {}
    queue = []
    for record in ranked_records:
        row = candidate_by_id[str(record["tile_id"])]
        gid = int(row["tile_source_gid"])
        if per_source and source_counts.get(gid, 0) >= per_source:
            continue
        source_counts[gid] = source_counts.get(gid, 0) + 1
        source_img = source_dset.imgs[gid]
        source_fpath = Path(source_dset.get_image_fpath(gid)).resolve()
        source_bbox_raw = _bbox_tile_to_source(row, record.get("top_bbox_xyxy"))
        source_bbox = None
        if source_bbox_raw is not None:
            width = float(source_img.get("width", 0))
            height = float(source_img.get("height", 0))
            source_bbox = [
                min(max(float(source_bbox_raw[0]), 0.0), width),
                min(max(float(source_bbox_raw[1]), 0.0), height),
                min(max(float(source_bbox_raw[2]), 0.0), width),
                min(max(float(source_bbox_raw[3]), 0.0), height),
            ]
        anns = list(source_dset.annots(gid=gid).objs)
        target_anns = [ann for ann in anns if ann.get("category_id") in target_cids]
        overlapping = 0
        if source_bbox is not None:
            overlapping = sum(
                _intersection_area_ltrb(source_bbox, ann_box) > 0
                for ann in target_anns
                if (ann_box := _ann_ltrb(ann)) is not None
            )
        sidecar = source_fpath.with_suffix(".json")
        item = {
            "rank": len(queue) + 1,
            "max_score": float(record.get("max_score", 0.0)),
            "tile_id": str(row["tile_id"]),
            "source_gid": gid,
            "source_name": source_img.get("name"),
            "source_file_name": source_img.get("file_name"),
            "source_fpath": str(source_fpath),
            "source_kwcoco": str(source_kwcoco),
            "adjacent_json_sidecar": str(sidecar),
            "adjacent_json_sidecar_exists": sidecar.is_file(),
            "negative_origin": row.get("negative_origin"),
            "tile_scale_name": row.get("tile_scale_name"),
            "tile_actual_scale_xy": row.get("tile_actual_scale_xy"),
            "tile_extent_xyxy_in_source": row.get("tile_extent_xyxy_in_source"),
            "top_label": record.get("top_label"),
            "top_bbox_xyxy_in_tile": record.get("top_bbox_xyxy"),
            "top_bbox_xyxy_in_source_unclipped": source_bbox_raw,
            "top_bbox_xyxy_in_source": source_bbox,
            "num_target_truth_on_source": len(target_anns),
            "num_target_truth_overlapping_prediction": int(overlapping),
            "target_truth_annotation_ids": [ann.get("id") for ann in target_anns],
            "review_status": "unreviewed",
            "review_note": "",
        }
        queue.append(item)
        if len(queue) >= int(config.top_n):
            break

    # Diagnostic source-level KWCoco: preserve current truth, then overlay review proposals.
    source_gids = sorted({item["source_gid"] for item in queue})
    review_dset = source_dset.subset(source_gids)
    review_dset.reroot(absolute=True)
    existing = next(
        (cat for cat in review_dset.dataset.get("categories", []) if cat.get("name") == _REVIEW_CATEGORY),
        None,
    )
    review_cid = existing["id"] if existing else review_dset.add_category(
        name=_REVIEW_CATEGORY, color="red",
    )
    for item in queue:
        bbox = _ltrb_to_xywh(item.get("top_bbox_xyxy_in_source"))
        if bbox is None:
            continue
        review_dset.add_annotation(
            image_id=item["source_gid"], category_id=review_cid, bbox=bbox,
            score=item["max_score"], review_rank=item["rank"],
            mined_tile_id=item["tile_id"],
            hard_negative_review=True,
        )
    review_dset.dataset.setdefault("info", []).append({
        "name": "kwcoco_detector_kit.data.review_mine",
        "source_kwcoco": str(source_kwcoco),
        "source_dataset_fingerprint_indexed": indexed_source_fingerprint,
        "source_dataset_fingerprint_current": current_source_fingerprint,
        "source_dataset_fingerprint_matches": source_fingerprint_matches,
        "candidate_index": str(candidate_index),
        "ledgers": [str(p) for p in ledger_paths],
        "selected_candidates": None if not config.selected_candidates else str(Path(config.selected_candidates).resolve()),
        "note": "Diagnostic only; edit canonical source truth, not this review dataset.",
    })
    review_fpath = dst_dpath / "review.kwcoco.zip"
    review_dset.fpath = str(review_fpath)
    review_dset.dump()

    if bool(config.make_previews):
        preview_dpath = dst_dpath / "previews"
        for item in queue:
            preview_name = f"rank_{item['rank']:04d}_gid_{item['source_gid']}.jpg"
            preview_fpath = preview_dpath / preview_name
            _render_preview(
                source_dset, item, preview_fpath,
                preview_max_dim=int(config.preview_max_dim),
                context_margin=float(config.context_margin),
                target_cids=target_cids,
            )
            item["preview_relpath"] = str(Path("previews") / preview_name)
        _write_html(queue, dst_dpath)

    queue_doc = {
        "schema_version": 1,
        "source_kwcoco": str(source_kwcoco),
        "source_dataset_fingerprint_indexed": indexed_source_fingerprint,
        "source_dataset_fingerprint_current": current_source_fingerprint,
        "source_dataset_fingerprint_matches": source_fingerprint_matches,
        "candidate_index": str(candidate_index),
        "ledgers": [str(p) for p in ledger_paths],
        "selected_candidates": None if not config.selected_candidates else str(Path(config.selected_candidates).resolve()),
        "review_kdoc": str(review_fpath),
        "num_items": len(queue),
        "items": queue,
    }
    queue_json = dst_dpath / "review_queue.json"
    queue_json.write_text(json.dumps(queue_doc, indent=2, sort_keys=True) + "\n")

    queue_tsv = dst_dpath / "review_queue.tsv"
    fields = [
        "rank", "max_score", "review_status", "review_note", "source_gid",
        "source_name", "source_fpath", "adjacent_json_sidecar",
        "adjacent_json_sidecar_exists", "tile_id", "negative_origin",
        "tile_scale_name", "tile_extent_xyxy_in_source",
        "top_bbox_xyxy_in_source", "num_target_truth_on_source",
        "num_target_truth_overlapping_prediction", "source_kwcoco",
    ]
    with queue_tsv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for item in queue:
            row = dict(item)
            for key in ["tile_extent_xyxy_in_source", "top_bbox_xyxy_in_source"]:
                row[key] = json.dumps(row.get(key))
            writer.writerow(row)

    print(f"hard-negative review: {len(queue)} items")
    print(f"  queue:  {queue_tsv}")
    print(f"  kdoc:   {review_fpath}")
    if bool(config.make_previews):
        print(f"  html:   {dst_dpath / 'index.html'}")
    return queue_json


__cli__ = ReviewMineConfig


if __name__ == "__main__":
    __cli__.main()
