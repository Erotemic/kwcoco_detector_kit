"""Truth-aware review of detector predictions in original source coordinates."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import kwconf

from kwcoco_detector_kit.data.truth_semantics import TruthSemantics


def ann_ltrb(ann):
    bbox = ann.get("bbox")
    if bbox is None and ann.get("segmentation") is not None:
        try:
            import kwimage
            bbox = kwimage.Segmentation.coerce(ann["segmentation"]).to_multi_polygon().box().to_coco()
        except Exception:
            bbox = None
    if bbox is None:
        return None
    x, y, w, h = map(float, bbox)
    return [x, y, x + w, y + h]


def intersection_area_ltrb(a, b):
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def iou_ltrb(a, b):
    inter = intersection_area_ltrb(a, b)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def classify_prediction(
    true_dset,
    gid,
    pred_bbox_ltrb,
    semantics: TruthSemantics,
    *,
    target_iou_thresh=0.5,
):
    """Classify one source-coordinate prediction against current source truth."""
    overlaps = []
    best_target_iou = 0.0
    roles_seen = set()
    has_unlocalized_ignore = False
    for ann in true_dset.annots(gid=gid).objs:
        role = semantics.annotation_role(true_dset, ann)
        box = ann_ltrb(ann)
        if box is None:
            # An uncertain annotation whose location cannot be recovered blocks a
            # definitive false-positive judgment for the entire source image.
            if role == "ignore":
                has_unlocalized_ignore = True
            continue
        inter = intersection_area_ltrb(pred_bbox_ltrb, box)
        if inter <= 0:
            continue
        roles_seen.add(role)
        name = semantics.category_name(true_dset, ann)
        row = {
            "annotation_id": ann.get("id"),
            "category_id": ann.get("category_id"),
            "category_name": name,
            "role": role,
            "intersection_area": float(inter),
            "iou": float(iou_ltrb(pred_bbox_ltrb, box)),
        }
        overlaps.append(row)
        if role == "target":
            best_target_iou = max(best_target_iou, row["iou"])

    # Missing-truth review must be conservative: an "unexplained" proposal
    # is useful only when it is spatially disjoint from all localized truth.
    # A low-IoU prediction that merely grazes an existing target is not a
    # missing-object candidate, even though it does not satisfy the configured
    # target match threshold.
    has_spatial_truth_overlap = bool(overlaps)
    if best_target_iou >= float(target_iou_thresh):
        classification = "matched_target"
    elif "ignore" in roles_seen or has_unlocalized_ignore:
        classification = "uncertain_region"
    elif "background" in roles_seen:
        classification = "known_distractor"
    elif "target" in roles_seen:
        classification = "overlapping_target"
    else:
        classification = "unexplained_prediction"

    return {
        "classification": classification,
        "best_target_iou": float(best_target_iou),
        "has_spatial_truth_overlap": has_spatial_truth_overlap,
        "overlapping_annotation_ids": [r["annotation_id"] for r in overlaps],
        "overlapping_category_names": [r["category_name"] for r in overlaps],
        "overlaps": overlaps,
        "has_unlocalized_ignore": bool(has_unlocalized_ignore),
    }


class PredictionReviewConfig(kwconf.Config):
    """Build a ranked source-linked truth review from prediction KWCoco."""

    true = kwconf.Value(None, required=True, help="canonical/current source truth KWCoco")
    pred = kwconf.Value(None, required=True, help="source-coordinate prediction KWCoco")
    dst_dpath = kwconf.Value(None, required=True, help="review output directory")
    target_categories = kwconf.Value(None, required=True, help="comma-separated positive detector classes")
    ignore_categories = kwconf.Value("", help="comma-separated uncertain source classes")
    uncategorized_annotation_policy = kwconf.Value(
        "ignore", choices=["background", "ignore", "error"]
    )
    default_non_target_policy = kwconf.Value(
        "background", choices=["background", "ignore", "error"]
    )
    unclassified_category_policy = kwconf.Value(
        "ignore", choices=["background", "ignore", "error"]
    )
    min_score = kwconf.Value(0.0, parser=float)
    top_n = kwconf.Value(500, parser=int)
    target_iou_thresh = kwconf.Value(0.5, parser=float)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        return build_prediction_review(config)


def _write_review_html(queue, dst_dpath):
    rows = []
    for item in queue:
        rows.append(
            "<tr>"
            f"<td>{item['rank']}</td>"
            f"<td>{item['score']:.5f}</td>"
            f"<td>{html.escape(item['classification'])}</td>"
            f"<td>{item['source_gid']}</td>"
            f"<td><code>{html.escape(item['source_fpath'])}</code></td>"
            f"<td><code>{html.escape(item['labelme_json'])}</code></td>"
            f"<td>{html.escape(', '.join(str(x) for x in item['overlapping_category_names']))}</td>"
            "</tr>"
        )
    page = """<!doctype html><meta charset=\"utf-8\"><title>Prediction truth review</title>
<style>body{font-family:sans-serif;margin:1rem}table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ccc;padding:.35rem;vertical-align:top}code{overflow-wrap:anywhere}</style>
<h1>Prediction truth review</h1>
<p>Diagnostic only. Edit canonical source annotations, not this bundle.</p>
<table><thead><tr><th>rank</th><th>score</th><th>class</th><th>gid</th><th>image</th><th>LabelMe</th><th>overlapping truth</th></tr></thead><tbody>
""" + "\n".join(rows) + "\n</tbody></table>\n"
    (dst_dpath / "index.html").write_text(page)


def build_prediction_review(config):
    import kwcoco

    true_fpath = Path(str(config.true)).expanduser().resolve()
    pred_fpath = Path(str(config.pred)).expanduser().resolve()
    dst_dpath = Path(str(config.dst_dpath)).expanduser().resolve()
    dst_dpath.mkdir(parents=True, exist_ok=True)
    true_dset = kwcoco.CocoDataset.coerce(str(true_fpath))
    pred_dset = kwcoco.CocoDataset.coerce(str(pred_fpath))
    semantics = TruthSemantics.coerce(
        target_categories=config.target_categories,
        ignore_categories=config.ignore_categories,
        uncategorized_annotation_policy=config.uncategorized_annotation_policy,
        default_non_target_policy=config.default_non_target_policy,
        unclassified_category_policy=config.unclassified_category_policy,
    )

    queue = []
    for ann in pred_dset.annots().objs:
        score = float(ann.get("score", 0.0))
        if score < float(config.min_score):
            continue
        gid = int(ann["image_id"])
        if gid not in true_dset.imgs:
            raise KeyError(
                f"prediction gid={gid} is absent from truth; source image IDs must be preserved"
            )
        bbox = ann_ltrb(ann)
        if bbox is None:
            continue
        source_fpath = Path(true_dset.get_image_fpath(gid)).resolve()
        truth = classify_prediction(
            true_dset,
            gid,
            bbox,
            semantics,
            target_iou_thresh=float(config.target_iou_thresh),
        )
        queue.append({
            "rank": 0,
            "score": score,
            "source_gid": gid,
            "source_image": true_dset.imgs[gid].get("file_name"),
            "source_fpath": str(source_fpath),
            "labelme_json": str(source_fpath.with_suffix(".json")),
            "labelme_json_exists": source_fpath.with_suffix(".json").is_file(),
            "prediction_ann_id": ann.get("id"),
            "prediction_bbox_xyxy": [float(v) for v in bbox],
            "prediction_has_segmentation": ann.get("segmentation") is not None,
            **truth,
            "review_status": "unreviewed",
            "review_note": "",
        })
    queue.sort(key=lambda row: (-row["score"], row["source_gid"], row["prediction_ann_id"] or -1))
    queue = queue[:max(0, int(config.top_n))]
    for rank, item in enumerate(queue, 1):
        item["rank"] = rank

    (dst_dpath / "review_queue.json").write_text(json.dumps({
        "schema": "kwcoco_detector_kit.prediction_review.v1",
        "true_kwcoco": str(true_fpath),
        "pred_kwcoco": str(pred_fpath),
        "truth_semantics": semantics.to_dict(),
        "items": queue,
    }, indent=2, sort_keys=True) + "\n")

    tsv_fields = [
        "rank", "score", "classification", "source_gid", "source_fpath",
        "labelme_json", "labelme_json_exists", "prediction_ann_id",
        "prediction_bbox_xyxy", "has_spatial_truth_overlap",
        "overlapping_annotation_ids", "overlapping_category_names",
        "best_target_iou", "review_status", "review_note",
    ]
    with open(dst_dpath / "review_queue.tsv", "w", newline="", encoding="utf8") as file:
        writer = csv.DictWriter(file, fieldnames=tsv_fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for item in queue:
            writer.writerow({
                **item,
                "prediction_bbox_xyxy": json.dumps(item["prediction_bbox_xyxy"]),
                "overlapping_annotation_ids": json.dumps(item["overlapping_annotation_ids"]),
                "overlapping_category_names": json.dumps(item["overlapping_category_names"]),
            })

    source_gids = sorted({item["source_gid"] for item in queue})
    review_dset = true_dset.subset(source_gids)
    review_dset.reroot(absolute=True)
    proposal_name = "__prediction_review__"
    existing = next(
        (c for c in review_dset.dataset.get("categories", []) if c.get("name") == proposal_name),
        None,
    )
    proposal_cid = existing["id"] if existing else review_dset.add_category(name=proposal_name)
    pred_ann_by_id = pred_dset.anns
    for item in queue:
        pred_ann = pred_ann_by_id[item["prediction_ann_id"]]
        new_ann = {
            "image_id": item["source_gid"],
            "category_id": proposal_cid,
            "bbox": list(pred_ann["bbox"]),
            "score": item["score"],
            "truth_classification": item["classification"],
            "source_prediction_ann_id": item["prediction_ann_id"],
        }
        if pred_ann.get("segmentation") is not None:
            new_ann["segmentation"] = pred_ann["segmentation"]
        review_dset.add_annotation(**new_ann)
    review_dset.dataset.setdefault("info", []).append({
        "type": "kwcoco_detector_kit.prediction_review",
        "diagnostic_only": True,
        "canonical_truth": str(true_fpath),
        "prediction_source": str(pred_fpath),
        "truth_semantics": semantics.to_dict(),
    })
    review_dset.fpath = str(dst_dpath / "review.kwcoco.zip")
    review_dset.dump()
    _write_review_html(queue, dst_dpath)
    print(f"prediction-review: wrote {len(queue)} ranked items to {dst_dpath}")
    return dst_dpath


__cli__ = PredictionReviewConfig
