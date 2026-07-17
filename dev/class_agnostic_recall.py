#!/usr/bin/env python3
"""
Class-agnostic detection recall, bucketed by TRUE category.

Separates "did we localize the animal" (detection) from "did we label it right"
(classification): predictions are matched to ground truth by IoU ONLY (predicted
class ignored), then recall is reported per true category. If a rare class has
high class-agnostic recall but low one-vs-rest AP, the model finds it and
mislabels it — a classification problem, not a detection problem.

Reads the raw pred_boxes / true_bbox_only kwcoco emitted by the kit eval. The
pred file has millions of low-score boxes, so we parse the JSON directly and
filter by score before matching (kwcoco's full object load is too slow here).

Usage
-----
    python dev/class_agnostic_recall.py \
        --pred  .../eval/<cand>/pred_boxes.kwcoco.zip \
        --true  .../eval/<cand>/true_bbox_only.kwcoco.zip \
        [--iou 0.5] [--thresholds 0.1,0.3,0.5]
"""
from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import kwconf


class RecallConfig(kwconf.Config):
    pred = kwconf.Value(None, required=True, help="pred_boxes kwcoco (.zip)")
    true = kwconf.Value(None, required=True, help="true_bbox_only kwcoco (.zip)")
    iou = kwconf.Value(0.5, parser=float, help="IoU match threshold")
    thresholds = kwconf.Value("0.1,0.3,0.5", help="comma-sep score thresholds to report")
    label = kwconf.Value("", help="optional label for the header")


def _load_json(zpath: Path) -> dict:
    zpath = Path(zpath)
    if zpath.suffix == ".zip":
        with zipfile.ZipFile(zpath) as zf:
            inner = next(n for n in zf.namelist() if n.endswith((".json", ".kwcoco")))
            with zf.open(inner) as f:
                return json.load(f)
    return json.loads(zpath.read_text())


def _iou_one_to_many(box, B):
    """IoU of one xywh box against an (n,4) xywh array."""
    ax0, ay0, aw, ah = box
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx0, by0 = B[:, 0], B[:, 1]
    bx1, by1 = bx0 + B[:, 2], by0 + B[:, 3]
    ix0 = np.maximum(ax0, bx0); iy0 = np.maximum(ay0, by0)
    ix1 = np.minimum(ax1, bx1); iy1 = np.minimum(ay1, by1)
    iw = np.clip(ix1 - ix0, 0, None); ih = np.clip(iy1 - iy0, 0, None)
    inter = iw * ih
    union = aw * ah + B[:, 2] * B[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


def main(argv=None) -> int:
    config = RecallConfig.cli(argv=argv)
    thresholds = sorted({float(t) for t in str(config.thresholds).split(",") if t.strip()})
    keep_min = min(thresholds)

    pred = _load_json(Path(config.pred).expanduser())
    true = _load_json(Path(config.true).expanduser())

    # Align pred<->true images by file BASENAME — ids differ across kwcoco
    # builds and paths differ (pred often absolute, true often relative).
    import os
    def _base(fn):
        return os.path.basename(fn) if fn else fn
    tname = {im["id"]: _base(im.get("file_name")) for im in true["images"]}
    pname = {im["id"]: _base(im.get("file_name")) for im in pred["images"]}
    fname_to_tid = {fn: i for i, fn in tname.items()}

    tcat = {c["id"]: c["name"] for c in true["categories"]}
    true_by_img = defaultdict(list)
    for ann in true["annotations"]:
        if "bbox" not in ann:
            continue
        true_by_img[ann["image_id"]].append((tcat[ann["category_id"]], ann["bbox"]))

    # pred grouped by the TRUE image id (via file_name), pre-filtered by score.
    pred_by_tid = defaultdict(list)
    n_pred_kept = 0
    for ann in pred["annotations"]:
        if "bbox" not in ann:
            continue
        s = float(ann.get("score", 1.0))
        if s < keep_min:
            continue
        tid = fname_to_tid.get(pname.get(ann["image_id"]))
        if tid is None:
            continue
        pred_by_tid[tid].append((s, ann["bbox"]))
        n_pred_kept += 1

    hdr = config.label or Path(config.pred).parts[-3]
    print(f"# class-agnostic detection recall by TRUE category — {hdr}")
    print(f"# iou={config.iou}  pred boxes kept (score>={keep_min}): {n_pred_kept}\n")

    cats = sorted({c for lst in true_by_img.values() for c, _ in lst})
    totals = defaultdict(int)
    # matched[thresh][cat]
    matched = {t: defaultdict(int) for t in thresholds}
    for tid, tlist in true_by_img.items():
        preds = pred_by_tid.get(tid, [])
        for t in thresholds:
            P = np.array([b for s, b in preds if s >= t], dtype=float)
            for cname, tb in tlist:
                if t == thresholds[0]:
                    totals[cname] += 1
                if len(P) and _iou_one_to_many(np.asarray(tb, float), P).max() >= config.iou:
                    matched[t][cname] += 1

    col = "  ".join(f"r@{t:g}" for t in thresholds)
    print(f"{'true category':18s} {'GT':>7s}   {col}")
    for c in sorted(cats, key=lambda c: -totals[c]):
        cells = "  ".join(f"{matched[t][c] / max(totals[c], 1):.3f}" for t in thresholds)
        print(f"{c:18s} {totals[c]:7d}   {cells}")
    # overall (any category)
    gt_all = sum(totals.values())
    over = "  ".join(
        f"{sum(matched[t].values()) / max(gt_all, 1):.3f}" for t in thresholds)
    print(f"{'ALL (detection)':18s} {gt_all:7d}   {over}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
