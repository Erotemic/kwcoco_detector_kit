#!/usr/bin/env python3
"""
Validate that an exported sea-lion ONNX package produces REAL detections on a
REAL annotated image, and render a GT-vs-prediction comparison to share.

The whole-images in the corpus are ~5760x3840; the detector trains on 1280
tiles, so running a whole frame downscaled to 1280 shrinks sea-lions below the
detector's small-object floor. This script instead crops a 1280x1280 window
centered on the densest cluster of ground-truth annotations (preferring pups,
the hard class), runs OnnxPredictor on that crop, and writes:

  * <out>/positive_crop.jpg        the crop itself (container-visible path)
  * <out>/gt_vs_pred.png           2-up: GT boxes | predicted boxes
  * <out>/crop_list.txt            single-line image list for the VIAME pipeline

So the same crop can be re-run through the actual VIAME pipeline for parity.

Usage
-----
    python dev/viame_demo_positive_crop.py \
        --kwcoco  .../scheme_applied/test.kwcoco.zip \
        --package .../export \
        --out     /data/users/jon.crall/kcd_sealion/viame_demo \
        [--gid 5903] [--window 1280] [--score-thresh 0.30]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def pick_densest_window(dset, gid, win, name2id):
    """Slide a win x win window; return (x0, y0) maximizing contained annots
    (pups break ties)."""
    img = dset.imgs[gid]
    W, H = int(img["width"]), int(img["height"])
    pup_id = name2id.get("pup")
    centers = []
    for aid in dset.gid_to_aids[gid]:
        x, y, w, h = dset.anns[aid]["bbox"]
        centers.append((x + w / 2.0, y + h / 2.0,
                        1 if dset.anns[aid]["category_id"] == pup_id else 0))
    centers = np.array(centers) if centers else np.zeros((0, 3))
    best, best_xy = (-1, -1), (0, 0)
    step = max(64, win // 8)
    for y0 in range(0, max(1, H - win) + 1, step):
        for x0 in range(0, max(1, W - win) + 1, step):
            if len(centers) == 0:
                break
            inside = ((centers[:, 0] >= x0) & (centers[:, 0] < x0 + win) &
                      (centers[:, 1] >= y0) & (centers[:, 1] < y0 + win))
            n = int(inside.sum())
            npup = int(centers[inside, 2].sum())
            if (n, npup) > best:
                best, best_xy = (n, npup), (x0, y0)
    return best_xy, best


def gt_boxes_in_window(dset, gid, x0, y0, win, name2id):
    """Return (ltrb Nx4, class_names) for GT annots inside the window, in crop
    coords."""
    id2name = {cid: c["name"] for cid, c in dset.cats.items()}
    boxes, names = [], []
    for aid in dset.gid_to_aids[gid]:
        x, y, w, h = dset.anns[aid]["bbox"]
        cx, cy = x + w / 2.0, y + h / 2.0
        if x0 <= cx < x0 + win and y0 <= cy < y0 + win:
            boxes.append([x - x0, y - y0, x - x0 + w, y - y0 + h])
            names.append(id2name[dset.anns[aid]["category_id"]])
    return np.array(boxes) if boxes else np.zeros((0, 4)), names


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kwcoco", required=True)
    ap.add_argument("--package", required=True, help="export dir or .onnx")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gid", type=int, default=None,
                    help="image id; default = densest-annotated image")
    ap.add_argument("--window", type=int, default=1280)
    ap.add_argument("--score-thresh", type=float, default=0.30)
    args = ap.parse_args()

    import kwcoco
    import kwimage
    from kwcoco_detector_kit.predictors.onnx import OnnxPredictor

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dset = kwcoco.CocoDataset(args.kwcoco)
    name2id = {c["name"]: cid for cid, c in dset.cats.items()}

    gid = args.gid
    if gid is None:
        gid = max(dset.images(), key=lambda g: len(dset.gid_to_aids[g]))
    print(f"[demo] image gid={gid}  file={dset.imgs[gid].get('file_name')}")

    (x0, y0), (n, npup) = pick_densest_window(dset, gid, args.window, name2id)
    print(f"[demo] densest {args.window}px window at ({x0},{y0}): "
          f"{n} GT sea-lions ({npup} pups) inside")

    fpath = dset.get_image_fpath(gid)
    full = kwimage.imread(fpath, space="rgb")
    H, W = full.shape[:2]
    x1, y1 = min(x0 + args.window, W), min(y0 + args.window, H)
    crop = full[y0:y1, x0:x1]
    crop_fpath = out / "positive_crop.jpg"
    kwimage.imwrite(str(crop_fpath), crop, space="rgb")
    (out / "crop_list.txt").write_text(str(crop_fpath) + "\n")
    print(f"[demo] wrote crop {crop.shape} -> {crop_fpath}")

    # --- run the detector ----------------------------------------------------
    pred = OnnxPredictor(args.package, device="cpu", score_thresh=args.score_thresh)
    print(f"[demo] category_names: {pred.category_names}")
    dets = pred.predict_image_kwimage(crop)
    print(f"[demo] predicted {len(dets)} detections >= {args.score_thresh}")
    if len(dets):
        for ci in sorted(set(map(int, dets.class_idxs))):
            cn = pred.category_names[ci] if ci < len(pred.category_names) else str(ci)
            print(f"         {cn:16s}: {int((dets.class_idxs == ci).sum())}")

    # --- render GT | Pred ----------------------------------------------------
    gt_ltrb, gt_names = gt_boxes_in_window(dset, gid, x0, y0, args.window, name2id)
    color = {"pup": "dodgerblue", "nonpup_sealion": "orange"}

    gt_canvas = crop.copy()
    if len(gt_ltrb):
        gt_dets = kwimage.Detections(
            boxes=kwimage.Boxes(gt_ltrb, "ltrb"),
            class_idxs=np.array([0 if nm == "pup" else 1 for nm in gt_names]),
            classes=["pup", "nonpup_sealion"],
        )
        gt_canvas = gt_dets.draw_on(gt_canvas, color=[color.get(nm, "lime") for nm in gt_names])

    pred_canvas = crop.copy()
    if len(dets):
        pnames = [pred.category_names[int(c)] for c in dets.class_idxs]
        pred_canvas = dets.draw_on(pred_canvas,
                                   color=[color.get(nm, "red") for nm in pnames])

    gt_canvas = kwimage.draw_header_text(gt_canvas, f"GROUND TRUTH ({len(gt_ltrb)})",
                                         color="white")
    pred_canvas = kwimage.draw_header_text(pred_canvas,
                                           f"PREDICTED ({len(dets)} @ thr {args.score_thresh})",
                                           color="white")
    stacked = kwimage.stack_images([gt_canvas, pred_canvas], axis=1, pad=8)
    viz_fpath = out / "gt_vs_pred.png"
    kwimage.imwrite(str(viz_fpath), stacked)
    print(f"[demo] wrote comparison -> {viz_fpath}")
    print("\n[demo] DONE")


if __name__ == "__main__":
    main()
