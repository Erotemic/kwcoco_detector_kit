"""
kwcoco eval driver — write predictions for every image in a test bundle,
then subprocess ``python -m kwcoco eval`` to compute detection metrics.

Output layout (mirrors prior project so eligibility.py finds it)::

  <kcd_root>/eval/<candidate_id>/
    pred_boxes.kwcoco.zip
    eval/
      detect_metrics.json
      confusion.kwcoco.zip
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Tuple


def _valid_detection_bbox(bbox) -> bool:
    """True iff ``bbox`` is a concrete kwcoco detection box."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    return all(v is not None for v in bbox)


def filter_bbox_only_kwcoco(src_fpath, dst_fpath) -> Tuple[Path, int, int]:
    """Write a copy of ``src_fpath`` with non-detection annotations removed.

    Some kwcoco datasets carry image-level/caption-only annotations or other
    task metadata in the annotation table. Those rows are valid for broader
    kwcoco workflows, but ``kwcoco eval``'s detection coercion expects every
    annotation row it sees to have a length-4 ``bbox``. Filtering at the eval
    boundary keeps this toolkit detection-focused without mutating the user's
    source dataset.

    Returns:
        ``(dst_fpath, kept, dropped)``.
    """
    import kwcoco

    src_fpath = Path(src_fpath)
    dst_fpath = Path(dst_fpath)
    if (
        dst_fpath.exists()
        and dst_fpath.stat().st_mtime >= src_fpath.stat().st_mtime
    ):
        dset = kwcoco.CocoDataset.coerce(str(dst_fpath))
        return dst_fpath, len(dset.dataset.get("annotations", [])), 0

    dset = kwcoco.CocoDataset.coerce(str(src_fpath))
    abs_image_fpaths = {}
    for img in dset.dataset.get("images", []):
        try:
            abs_image_fpaths[img["id"]] = str(dset.get_image_fpath(img["id"]))
        except Exception:
            pass
    drop_ids = []
    kept = 0
    for ann in list(dset.anns.values()):
        if _valid_detection_bbox(ann.get("bbox")):
            kept += 1
        else:
            drop_ids.append(ann["id"])
    for aid in drop_ids:
        dset.remove_annotation(aid)
    for img in dset.dataset.get("images", []):
        if img["id"] in abs_image_fpaths:
            img["file_name"] = abs_image_fpaths[img["id"]]

    dst_fpath.parent.mkdir(parents=True, exist_ok=True)
    # remove_annotation() invalidates the imgs/anns indexes (sets them to
    # None in modern kwcoco), and _update_fpath() -> reroot() needs
    # len(self.imgs). Rebuild the index before the save so the reroot path
    # doesn't trip TypeError: object of type 'NoneType' has no len().
    dset._build_index()
    dset._update_fpath(str(dst_fpath))
    dset.dump()
    return dst_fpath, kept, len(drop_ids)


def run_kwcoco_eval(
    *,
    trainer,
    workdir: Path,
    test_kwcoco: str,
    kcd_root: Path,
    candidate_id: str,
    category_name: str = "widget",
    score_thresh: float = 0.001,
    force: bool = False,
) -> Path:
    """Score every image in `test_kwcoco` with the trained model; eval.

    ``score_thresh`` defaults to 0.001 so the COCO AP integral sees the
    full precision-recall curve. Setting this above ~0.01 caps recall
    and artificially deflates AP (the prior 0.30 default cost ~0.07-0.10
    AP on shitspotter pico@416 vs. matching v4's evaluation).
    """
    import kwcoco

    workdir = Path(workdir)
    eval_root = Path(kcd_root) / "eval" / candidate_id
    eval_root.mkdir(parents=True, exist_ok=True)
    eval_inner = eval_root / "eval"
    eval_inner.mkdir(parents=True, exist_ok=True)
    metrics_fpath = eval_inner / "detect_metrics.json"

    if metrics_fpath.exists() and not bool(force):
        print(f"  reusing existing eval metrics: {metrics_fpath}")
        return metrics_fpath

    predictor = trainer.build_predictor(workdir, device="cpu")

    true = kwcoco.CocoDataset.coerce(str(test_kwcoco))
    pred = kwcoco.CocoDataset()
    pred.fpath = str(eval_root / "pred_boxes.kwcoco.zip")
    cat_id = pred.add_category(name=str(category_name))
    # Copy image rows but rewrite file_name to the absolute on-disk path so
    # the eval subprocess can reroot regardless of where pred_boxes.kwcoco.zip
    # lives. Without this, kwcoco reroots against the pred bundle's dir and
    # the relative file_name="raw_assets/foo.jpg" points at a nonexistent
    # path under the pred bundle's parent.
    for img in true.images().objs:
        new = {k: v for k, v in img.items() if k != "id"}
        try:
            abs_fpath = true.get_image_fpath(img["id"])
            new["file_name"] = str(abs_fpath)
        except Exception:
            pass
        pred.add_image(**new, id=img["id"])

    for gid in list(true.images()):
        coco_img = true.coco_image(gid)
        try:
            arr = coco_img.imdelay().finalize()
        except Exception as ex:
            print(f"  eval: failed to read gid {gid}: {ex}")
            continue
        H, W = arr.shape[:2]
        detections = predictor.predict_image(arr, (W, H))
        for det in detections:
            score = float(det.get("score", 0.0))
            if score < float(score_thresh):
                continue
            x1, y1, x2, y2 = det["bbox_xyxy"]
            pred.add_annotation(
                image_id=gid, category_id=cat_id,
                bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                score=score,
            )
    pred.dump()

    true_filtered, true_kept, true_dropped = filter_bbox_only_kwcoco(
        test_kwcoco, eval_root / "true_bbox_only.kwcoco.zip")
    pred_filtered, pred_kept, pred_dropped = filter_bbox_only_kwcoco(
        pred.fpath, eval_root / "pred_boxes_bbox_only.kwcoco.zip")
    if true_dropped or pred_dropped:
        print(
            "  eval bbox filter: "
            f"true kept={true_kept} dropped={true_dropped}; "
            f"pred kept={pred_kept} dropped={pred_dropped}"
        )

    cmd = [
        sys.executable, "-m", "kwcoco", "eval",
        "--true_dataset", str(true_filtered),
        "--pred_dataset", str(pred_filtered),
        "--out_dpath", str(eval_inner),
        "--out_fpath", str(metrics_fpath),
        "--draw", "False",
        "--iou_thresh", "0.5",
    ]
    # The confusion sidecar pass inside kwcoco eval can raise a reroot
    # exception in some asset layouts, but the metrics JSON is written
    # before that step. Tolerate a non-zero subprocess exit when the
    # metrics file landed on disk anyway — same recovery pattern as the
    # ONNX export's onnxsim crash handling.
    result = subprocess.run(cmd)
    if result.returncode != 0 and not metrics_fpath.exists():
        raise subprocess.CalledProcessError(result.returncode, cmd)
    if result.returncode != 0:
        print(
            f"  kwcoco eval exited {result.returncode} but {metrics_fpath} "
            "is present — recovering metrics."
        )
    print(f"  wrote {metrics_fpath}")
    return metrics_fpath
