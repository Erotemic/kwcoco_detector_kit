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


def run_kwcoco_eval(
    *,
    trainer,
    workdir: Path,
    test_kwcoco: str,
    kcd_root: Path,
    candidate_id: str,
    category_name: str = "widget",
    score_thresh: float = 0.30,
) -> Path:
    """Score every image in `test_kwcoco` with the trained model; eval."""
    import kwcoco

    workdir = Path(workdir)
    eval_root = Path(kcd_root) / "eval" / candidate_id
    eval_root.mkdir(parents=True, exist_ok=True)
    eval_inner = eval_root / "eval"
    eval_inner.mkdir(parents=True, exist_ok=True)

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

    metrics_fpath = eval_inner / "detect_metrics.json"
    cmd = [
        sys.executable, "-m", "kwcoco", "eval",
        "--true_dataset", str(test_kwcoco),
        "--pred_dataset", str(pred.fpath),
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
