"""
Score a single checkpoint against a kwcoco bundle, then apply the kit's
distractor-pruning eval pass.

Use case: you have multiple checkpoints from one training run (best_stg1,
best_stg2, last, intermediate snapshots) and want to know which is best
under the distractor-pruned metric — without trusting DEIMv2's in-train
selection which uses the with-distractor metric.

The kit's standard `run_kwcoco_eval` uses `trainer.build_predictor(workdir)`
which auto-picks one checkpoint (best_stg2 > best_stg1 > last). This helper
takes a specific `ckpt_fpath` instead, so a caller can iterate.

Outputs land in a per-checkpoint sub-folder under the eval root so multiple
checkpoints can be scored without overwriting each other.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


def score_one_checkpoint(
    *,
    trainer,
    workdir: Path,
    ckpt_fpath: Path,
    eval_target_kwcoco: str,
    eval_root: Path,
    candidate_id: str,
    category_names: Sequence[str],
    distractor_classes: Optional[Sequence[str]] = None,
    score_thresh: float = 0.001,
    device: str = "cpu",
    force: bool = False,
) -> Path:
    """Run predict + eval for a specific (checkpoint, test bundle) pair.

    Writes under ``<eval_root>/<ckpt_stem>/``:
      - pred_boxes.kwcoco.zip
      - pred_boxes_bbox_only.kwcoco.zip
      - true_bbox_only.kwcoco.zip
      - eval/detect_metrics.json
      - eval/detect_metrics.<distractors>.json   (if distractor_classes set)

    Returns the path to the per-checkpoint eval dir.
    """
    import kwcoco
    from kwcoco_detector_kit.eval.kwcoco_eval import (
        filter_bbox_only_kwcoco,
        _distractor_sidecar_fpath,
        _rerun_eval_dropping_distractors,
    )

    workdir = Path(workdir)
    eval_root = Path(eval_root)
    ckpt_fpath = Path(ckpt_fpath)
    ckpt_stem = ckpt_fpath.stem
    out_root = eval_root / ckpt_stem
    out_root.mkdir(parents=True, exist_ok=True)
    eval_inner = out_root / "eval"
    eval_inner.mkdir(parents=True, exist_ok=True)

    metrics_fpath = eval_inner / "detect_metrics.json"
    if metrics_fpath.exists() and not bool(force):
        print(f"  reusing {metrics_fpath}")
        if distractor_classes:
            sidecar_fpath = _distractor_sidecar_fpath(metrics_fpath, distractor_classes)
            if sidecar_fpath.exists():
                return out_root
        else:
            return out_root

    category_names = list(category_names)

    # Build a predictor pinned to this specific checkpoint. The trainer's
    # build_predictor() autoselects; we go through the DEIMv2Predictor
    # constructor directly with our chosen ckpt.
    from kwcoco_detector_kit.trainers.deimv2 import DEIMv2Predictor
    cfg_fpath = workdir / "generated_configs" / "train.yml"
    if not cfg_fpath.exists():
        raise FileNotFoundError(
            f"missing {cfg_fpath} -- per-checkpoint eval needs the original "
            "generated train.yml to build the predictor"
        )
    predictor = DEIMv2Predictor(ckpt_fpath, cfg_fpath, device=device)

    # Predict every image in the eval target.
    true = kwcoco.CocoDataset.coerce(str(eval_target_kwcoco))
    pred = kwcoco.CocoDataset()
    pred.fpath = str(out_root / "pred_boxes.kwcoco.zip")
    label_to_cat_id = [pred.add_category(name=name) for name in category_names]
    n_labels = len(label_to_cat_id)
    for img in true.images().objs:
        new = {k: v for k, v in img.items() if k != "id"}
        try:
            new["file_name"] = str(true.get_image_fpath(img["id"]))
        except Exception:
            pass
        pred.add_image(**new, id=img["id"])

    dropped = 0
    n_imgs_total = len(list(true.images()))
    for n, gid in enumerate(list(true.images())):
        coco_img = true.coco_image(gid)
        try:
            arr = coco_img.imdelay().finalize()
        except Exception as ex:
            print(f"  warn: read failed for gid={gid}: {ex}")
            continue
        H, W = arr.shape[:2]
        dets = predictor.predict_image(arr, (W, H))
        for det in dets:
            score = float(det.get("score", 0.0))
            if score < float(score_thresh):
                continue
            label = int(det.get("label", 0))
            if label < 0 or label >= n_labels:
                dropped += 1
                continue
            x1, y1, x2, y2 = det["bbox_xyxy"]
            pred.add_annotation(
                image_id=gid, category_id=label_to_cat_id[label],
                bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                score=score,
            )
        if (n + 1) % 25 == 0 or n + 1 == n_imgs_total:
            print(f"  [{ckpt_stem}] predicted {n+1}/{n_imgs_total}", flush=True)
    if dropped:
        print(f"  warn: dropped {dropped} out-of-range labels")
    pred.dump()

    true_filt, _, _ = filter_bbox_only_kwcoco(
        eval_target_kwcoco, out_root / "true_bbox_only.kwcoco.zip"
    )
    pred_filt, _, _ = filter_bbox_only_kwcoco(
        pred.fpath, out_root / "pred_boxes_bbox_only.kwcoco.zip"
    )

    cmd = [
        sys.executable, "-m", "kwcoco", "eval",
        "--true_dataset", str(true_filt),
        "--pred_dataset", str(pred_filt),
        "--out_dpath", str(eval_inner),
        "--out_fpath", str(metrics_fpath),
        "--draw", "False",
        "--iou_thresh", "0.5",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0 and not metrics_fpath.exists():
        raise subprocess.CalledProcessError(result.returncode, cmd)
    print(f"  wrote {metrics_fpath}")

    if distractor_classes:
        sidecar_fpath = _distractor_sidecar_fpath(metrics_fpath, distractor_classes)
        if sidecar_fpath.exists() and not bool(force):
            print(f"  reusing {sidecar_fpath}")
        else:
            _rerun_eval_dropping_distractors(
                true_fpath=true_filt,
                pred_fpath=pred_filt,
                distractor_names=distractor_classes,
                out_fpath=sidecar_fpath,
                score_thresh=score_thresh,
                test_kwcoco=str(eval_target_kwcoco),
                candidate_id=candidate_id,
                category_names=category_names,
            )
            print(f"  wrote {sidecar_fpath}")
    return out_root
