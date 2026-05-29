#!/usr/bin/env python3
"""
Recompute sea-lion detection AP from an existing eval output dir, with
distractor classes pruned from both ground-truth and predictions.

For the sea-lion project, distractor = northern_fur_seal — the model
learns it as its own class so it can discriminate, but the mission
treats it as a non-target. The kit's eval step now does this
automatically (see kwcoco_detector_kit/eval/kwcoco_eval.py +
docs/class_schemes.yaml::*.distractor_classes); this script exists to
backfill the metric on runs that finished before that plumbing landed.

Reads pred_boxes.kwcoco.zip + true_bbox_only.kwcoco.zip from the eval
dir, prunes the named distractor classes from both, reruns the kwcoco
CocoEvaluator, and prints the corrected AP. The original
detect_metrics.json on disk is NOT modified; use --write-json to write
a sibling detect_metrics.<distractors>.json sidecar (matching the
filename convention that eligibility's selection function prefers).

Usage:
    python3 projects/viame_sealions_2026/scripts/recompute_sealion_ap.py \\
        --eval-dir /data/users/jon.crall/kcd_sealion/runs/lifestage_6cls_.../eval/<candidate>/

    # Override the distractor list (default: northern_fur_seal):
    #   --distractor-class some_class --distractor-class another
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import kwcoco


DEFAULT_DISTRACTORS = ("northern_fur_seal",)


def _prune_classes(dset: kwcoco.CocoDataset, distractor_names) -> kwcoco.CocoDataset:
    """Return a copy of `dset` with annotations whose category name is in
    `distractor_names` removed. Categories themselves stay so the eval doesn't
    complain about class-id remapping; only annotations are dropped.
    """
    distractor_names = set(distractor_names)
    out = dset.copy()
    cat_ids_to_drop = {c["id"] for c in out.dataset["categories"]
                       if c["name"] in distractor_names}
    if not cat_ids_to_drop:
        return out
    keep = [a for a in out.dataset["annotations"]
            if a.get("category_id") not in cat_ids_to_drop]
    n_before = len(out.dataset["annotations"])
    out.dataset["annotations"] = keep
    out._build_index()
    print(f"  dropped {n_before - len(keep)} / {n_before} annotations "
          f"(classes: {sorted(distractor_names)})")
    return out


def _find_bundles(eval_dir: Path):
    true_fpath = eval_dir / "true_bbox_only.kwcoco.zip"
    pred_fpath = eval_dir / "pred_boxes.kwcoco.zip"
    pred_bbox_fpath = eval_dir / "pred_boxes_bbox_only.kwcoco.zip"
    if not true_fpath.exists():
        raise FileNotFoundError(true_fpath)
    if pred_bbox_fpath.exists():
        pred_fpath = pred_bbox_fpath
    elif not pred_fpath.exists():
        raise FileNotFoundError(pred_fpath)
    return true_fpath, pred_fpath


def recompute(eval_dir: Path, distractor_names, write_json: bool = False):
    from kwcoco.coco_evaluator import CocoEvaluator

    true_fpath, pred_fpath = _find_bundles(eval_dir)
    print(f"GT   : {true_fpath}")
    print(f"pred : {pred_fpath}")

    true = kwcoco.CocoDataset.coerce(true_fpath)
    pred = kwcoco.CocoDataset.coerce(pred_fpath)
    print(f"before prune: GT n_ann={true.n_annots}, pred n_ann={pred.n_annots}")

    print(f"pruning classes from GT: {sorted(distractor_names)}")
    true_pruned = _prune_classes(true, distractor_names)
    print(f"pruning classes from pred: {sorted(distractor_names)}")
    pred_pruned = _prune_classes(pred, distractor_names)

    config = {
        "true_dataset": true_pruned,
        "pred_dataset": pred_pruned,
        "iou_thresh": 0.5,
        "area_range": "all",
    }
    coco_eval = CocoEvaluator(config)
    coco_eval._init()
    results = coco_eval.evaluate()

    # results may be a CocoSingleResult or a dict-of-results keyed by
    # area_range/iou; normalise.
    if hasattr(results, "items"):
        items = list(results.items())
    else:
        items = [(getattr(results, "reskey", "result"), results)]

    print()
    print(f"=== recomputed (distractors pruned: {sorted(distractor_names)}) ===")
    summary = {}
    for reskey, single in items:
        nocls = getattr(single, "nocls_measures", None)
        ovr = getattr(single, "ovr_measures", None)
        if nocls is None and hasattr(single, "__getitem__"):
            nocls = single["nocls_measures"]
            ovr = single["ovr_measures"]
        nocls_ap = float(nocls["ap"])
        nocls_auc = float(nocls["auc"])
        print(f"[{reskey}]")
        print(f"  class-agnostic AP (nocls.ap): {nocls_ap:.4f}")
        print(f"  class-agnostic AUC          : {nocls_auc:.4f}")
        per_class = {}
        if ovr is not None:
            print(f"  per-class AP:")
            for cname, m in ovr.items():
                ap = float(m["ap"])
                pos = float(m["realpos_total"])
                marker = "  (no positives — excluded)" if pos == 0 else ""
                print(f"    {cname}: ap={ap:.4f} realpos={int(pos)}{marker}")
                per_class[cname] = {"ap": ap, "realpos": pos}
        summary[str(reskey)] = {
            "nocls_ap": nocls_ap,
            "nocls_auc": nocls_auc,
            "per_class": per_class,
            "distractor_classes": sorted(distractor_names),
        }

    if write_json:
        # Match the kit's sidecar filename convention so eligibility's
        # selection function (kwcoco_detector_kit/orchestration/
        # eligibility.py::_find_eval_ap) picks this up automatically.
        suffix = "_".join(sorted(distractor_names))
        dst = eval_dir / "eval" / f"detect_metrics.{suffix}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nwrote {dst}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", type=Path, required=True,
                   help="eval output dir containing true_bbox_only.kwcoco.zip "
                        "+ pred_boxes.kwcoco.zip")
    p.add_argument("--distractor-class", action="append", default=None,
                   help="category name to treat as a distractor and prune "
                        "from both GT+pred before scoring; repeatable "
                        "(default: northern_fur_seal). These are classes "
                        "the model learns but the mission treats as "
                        "non-targets.")
    p.add_argument("--write-json", action="store_true",
                   help="write the recomputed metrics to "
                        "<eval-dir>/eval/detect_metrics.<distractors>.json "
                        "(filename convention eligibility's selection logic "
                        "looks for, so the corrected AP picks up automatically)")
    args = p.parse_args()
    distractors = (tuple(args.distractor_class)
                   if args.distractor_class else DEFAULT_DISTRACTORS)
    recompute(args.eval_dir, distractors, write_json=args.write_json)


if __name__ == "__main__":
    main()
