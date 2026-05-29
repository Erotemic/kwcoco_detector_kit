#!/usr/bin/env python3
"""
Recompute sea-lion detection AP from an existing eval output dir, with
NFS (northern_fur_seal) excluded from both ground-truth and predictions.

Use case: the kit's eval step writes pred_boxes.kwcoco.zip +
true_bbox_only.kwcoco.zip and a detect_metrics.json built by kwcoco's
CocoEvaluator. The class-agnostic AP in that json counts NFS as a
positive — which contradicts the operational rule "NFS is a negative
when scoring sea-lion detection ability." This script reads the same
two kwcoco bundles, filters NFS, reruns CocoEvaluator, and prints the
corrected AP.

It does NOT modify the on-disk detect_metrics.json. Use --write-json to
write a sibling `detect_metrics.sealion.json` alongside the original.

Usage:
    python3 projects/viame_sealions_2026/scripts/recompute_sealion_ap.py \\
        --eval-dir /data/users/jon.crall/kcd_sealion/runs/lifestage_6cls_.../eval/<candidate>/

    # Pass an extra --exclude-cat for any non-target class you also
    # want pruned (default: northern_fur_seal).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import kwcoco


DEFAULT_EXCLUDE = ("northern_fur_seal",)


def _prune_classes(dset: kwcoco.CocoDataset, exclude_names) -> kwcoco.CocoDataset:
    """Return a copy of `dset` with annotations whose category name is in
    `exclude_names` removed. Categories themselves stay so the eval doesn't
    complain about class-id remapping; only annotations are dropped.
    """
    exclude_names = set(exclude_names)
    out = dset.copy()
    cat_ids_to_drop = {c["id"] for c in out.dataset["categories"]
                       if c["name"] in exclude_names}
    if not cat_ids_to_drop:
        return out
    keep = [a for a in out.dataset["annotations"]
            if a.get("category_id") not in cat_ids_to_drop]
    n_before = len(out.dataset["annotations"])
    out.dataset["annotations"] = keep
    out._build_index()
    print(f"  dropped {n_before - len(keep)} / {n_before} annotations "
          f"(classes: {sorted(exclude_names)})")
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


def recompute(eval_dir: Path, exclude_names, write_json: bool = False):
    from kwcoco.coco_evaluator import CocoEvaluator

    true_fpath, pred_fpath = _find_bundles(eval_dir)
    print(f"GT   : {true_fpath}")
    print(f"pred : {pred_fpath}")

    true = kwcoco.CocoDataset.coerce(true_fpath)
    pred = kwcoco.CocoDataset.coerce(pred_fpath)
    print(f"before prune: GT n_ann={true.n_annots}, pred n_ann={pred.n_annots}")

    print(f"pruning classes from GT: {sorted(exclude_names)}")
    true_pruned = _prune_classes(true, exclude_names)
    print(f"pruning classes from pred: {sorted(exclude_names)}")
    pred_pruned = _prune_classes(pred, exclude_names)

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
    print("=== recomputed (NFS excluded) ===")
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
            "excluded_classes": sorted(exclude_names),
        }

    if write_json:
        dst = eval_dir / "eval" / "detect_metrics.sealion.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nwrote {dst}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", type=Path, required=True,
                   help="eval output dir containing true_bbox_only.kwcoco.zip "
                        "+ pred_boxes.kwcoco.zip")
    p.add_argument("--exclude-cat", action="append", default=None,
                   help="category name to drop from both GT+pred before "
                        "scoring; repeatable (default: northern_fur_seal)")
    p.add_argument("--write-json", action="store_true",
                   help="write the recomputed metrics to "
                        "<eval-dir>/eval/detect_metrics.sealion.json")
    args = p.parse_args()
    exclude = tuple(args.exclude_cat) if args.exclude_cat else DEFAULT_EXCLUDE
    recompute(args.eval_dir, exclude, write_json=args.write_json)


if __name__ == "__main__":
    main()
