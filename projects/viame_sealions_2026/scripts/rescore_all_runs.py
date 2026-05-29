#!/usr/bin/env python3
"""
Rescore every existing run with the kit's *production* distractor-pruning
code path, using the pred_boxes / true_bbox_only kwcocos that earlier runs
already wrote to disk.

Two purposes in one script:
  1. Backfill — runs that completed before the distractor-pruning eval
     step landed don't have detect_metrics.<distractors>.json sidecars.
     This walks every run dir under $KCD_TRAINING_ROOT/runs/, derives
     the distractor list from the scheme's class_schemes.yaml entry,
     and writes the sidecar by calling the same internal helper
     (kwcoco_detector_kit.eval.kwcoco_eval._rerun_eval_dropping_distractors)
     that the production pipeline now invokes on every fresh run.
  2. Test — verifies the kit's distractor logic produces sensible numbers
     against real data. Prints a before/after AP comparison per run.

Does not re-run inference; reuses the pred_boxes.kwcoco.zip the original
eval wrote. So this is pure CPU work and finishes in seconds per run.

Usage (from the kit root):
    python3 projects/viame_sealions_2026/scripts/rescore_all_runs.py

    # restrict to a subset:
    python3 projects/viame_sealions_2026/scripts/rescore_all_runs.py \\
        --runs-glob 'lifestage_*'

    # dry run (no sidecar written):
    python3 projects/viame_sealions_2026/scripts/rescore_all_runs.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


def _scheme_distractors(schemes_yaml: Path, scheme_name: str):
    data = yaml.safe_load(schemes_yaml.read_text()) or {}
    scheme = (data.get("schemes") or {}).get(scheme_name) or {}
    return list(scheme.get("distractor_classes") or [])


def _detect_scheme_from_run_name(run_name: str, schemes_yaml: Path):
    """Match the run dir name against the known scheme prefixes."""
    data = yaml.safe_load(schemes_yaml.read_text()) or {}
    schemes = list((data.get("schemes") or {}).keys())
    schemes.sort(key=len, reverse=True)
    for s in schemes:
        if run_name.startswith(s + "_"):
            return s
    return None


def _read_ap(metrics_fpath: Path):
    """Pull the class-agnostic AP out of a detect_metrics.*.json."""
    try:
        d = json.loads(metrics_fpath.read_text())
    except Exception:
        return None
    # The kit's main detect_metrics.json has area_range=all,iou_thresh=0.5
    # at the top; recompute_sealion_ap.py-style sidecars store the value
    # under the same key but with a flat nocls_ap field.
    for k, v in (d.items() if isinstance(d, dict) else []):
        if not isinstance(v, dict):
            continue
        if "nocls_ap" in v:
            return float(v["nocls_ap"])
        nm = v.get("nocls_measures")
        if isinstance(nm, dict) and nm.get("ap") is not None:
            return float(nm["ap"])
    return None


def rescore_one(eval_dir: Path, distractor_names, dry_run=False):
    from kwcoco_detector_kit.eval.kwcoco_eval import (
        _distractor_sidecar_fpath,
        _rerun_eval_dropping_distractors,
    )

    pred_bbox = eval_dir / "pred_boxes_bbox_only.kwcoco.zip"
    pred_full = eval_dir / "pred_boxes.kwcoco.zip"
    pred_fpath = pred_bbox if pred_bbox.exists() else pred_full
    true_fpath = eval_dir / "true_bbox_only.kwcoco.zip"
    metrics_fpath = eval_dir / "eval" / "detect_metrics.json"
    if not (pred_fpath.exists() and true_fpath.exists() and metrics_fpath.exists()):
        return None, "missing pred/true/metrics"

    sidecar_fpath = _distractor_sidecar_fpath(metrics_fpath, distractor_names)
    if not distractor_names:
        return ("noop", _read_ap(metrics_fpath), None, sidecar_fpath)

    if dry_run:
        return ("would-write", _read_ap(metrics_fpath), None, sidecar_fpath)

    _rerun_eval_dropping_distractors(
        true_fpath=true_fpath,
        pred_fpath=pred_fpath,
        distractor_names=distractor_names,
        out_fpath=sidecar_fpath,
        score_thresh=0.001,
        test_kwcoco=str(true_fpath),
        candidate_id=eval_dir.parent.name,
        category_names=[],
    )
    return (
        "wrote",
        _read_ap(metrics_fpath),
        _read_ap(sidecar_fpath),
        sidecar_fpath,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-root",
        default=os.environ.get(
            "KCD_TRAINING_ROOT", "/data/users/jon.crall/kcd_sealion"
        ) + "/runs",
        help="root dir containing per-run subdirs (default: $KCD_TRAINING_ROOT/runs)",
    )
    p.add_argument(
        "--schemes-yaml",
        default=str(Path(__file__).resolve().parent.parent / "docs" / "class_schemes.yaml"),
        help="path to class_schemes.yaml",
    )
    p.add_argument(
        "--runs-glob",
        default="*",
        help="glob pattern for run names to include (default: '*')",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be written; don't actually score")
    args = p.parse_args()

    runs_root = Path(args.runs_root)
    schemes_yaml = Path(args.schemes_yaml)

    if not runs_root.exists():
        raise SystemExit(f"runs root does not exist: {runs_root}")
    if not schemes_yaml.exists():
        raise SystemExit(f"schemes yaml does not exist: {schemes_yaml}")

    runs = sorted(d for d in runs_root.glob(args.runs_glob) if d.is_dir())
    print(f"# scanning {len(runs)} run dir(s) under {runs_root}")
    if args.dry_run:
        print("# DRY RUN -- no sidecars will be written")

    rows = []
    for run_dir in runs:
        run_name = run_dir.name
        scheme = _detect_scheme_from_run_name(run_name, schemes_yaml)
        if scheme is None:
            rows.append((run_name, "?", "-", "?", "skip: no scheme matched"))
            continue
        distractors = _scheme_distractors(schemes_yaml, scheme)
        eval_dirs = sorted(run_dir.glob("eval/*/"))
        if not eval_dirs:
            rows.append((run_name, scheme, ",".join(distractors) or "-",
                         "?", "skip: no eval/<candidate>/ dir"))
            continue
        for eval_dir in eval_dirs:
            try:
                result = rescore_one(eval_dir, distractors, dry_run=args.dry_run)
            except Exception as ex:
                rows.append((run_name, scheme, ",".join(distractors) or "-",
                             "FAIL", f"{type(ex).__name__}: {ex}"))
                continue
            if result is None or result[0] is None:
                rows.append((run_name, scheme, ",".join(distractors) or "-",
                             "skip", result[1] if isinstance(result, tuple) else "n/a"))
                continue
            status, ap_before, ap_after, sidecar = result
            if status == "noop":
                ap_after_s = "-"
                detail = "(no distractors in scheme)"
            elif status == "would-write":
                ap_after_s = "would-compute"
                detail = f"-> {sidecar.name}"
            else:
                ap_after_s = f"{ap_after:.4f}" if ap_after is not None else "?"
                detail = f"-> {sidecar.name}"
            ap_before_s = f"{ap_before:.4f}" if ap_before is not None else "?"
            rows.append((run_name, scheme, ",".join(distractors) or "-",
                         f"{ap_before_s} -> {ap_after_s}", detail))

    # Tabular report
    cols = ["run_name", "scheme", "distractors", "AP (before -> after)", "detail"]
    widths = [max(len(c), *(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    sep = "  "
    print()
    print(sep.join(c.ljust(w) for c, w in zip(cols, widths)))
    print(sep.join("-" * w for w in widths))
    for r in rows:
        print(sep.join(str(c).ljust(w) for c, w in zip(r, widths)))


if __name__ == "__main__":
    main()
