#!/usr/bin/env python3
"""
Re-score one trained run's checkpoint with WHOLE-IMAGE vs TILED eval and
print the class-agnostic detection-AP delta.

This quantifies the train/eval resolution-mismatch fix (eval/tiled_predictor.py)
on an EXISTING checkpoint — no retraining, no GPU-hours beyond two eval passes.
It scores the same ``best_*.pth`` twice into separate output dirs so both
metrics survive for comparison:

    <out_root>/eval/wholeimage/eval/detect_metrics.json
    <out_root>/eval/tiled/eval/detect_metrics.json

``run_kwcoco_eval`` takes ``workdir`` (where the checkpoint lives) and
``kcd_root`` (where eval output goes) independently, so pointing both passes
at the real workdir while redirecting output keeps the run's own eval/ dir
untouched.

Run it INSIDE the kit docker image (it needs torch + the trained config).
Mount the host kit so the new tiled-eval code is live without a rebuild:
KCD_DEV_MOUNT_KIT=1 (see scripts/rescore_tiled_compare.sh which wraps this).

Example (inside container)::

    python3 projects/viame_sealions_2026/scripts/rescore_tiled_compare.py \\
        --kcd_root /data/users/jon.crall/kcd_sealion/runs/pup_vs_nonpup_deimv2_dinov3_s_2gpu_arisia_gen005_v2splits \\
        --category_names pup,nonpup_sealion \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_nocls_ap(metrics_fpath: Path):
    """Pull the class-agnostic AP out of a detect_metrics.*.json."""
    try:
        d = json.loads(Path(metrics_fpath).read_text())
    except Exception:
        return None
    for _k, v in (d.items() if isinstance(d, dict) else []):
        if not isinstance(v, dict):
            continue
        if "nocls_ap" in v:
            return float(v["nocls_ap"])
        nm = v.get("nocls_measures")
        if isinstance(nm, dict) and nm.get("ap") is not None:
            return float(nm["ap"])
    return None


def _score_one(trainer, *, workdir, test_kwcoco, out_root, candidate_id,
               category_names, distractors, tiled, device, window, overlap,
               force=True):
    from kwcoco_detector_kit.eval.kwcoco_eval import run_kwcoco_eval
    metrics = run_kwcoco_eval(
        trainer=trainer,
        workdir=workdir,
        test_kwcoco=str(test_kwcoco),
        kcd_root=out_root,
        candidate_id=candidate_id,
        category_names=category_names,
        force=force,
        distractor_classes=distractors or None,
        tiled_eval=tiled,
        tiled_window=window,
        tiled_overlap=overlap,
        device=device,
    )
    return Path(metrics)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kcd_root", type=Path, required=True,
                   help="the run dir (holds runs/<cid>/best_*.pth + scheme_applied/)")
    p.add_argument("--variant", default="deimv2_dinov3_s")
    p.add_argument("--input_hw", default="640,640",
                   help="HxW used at train time; sets candidate_id. e.g. 640,640")
    p.add_argument("--train_policy", default="fixed")
    p.add_argument("--category_names", required=True,
                   help="comma-separated, in trained class-index order "
                        "(e.g. pup,nonpup_sealion or sealion)")
    p.add_argument("--distractor_classes", default="",
                   help="comma-separated NFS-style distractors to exclude from AP "
                        "(empty for schemes that DROP NFS, e.g. pup_vs_nonpup)")
    p.add_argument("--test_kwcoco", type=Path, default=None,
                   help="default: <kcd_root>/scheme_applied/test.kwcoco.zip")
    p.add_argument("--workdir", type=Path, default=None,
                   help="default: <kcd_root>/runs/<candidate_id>")
    p.add_argument("--out_root", type=Path, default=None,
                   help="default: <kcd_root>/tiled_compare")
    p.add_argument("--device", default="cuda", help="cpu or cuda")
    p.add_argument("--window", type=int, default=None,
                   help="tiled window size (square); default = model eval_spatial_size")
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--force_wholeimage", action="store_true",
                   help="recompute the whole-image baseline even if its "
                        "detect_metrics.json already exists (default: reuse it)")
    args = p.parse_args()

    H, W = (int(x) for x in str(args.input_hw).replace("x", ",").split(","))
    candidate_id = f"{args.variant}_{H}x{W}_{args.train_policy}"
    workdir = args.workdir or (args.kcd_root / "runs" / candidate_id)
    test_kwcoco = args.test_kwcoco or (args.kcd_root / "scheme_applied" / "test.kwcoco.zip")
    out_root = args.out_root or (args.kcd_root / "tiled_compare")
    category_names = [s.strip() for s in args.category_names.split(",") if s.strip()]
    distractors = [s.strip() for s in args.distractor_classes.split(",") if s.strip()]

    for label, path in [("workdir", workdir), ("test_kwcoco", test_kwcoco)]:
        if not Path(path).exists():
            p.error(f"{label} not found: {path}")

    from kwcoco_detector_kit.trainers._registry import get_trainer
    trainer = get_trainer("deimv2")

    print(f"candidate_id : {candidate_id}")
    print(f"workdir      : {workdir}")
    print(f"test_kwcoco  : {test_kwcoco}")
    print(f"out_root     : {out_root}")
    print(f"categories   : {category_names}  distractors: {distractors or '-'}")
    print(f"device       : {args.device}  window: {args.window or 'eval_spatial_size'}  overlap: {args.overlap}")
    print()

    print("=== pass 1/2: WHOLE-IMAGE (baseline) ===")
    whole = _score_one(trainer, workdir=workdir, test_kwcoco=test_kwcoco,
                       out_root=out_root, candidate_id="wholeimage",
                       category_names=category_names, distractors=distractors,
                       tiled=False, device=args.device, window=None, overlap=args.overlap,
                       force=bool(args.force_wholeimage))
    print("\n=== pass 2/2: TILED (windowed) ===")
    tiled = _score_one(trainer, workdir=workdir, test_kwcoco=test_kwcoco,
                      out_root=out_root, candidate_id="tiled",
                      category_names=category_names, distractors=distractors,
                      tiled=True, device=args.device, window=args.window, overlap=args.overlap)

    def _pair(metrics_fpath):
        m = Path(metrics_fpath)
        ap = _read_nocls_ap(m)
        sidecar = None
        if distractors:
            suffix = "_".join(sorted(distractors))
            sc = m.with_name(f"detect_metrics.{suffix}.json")
            if sc.exists():
                sidecar = _read_nocls_ap(sc)
        return ap, sidecar

    w_ap, w_nfs = _pair(whole)
    t_ap, t_nfs = _pair(tiled)

    def _fmt(x):
        return f"{x:.4f}" if isinstance(x, float) else "  n/a "

    print("\n" + "=" * 56)
    print("class-agnostic detection AP   whole-image -> tiled")
    print("-" * 56)
    print(f"  AP (all anns)               {_fmt(w_ap)}  ->  {_fmt(t_ap)}")
    if distractors:
        print(f"  AP (NFS-excluded, selection){_fmt(w_nfs)}  ->  {_fmt(t_nfs)}")
    if isinstance(w_ap, float) and isinstance(t_ap, float):
        delta = t_ap - w_ap
        mult = (t_ap / w_ap) if w_ap > 0 else float("inf")
        print("-" * 56)
        print(f"  delta {delta:+.4f}   ({mult:.2f}x)")
    print("=" * 56)
    print(f"\nmetrics written under: {out_root}/eval/{{wholeimage,tiled}}/eval/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
