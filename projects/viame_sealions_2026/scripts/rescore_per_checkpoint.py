#!/usr/bin/env python3
"""
Re-score every saved checkpoint in a run dir against vali (and optionally
test), with distractor pruning, and report the per-checkpoint AP curve.

Why: DEIMv2's in-train CocoEvaluator uses the with-distractor metric to
pick best_stg{1,2}.pth. The "best" checkpoint under our corrected metric
(distractors pruned) may be a different epoch's snapshot. This tool runs
inference for each *.pth in the run dir and writes per-checkpoint
metrics so we can compare.

For our current 3-checkpoint-per-run reality (best_stg1, best_stg2,
last), this gives 3 data points. Once the kit's image is patched to
save intermediate snapshots, the same tool will work for N data points
with no code change.

Usage (from kit root):
    python3 projects/viame_sealions_2026/scripts/rescore_per_checkpoint.py \\
        --run-dir /data/users/jon.crall/kcd_sealion/runs/<run_name> \\
        --device cuda          # or cpu (default)

Slurm wrapper for arisia:
    bash projects/viame_sealions_2026/scripts/submit_rescore_per_checkpoint.sh \\
        <run_name>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _scheme_from_run_name(run_name: str, schemes_yaml: Path):
    data = yaml.safe_load(schemes_yaml.read_text()) or {}
    schemes = list((data.get("schemes") or {}).keys())
    schemes.sort(key=len, reverse=True)
    for s in schemes:
        if run_name.startswith(s + "_"):
            return s, ((data["schemes"][s]).get("distractor_classes") or [])
    return None, []


def _read_nocls_ap(metrics_fpath: Path):
    """Tolerant of both the kwcoco-eval schema and the compact recompute schema."""
    if not metrics_fpath.exists():
        return None
    try:
        d = json.loads(metrics_fpath.read_text())
    except Exception:
        return None

    def find(node):
        if isinstance(node, dict):
            if "nocls_measures" in node and isinstance(node["nocls_measures"], dict):
                v = node["nocls_measures"].get("ap")
                if v is not None:
                    return float(v)
            if "nocls_ap" in node and not isinstance(node["nocls_ap"], dict):
                v = node["nocls_ap"]
                if v is not None:
                    return float(v)
            for v in node.values():
                r = find(v)
                if r is not None:
                    return r
        elif isinstance(node, list):
            for v in node:
                r = find(v)
                if r is not None:
                    return r
        return None

    return find(d)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="<kcd_root>/runs/<run_name>/ (the dir that holds runs/<candidate>/, scheme_applied/, eval/...)")
    p.add_argument("--eval-target", choices=("vali", "test", "both"), default="vali",
                   help="which kwcoco bundle to score against (default: vali)")
    p.add_argument("--device", default="cpu",
                   help="cpu or cuda; cuda needs a GPU + torch.cuda available")
    p.add_argument("--force", action="store_true",
                   help="rescore even if per-checkpoint metrics already exist")
    p.add_argument(
        "--schemes-yaml",
        default=str(Path(__file__).resolve().parent.parent / "docs" / "class_schemes.yaml"),
    )
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    run_name = run_dir.name
    scheme, distractors = _scheme_from_run_name(run_name, Path(args.schemes_yaml))
    if scheme is None:
        raise SystemExit(f"could not match run name {run_name!r} to a scheme")
    print(f"# scheme: {scheme}; distractors: {distractors or '(none)'}")

    # Each run has one or more candidate subdirs under runs/.
    candidates_root = run_dir / "runs"
    if not candidates_root.exists():
        raise SystemExit(f"missing {candidates_root}")
    candidate_dirs = sorted(d for d in candidates_root.iterdir() if d.is_dir())
    if not candidate_dirs:
        raise SystemExit(f"no candidates under {candidates_root}")

    # Find the scheme_applied vali/test bundles (kit convention).
    scheme_applied = run_dir / "scheme_applied"
    targets = []
    if args.eval_target in ("vali", "both"):
        targets.append(("vali", scheme_applied / "vali.kwcoco.zip"))
    if args.eval_target in ("test", "both"):
        targets.append(("test", scheme_applied / "test.kwcoco.zip"))
    for label, fp in targets:
        if not fp.exists():
            raise SystemExit(f"missing {label} kwcoco at {fp}")

    # Read category_names from the candidate's policy.json (authoritative).
    from kwcoco_detector_kit.trainers.deimv2 import DEIMv2Trainer
    from kwcoco_detector_kit.eval.per_checkpoint_eval import score_one_checkpoint
    trainer = DEIMv2Trainer()

    summary_rows = []
    for cand_dir in candidate_dirs:
        candidate_id = cand_dir.name
        policy_fpath = cand_dir / "policy.json"
        category_names = None
        if policy_fpath.exists():
            try:
                policy = json.loads(policy_fpath.read_text())
                category_names = policy.get("category_names")
            except Exception:
                pass
        if not category_names:
            # Fall back to scheme target_order
            data = yaml.safe_load(Path(args.schemes_yaml).read_text()) or {}
            category_names = ((data.get("schemes") or {}).get(scheme) or {}).get("target_order") or []
        if not category_names:
            print(f"  warn: no category_names for {candidate_id}; skipping")
            continue
        print(f"# candidate: {candidate_id}; category_names={category_names}")

        ckpts = sorted(cand_dir.glob("*.pth"))
        # Skip ONNX-export sidecars (e.g. best_stg2.onnx.data)
        ckpts = [c for c in ckpts if c.suffix == ".pth"]
        if not ckpts:
            print(f"  no .pth checkpoints in {cand_dir}")
            continue
        print(f"  found {len(ckpts)} checkpoints: {[c.name for c in ckpts]}")

        for label, target_fpath in targets:
            eval_root = run_dir / "per_checkpoint_eval" / label / candidate_id
            print(f"# scoring against {label} -> {eval_root}")
            for ckpt in ckpts:
                try:
                    score_one_checkpoint(
                        trainer=trainer,
                        workdir=cand_dir,
                        ckpt_fpath=ckpt,
                        eval_target_kwcoco=str(target_fpath),
                        eval_root=eval_root,
                        candidate_id=candidate_id,
                        category_names=category_names,
                        distractor_classes=distractors,
                        device=args.device,
                        force=args.force,
                    )
                except Exception as ex:
                    print(f"  {ckpt.name}: FAILED -- {type(ex).__name__}: {ex}")
                    summary_rows.append((candidate_id, label, ckpt.stem, "FAIL", "FAIL"))
                    continue
                # Read back AP values.
                ckpt_eval_root = eval_root / ckpt.stem
                base_ap = _read_nocls_ap(ckpt_eval_root / "eval" / "detect_metrics.json")
                suffix = "_".join(sorted(distractors)) if distractors else ""
                sidecar_ap = None
                if suffix:
                    sidecar_ap = _read_nocls_ap(
                        ckpt_eval_root / "eval" / f"detect_metrics.{suffix}.json"
                    )
                summary_rows.append((
                    candidate_id, label, ckpt.stem,
                    f"{base_ap:.4f}" if base_ap is not None else "?",
                    f"{sidecar_ap:.4f}" if sidecar_ap is not None else ("-" if not suffix else "?"),
                ))

    print()
    cols = ["candidate", "split", "checkpoint", "nocls AP (full)", "nocls AP (no distractors)"]
    widths = [max(len(c), *(len(str(r[i])) for r in summary_rows or [["", "", "", "", ""]])) for i, c in enumerate(cols)]
    sep = "  "
    print(sep.join(c.ljust(w) for c, w in zip(cols, widths)))
    print(sep.join("-" * w for w in widths))
    for r in summary_rows:
        print(sep.join(str(c).ljust(w) for c, w in zip(r, widths)))


if __name__ == "__main__":
    main()
