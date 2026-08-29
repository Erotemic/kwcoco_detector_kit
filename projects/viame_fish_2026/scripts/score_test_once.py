"""Score one checkpoint per run on the test split, under one protocol.

Purpose is the record, not a decision: the tiling hypothesis is already
falsified on vali, and this puts numbers on the held-out split next to it so a
future brainstorm can see how far the confident predictions actually missed.

Two things here are about measurement quality rather than holdout hygiene, and
both matter:

* **The checkpoint is chosen on vali, not here.** A test number on an arbitrary
  checkpoint is uninterpretable. gen006 was never ranked under deployment
  geometry -- its ``best_stg1.pth`` is whatever DEIMv2's in-loop TILE-level
  validation preferred, and gen007 showed those disagree (in-loop picked epoch
  27; deployment geometry picked epoch 6). So each run's checkpoint comes from
  the vali ranking summary, and a run with staged epochs but no vali entry is
  an error rather than a guess.

* **One protocol across all four runs.** The existing gen001/gen003 test numbers
  are WHOLE-IMAGE, taken before true-tiled inference existed. Comparing a
  true-tiled gen006 number against them would repeat the protocol mismatch this
  project has already been burned by, so everything is rescored true-tiled
  here.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from datetime import datetime, timezone

import kwcoco_detector_kit.trainers  # noqa: F401  -- registers the plugins
from kwcoco_detector_kit.eval.kwcoco_eval import run_kwcoco_eval
from kwcoco_detector_kit.selection.scoring import measures_from_detect_metrics
from kwcoco_detector_kit.trainers._registry import get_trainer


def winners_from_vali(summary_fpath) -> dict:
    """``{run: (label, vali_ap, stride)}`` from score_epochs.py's summary.

    The stride is carried so the output can say which ranking chose the
    checkpoint: a stride-8 ranking is a fine way to CHOOSE and is never itself
    a reportable AP.
    """
    p = pathlib.Path(summary_fpath)
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    stride = int(data.get("stride", 1))
    best: dict = {}
    for row in data.get("rows", []):
        run, ap = row["run"], float(row["ap"])
        if run not in best or ap > best[run][1]:
            best[run] = (row["label"], ap, stride)
    return best


def checkpoint_for(workdir: pathlib.Path, label: str):
    """Resolve a vali-chosen label to a weights file.

    ``autoselect`` defers to the trainer, correct for a run with no per-epoch
    staging. Anything else must exist: silently falling back would score a
    different checkpoint than the summary records.
    """
    if label == "autoselect":
        return None
    ckpt = workdir / "staging" / f"{label}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"vali chose {label} but {ckpt} is missing; refusing to score a "
            "different checkpoint than the one selected")
    return ckpt


def main() -> int:
    test_kwcoco = os.environ["KCD_TEST_KWCOCO"]
    runs_dpath = pathlib.Path(os.environ["KCD_RUNS_DPATH"])
    out_root = pathlib.Path(os.environ["KCD_TEST_SCORE_OUT"])
    candidate_id = os.environ.get(
        "KCD_CANDIDATE_ID", "deimv2_dinov3_x_1024x1024_fixed")
    runs = os.environ["KCD_TEST_SCORE_RUNS"].split()
    window = int(os.environ["KCD_TILED_EVAL_WINDOW"])
    overlap = float(os.environ.get("KCD_TILED_EVAL_OVERLAP", "0.25"))
    if window < 64:
        raise SystemExit(f"implausible eval window {window}px -- check paths.sh")

    winners = winners_from_vali(os.environ.get("KCD_VALI_SUMMARY", ""))
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for run in runs:
        workdir = runs_dpath / run / "runs" / candidate_id
        if not workdir.is_dir():
            print(f"[{run}] SKIP -- no workdir at {workdir}", file=sys.stderr)
            continue
        staged = sorted((workdir / "staging").glob("epoch_*.pth"))
        if run in winners:
            label, vali_ap, vali_stride = winners[run]
            chosen_by = f"vali stride-{vali_stride} ranking (AP@0.5 {vali_ap:.4f})"
        elif staged:
            raise SystemExit(
                f"[{run}] has {len(staged)} staged epochs but no vali entry. "
                "Rank it on vali first -- picking one here would be choosing "
                "on test, which is the one thing that would make these numbers "
                "meaningless.")
        else:
            label, chosen_by = "autoselect", "trainer autoselect (no staging)"

        ckpt = checkpoint_for(workdir, label)
        category_names = json.loads(
            (workdir / "policy.json").read_text())["category_names"]

        evalroot = out_root / f"w{window}_o{overlap}_bf16" / run / label
        evalroot.mkdir(parents=True, exist_ok=True)
        print(f"[{run}] {label} -- chosen by {chosen_by}", flush=True)
        metrics = run_kwcoco_eval(
            trainer=get_trainer("deimv2"),
            workdir=workdir,
            checkpoint=ckpt,
            test_kwcoco=test_kwcoco,
            kcd_root=evalroot,
            candidate_id=run,
            category_names=category_names,
            # Never reuse a stored result: gen001/gen003 already carry
            # detect_metrics.json from WHOLE-IMAGE test eval, and silently
            # reporting one of those as a true-tiled number is the exact
            # protocol mismatch this run exists to remove.
            force=True,
            tiled_eval=True,
            tiled_window=window,
            tiled_overlap=overlap,
            tiled_keep_full=True,
            tiled_batch=int(os.environ.get("KCD_TILED_EVAL_BATCH", "64")),
            device=os.environ.get("KCD_EVAL_DEVICE", "cuda"),
            read_workers=int(os.environ.get("KCD_EVAL_READ_WORKERS", "4")),
        )
        ap = measures_from_detect_metrics(metrics).get("AP@0.5")
        if ap is None:
            print(f"  WARN: no AP@0.5 block in {metrics}", file=sys.stderr)
            continue
        rows.append({"run": run, "label": label, "test_ap": float(ap),
                     "chosen_by": chosen_by, "metrics": str(metrics)})
        print(f"  TEST AP@0.5 {float(ap):.4f}", flush=True)

    if not rows:
        print("no results -- nothing to summarise", file=sys.stderr)
        return 1

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_fpath = out_root / f"test_summary_w{window}_o{overlap}.json"
    out_fpath.write_text(json.dumps(
        {"scored_at": stamp, "split": test_kwcoco, "window": window,
         "overlap": overlap, "amp": os.environ.get("KCD_AMP_DTYPE", "bfloat16"),
         "protocol": "true-tiled, keep_full, cross-window NMS 0.5, full test split",
         "rows": rows}, indent=2) + "\n")

    print()
    print("=" * 78)
    print(f" TEST -- true-tiled {window}px, overlap {overlap}, bf16, full split")
    print("=" * 78)
    print(f"  {'run':46} {'ckpt':12} {'test':>8}  chosen by")
    for r in sorted(rows, key=lambda r: -r["test_ap"]):
        print(f"  {r['run'][:46]:46} {r['label']:12} {r['test_ap']:8.4f}  {r['chosen_by']}")
    print()
    print(f"  wrote {out_fpath}")
    print("  Journal these next to the vali numbers -- the point is the gap")
    print("  between what the tiling hypothesis predicted and what it delivered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
