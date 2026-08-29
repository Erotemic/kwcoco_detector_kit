"""Score checkpoints on the full vali split under ONE frozen protocol.

Why this file exists
--------------------
Two questions have to be answered with the same ruler:

  * what is ``B``, the baseline gen006 has to beat (gen001, gen003); and
  * which gen006 epoch is actually best under DEPLOYMENT geometry.

gen006's in-training selection ranked epochs by DEIMv2's own tile-level AP at
the model's input size. That is not the geometry the detector runs at, so its
"best epoch 4" is a claim about tiles, not about the deployed system. Answering
both questions in one pass, under one protocol, is what makes the comparison
mean anything -- and it needs no retraining.

The frozen protocol
-------------------
    true tiled, source window = KCD_TILED_EVAL_WINDOW px, overlap 0.25,
    keep_full, cross-window NMS 0.5, per-window NMS off
    full 46-sequence VALI split (never test)
    bf16
    each checkpoint under ITS OWN preprocessing contract

That last point is automatic rather than a flag: DEIMv2Predictor recovers
mean/std from the generated train.yml of the run being scored. gen001 and
gen003 predate DINO normalization, carry no Normalize op, and are scored
unnormalized -- which is correct for them. gen006 carries one and is scored
with it. See _launch_baseline_vali.sh for why that confound is the right call.

Two stages, because full vali x 16 checkpoints is not affordable
----------------------------------------------------------------
The vali split is 35,111 images. Scoring one checkpoint over all of them under
true-tiled 1229px inference is hours of GPU; sixteen of them is days. So the
selection is run in two stages, and ``KCD_EVAL_STRIDE`` is what separates them:

  stage 1  KCD_EVAL_STRIDE=8   all 14 gen006 epochs + both baselines
           ranks epochs against each other cheaply
  stage 2  KCD_EVAL_STRIDE=1   the 2-3 finalists + both baselines
           the number that goes in the journal

This is sound as long as the roles are not confused. Stage 1 ranks checkpoints
WITHIN one run, where every candidate sees the identical subsample, so the
comparison is paired and the subsample's own bias cancels. Stage 2 is where B
and the winner are actually established. A stage-1 AP is never reported as B
and never compared against a stage-2 AP -- the summary json records the stride
so the two cannot be silently mixed.

The subsample takes every Nth image in kwcoco order, which is frame order
within sequence, so all 46 sequences stay represented rather than sampling a
subset of sequences. Fewer FRAMES weakens the estimate far less than fewer
SEQUENCES would: those 35,111 frames carry only ~2,140 tracks, so neighbouring
frames are close to redundant.

What is scored per run
----------------------
A run with a ``staging/`` directory is scored at EVERY staged epoch; a run
without one is scored at its autoselected checkpoint. So gen001/gen003
contribute one row each and gen006 contributes fourteen, from a single
invocation and a single protocol.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import kwcoco_detector_kit.trainers  # noqa: F401  -- registers the plugins
from kwcoco_detector_kit.eval.kwcoco_eval import run_kwcoco_eval
from kwcoco_detector_kit.selection.scoring import measures_from_detect_metrics
from kwcoco_detector_kit.trainers._registry import get_trainer


def preprocessing_contract(workdir: pathlib.Path) -> str:
    """Report, rather than assume, how this checkpoint's inputs are scaled."""
    cfg_fpath = workdir / "generated_configs" / "train.yml"
    if not cfg_fpath.exists():
        return "unknown (no generated train.yml)"
    import yaml
    cfg = yaml.safe_load(cfg_fpath.read_text()) or {}
    try:
        ops = cfg["val_dataloader"]["dataset"]["transforms"]["ops"]
    except (KeyError, TypeError):
        ops = []
    for op in ops or []:
        if isinstance(op, dict) and op.get("type") == "Normalize":
            return f"normalized mean={op.get('mean')} std={op.get('std')}"
    return "unnormalized"


def checkpoints_for(workdir: pathlib.Path):
    """(label, checkpoint_or_None) pairs for one run.

    ``None`` means "let the trainer autoselect", which is what a run without
    per-epoch staging can offer. Staged epochs are returned sorted by name,
    which is epoch order because det_solver zero-pads (``epoch_0004.pth``).
    """
    staging = workdir / "staging"
    staged = sorted(staging.glob("epoch_*.pth")) if staging.is_dir() else []
    if staged:
        return [(p.stem, p) for p in staged]
    return [("autoselect", None)]


def subsampled_target(vali: str, stride: int, out_root: pathlib.Path) -> str:
    """Materialise every Nth image of ``vali`` as its own kwcoco bundle.

    Written to disk rather than filtered in memory so that the exact eval
    target is an inspectable artifact: a stage-1 ranking is only meaningful if
    every checkpoint saw the SAME images, and a file on disk proves that in a
    way a re-derived filter does not.
    """
    if stride <= 1:
        return vali
    import kwcoco
    out_fpath = out_root / f"vali_stride{stride}.kwcoco.zip"
    if out_fpath.exists():
        print(f"  reusing subsample {out_fpath}")
        return str(out_fpath)
    full = kwcoco.CocoDataset.coerce(str(vali))
    gids = list(full.images())[::stride]
    sub = full.subset(gids, copy=True)
    sub.fpath = str(out_fpath)
    out_fpath.parent.mkdir(parents=True, exist_ok=True)
    sub.dump()
    print(f"  wrote subsample {out_fpath}: {len(gids)} of {len(list(full.images()))} images")
    return str(out_fpath)


def main() -> int:
    vali = os.environ["KCD_VALI_KWCOCO"]
    runs_dpath = pathlib.Path(os.environ["KCD_RUNS_DPATH"])
    out_root = pathlib.Path(os.environ["KCD_BASELINE_OUT"])
    candidate_id = os.environ.get(
        "KCD_CANDIDATE_ID", "deimv2_dinov3_x_1024x1024_fixed")
    runs = os.environ["KCD_BASELINE_RUNS"].split()

    window = int(os.environ["KCD_TILED_EVAL_WINDOW"])
    overlap = float(os.environ.get("KCD_TILED_EVAL_OVERLAP", "0.25"))
    if window < 64:
        raise SystemExit(f"implausible eval window {window}px -- check paths.sh")

    stride = int(os.environ.get("KCD_EVAL_STRIDE", "1"))
    out_root.mkdir(parents=True, exist_ok=True)
    target = subsampled_target(vali, stride, out_root)
    stage = "stage1-ranking" if stride > 1 else "stage2-final"
    print(f"  stride {stride} ({stage}) target={target}", flush=True)

    rows = []
    for run in runs:
        workdir = runs_dpath / run / "runs" / candidate_id
        if not workdir.is_dir():
            print(f"[{run}] SKIP -- no workdir at {workdir}", file=sys.stderr)
            continue
        policy = json.loads((workdir / "policy.json").read_text())
        category_names = policy["category_names"]
        contract = preprocessing_contract(workdir)
        todo = checkpoints_for(workdir)
        print(f"[{run}] {len(todo)} checkpoint(s), "
              f"categories={category_names}, preprocessing={contract}",
              flush=True)

        for label, ckpt in todo:
            # A DISTINCT eval root per checkpoint. run_kwcoco_eval keys its
            # outputs on candidate_id alone, so a shared root would make all
            # fourteen epochs overwrite one another -- or, with force=False,
            # report epoch 0's metrics fourteen times.
            evalroot = (out_root / f"w{window}_o{overlap}_bf16_s{stride}"
                        / run / label)
            evalroot.mkdir(parents=True, exist_ok=True)
            print(f"  [{run}/{label}] scoring ...", flush=True)
            metrics = run_kwcoco_eval(
                trainer=get_trainer("deimv2"),
                workdir=workdir,
                checkpoint=ckpt,
                test_kwcoco=target,
                kcd_root=evalroot,
                candidate_id=run,
                category_names=category_names,
                # ALWAYS recompute. run_kwcoco_eval reuses a stored
                # detect_metrics.json without checking that it was produced
                # with this window, overlap, NMS setting, dtype or
                # preprocessing. gen001 and gen003 both already carry metrics
                # from WHOLE-IMAGE eval on the TEST split; reporting one of
                # those as the frozen 1229px vali baseline would poison B at
                # its root.
                force=True,
                tiled_eval=True,
                tiled_window=window,
                tiled_overlap=overlap,
                tiled_keep_full=True,
                tiled_batch=int(os.environ.get("KCD_TILED_EVAL_BATCH", "64")),
                device=os.environ.get("KCD_EVAL_DEVICE", "cuda"),
                read_workers=int(os.environ.get("KCD_EVAL_READ_WORKERS", "4")),
            )
            # Read AP the same way SELECTION reads it -- the exact
            # 'area_range=all,iou_thresh=0.5' block plus the distractor
            # sidecar rule. Computing it the same way is not enough; B and the
            # candidate have to be *read* the same way.
            ap = measures_from_detect_metrics(metrics).get("AP@0.5")
            if ap is None:
                print(f"    WARN: no AP@0.5 block in {metrics}", file=sys.stderr)
                continue
            rows.append({"run": run, "label": label, "ap": float(ap),
                         "contract": contract, "metrics": str(metrics)})
            print(f"    AP@0.5 {float(ap):.4f}", flush=True)

    if not rows:
        print("no results -- nothing to summarise", file=sys.stderr)
        return 1

    summary = out_root / f"summary_w{window}_o{overlap}_bf16_s{stride}.json"
    # MERGE rather than overwrite. Scoring one more run later -- gen006 after
    # gen001/gen003/gen007, say -- would otherwise silently drop every row it
    # did not recompute, and the next reader would think those runs were never
    # measured. Rows for the runs scored in THIS pass replace their previous
    # entries; everything else is carried forward. Safe because the filename
    # already pins window, overlap and stride, so a merge can only ever join
    # rows measured under the same protocol.
    merged = rows
    if summary.exists():
        try:
            prior = json.loads(summary.read_text()).get("rows", [])
        except (json.JSONDecodeError, OSError) as ex:
            print(f"  WARN: could not read {summary} ({ex}); overwriting",
                  file=sys.stderr)
            prior = []
        scored_now = {r["run"] for r in rows}
        carried = [r for r in prior if r["run"] not in scored_now]
        if carried:
            print(f"  carrying forward {len(carried)} row(s) from "
                  f"{len({r['run'] for r in carried})} previously-scored run(s)")
        merged = carried + rows
    summary.write_text(json.dumps(
        {"window": window, "overlap": overlap, "split": target,
         "stride": stride, "stage": stage, "full_split": vali,
         "amp": os.environ.get("KCD_AMP_DTYPE", "bfloat16"), "rows": merged},
        indent=2) + "\n")

    print()
    print("=" * 78)
    print(f" vali stride={stride} ({stage}), true-tiled {window}px "
          f"overlap {overlap}, bf16")
    print("=" * 78)
    for r in sorted(rows, key=lambda r: -r["ap"]):
        print(f"  {r['run'][:44]:44} {r['label']:12} AP@0.5 {r['ap']:.4f}")
    print()
    print(f"  wrote {summary}")
    print()
    print("  B is the best row among the BASELINE runs (gen001/gen003) only --")
    print("  a gen006 epoch is the candidate, never part of its own baseline.")
    print("  Do NOT touch the test split until one checkpoint is chosen here.")
    if stride > 1:
        print()
        print(f"  STAGE 1 ONLY (stride {stride}). These AP values rank checkpoints")
        print("  against each other on a subsample; none of them is B, and none")
        print("  is comparable to a stride-1 number. Re-run the top 2-3 with")
        print("  KCD_EVAL_STRIDE=1 before recording anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
