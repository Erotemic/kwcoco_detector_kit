#!/usr/bin/env python3
"""
Extract training curves from a DEIMv2 run's TensorBoard event files
and emit a CSV + a small matplotlib PNG. Tells you whether the run is
converging, overfit, or undercooked without firing up tensorboard.

DEIMv2 logs:
  - Loss/total            train loss per iter
  - Loss/<component>      per-loss-term train values per iter (lots)
  - Lr/pg_<i>             learning rates per param group per iter
  - Test/coco_eval_bbox_0 vali AP@[0.5:0.95] per epoch  ← the curve
  - Test/<k>_<i>          other vali stats per epoch

Usage:
    python3 projects/viame_sealions_2026/scripts/plot_training_curves.py \\
        --run-dir /data/users/jon.crall/kcd_sealion/runs/<run_name>

    # tail mode — polls the events file every 60s, useful for monitoring
    # an in-progress run from a separate terminal:
    python3 ... --run-dir <run_name> --tail 60
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path


def _read_events(events_fpaths):
    """Stream every scalar value across all event files in time order."""
    try:
        from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    except ImportError as ex:
        raise SystemExit(
            "tensorboard package required; pip install tensorboard"
        ) from ex

    # Modern tensorboard writes scalars as tensor protos; legacy writers
    # used simple_value. Support both.
    try:
        from tensorboard.util import tensor_util as _tu
        _tensor_to_float = lambda t: float(_tu.make_ndarray(t).item())  # noqa: E731
    except ImportError:
        import numpy as _np
        def _tensor_to_float(t):
            # Fall back to the protobuf's float_val field for simple cases.
            if t.float_val:
                return float(t.float_val[0])
            if t.double_val:
                return float(t.double_val[0])
            return float("nan")

    rows = []
    for fpath in events_fpaths:
        loader = EventFileLoader(str(fpath))
        for event in loader.Load():
            for v in event.summary.value:
                value = None
                if v.HasField("simple_value"):
                    value = float(v.simple_value)
                elif v.HasField("tensor"):
                    try:
                        value = _tensor_to_float(v.tensor)
                    except Exception:
                        continue
                if value is None:
                    continue
                rows.append({
                    "wall_time": event.wall_time,
                    "step": event.step,
                    "tag": v.tag,
                    "value": value,
                    "events_fpath": fpath.name,
                })
    return rows


def _scan_run_dir(run_dir: Path):
    """Find DEIMv2 summary/events.* files under the run dir.

    Run dir layout (kit convention):
      <run_dir>/runs/<candidate>/summary/events.out.tfevents.*
    """
    candidates = sorted(run_dir.glob("runs/*/summary/events.out.tfevents.*"))
    if not candidates:
        raise SystemExit(
            f"no DEIMv2 tfevents files under {run_dir}/runs/*/summary/"
        )
    return candidates


def _to_csv(rows, dst_fpath: Path):
    keys = ["wall_time", "step", "tag", "value", "events_fpath"]
    with open(dst_fpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _make_plot(rows, dst_png: Path):
    """One matplotlib PNG: train loss (left y) + vali AP (right y) over step."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  matplotlib not available; skipping PNG (CSV at {dst_png.with_suffix('.csv')})")
        return

    train_loss = sorted(
        [(r["step"], r["value"]) for r in rows if r["tag"] == "Loss/total"],
        key=lambda t: t[0],
    )
    vali_ap = sorted(
        # DEIMv2 stamps `coco_eval_bbox_0` per epoch in `Test/`.
        [(r["step"], r["value"]) for r in rows
         if r["tag"] in ("Test/coco_eval_bbox_0", "Test/coco_eval_bbox")],
        key=lambda t: t[0],
    )

    fig, ax1 = plt.subplots(figsize=(11, 5))
    if train_loss:
        steps, vals = zip(*train_loss)
        # Downsample for plotting if there are >10k points (every iter
        # is a lot). Min/max envelope over windows preserves the shape.
        if len(steps) > 4000:
            import numpy as np
            arr = np.array(vals)
            w = len(arr) // 2000
            ax1.plot(steps[::w], arr[::w], color="tab:blue", alpha=0.4, linewidth=0.8,
                     label="train Loss/total (subsampled)")
            # Also overlay a rolling mean for trend
            kernel = np.ones(w) / w
            smoothed = np.convolve(arr, kernel, mode="valid")
            ax1.plot(steps[w-1:], smoothed, color="tab:blue", linewidth=1.5,
                     label="train Loss/total (rolling mean)")
        else:
            ax1.plot(steps, vals, color="tab:blue", linewidth=1.0,
                     label="train Loss/total")
        ax1.set_xlabel("global step (iter)")
        ax1.set_ylabel("train loss", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

    if vali_ap:
        ax2 = ax1.twinx()
        steps2, vals2 = zip(*vali_ap)
        # Vali eval steps are recorded as epoch numbers. To put both on
        # the same x scale we approximate epoch -> iter using iters per
        # epoch (= max train step / num epochs). If we can infer it,
        # rescale; else plot on a second x with right-side axis.
        max_train_step = max((r["step"] for r in rows if r["tag"] == "Loss/total"), default=0)
        max_epoch = max(steps2) if steps2 else 0
        if max_train_step and max_epoch:
            iters_per_epoch = max_train_step / max(max_epoch, 1)
            steps2_iters = [s * iters_per_epoch for s in steps2]
        else:
            steps2_iters = steps2
        ax2.plot(steps2_iters, vals2, marker="o", color="tab:red", linewidth=1.5,
                 label="vali coco_eval_bbox")
        ax2.set_ylabel("vali AP@[0.5:0.95]", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.legend(loc="upper right")
        if train_loss:
            ax1.legend(loc="upper left")

    ax1.set_title(f"DEIMv2 training curves\n{dst_png.parent.name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(dst_png, dpi=120)
    plt.close(fig)


def _summarize(rows):
    """Quick human-readable summary: last train loss, best/last vali AP,
    a verdict on cooked-ness."""
    train_steps = [r["step"] for r in rows if r["tag"] == "Loss/total"]
    train_loss = [r["value"] for r in rows if r["tag"] == "Loss/total"]
    vali_pairs = [(r["step"], r["value"]) for r in rows
                  if r["tag"] in ("Test/coco_eval_bbox_0", "Test/coco_eval_bbox")]
    vali_pairs.sort()

    print(f"  train Loss/total samples : {len(train_loss)}")
    if train_loss:
        n = min(500, len(train_loss))
        first = sum(train_loss[:n]) / n
        last = sum(train_loss[-n:]) / n
        print(f"    first {n} mean : {first:.4f}")
        print(f"    last  {n} mean : {last:.4f}   ({(last-first)/first*100:+.1f}%)")

    print(f"  vali AP samples (epochs) : {len(vali_pairs)}")
    if vali_pairs:
        for ep, ap in vali_pairs:
            print(f"    epoch {int(ep):2d}  vali AP = {ap:.4f}")
        best_ep, best_ap = max(vali_pairs, key=lambda t: t[1])
        last_ep, last_ap = vali_pairs[-1]
        plateau = sum(
            1 for ep, ap in vali_pairs if ep > best_ep and ap <= best_ap + 1e-4
        )
        verdict = []
        if last_ep == best_ep:
            verdict.append("vali AP still climbing at last epoch")
            verdict.append("→ UNDERCOOKED, train longer")
        elif plateau >= 5:
            verdict.append(f"vali AP plateaued at epoch {int(best_ep)}, "
                           f"{plateau} flat epochs after → OVERCOOKED (or saturated)")
        elif last_ap < best_ap - 0.01:
            verdict.append(f"vali AP fell {best_ap - last_ap:.4f} after epoch {int(best_ep)}")
            verdict.append("→ OVERFITTING")
        else:
            verdict.append("vali AP near-converged at last epoch → looks fine")
        print(f"  best vali AP             : {best_ap:.4f} @ epoch {int(best_ep)}")
        print(f"  last vali AP             : {last_ap:.4f} @ epoch {int(last_ep)}")
        for v in verdict:
            print(f"  {v}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="<kcd_root>/runs/<run_name>/")
    p.add_argument("--out-prefix", type=Path, default=None,
                   help="output PNG/CSV prefix; default writes alongside the events file")
    p.add_argument("--tail", type=float, default=0,
                   help="poll the events file every TAIL seconds and rerun "
                        "(useful for monitoring an in-progress run)")
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    if args.out_prefix is None:
        # one PNG/CSV per candidate
        candidates = sorted((run_dir / "runs").glob("*/summary"))
        if not candidates:
            raise SystemExit(f"no candidate summary dirs under {run_dir}/runs/")
        out_dir = candidates[0]
        out_prefix = out_dir / "training_curves"
    else:
        out_prefix = args.out_prefix

    while True:
        events = _scan_run_dir(run_dir)
        print(f"[{time.strftime('%H:%M:%S')}] reading {len(events)} event file(s) under {run_dir}")
        rows = _read_events(events)
        print(f"  {len(rows)} scalar samples")

        csv_fpath = Path(str(out_prefix) + ".csv")
        png_fpath = Path(str(out_prefix) + ".png")
        _to_csv(rows, csv_fpath)
        _make_plot(rows, png_fpath)
        print(f"  wrote {csv_fpath}")
        print(f"  wrote {png_fpath}")
        _summarize(rows)

        if args.tail <= 0:
            break
        print(f"  --tail {args.tail}s: sleeping...")
        time.sleep(args.tail)


if __name__ == "__main__":
    main()
