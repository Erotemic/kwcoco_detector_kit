#!/usr/bin/env python3
"""
Parse DEIMv2 training logs for GPU memory usage.

DEIMv2 prints per-GPU memory at the end of every logged training step:
    cur mem: <MB>  max mem: <MB>

where max_mem is the monotonically-increasing high-water mark
(torch.cuda.max_memory_allocated()) since the last reset, and cur_mem is the
live allocation at that step. The counter is reset at the start of each epoch
by DEIMv2's train_one_epoch() loop.

Usage:
    python dev/analyze_gpu_memory.py --log-file <path> [--out-json <path>]

Output:
    Per-epoch table of {step, cur_mem_mb, max_mem_mb} printed to stdout, and
    optionally written as a JSON data file for archival in journals.

This is a posthoc analysis; the script never writes to the training directory.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


STEP_PATTERN = re.compile(
    r'\[(\d+)/(\d+)\]'
    r'.*?'
    r'cur mem:\s+(\d+(?:\.\d+)?)'
    r'\s+max mem:\s+(\d+(?:\.\d+)?)'
)
EPOCH_PATTERN = re.compile(r'Epoch:\s*\[(\d+)\]')

# Match "Test:" lines that appear during eval (also emit cur/max mem)
EVAL_PATTERN = re.compile(r'Test:.*?cur mem:\s+(\d+(?:\.\d+)?)\s+max mem:\s+(\d+(?:\.\d+)?)')

COCO_MAP_PATTERN = re.compile(r'COCO eval.*AP.*?=\s*([\d.]+)', re.IGNORECASE)
TILED_AP_PATTERN = re.compile(r'AP_tiled[^:]*:\s*([\d.]+)', re.IGNORECASE)

# Match inline mAP lines like "mAP: 0.2291  ..."
MAP_LINE_PATTERN = re.compile(r'\bAverage Precision.*?@\[\s*IoU=0\.50:0\.95\b.*?=\s*([\d.]+)')


def parse_log(log_path: Path) -> dict:
    """
    Extract per-epoch and per-step memory statistics from a DEIMv2 log.

    Returns a dict with:
        epochs: list of per-epoch dicts, each containing:
            epoch: int
            steps: list of {step, cur_mem_mb, max_mem_mb}
            peak_cur_mem_mb: max cur_mem across the epoch
            peak_max_mem_mb: max max_mem across the epoch (== final max_mem value)
            final_max_mem_mb: the max_mem value at the last step of the epoch
        eval_mem_mb: list of {epoch, cur_mem_mb, max_mem_mb} from eval steps
        global_peak_mb: single maximum max_mem across the entire run
    """
    epoch_steps: dict[int, list] = defaultdict(list)
    eval_rows = []
    current_epoch = None

    with open(log_path) as fh:
        for line in fh:
            # Detect epoch header (may appear on same line as first step)
            em = EPOCH_PATTERN.search(line)
            if em:
                current_epoch = int(em.group(1))

            # Training step memory
            sm = STEP_PATTERN.search(line)
            if sm and current_epoch is not None:
                step = int(sm.group(1))
                cur_mem = float(sm.group(3))
                max_mem = float(sm.group(4))
                epoch_steps[current_epoch].append({
                    'step': step,
                    'cur_mem_mb': cur_mem,
                    'max_mem_mb': max_mem,
                })

            # Eval step memory (Test: prefix)
            ev = EVAL_PATTERN.search(line)
            if ev and current_epoch is not None:
                eval_rows.append({
                    'epoch': current_epoch,
                    'cur_mem_mb': float(ev.group(1)),
                    'max_mem_mb': float(ev.group(2)),
                })

    epochs = []
    global_peak = 0.0
    for ep in sorted(epoch_steps.keys()):
        steps = epoch_steps[ep]
        if not steps:
            continue
        peak_cur = max(s['cur_mem_mb'] for s in steps)
        peak_max = max(s['max_mem_mb'] for s in steps)
        final_max = steps[-1]['max_mem_mb']
        global_peak = max(global_peak, peak_max)
        epochs.append({
            'epoch': ep,
            'n_steps': len(steps),
            'steps': steps,
            'peak_cur_mem_mb': peak_cur,
            'peak_max_mem_mb': peak_max,
            'final_max_mem_mb': final_max,
        })

    # Include eval rows in global peak check
    if eval_rows:
        global_peak = max(global_peak, max(r['max_mem_mb'] for r in eval_rows))

    return {
        'log_file': str(log_path),
        'epochs': epochs,
        'eval_mem': eval_rows,
        'global_peak_mb': global_peak,
    }


def print_summary(result: dict, gpu_capacity_mb: float = 98304.0) -> None:
    """Pretty-print the per-epoch memory table."""
    gpu_gb = gpu_capacity_mb / 1024
    print(f"\nGPU capacity: {gpu_capacity_mb:.0f} MB ({gpu_gb:.1f} GB)")
    print(f"Log: {result['log_file']}")
    print()
    print(f"{'Epoch':>6}  {'Steps':>6}  {'peak cur_mem':>13}  {'peak max_mem':>13}  {'headroom':>10}  {'headroom%':>10}")
    print("-" * 72)
    for ep in result['epochs']:
        e = ep['epoch']
        ns = ep['n_steps']
        pcur = ep['peak_cur_mem_mb']
        pmax = ep['peak_max_mem_mb']
        headroom = gpu_capacity_mb - pmax
        head_pct = 100.0 * headroom / gpu_capacity_mb
        print(f"{e:>6}  {ns:>6}  {pcur:>10.0f} MB  {pmax:>10.0f} MB  {headroom:>7.0f} MB  {head_pct:>8.1f}%")

    global_peak = result['global_peak_mb']
    headroom = gpu_capacity_mb - global_peak
    head_pct = 100.0 * headroom / gpu_capacity_mb
    print("-" * 72)
    print(f"{'GLOBAL PEAK':>6}  {'':>6}  {'':>13}  {global_peak:>10.0f} MB  {headroom:>7.0f} MB  {head_pct:>8.1f}%")

    if result['eval_mem']:
        eval_peak = max(r['max_mem_mb'] for r in result['eval_mem'])
        eval_headroom = gpu_capacity_mb - eval_peak
        print(f"\nEval phase peak max_mem: {eval_peak:.0f} MB  ({eval_headroom:.0f} MB headroom = {100*eval_headroom/gpu_capacity_mb:.1f}%)")

    print()
    print("Batch-size scaling estimate (linear rule, per-GPU):")
    baseline_batch = 2
    baseline_peak = global_peak
    for target_batch in [2, 3, 4, 6, 8]:
        scale = target_batch / baseline_batch
        # Memory scales roughly linearly with batch for activation tensors;
        # model params + optimizer states are constant. Approximate: model
        # overhead ~constant, activations scale ~linearly.
        # We estimate: mem(b) ≈ overhead + activations_per_sample * b
        # With two data points we can't separate them, so use a conservative
        # linear model: mem(b) ≈ peak * (b / baseline).
        estimated = baseline_peak * scale
        fit = gpu_capacity_mb - estimated
        safe = "SAFE" if fit > 0 else "OOM"
        print(f"  batch={target_batch}/GPU  estimated peak ≈ {estimated:.0f} MB  headroom ≈ {fit:.0f} MB  [{safe}]")

    print()
    print("Note: The linear model overestimates for large batches because model")
    print("parameters + optimizer state (~constant) dominate at small batch.")
    print("Actual headroom will be larger than estimated; treat as a lower bound.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--log-file', required=True, help='Path to slurm .out log')
    ap.add_argument('--out-json', help='Optional path to write parsed data as JSON')
    ap.add_argument('--gpu-gb', type=float, default=96.0,
                    help='GPU VRAM in GB (default: 96 for RTX PRO 6000)')
    args = ap.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"ERROR: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    result = parse_log(log_path)

    gpu_capacity_mb = args.gpu_gb * 1024
    print_summary(result, gpu_capacity_mb)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Strip full step lists to keep file small; keep epoch-level summaries
        export = {
            'log_file': result['log_file'],
            'gpu_capacity_mb': gpu_capacity_mb,
            'global_peak_mb': result['global_peak_mb'],
            'epochs': [
                {k: v for k, v in ep.items() if k != 'steps'}
                for ep in result['epochs']
            ],
            'eval_mem_peak_mb': (
                max(r['max_mem_mb'] for r in result['eval_mem'])
                if result['eval_mem'] else None
            ),
        }
        with open(out_path, 'w') as fh:
            json.dump(export, fh, indent=2)
        print(f"Data written to: {out_path}")


if __name__ == '__main__':
    main()
