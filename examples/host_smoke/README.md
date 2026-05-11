# host_smoke — exercise the main codepaths on a real GPU

Companion to [`examples/kwcoco_demo/run_smoke.sh`](../kwcoco_demo/run_smoke.sh) (the CPU smoke). This dir holds drivers meant to run **on your dev machine** with a real GPU, against synthetic kwcoco data only — not designed to stress GPU memory or CPU, just to validate that the orchestration paths reach the upstream trainer subprocesses and come back with the right artifacts.

## What it tests

| stage | trainer | hardware | what it exercises |
|---|---|---|---|
| 1. env probe | — | CPU | `nvidia-smi`, kit import, `check-env`, `_tier.detect_tier()` |
| 2. CPU baseline | `mock_tiny` | CPU | sanity that the kit's plumbing still works |
| 3. **GPU cell** | `deimv2_hgnetv2_atto` | GPU (`CUDA_VISIBLE_DEVICES`) | the real DEIMv2 trainer subprocess; YAML config-gen → MSCOCO export → `train.py` subprocess → checkpoint → ONNX export → kwcoco eval → bench → manifest row |
| 4. round-loop | `mock_tiny` | CPU | `data.merge` → in-process training → `data.mine` → next round's merge — the hard-negative mining cycle. CPU only to keep GPU time low |
| 5. unified manifest | — | CPU | aggregate stages 3 + 4 into a single eligibility manifest with both candidates side-by-side |

## Run

```bash
# Defaults: CUDA_VISIBLE_DEVICES=1, KCD_DEIMV2_REPO_DPATH from prior-project layout
bash examples/host_smoke/run_gpu_smoke.sh

# Override the workspace + GPU index
KCD_ROOT=/scratch/kcd_gpu_smoke CUDA_VISIBLE_DEVICES=0 \
    bash examples/host_smoke/run_gpu_smoke.sh

# Skip individual stages (any combination):
SKIP_DEIMV2=1 bash examples/host_smoke/run_gpu_smoke.sh    # only mock_tiny + round-loop
SKIP_ROUND_LOOP=1 bash examples/host_smoke/run_gpu_smoke.sh
SKIP_CPU_BASELINE=1 bash examples/host_smoke/run_gpu_smoke.sh
```

## Time + resource budget

On a 24 GB consumer GPU with a warm HGNetv2 backbone cache, the full battery runs in **~60–90 s**. Peak VRAM for the DEIMv2 atto cell is well under 4 GB (256×256 input × batch 4 × 1 epoch × 8 synth images). The script never touches >50 MB of disk under `$KCD_ROOT`.

## First-run caveat — backbone download

DEIMv2's HGNetv2 backbone is initialized from upstream pretrained weights downloaded by the framework on first use (~10–50 MB depending on variant). If your host has no network, expect the DEIMv2 stage to fail at backbone init — set `SKIP_DEIMV2=1` to bypass.

## Prerequisites

- `pip install -e ".[deimv2]"` from the kit's root.
- `$KCD_DEIMV2_REPO_DPATH` set to a DEIMv2 checkout. The script defaults to `$HOME/code/shitspotter/tpl/DEIMv2`; override if yours lives elsewhere.

If `--check-env` reports any missing modules, the script keeps going but prints the install command for each.
