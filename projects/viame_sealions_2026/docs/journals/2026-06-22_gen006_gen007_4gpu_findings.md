# gen006/gen007 4-GPU findings: batch scaling, LR, and ONNX export (2026-06-22)

Jobs: 25 (gen006 4-GPU), 26 (gen007 4-GPU). Both on aiq-gpu, 4× RTX PRO 6000
Blackwell 96 GB. Companion to `2026-06-21_run_catalog_and_memory_analysis.md`.

---

## Results: all three 1280px runs

All numbers are standard (whole-image) eval. Tiled eval is pending for all three.

| job | run | b/gpu | total_b | LR    | backbone LR | ep | nocls_AP | pup_AP | nonpup_AP | peak_MB | wall_h |
|-----|-----|-------|---------|-------|-------------|-----|----------|--------|-----------|---------|--------|
| 23  | gen006 2-GPU | 2 |  4 | 4e-4 | 2e-5 | 30 | 0.899 | 0.875 | 0.920 | 23,824 | 20.5 |
| 25  | gen006 4-GPU | 4 | 16 | 4e-4 | 2e-5 | 45 | 0.882 | 0.869 | 0.907 | 50,808 |  9.9 |
| 26  | gen007 4-GPU | 6 | 24 | 1e-3 | 5e-5 | 45 | 0.892 | 0.870 | 0.910 | 55,131 |  8.3 |

Gen007 (sqrt-scaled LR) recovered most of the gen006 4-GPU regression, but both
4-GPU runs trail the 2-GPU baseline on standard eval by 0.007–0.017 nocls AP.
The 2-GPU baseline (job 23) remains the best standard-eval result at 1280px.

---

## Why gen007 > gen006 4-GPU, but neither beats gen006 2-GPU

**LR / batch interaction**: total batch went from 4 (2-GPU × 2/GPU) to 16
(4-GPU × 4/GPU = gen006) and to 24 (4-GPU × 6/GPU = gen007) while the standard
scaling rule suggests LR should grow with the batch.  Gen006 kept LR at 4e-4;
gen007 used √(24/4) × 4e-4 = 1e-3 (sqrt rule).  Gen007 improves over gen006
(+0.010 nocls) but still doesn't match the 2-GPU baseline.

Two remaining hypotheses for the gap vs job 23:

1. **sqrt rule isn't enough**: linear scaling (6× → 2.4e-3) might be needed.
   Not tested.

2. **Fewer gradient updates**: the 2-GPU 30ep run had N/4 × 30 = 7.5N total
   steps. Gen006 4-GPU had N/16 × 45 = 2.8N. Gen007 had N/24 × 45 = 1.9N.
   The 4-GPU runs have substantially fewer gradient updates even with more epochs.
   Adding epochs (60+) or using a warmup strategy that accounts for this may help.

Neither is decisive without a controlled experiment. Tiled eval may also shift
the comparison — the whole-image metric at 1280px is not the selection criterion.

---

## Memory model correction: the linear fit was wrong for 1280px

The 2026-06-21 design journal predicted batch=6/GPU peak at **77,792 MB** using
a linear fit to the two existing data points (batch=2: 23,824 MB; batch=4:
50,808 MB). The actual was **55,131 MB** — a 41% overestimate.

### Per-epoch peak memory for the three 1280px runs

| ep | b=2/gpu (job 23) | b=4/gpu (job 25) | b=6/gpu (job 26) |
|----|----------------:|----------------:|----------------:|
|  0 |          10,182 |          19,142 |          28,124 |
|  1 |          12,393 |          19,142 |          28,124 |
|  2 |          12,393 |          19,142 |          28,124 |
|  3 |          12,393 |          23,609 |          34,825 |
|  4 |          21,683 |          23,609 |          34,825 |
|  5 |          21,683 |          26,482 |          34,825 |
|  6 |          21,683 |          26,482 |          34,825 |
|  7 |          21,683 |          37,550 |          34,825 |
| … |            …   |           …    |            …   |
| 14 |         23,824 |          37,550 |          47,225 |
| 20 |         23,824 |          50,808 |          47,225 |
| 23 |         23,824 |          50,808 |          55,131 |
| 29 |         23,824 |          50,808 |          55,131 |
| 44 |             —  |          50,808 |          55,131 |

Memory grows in **step-wise discrete jumps**, not smoothly. The jumps don't map
cleanly to the documented augmentation schedule epochs (Mosaic at ep4,
MixUp at ep4–29, CopyBlend at ep4–50). The epochs where jumps occur shift
with batch size, suggesting the steps are driven by EMA cache warming, training
dynamic state, or the particular augmentation blends hitting memory high-water
marks at different rates.

**Key corrected insight**: going from batch=4 to batch=6 at 1280px adds only
~4.3 GB peak (55,131 − 50,808), not 26.9 GB as the linear model predicted.  The
early-training data points are heavily influenced by Mosaic compositing (which
multiplies batch by 4 in terms of pixels loaded), so the slope fitted to them
overestimates the cost of additional batch size in later training.

**Practical ceiling**: batch=6/GPU is verified safe at 56% of 96 GB.  Batch=8
is unverified — extrapolation is unreliable — but the slow growth from b=4 to
b=6 suggests substantial headroom remains.

---

## ONNX export issues resolved

### Issue 1: CPU OOM during export (batch=32 hardcoded in DEIMv2)

`tools/deployment/export_onnx.py` used `torch.rand(32, 3, *img_size)` for the
ONNX trace. At 1280px dinov3_x, batch=32 exhausts CPU RAM (SIGKILL, no .onnx
written). **Fixed**: changed to `batch=1`. The batch dim is marked dynamic so
the exported model accepts any batch at inference. (DEIMv2 submodule commit
`3ce0015`, kit commit `e48ca1d`.)

### Issue 2: onnxsim crash on dinov3 models

`onnxsim` fails with `IndexError: Input .../rope_embed/... is undefined!` on
dinov3-based ONNX models. **Fixed**: `kwcoco_detector_kit/export/onnx.py` now
skips `--simplify` when `"dinov3"` appears in the workdir's `policy.json`
variant. Other backbones (hgnetv2, etc.) still go through onnxsim. (Kit commit
`4e3f03d`.)

With both fixes, gen007's export completed cleanly:
```
Check export onnx model done...
[export.onnx] (no simplify — dinov3 variant)
[sweep] deimv2_dinov3_x_1280x1280_fixed ok
```

---

## What's next

1. **Tiled eval on all three 1280px checkpoints** — the definitive comparison
   metric for pup detection is tiled AP, not whole-image AP.
2. **Linear LR scaling test** — if running another 4-GPU X@1280 with LR=2.4e-3
   (linear 6× from base), compare to gen007's sqrt-scaled 1e-3.
3. **More epochs with proper LR** — gen007's 1.9N total gradient steps vs the
   2-GPU 7.5N may explain the remaining gap; a 90-epoch run at batch=6 LR=1e-3
   would equalize step count.
