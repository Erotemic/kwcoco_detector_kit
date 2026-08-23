# The stop_epoch collision, and the first completed schedule (2026-08-23)

Three days that started with gen002 burning four GPUs to produce nothing, and
ended with the first fish run to finish its schedule and a held-out number that
matches gen001's. The interesting part is in between: the bug that killed
gen002 was introduced by the commit that was supposed to fix short schedules.

## Outcome first

| run | dtype | schedule | vali AP | **held-out test AP50** |
|---|---|---|---|---|
| gen001 (job 293) | fp16 | 20, died at 13 | 0.5440 | 0.7272 |
| gen002 (job 489) | fp16 | 12, died at 1 | 0.5443 (epoch 1) | — |
| **gen003 (job 490)** | bf16 | **24 of 24** | 0.5406 | **0.7285** |

gen003 is the first completed schedule on this corpus. Its held-out test AP50
of **0.7285** against gen001's 0.7272, on the identical 69-sequence / 33,434
image split, is a marginal improvement well inside noise.

The vali number went the *other* way (0.5406 vs 0.5440). That is worth sitting
with: **vali did not predict test.** It has been the weaker guide throughout
this project, and gen001's vali is additionally flattered by a mechanism
described below.

## What killed gen002

Partway through epoch 1 the forward went all-NaN and never came back. The run
then trained 11 more epochs to AP 0.000, holding four GPUs for ~11 hours.

Two unrelated things collide at epoch 1:

1. `train_policy=fixed` pins `collate_fn.stop_epoch = 1` (`deimv2.py:287`), and
   at `epoch == stop_epoch` DEIMv2 reloads model, optimizer, **GradScaler** and
   EMA state from `best_stg1.pth` (`det_solver.py:83-86`).
2. The augmentation policy `[1, 6, 11]` turned Mosaic / ZoomOut / IoUCrop /
   PhotometricDistort on for the first time — also at epoch 1.

So the optimizer and the fp16 loss scale were reset to their epoch-0 (NoAug)
values at the exact step the input distribution changed.

**The evidence is 2-for-2.** Both NaN excursions this project has seen land in
the first epoch with augmentation enabled, at almost the same step:

| run | policy | first NaN |
|---|---|---|
| gen001 | `[4, 78, 148]` | epoch 4, step 7500 / 10464 |
| gen002 | `[1, 6, 11]` | epoch 1, step ~7700 / 10464 |

gen001's reload was three epochs before its boundary, so the two events were
separated — and it **recovered** at epoch 5. gen002's collided, and it never
escaped: it reloaded good weights at the end of each epoch and re-NaN'd inside
the next one.

### The commit that caused it

`2b3a911` scaled the augmentation policy to the schedule length, fixing a real
bug (short runs never reached the clean final stage). Its clamp is
`e0 = max(1, ...)` — which lands on **1 for every schedule shorter than ~80
epochs**. Before that commit the policy was upstream's raw `[4, 78, 148]` and
no collision was possible. The fix for short schedules quietly created a
different defect on all of them, and it affected the sea-lion project too.

### Why it was not self-limiting

Once the weights are NaN, every loss reads exactly `0.0000`. The gradients are
therefore 0 and **finite** — so `GradScaler` finds nothing to skip, and
upstream's own `math.isfinite(loss_value)` guard 70 lines further down never
fires. Meanwhile DEIMv2's NaN detector dumped an 819 MB `NaN.pth` and printed a
300x4 tensor *every step*: 500,000 log lines, 13.9 MB, and 3x the step time.

A run that is alive, consuming four GPUs, and provably producing nothing looks
identical to a healthy one in a `tail -f`.

## Fixes

| commit | change |
|---|---|
| `a5cee5b` | no augmentation boundary may coincide with `stop_epoch` |
| DEIMv2 `32c3cd7` | abort on non-finite `pred_boxes` instead of dumping and continuing |
| `09298ff` | `run-health`: diagnose a run from its log |
| `03e48a9` | cast autocast outputs to float32 before they reach numpy |
| DEIMv2 `4d7d253`, `db19ab9` | revert the AMP default to fp16 |

The collision fix is general rather than special-casing `e0`: `multiscale` pins
`stop_epoch = num_epochs - 4` and could collide at `e1` or `e2` instead. Where
no four-stage schedule fits without a collision — only 2-3 epoch smoke tests —
the policy collapses to `[0, 0, 0]`, disabling augmentation entirely, which is
strictly safer than recreating the collision.

## bf16 was a mistake

gen003 changed three things at once, and only two were justified. bf16 was
adopted on the theory that fp16's ~65504 ceiling caused the NaN excursions. It
did not — the collision explains them.

DEIMv2 is an **fp16 recipe**:

- bfloat16 appears nowhere in its training path (the only hits are a
  `layernormbf16` alias inside the vendored DINOv3 backbone and an unused fp8
  path);
- it constructs a `GradScaler` unconditionally (`configs/runtime.yml:11`) —
  machinery whose sole purpose is keeping fp16 gradients out of underflow, and
  a vestige under bf16;
- every published COCO number comes from plain `--use-amp`, i.e. torch's fp16
  default.

bf16 also gives up three mantissa bits (7 vs 10) in a model whose localization
is fine-grained-distribution based (`reg_max: 32`, `loss_fgl`, `loss_ddf`).
gen003 finished below both fp16 runs on vali. The default is reverted; the knob
(`KCD_AMP_DTYPE`) survives as an escape hatch, and gen003's own scripts pin
`bfloat16` so re-running them reproduces gen003.

**It also broke scoring.** Job 490 trained all 24 epochs and then died on the
first image of the test pass with `TypeError: Got unsupported ScalarType
BFloat16`. numpy has a `float16` dtype but no `bfloat16`, so every `.numpy()`
downstream of the eval autocast worked for exactly as long as that autocast was
fp16. Training was never at risk; job 491 re-scored the finished checkpoint.

## Two things that will mislead you later

**Vali is not comparable across these runs.** DEIMv2 reloads `best_stg1.pth`
after *every* non-improving eval (`det_solver.py:213-217`). gen001's loss curve
sawtooths because of it — 36.66 down to 34.71, then back to 36.78 at epoch 9 —
and its AP is repeatedly restored to its best weights. gen003 never triggered
that path; its trajectory is monotone. Comparing the two vali numbers compares
different mechanisms.

**"Average Recall" reads as a much better AP if you grep loosely.** The AR
lines share the entire bracket text with the AP lines and differ only in the
`Average Precision (AP)` / `Average Recall (AR)` prefix. Alternating 0.53 /
0.62 values in a log are AP and AR for one epoch, not a raw model and an EMA
model. There is one eval per epoch. This cost an hour of confusion and is now
pinned by a regression test.

## VRAM, measured

`MetricLogger` records `torch.cuda.max_memory_allocated()`. gen003 at batch
16/GPU on 96 GB cards:

| | |
|---|---|
| peak allocated | **37.8 GB of 96** |
| working set between steps | 1.26 GB (model + EMA + optimizer) |
| activations | 2.28 GB/sample |
| growth | 36.9 GB at epoch 0 step 0 -> 37.8 GB by epoch 14, then flat |

Turning augmentation on at epoch 2 added ~100 MB. No fragmentation creep over
24 epochs. Batch 32/GPU would fit in ~77 GB.

**But bigger batches are the wrong lever.** Throughput scales sub-linearly with
batch while step count scales inversely, so across a fixed 48 h: batch 64 gives
54 epochs / 212k steps, batch 128 gives 62 epochs / 122k steps. Doubling the
batch buys 16% more images per hour and halves the updates.

It is tempting to read gen003 as update-starved — gen001 reached vali 0.5440
on ~136k steps, gen003 only 0.5406 on ~94k. **The test numbers say otherwise.**
On held-out data gen003 scored *higher on fewer steps*: 0.7285 at ~94k against
gen001's 0.7272 at ~136k. Per update it was the more efficient run, and the
batch increase cost nothing measurable.

What is true is that gen003 was **still climbing when its schedule ended** —
vali ran 0.537, 0.539, 0.540, 0.541 over its last epochs before the final NoAug
one. That, not update starvation, is the case for a longer schedule.

## Lessons

1. **A boundary that lands on another mechanism's boundary is a bug**, even
   when both are individually correct. Neither the reload nor the augmentation
   switch is wrong; their coincidence is.
2. **A fix scoped to one symptom can create another.** `2b3a911` was right
   about short schedules and wrong about where its clamp would land.
3. **Failing loudly is a feature.** The difference between an 11-hour waste and
   a 20-minute one was a `raise` that upstream declined to write.
4. **Change one variable.** gen003 moved dtype, batch and schedule together, so
   its 0.5406 cannot be attributed. gen004 moves only the schedule.
5. **Trust the held-out split.** Vali said gen003 was worse; test said it was
   marginally better. Only one of those is a generalization measure.

## Next

`gen004` (`submit_train_..._gen004_long.sh`): fresh from COCO, **fp16**, batch
16/GPU unchanged, **48 epochs** (~44 h, ~212k steps — 1.6x gen001, 2.3x
gen003). Policy `[2, 25, 47]`, no collision. It is also the missing control:
no run has yet combined fp16 with the collision fixed.

Check `nvidia-smi` before submitting. A stray vLLM server remains the leading
unproven explanation for job 296's 8.4x slow steps and job 489's unexplained
2 h 03 m mid-epoch freeze between steps 4000 and 4500.

Monitor with:

```bash
bash projects/viame_fish_2026/scripts/run_health.sh --watch --num_epochs 48
```
