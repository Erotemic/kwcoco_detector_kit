# 2026-06-04 — DEIMv2 training internals: AMP, LR, Mosaic, OOM

## Context

While diagnosing gen004 OOMs ([[2026-06-04_gen004_forensic_and_resume]])
we uncovered several DEIMv2 internals that affect every kit-launched
run and that are easy to misread from the surface YAML. Recording
them here so future agents can sanity-check a config without
re-deriving from scratch.

## 1. AMP was silently disabled in every kit run prior to commit `c43bf8f`

**The bug chain:**

1. Kit's submit script: `export KCD_USE_AMP=true`
2. Kit's `_launch_train.sh`: `--use_amp "$KCD_USE_AMP"` to sweep
3. Kit's `trainers/deimv2.py:_build_train_yml`: writes
   `use_amp: true` to the generated `train.yml`
4. Kit's `launch()`: subprocesses
   `python -m torch.distributed.run ... train.py -c train.yml`
   **without** `--use-amp`
5. DEIMv2's `train.py:72`: defines `--use-amp` as
   `action='store_true'`, defaulting `args.use_amp` to **False**
   (not None) when absent
6. DEIMv2's `train.py:41-42`: forwards `args.__dict__` to
   `YAMLConfig(**kwargs)` via
   `{k: v for k, v in args.__dict__.items() if v is not None}`.
   `False is not None`, so `use_amp=False` enters update_dict.
7. `YAMLConfig.__init__`:
   `cfg = merge_dict(cfg, kwargs)` — kwargs override the YAML.
8. The resulting cfg has `use_amp: False` despite the YAML saying
   true.

**Net effect**: every kit-launched DEIMv2 run trained at FP32
regardless of the user's `KCD_USE_AMP` setting. Caught when
2577 OOM'd at 45.76 GB / 47.4 GB on GPU 1 — AMP would have
halved activation memory and the OOM wouldn't have happened.

**Fix** (`c43bf8f`): kit-side, no DEIMv2 patch. In
`trainers/deimv2.py:launch()`, introspect `use_amp` from the
generated YAML and append `--use-amp` to the `train.py` argv
when true:

```python
import yaml
with open(cfg_fpath) as _f:
    _yaml_top = yaml.safe_load(_f) or {}
if _yaml_top.get("use_amp", False):
    args.append("--use-amp")
```

**How to verify the fix took effect**: grep the slurm log's
`cfg:` dump line for `'use_amp': True` at both top level AND
inside `yaml_cfg`. Both must be True. The `scaler` block alone
isn't sufficient evidence — DEIMv2 builds a GradScaler when
`use_amp` is True, but `scaler.enabled` reflects only the
scaler's autocast state, not whether AMP is actually wired into
training.

## 2. Mosaic / RandomZoomOut / RandomIoUCrop are 4-stage scheduled, not "stop at epoch N"

The `policy = {'name': 'stop_epoch', 'epoch': [4, 78, 148], 'ops':
[Mosaic, RandomPhotometricDistort, RandomZoomOut, RandomIoUCrop]}`
config is easy to read as "Mosaic stops at epoch 4." It's
actually a 4-stage scheduler implemented in
`tpl/DEIMv2/engine/data/transforms/container.py:74-107`:

| stage | epoch range | Mosaic | RPD / RZO / IoUCrop |
|---|---|---|---|
| 1 (warmup) | `epoch < 4` | OFF | OFF |
| 2 (heavy aug) | `4 ≤ epoch < 78` | ON with `mosaic_prob=0.5` | ON |
| 3 (mid) | `78 ≤ epoch < 148` | OFF | ON |
| 4 (tail no-aug) | `epoch ≥ 148` | OFF | OFF |

When Mosaic is active, RZO and IoUCrop are mutually exclusive
with it (skipped to avoid double-augmentation conflicts).

Practical consequences:
* For a 30-epoch run, we're in stage 2 for epochs 4-29 — Mosaic
  is active most of the run, not "stopped early."
* The early-epoch mAP plateau in 2570 (hgnetv2_n) at 0.006 was
  NOT explained by Mosaic going away — Mosaic is on from epoch
  4. The plateau reflects model capacity.
* When resuming mid-run (e.g. 2577 → resume at epoch 5),
  Mosaic IS still active.

MixUp and CopyBlend live in the collate function (separate from
the policy.ops machinery):
* MixUp: `mixup_epochs=[4, 64]` — active in epochs 4-63
* CopyBlend: `copyblend_epochs=[4, 120]` — active in epochs 4-119
* Collate `stop_epoch=120` — after that, no mixup/copyblend

For 30-epoch runs both are active for the entire post-warmup run.

## 3. FlatCosineLR with `flat_epoch > epoches` is constant LR for the entire training

The dinov3_s upstream config has `flat_epoch=64`, `warmup_iter=2000`.
Kit overrides `epoches=30` but NOT `flat_epoch`. The
FlatCosineLR has three phases (warmup → flat → cosine):

| iters/epochs | phase | LR |
|---|---|---|
| iter `< 2000` (~0.05 epoch) | warmup | linear 0 → target |
| iter `≥ 2000`, epoch `< 64` | flat | constant target |
| epoch `≥ 64` | cosine | annealing 0 → target |

With `epoches=30` and `flat_epoch=64`, the cosine phase never
triggers. The entire run after the brief iter-warmup is at the
flat (constant) LR.

Practical consequences:
* Changing iters-per-epoch (via `max_oversample`) doesn't
  break the LR schedule — there's no curve being interpolated
  against epoch count.
* The "FlatCosine" name is misleading for kit configs; in
  practice it's a constant-LR-after-warmup schedule.
* If we ever want cosine annealing, kit needs to override
  `flat_epoch` (probably to `epoches/2` ≈ 15) when emitting
  the YAML.

## 4. OOM math for dinov3_s + 640 + batch=16 + 2-GPU DDP

Observed memory peaks at OOM for the dinov3_s gen004 runs:

| Run | mode | OOM at | peak max_mem |
|---|---|---|---|
| 2577 | FP32, multiscale 512-768 | epoch 6 backward | 45.76 GB |
| 2578 | **FP16**, multiscale 512-768 | epoch 5 iter 500 | 44.79 GB |
| 2579 | **FP16**, **fixed 640**, resume | epoch 8 (mid-batch spike) | 42.82 GB |

2579 ran THREE full epochs (5-7) at steady-state ~31-33 GB
before hitting a transient peak at 42.82 GB and OOMing on a
4.68 GB allocation. So fixed 640 reduces the typical peak
substantially but the worst-case batch (lots of GT, IoUCrop
producing a lot of crops, etc.) still flirts with the 47 GB
ceiling. Reproducibly. Lesson: at batch_per_gpu=16,
dinov3_s + 640 is too close to the ceiling regardless of
multiscale on/off.

AMP saved ~1 GB. Less than expected because:

```
   dinov3_s weights (FP32)            ~5 GB   (AMP keeps FP32 master)
   AdamW state (m, v) at FP32         ~10 GB  (AMP doesn't optimize this)
   EMA model (full duplicate at FP32) ~5 GB   (AMP doesn't help)
   ───────────────────────────────────────────
   non-AMP-helpable subtotal          ~20 GB
   activations + gradients @640 + AMP ~10 GB  (~halved from ~20 GB FP32)
   matcher / loss / aux heads peak    ~5 GB
   working buffers                    ~3 GB
   ───────────────────────────────────────────
   peak realistic @640 fixed          ~38 GB  → fits in 47.4 GB
   peak realistic @768 multiscale     ~50 GB  → OOM
```

The 768×768 batches hit when `multiscale_512_768` samples the
upper end of its range. At 768²/640² = 1.44× pixels, activations
scale 1.44×, pushing the total over 47.4 GB.

**Mitigation options**, learned the hard way:

1. ~~`KCD_TRAIN_POLICY=fixed` alone~~ — NOT sufficient. 2579
   ran 3 full epochs at fixed 640 then OOMd on a transient
   peak. Reduces typical peak but worst-case batch still
   hits the ceiling.
2. **`KCD_PER_GPU_BATCH=8` is the actual fix**. Halves
   activation memory in worst-case batches → ~7-10 GB
   headroom. Adopted in both gen004 dinov3_s scripts after
   2579. Trade-off: 2× iters/epoch and ~half per-iter
   throughput. With `max_oversample=1` keeping epochs short
   (~1600 iters at batch 8), wallclock is still manageable
   (4-8h per 30-epoch run).
3. `multiscale_512_640` — cap upper end; some downscale
   diversity, no upscale. Untested but probably equivalent
   to fixed 640 in memory terms.
4. Disable EMA (`use_ema: false` in YAML) — saves ~5 GB but
   may hurt convergence; EMA is usually worth the memory.
   Untested.
5. Gradient checkpointing on the decoder — saves activation
   memory at cost of 30-50% wallclock per backward pass. Not
   currently exposed; would need a DEIMv2 patch.

**Conclusion**: at dinov3_s + 640 on A6000 (47 GB), batch 16
per GPU is too aggressive. Batch 8 is the sweet spot — model
fits with headroom, the shorter epoch from max_oversample=1
absorbs the 2× iter overhead.

## 5. Other knobs to know about

**`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** —
the kit's trainer sets this via `env.setdefault(...)` so it
takes effect unless overridden. Helps fragmentation but
doesn't help true memory pressure. In 2578 the OOM message
showed only 187 MB "reserved but unallocated" — fragmentation
wasn't the issue, real allocation was.

**`find_unused_parameters: True`** — set by the kit for
DEIMv2's DDP wrapping. Required because the DEIM decoder
gates some heads conditionally. Adds DDP overhead but no
memory cost worth worrying about.

**`sync_bn: True`** — synchronizes BN stats across DDP ranks.
Modest extra communication per batch; not the OOM cause.

**`num_workers`** — DataLoader workers, each holds a
shuffle buffer and decode workspace. Each worker is ~1-2 GB
host RAM, not GPU. Counted in [[feedback-arisia-resource-budgets]]
for the host KCD_MEM allocation.

## How to validate a fresh kit run

Quick sanity grep on the first slurm log output (within 5 min
of submission):

```bash
LOG=/data/users/jon.crall/slurm_logs/<run>-<jobid>.out

# AMP active end-to-end
grep -oE "'use_amp': (True|False)" "$LOG" | head -2
# Both should print 'use_amp': True

# Balance applied (if KCD_BALANCE_TARGET_JSON was set)
grep "balance_mscoco: wrote" "$LOG"
# Should show your target_distribution and actual bucket counts

# max_oversample took effect
grep -E "max_oversample|target_size" "$LOG"

# Augmentation policy active
grep -oE "'policy': \{[^}]+\}" "$LOG" | head -1

# First few iters' memory growth
grep -oE "max mem: [0-9]+" "$LOG" | head -5
# At iter 0 max mem should be ~1/3 of GPU. Growth across iters is
# expected (caching, EMA buildup); plateau by iter ~500.
```

## Open

* Should the kit override `flat_epoch` to bring cosine into
  play for short runs? Worth ablating once we have a stable
  dinov3_s + balance baseline.
* Is there a DEIMv2 patch to add `--use-amp` to `default=None`
  so the YAML wins when CLI is absent? Cleaner than the kit
  introspecting YAML. Upstream PR worth considering.
