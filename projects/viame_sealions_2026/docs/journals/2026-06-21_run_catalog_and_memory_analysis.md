# Run catalog and GPU memory analysis across all historical runs (2026-06-21)

Script: `dev/catalog_runs.py`
Catalog: `dev/run_catalog.json`, `docs/journals/data/run_catalog.md`

---

## Corpus summary

102 training job records from 29 unique run names spanning:
- Schemes: `pup_vs_nonpup`, `single_sealion`, `lifestage_6cls`, `full_8cls`
- Backbones: `hgnetv2_n`, `dinov3_s`, `dinov3_x`
- Resolutions: 320, 640, 1280 px
- Hosts: arisia (A6000, 48 GB), aiq-gpu (Blackwell, 96 GB), namek (3090, 24 GB)
- Generations: v1–v6 (old naming) + gen002–gen006 (new naming)

Outcome breakdown (per job, not per run_name):

| outcome    | jobs |
|------------|------|
| unknown    | 37   |
| killed     | 30   |
| oom        | 16   |
| completed  | 14   |
| partial    | 3    |
| nan_diverge| 2    |

`unknown` = log has the run-config header but no training steps appeared (Docker
startup failure, slurm queue never started, or node allocated with no GPU
resources). These dominated the gen002-era arisia runs when we were still
shaking down the pipeline.

---

## AP progression: pup_vs_nonpup (primary scheme)

All APs are standard (whole-image) eval unless noted. Tiled eval pending for
all 1280px runs.

| generation | backbone   | res  | gpus | b/gpu | nocls_AP | pup_AP | notes |
|-----------|------------|------|------|-------|----------|--------|-------|
| gen002    | hgnetv2_n  | 320  |  1   |  16   | 0.025    | 0.000  | training_ready_v1 (20% subset) |
| v6        | hgnetv2_n  | 320  |  1   |  16   | 0.199    | 0.010  | training_ready_v1, fixed LR/batch |
| gen004    | dinov3_s   | 640  |  2   |   8   | 0.565    | 0.102  | training_ready_v1, balanced corpus |
| gen005    | dinov3_s   | 640  |  2   |   8   | 0.861    | 0.840  | **detection_v1 full corpus** |
| gen005    | dinov3_x   | 640  |  4   |  16   | 0.892    | 0.863  | detection_v1 |
| gen006    | dinov3_x   | 1280 |  2   |   2   | 0.899    | 0.875  | detection_v1; job 23 |
| gen006    | dinov3_x   | 1280 |  4   |   4   | 0.882    | 0.869  | LR unchanged at 4e-4; job 25 |
| gen007    | dinov3_x   | 1280 |  4   |   6   | 0.892    | 0.870  | LR=1e-3 (sqrt-scaled); job 26 |

Key inflection points:

- **gen004→gen005 (+0.74 pup AP)**: dataset change is the dominant factor.
  `training_ready_v1` was a 20% subset; `detection_v1` is the full 2007–2024
  corpus (6.5 k train images). Backbone and LR also changed simultaneously so
  we can't fully decompose, but the corpus change is the binding lever.

- **dinov3_s→dinov3_x at 640px (+0.023 pup AP)**: capacity helps everywhere,
  not disproportionately for pups.  Consistent with gen005 analysis.

- **640px→1280px at dinov3_x (+0.012 pup AP)**: small improvement on the
  whole-image eval (which is what the gen006 1280 run reports).  The tiled
  eval at 1280 would likely show more gain for small objects (pups), but that
  number isn't yet in the catalog for this run.

### Cross-scheme comparison (gen005 era, dinov3_s@640, detection_v1 corpus)

| scheme       | backbone  | res  | nocls_AP | notes |
|-------------|-----------|------|----------|-------|
| single_sealion | dinov3_s | 640 | 0.864  | class-agnostic |
| pup_vs_nonpup  | dinov3_s | 640 | 0.861  | 2-class |
| lifestage_6cls | dinov3_s | 640 | 0.810  | 6-class |
| full_8cls      | dinov3_x | 640 | 0.829  | 8-class, X backbone |

More classes → lower nocls_AP. The full_8cls run uses dinov3_x (larger)
which partially compensates for the harder task but doesn't fully close the gap.

---

## GPU memory analysis

### Where memory spikes happen

Across all dinov3_x runs, two consistent jumps appear:

**Jump 1 — epoch 3→4: Mosaic activation (primary spike)**

The DEIMv2 augmentation policy schedule triggers Mosaic, MixUp, and
CopyBlend at epoch 4 (policy `epoch: [4, 78, 148]` for a 30-epoch run at
1280px; `[4, 29, 50]` at 640px).  The memory jump at epoch 3→4 is the
largest single event in every run.

| config              | ep0 MB | ep4 MB | ep3→4 jump | % of GPU |
|--------------------|--------|--------|-----------|----------|
| dinov3_s@640 2×8   |  6,099 | 11,255 |  +5,156   |  10.5%   |
| dinov3_x@640 2×8   | 13,400 | 17,305 |  +3,905   |   4.0%   |
| dinov3_x@640 4×16  | 28,524 | 34,772 |  +6,248   |   6.4%   |
| dinov3_x@640 4×64* | 25,785 | 41,219 | +15,434   |  15.7%   |
| dinov3_x@1280 2×2  | 10,182 | 21,683 |  +9,501   |   9.7%   |

`*` full_8cls run (64 total = 16/GPU × 4 GPUs, 8 classes)

The 1280px run's epoch 3→4 jump (+9.5 GB) is notably larger than the 640px
run at comparable per-GPU batch (+6.2 GB) because Mosaic generates 4× the
pixel data at 1280px vs 640px.

**Jump 2 — epoch 14→15 (secondary, 1280px only)**

At 1280px with `mixup_epochs: [4, 29]`, the MixUp window is active from
epoch 4 through 28.  An additional jump appears at epoch 14→15 in the
1280px run (+2,141 MB).  This coincides with the `flat_epoch=15` LR
transition but the mechanism is more likely EMA cache warming or a
secondary augmentation parameter change.  This jump does NOT appear in
640px runs with `mixup_epochs: [4, 14]`.

### Memory profiles by backbone and resolution

| backbone  | res  | host  | b/gpu | total_b | ep0 MB | ep4 MB | peak MB | vram MB | util% |
|-----------|------|-------|-------|---------|--------|--------|---------|---------|-------|
| dinov3_s  | 640  | arisia|  8    |  16     |  6,099 | 11,255 | 12,240  | 49,152  |  25%  |
| dinov3_s  | 640  | namek |  8    |   8     |  6,897 | 10,201 | 10,201  | 24,576  |  41%  |
| dinov3_x  | 640  | aiq   |  8    |  16     | 13,400 | 17,305 | 23,456  | 98,304  |  24%  |
| dinov3_x  | 640  | aiq   | 16    |  64     | 28,524 | 34,772 | 43,328  | 98,304  |  44%  |
| dinov3_x  | 640  | aiq   | 16    |  64†   | 25,785 | 41,219 | 52,735  | 98,304  |  54%  |
| dinov3_x  | 1280 | aiq   |  2    |   4     | 10,182 | 21,683 | 23,824  | 98,304  |  24%  |
| dinov3_x  | 1280 | aiq   |  4    |  16     | 19,142 | 23,609 | 50,808  | 98,304  |  52%  |
| dinov3_x  | 1280 | aiq   |  6    |  24     | 28,124 | 34,825 | 55,131  | 98,304  |  56%  |

`†` full_8cls (8 classes adds ~9 GB at peak vs 2-class config at same batch)

### Scaling rule: model overhead vs activation cost

From the two dinov3_x@640 points on aiq (8/GPU and 16/GPU):

```
ep0 overhead (no augmentation): 13,400 MB  (consistent across both)
ep4 peak at batch=8:  17,305 MB  → activation component = 3,905 MB
ep4 peak at batch=16: 34,772 MB  → activation component = 21,372 MB
```

The activation component does NOT scale linearly (3.9 → 21.4 is ~5.5×, not
2×), which suggests the Mosaic augmentation's cache and composite image
generation contribute significantly beyond just the batch gradient tensors.

### Three-point 1280px memory model (corrected 2026-06-22)

Three measured data points for dinov3_x@1280 on Blackwell 96 GB:

| b/gpu | total_b | ep0 MB  | ep4 MB  | peak MB | peak epoch |
|-------|---------|---------|---------|---------|------------|
|  2    |   4     | 10,182  | 21,683  | 23,824  | ep15+      |
|  4    |  16     | 19,142  | 23,609  | 50,808  | ep20+      |
|  6    |  24     | 28,124  | 34,825  | 55,131  | ep23+      |

The linear model fitted to the first two points (`peak ≈ −3,160 + 13,492 × b`)
predicted 77,792 MB at batch=6 — the actual was 55,131 MB, a 41% overestimate.

**Why the linear model fails at 1280px**: memory grows in discrete step-wise
jumps during training, not at a predictable per-batch rate. The per-epoch peak
memory for the batch=4 and batch=6 runs tracks almost identically from ep3–ep13,
then batch=6 pulls ahead slowly through late training. The initial Mosaic phase
(ep0–3) scales more directly with batch (compositing 4×batch images), but the
steady-state peak is dominated by later-epoch activation patterns that grow more
slowly with batch size.

**Practical rule for 1280px dinov3_x on Blackwell 96 GB:**
- batch=4/GPU → peak ~51 GB (52%)  ← measured
- batch=6/GPU → peak ~55 GB (56%)  ← measured
- batch=8/GPU → actual headroom is roughly 40 GB; batch=8 may be feasible
  but the non-linear growth makes a firm prediction unreliable without a test run

The safe ceiling on 45-epoch steady-state is 55 GB at batch=6. For planning
purposes, treat batch=6 as the verified safe point and batch=8 as untested.

### Number of classes and memory

Comparing pup_vs_nonpup (2 classes) vs full_8cls (8 classes) at dinov3_x@640,
batch=16/GPU, 4 GPUs:

| config        | classes | peak MB | vs 2-class |
|--------------|---------|---------|-----------|
| pup_vs_nonpup | 2      | 43,328  | baseline  |
| full_8cls     | 8      | 52,735  | +9,407 MB |

The 8-class decoder adds ~9.4 GB at this batch size.  Class count affects the
classification head and matcher, not the backbone or encoder — so the overhead
is relatively flat with respect to resolution.

---

## The v1–v6 OOM era (hgnetv2_n on arisia)

The early arisia runs spent most of their job attempts failing with OOM before
finding the stable operating point.  Reconstructed from the catalog:

| version | batch/gpu | peak_MB | vram_MB | result     |
|---------|-----------|---------|---------|------------|
| v1      | 32        | —       | 49,152  | killed (time limit) |
| v2      | 128       | —       | 49,152  | killed (time limit) |
| v3      | 64        | —       | 49,152  | OOM ep=1   |
| v4      | 48        | —       | 49,152  | OOM ep=1   |
| v5      | 32        | —       | 49,152  | OOM ep=6   |
| v6      | 16        | ~13 GB  | 49,152  | **completed** |

The OOMs happening at ep=1 (not ep=0) are consistent with the memory
analysis: ep=0 runs lean (no Mosaic), ep=1 starts the augmentation ramp, and
the peak only hits at ep=4.  v5 with batch=32 survived through ep=6 (past the
Mosaic activation) before OOMing.  The working point for hgnetv2_n on arisia
is batch=16/GPU (≈13 GB at 320px with a 48 GB GPU).

---

## What the catalog does NOT cover

- **Tiled eval for any 1280px run**: gen006 2-GPU (job 23), gen006 4-GPU (job 25),
  and gen007 (job 26) all have only standard/whole-image eval results.  Tiled eval
  at 1280px would show higher AP-small for pups.
- **per-epoch LR**: captured in step logs but not extracted into the catalog.
  LR schedule diagnostics are in `2026-06-21_gpu_memory_analysis_and_4gpu_design.md`.
- **namek best_stg2.pth**: the `pup_vs_nonpup_deimv2_dinov3_s_1gpu_namek_gen006_sampler`
  partial run has no eval result in the catalog; checkpoint scoring pending.

---

## 4-GPU scaling comparison (standard/whole-image eval, updated 2026-06-22)

Three 4-GPU variants of the X@1280 run:

| job | run | b/gpu | total_b | LR   | ep | nocls_AP | pup_AP | wall_h |
|-----|-----|-------|---------|------|----|----------|--------|--------|
| 23  | gen006 2-GPU | 2 | 4  | 4e-4 | 30 | 0.899 | 0.875 | 20.5 |
| 25  | gen006 4-GPU | 4 | 16 | 4e-4 | 45 | 0.882 | 0.869 |  9.9 |
| 26  | gen007 4-GPU | 6 | 24 | 1e-3 | 45 | 0.892 | 0.870 |  8.3 |

Gen007 (sqrt-scaled LR) recovered most of the gen006 4-GPU regression but
remains a few points below the 2-GPU baseline on standard eval.  All three
need tiled eval for a definitive comparison.  The LR/batch interaction is
likely still the limiting factor — gen007 used sqrt(6)×LR; linear scaling
(6×LR = 2.4e-3) was not tested.
