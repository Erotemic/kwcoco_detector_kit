# 2026-06-01 — gen004: class-balanced JPEG runs

## Context

[[2026-06-01_gen002_three_scheme_results]] showed gen002/gen003
under-trained pup because the universal corpus is ~80% background
and ~1-2% pup. The kit ships two backends now:

1. **WDS** — `bucket_weights` re-weights at sample-pick time. The
   underlying `WeightedChunkMix` had a latent dead-source bug that
   would have silently neutralized any non-uniform weighting until
   it was fixed today (kwcoco_dataloader `759fdcf`). Now correct,
   but never validated end-to-end against a non-uniform
   distribution in production.

2. **JPEG** — `balance_mscoco` duplicates image entries in the
   on-disk MSCOCO at composition time. Shipped today
   (kit `4c338c8`); 6 E2E tests pass; no DEIMv2 changes required.

User's call (this session): for gen004, use the JPEG backend. It's
what we just tested, the balanced MSCOCO is a literal file you can
diff between recipes, and SSD-resident JPEG is faster than
SSD-resident WDS anyway ("Wow! we are cooking on the GPU now!" —
job 2565 on SSD-served data, gen003).

## The two gen004 runs

Both target `pup_vs_nonpup` (P1, binding constraint). Same
balance target so Run 2 - Run 1 isolates "bigger model + 2-GPU"
from "balance alone."

### Run 1 — ablation (1 GPU)

`submit_train_pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_gen004_balanced.sh`

| Knob | Value | Note |
|---|---|---|
| model | deimv2_hgnetv2_n | same as gen003 |
| input | 320 × 320 | same as gen003 |
| total batch | 16 (= 16/GPU × 1) | same as gen003 |
| LR | 5.66e-4 | same as gen003 |
| backbone_lr | 2.83e-4 | same as gen003 |
| epochs | 30 | same as gen003 |
| backend | JPEG (CocoDetection) | gen003 was WDS |
| balance target | `{<empty>: 0.4, pup: 0.2, nonpup: 0.4}` | NEW |
| target_size | len(src) | default |

This is the cleanest possible ablation: only data composition
moves. If it beats gen002's 0.025, the lift is unambiguously from
balancing.

### Run 2 — bigger leap (2 GPUs)

`submit_train_pup_vs_nonpup_deimv2_hgnetv2_s_2gpu_arisia_gen004_balanced.sh`

| Knob | Value (delta vs Run 1) |
|---|---|
| model | **deimv2_hgnetv2_s** (~3-4× params) |
| GPUs | **2** |
| total batch | **32** (= 16/GPU × 2) |
| LR | **8.00e-4** (sqrt(2) × Run 1's 5.66e-4) |
| backbone_lr | **4.00e-4** (sqrt(2) × Run 1's 2.83e-4) |
| balance target | SAME as Run 1 |

Three independent improvements stacked on Run 1:
capacity, gradient stability, and continued data balancing.

## Wiring

Three small edits land in this commit:

* `kwcoco_detector_kit/data/balance_mscoco.py` already exists
  (`4c338c8`); no change.

* `projects/.../scripts/_launch_train.sh` — new step 2c, runs
  between apply_scheme and the sweep when
  `KCD_BALANCE_TARGET_JSON` is set. Exports apply_scheme'd
  kwcoco -> MSCOCO, runs balance_mscoco, repoints TRAIN_KWCOCO at
  the balanced .mscoco.json (which `_ensure_mscoco` passes
  through unchanged).

* `projects/.../scripts/paths.sh` + `fetch_pretrained.sh` — add
  `deimv2_hgnetv2_s` entries (HF repo
  `Intellindust/DEIMv2_HGNetv2_S_COCO`). Run-2 prerequisite.

## Pre-submit checklist

1. Confirm gen003 single_sealion 2565 (currently training) has
   finished or will be done before submitting. We have 3 GPUs
   available; if 2565 is still using one, hold Run 2.
2. Fetch hgnetv2_s pretrained:
   ```
   bash projects/viame_sealions_2026/scripts/fetch_pretrained.sh deimv2_hgnetv2_s
   ```
3. (Optional) Pre-warm the apply_scheme+balance steps locally so
   the slurm job starts straight into training:
   ```
   KCD_DATA_PREP_ONLY=1 bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_gen004_balanced.sh
   ```
   (this still tiles + applies scheme; the balance step is per-run
   and runs inside the slurm job to avoid stale artifacts).
4. Submit Run 1 first, monitor for a few iters via
   `dev/check_resources.py`, then submit Run 2 once GPU pinning
   looks healthy.

## What to compare

Headline metric: **kit AP** (class-agnostic, NFS-excluded; see
[[feedback-detection-ap-is-selection-criterion]]). Diagnostic:
per-class AP, especially pup.

| Run | Pup AP target |
|---|---|
| gen002 pup_vs_nonpup baseline | 0.025 (already on disk) |
| gen004 Run 1 (balance only) | beat 0.025; goal ≥ 0.05 |
| gen004 Run 2 (balance + bigger + 2-GPU) | goal ≥ 0.10 |

If Run 1 doesn't beat 0.025, the balance hypothesis is wrong and
we should re-examine whether pup_vs_nonpup needs a different lever
(resolution, multi-scale tiles, etc.) before re-running.

If Run 1 wins but Run 2 doesn't add a meaningful delta, capacity
isn't the bottleneck and we should not promote hgnetv2_s globally.

## Open questions / risks

* **Re-tile required?** No — gen003's universal tile cache is
  reused; only the MSCOCO export + balance step is new. Saves
  ~2h vs a re-tile.
* **target_size = len(src)** means each pup tile gets repeated
  ~10-20× per epoch. Augmentations (Mosaic with cache, MixUp,
  CopyBlend, photometric) provide stochastic differentiation, but
  there's a real risk of overfitting on the small pup pool.
  Watching the in-train COCO mAP curve for pup specifically will
  tell us early.
* **Validation set is NOT balanced** — evaluation is on the
  original composition, so we're measuring real-world AP, not
  inflated-by-balance AP.
