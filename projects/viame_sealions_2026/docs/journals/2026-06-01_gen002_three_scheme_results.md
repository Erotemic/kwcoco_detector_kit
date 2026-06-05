# 2026-06-01 — gen002 results across all three schemes
> [!WARNING]
> **2026-06-05 retroactive correction.** Every training run referenced
> below used `training_ready_v1/` — a 1,314-image, 2021–2024-only,
> n_cats=1 subset of the actual corpus. The authoritative bundles at
> `/data/Public/VIAME/viame_sealions_2026/unpacked/*_norm.kwcoco.zip`
> have 6,462 train images across 14 years (2007–2024) with 9 proper
> kwcoco categories. All AP numbers in this journal are SUBSET-only
> and not comparable to gen005+ runs trained on the real corpus. The
> recipe-level findings (dinov3_s + balance beats hgnetv2_n, EMA
> beats best_stg1, multiscale OOMs at 2-GPU 512–768) likely
> generalize; the absolute AP numbers (kit AP 0.5xx, in-train mAP
> 0.1xx) do not. See `2026-06-05_corpus_audit_wrong_bundle.md`.


## Context

[[2026-05-30_gen002_wds_pipeline_shakedown]] got the WDS pipeline
working on `pup_vs_nonpup`. This entry records what training across
all three schemes (pup_vs_nonpup, single_sealion, lifestage_6cls)
actually produced. Headline: **gen002 helps where pup detection was
the binding constraint, hurts everywhere else.**

## Results

### Completed

| Run | Scheme | Path | Train time | Epoch reached | kit AP | best in-train mAP (epoch) |
|---|---|---|---|---|---|---|
| 2545 | single_sealion v5 | gen001 (CocoDetection) | 1d 19h | 30/30 | **0.177** | 0.0485 @ ep 11 |
| 2552 | pup_vs_nonpup gen002 | WDS (full corpus) | 18h | 30/30 | **0.025** | 0.0073 @ ep 21 |

### Live (epochs 26-29 of 30; rsynced 2026-06-01 14:27)

| Run | Scheme | Path | Best so far (in-train mAP) | Projected kit AP* |
|---|---|---|---|---|
| 2558 | single_sealion gen002 | WDS (full corpus) | 0.0120 @ ep 21 | ~0.044 |
| 2561 | lifestage_6cls gen002 | WDS (full corpus) | 0.0056 @ ep 19 | ~0.020 |

*Projected = `in_train_mAP × (v5's kit_AP / v5's best_in_train) = in_train × 3.65`.
Crude, but the kit's NFS-excluded class-agnostic eval has been roughly
3-4× the COCO-style mAP across all our prior runs.

### Update 2026-06-01 18:36 — single_sealion gen002 finished

| Run | Train time | best in-train mAP | **actual kit AP** | projected | error |
|---|---|---|---|---|---|
| 2558 single_sealion gen002 | 27h | 0.0120 @ ep 21 | **0.024** | ~0.044 | 1.8× overestimate |

The projection was way off — the kit-AP-to-mAP ratio was **~2×** for
gen002, not the ~3.65× we saw for gen001 v5. That's its own finding:
gen002 produces predictions that are more uniformly low-confidence
(so the kit's NFS-exclusion + class-agnostic re-eval can't promote
them as much as it can for gen001 detections). Likely a downstream
effect of the corpus-composition imbalance — the model isn't
confidently detecting anything.

So the corrected verdict for single_sealion is **0.024 vs v5's 0.177 = 86% regression** (7.4× worse), not 4× worse. Gen002 is much more strongly contraindicated for single_sealion than the in-train mAP alone would suggest.

### Zombies

| Run | Scheme | Stuck since | Cause |
|---|---|---|---|
| 2553 | single_sealion gen002 | 2026-05-30 17:49 | hung at iter 0 — likely the GPU-collision bug fixed in `03dfa2e` |
| 2556 | lifestage_6cls gen002 | 2026-05-31 11:29 | same pattern |

Both have been holding GPU memory hostage for days. Need scancel +
docker stop.

## Key research findings

### Finding 1: gen002 improves pup_vs_nonpup, regresses single_sealion

The memory entry [[project-pup-is-binding-constraint]] said pup AP
was ~0.01 at hgnetv2_n+320 and needed resolution+capacity. gen002
delivered: pup_vs_nonpup went from ~0.01 (gen001) to **0.025**
(gen002). 2.5x lift, explained by gen002's full-resolution tiles
surfacing more pup samples that gen001's subsampled tile bundle was
missing.

But single_sealion REGRESSED hard: v5 (gen001) was **0.177**;
gen002's projected kit AP is **~0.044** — a 4x loss. lifestage_6cls
projects to **~0.02**, also weak.

### Finding 2: the trade-off is class balance, not pipeline

gen002 trains on the full universal tile corpus (~766k samples) vs
gen001's curated subset (~14k iters/epoch worth). The full corpus
contains many more "easy" background-only tiles. For schemes where
positives were already abundant (single_sealion sees any sealion
as a positive, lifestage_6cls has 6 classes covering most sealion
appearances), the extra negatives **dilute** the training signal
and hurt class-agnostic detection AP. For pup_vs_nonpup, where pup
was rare, more pup-containing tiles outweigh the background dilution.

So gen002 is the right path WHEN the binding constraint is "not enough
positives," and the wrong path when "lots of negatives crowd out
useful gradient." The WDS streaming isn't the problem; the corpus
composition is.

### Finding 3: per-iter speed of WDS is genuinely faster

After warmup, gen002's WDS path runs at ~0.05 s/iter steady state
(vs v5's gen001 path at ~0.30 s/iter). Even with 3.3× more iters
per epoch, gen002 finishes faster wall-clock (18h vs 43h for
single_sealion). The WDS reader's HDD-friendly sequential streaming
+ page cache absorbs the entire corpus working set on arisia (126G
RAM, ~110G of tile data) by epoch 2-3, so all subsequent epochs are
served warm.

### Finding 4: in-train COCO mAP isn't the right metric

Notice the 4× ratio between in-train mAP (with distractors like NFS
counted as positives) and the kit's NFS-excluded class-agnostic AP.
This held for both completed runs. The in-train COCO eval also
selects checkpoints that aren't optimal under the kit's eval, per
[[2026-05-29_per_checkpoint_rescoring_results]]. Worth re-running
the rescore tool on 2558 and 2561 when they finish to find the
ACTUAL best checkpoint instead of just trusting `best_stg2.pth`.

## What to do next

1. **Wait for 2558 + 2561 to finish** (~few hours each). Then run
   `scripts/submit_rescore_per_checkpoint.sh` on both to confirm
   the projected kit APs and find the real best checkpoint per run.
2. **Kill 2553 and 2556**. They're zombies; the docker containers
   are holding GPU memory but making no progress.
3. **Don't promote gen002 globally**. The data composition issue
   needs solving first. Options to investigate before the next round:
   - Subsample backgrounds in the WDS shards (skip empty tiles
     during tiling, or build a balanced shard set)
   - Use `distractor_classes` plumbing to downweight common
     classes during training, not just at eval
   - Two-stage curriculum: pretrain on full WDS (broad coverage),
     finetune on gen001's balanced subset
4. **Promote gen002 for pup_vs_nonpup only** — it's a real win
   there. The journal entry for that should call out 0.025 as the
   new baseline.
5. **Don't bother resubmitting single_sealion or lifestage gen002**
   under the current corpus until corpus balance is fixed. v5 is
   the better single_sealion model.

## State at end of session

- 2545 single_sealion v5: DONE, AP=0.177 (the model to deploy/compare)
- 2552 pup_vs_nonpup gen002: DONE, AP=0.025 (gen002 baseline for pup)
- 2558 single_sealion gen002: epoch 29, ~30 min remaining
- 2561 lifestage_6cls gen002: epoch 26, ~3h remaining
- 2553, 2556: zombies, kill them

## How to resume

When 2558 and 2561 finish, rerun rescoring + add their final kit AP
to the table above. Compare the rescored-best-checkpoint AP to
`best_stg2.pth`'s AP for both; if rescoring picks an earlier epoch
significantly higher AP, the in-train COCO eval is misselecting
checkpoints and we should write a kit eval-during-training that
matches the deployment metric (NFS-excluded class-agnostic AP).
