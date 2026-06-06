# 2026-05-29 — Per-checkpoint vali rescoring under the corrected metric
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

[[2026-05-29_nfs_must_be_negative_and_first_per_class_ap]] established
that the test-time eval needed NFS pruned before scoring. The
follow-on question: was the *checkpoint* the kit picked
(best_stg2.pth) actually the best one under the corrected metric, or
did DEIMv2's in-train CocoEvaluator (which uses the with-distractor
metric) bias selection towards the wrong epoch?

This was the user-accepted "Option B" from earlier — save more
checkpoints + add a re-eval tool. The save-more-checkpoints half is
deferred (DEIMv2's checkpoint save logic gates on `epoch <
stop_epoch=1` so today we only get best_stg1, best_stg2, and last per
run); the re-eval tool shipped today.

## What we ran

`projects/viame_sealions_2026/scripts/submit_rescore_per_checkpoint.sh`
on lifestage_6cls v4 and pup_vs_nonpup v6, against their respective
vali kwcocos. The script loads each of the three saved checkpoints
via `DEIMv2Predictor`, runs full inference on vali, then scores both
with the standard kwcoco eval and (when the scheme has distractors)
the kit's distractor-pruning second pass. Slurm jobs 2519 and 2520
on arisia A6000s; each ran in ~5 min.

## Results

**lifestage_6cls_v4 (vali, NFS as distractor):**

| ckpt        | nocls AP (full) | nocls AP (NFS pruned) |
|-------------|----------------:|----------------------:|
| best_stg1   | 0.0419          | 0.0466                |
| **best_stg2** | **0.0862**    | **0.0924**            |
| last        | 0.0419          | 0.0466                |

Per-class AP at best_stg2 (vali, NFS pruned):
- bull 0.265, female 0.057, subadult_male 0.050, juvenile 0.011,
  **pup 0.000**.

**pup_vs_nonpup_v6 (vali, no distractors):**

| ckpt        | nocls AP (full) |
|-------------|----------------:|
| best_stg1   | 0.0772          |
| **best_stg2** | **0.1578**    |
| last        | 0.0772          |

Per-class AP at best_stg2 (vali):
- nonpup_sealion 0.102, **pup 0.000**.

## Findings

1. **DEIMv2's in-train selection agrees with the corrected metric.**
   best_stg2 is the right pick on both runs, under both the full and
   the distractor-pruned metric. The classification penalty in the
   with-distractor metric didn't push DEIMv2 towards a worse
   checkpoint, at least for these two runs. Saves us from needing to
   patch DEIMv2's CocoEvaluator (the deferred "Option C") for now.

2. **`last.pth` is misnamed in our setup — it's effectively
   epoch-0.pth.** Both runs show `last.pth` AP == `best_stg1.pth` AP
   to the nearest 0.0001. Reason: DEIMv2's save loop at
   `tpl/DEIMv2/engine/solver/det_solver.py:106` only updates
   `last.pth` when `epoch < self.train_dataloader.collate_fn.stop_epoch`,
   and our `fixed`-policy training sets `stop_epoch=1`. So `last.pth`
   only got written at epoch 0 and was never updated as training
   continued. The slurm walltime kill therefore didn't preserve a
   late-epoch state — the resume from `last.pth` started from epoch 0,
   not from epoch 23 as we'd assumed.

   This has two downstream implications:
   - **Resume semantics for our config are different than we
     expected.** Job 2515 (the lifestage resume) is *retraining* from
     epoch 0, not finishing the last few epochs. It will eventually
     get back to best_stg2-quality somewhere around epoch 20-25 again.
   - **Per-epoch checkpoint coverage is even worse than 3 candidates
     per run** — really it's 2 distinct states (best_stg1 ≈ last,
     plus best_stg2). The Option-B "more checkpoints" work needs a
     DEIMv2 patch (lift the `epoch < stop_epoch` guard on the
     intermediate save) to actually deliver more snapshots.

3. **Vali AP is lower than test AP on both runs, by similar ratios.**
   - lifestage: vali 0.092 vs test 0.116 (distractor-pruned)
   - pup: vali 0.158 vs test 0.199

   So our test set is genuinely a bit easier than vali — small NOAA
   bundle that gets validation/test pre-split, presumably non-iid.
   Both numbers move together so this isn't masking a metric bug.

4. **Pup AP = 0.000 on vali too.** Pup remains the binding constraint
   ([[project-pup-is-binding-constraint]]). Same finding from two
   different splits, both schemes. Resolution + capacity wall, not a
   hyperparameter problem.

## Lessons

- **Re-eval is cheap; rely on it.** ~5 min on one GPU per run to
  score every saved checkpoint. Now in the toolkit
  (`per_checkpoint_eval.py` + `rescore_per_checkpoint.py` +
  `submit_rescore_per_checkpoint.sh`). Use it after every training
  run before drawing conclusions about checkpoint quality.
- **`last.pth` is not what we thought.** When resuming a walltime'd
  run, the resume from `last.pth` may regress to epoch 0 state. This
  is a DEIMv2-config interaction (stop_epoch=1 in our fixed policy),
  not a kit-level bug. Workaround until the deferred Option C: copy
  `best_stg2.pth` over `last.pth` before resuming if we want to
  actually pick up from the best mid-training state. Or wait for
  the Option B / C patches.
- **DEIMv2's selection isn't measurably broken by the distractor
  metric** in our current runs, but we can't generalize from 2 data
  points. Future runs should still get the rescore pass to verify.

## Code refs

- `kwcoco_detector_kit/eval/per_checkpoint_eval.py::score_one_checkpoint`
- `projects/viame_sealions_2026/scripts/rescore_per_checkpoint.py`
- `projects/viame_sealions_2026/scripts/submit_rescore_per_checkpoint.sh`
- Slurm jobs: 2519 (lifestage_v4 vali) → 51 KB log;
  2520 (pup_v6 vali) → 21 KB log.

## Next

1. Let job 2515 (lifestage resume) and 2508 (single_v5) finish.
2. Rescore single_v5 against vali once it finishes — gives us a
   matched-config 1-cls baseline for comparison.
3. Once the lifestage resume completes (~30h from epoch 1), rescore
   the new best_stg2. Quite possible it won't beat the original
   best_stg2 from job 2504, since we're essentially redoing the same
   training run from epoch 0.
4. Consider the deferred DEIMv2 patch
   ([[project-pup-is-binding-constraint]]) to lift the
   `epoch < stop_epoch` guard on intermediate saves and on `last.pth`
   updates. That's a 1-line change in the submodule fork; once it
   lands, future runs save real per-epoch state and the rescore tool
   immediately becomes useful for picking the right epoch.
