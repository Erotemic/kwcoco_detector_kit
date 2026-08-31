# 2026-08-31 — the tiling hypothesis, falsified on both splits

Four generations, one protocol, both splits. This is the entry to read before
proposing the next fish experiment.

## The measurement

True-tiled inference, 1229 px source window, overlap 0.25, keep_full,
cross-window NMS 0.5, bf16. Every checkpoint chosen on **vali**; test scored
afterwards and used for nothing.

| run | trained on | checkpoint | vali AP@0.5 | test AP@0.5 |
|---|---|---|---|---|
| **gen003** | whole frames | autoselect | **0.7689** | **0.7012** |
| gen001 | whole frames | autoselect | 0.7658 | 0.6981 |
| gen006 | 1229px tiles | epoch 3 | 0.7526 | 0.6958 |
| gen007 | tiles + seq/track balance | epoch 6 | 0.7311 | 0.6763 |

Vali stride 8 (a ranking); test the full 33,434-image split.

## Two things this settles

**1. Tile-scale training did not help. It hurt, monotonically.**

Both whole-frame runs beat both tile-trained runs, on both splits. The ordering
is not marginal — gen003 leads gen007 by 0.038 on vali and 0.025 on test.

This is the *opposite* of the premise gen005/gen006/gen007 were built on: that
training at native tile scale would close the resolution gap for small fish and
improve deployment performance. The tile-trained models see, at inference,
exactly the geometry they trained on. The whole-frame models see a scale they
never trained at. The whole-frame models still win.

**2. gen007 regressed against its own predecessor.**

gen007 was designed to improve on gen006 and came in 0.0215 lower on vali and
0.0195 lower on test. Sequence/track balancing, lighter augmentation and the
halved LR did not merely fail to help; together they made a worse model.

## The most useful lesson: the mechanisms all worked

gen007 is not a story about a broken implementation. Every mechanism did
precisely what it was designed to do, and was measured doing it:

- effective sequences per epoch **81 → 195**, effective tracks **2,901 → 5,461**
- duplicate draws per epoch **19.0% → 0**, zero cross-rank overlap
- negative fraction held at 21.1% against a corpus rate of 20.8%
- zero NaN across 34 epochs and 18h44m, where three earlier fp16 runs died
- the schedule resolved exactly as specified: policy [2, 15, 26], stop 26,
  matcher 23, tail 8, mixup/copyblend disabled

Every one of those was verified on the real corpus before launch. **The
intervention was a mechanical success and an outcome failure.** Tripling the
diversity of the sampled distribution bought nothing, and the accompanying
changes cost real accuracy.

That is the lesson worth carrying: a diversity metric improving is not evidence
that generalisation will improve. We had strong measurements of the *mechanism*
and no measurement at all of the *outcome* until the run finished.

## What was predicted, and what happened

| prediction | outcome |
|---|---|
| gen007 ≥ B + 0.01 = 0.7789 (success criterion, pre-registered) | 0.7311 — missed by 0.048 |
| ~78k primary updates needed, sized from gen006's peak | peaked at ~21k; the last 57k bought nothing |
| tiling closes the small-object resolution gap | whole-frame training wins on both splits |
| balancing the sampler improves generalisation | worse than the unbalanced predecessor |

## Two methodological results that DID hold

**Vali predicted test perfectly.** The rank order is identical on both splits,
and the vali→test gap is nearly constant (−0.057 to −0.068). The frozen
true-tiled vali protocol is a sound selection proxy — every decision made on it
would have been made the same way on test.

**Selecting under deployment geometry beat in-loop selection.** DEIMv2's own
tile-level validation picked gen007 epoch 27; deployment geometry picked epoch
6, which scores 0.0037 higher. For gen006, in-loop picked epoch 4 and
deployment geometry picked epoch 3. Small margins, but the staging-plus-
true-tiled-selection apparatus is doing real work and should be kept.

## The open question, which is now the most important one

For gen001 the *older, whole-image* eval reported **vali 0.8060 / test 0.7272
AP@0.5**. The true-tiled protocol reports **0.7658 / 0.6981** for the same run.
Tiled inference scores that model roughly 4 points LOWER on both splits.

If that survives scrutiny it means the tiling program was a net negative at
**inference** as well as at training — i.e. the whole pivot, not just the
training half. Before anyone acts on it, confirm that the two numbers are
actually comparable: they came from different eval entry points
(`_launch_export_score.sh` vs `run_kwcoco_eval`) and may differ in
`score_thresh`, `max_dets` or distractor handling. Cross-protocol comparison is
the exact mistake that made the earlier RF-DETR comparison useless, and this
entry should not be the place it gets repeated.

**Do not treat the −4 points as established.** Treat it as the next thing to
measure: one model, one eval code path, whole-image vs true-tiled.

## For the next brainstorm

1. The burden of proof has shifted. Tiling is not a neutral default here; it
   has lost twice. A proposal that keeps it needs to say why this evidence does
   not apply.
2. gen003's recipe — whole frames, 1024×1024, batch 32, ~94k updates — is the
   thing to beat, and nothing since has beaten it.
3. Saturation is early and consistent. gen006 peaked at epoch 3–4, gen007 at
   epoch 6 under deployment geometry. Long schedules have not paid off in any
   run; budget accordingly and stage every epoch.
4. Measure the outcome, not the mechanism. gen007's diversity statistics were
   excellent and irrelevant.

## Provenance

- vali: `baseline_vali/summary_w1229_o0.25_bf16_s8.json` (50 rows, 4 runs)
- test: `test_score/test_summary_w1229_o0.25.json`
- logs: `slurm_logs/test_scores_20260829_154141/`, `gen007_e2e_20260827_222454/`
- image `kit_sha a7ef134`, DEIMv2 `1e6339d`, kwcoco_dataloader `447fc78`

Test had already been scored twice and consulted for four decisions before
this; that history is in `training_runs.yaml:holdout_discipline`. These numbers
were taken for the record, after selection, and were used to decide nothing.
