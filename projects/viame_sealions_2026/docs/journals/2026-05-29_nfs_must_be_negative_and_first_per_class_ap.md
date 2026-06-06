# 2026-05-29 — NFS must count as a negative; pup is the binding constraint
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

The v6 (pup) and v4 (lifestage) runs finally completed (pup_v6 to
completion; lifestage_v4 walltime'd mid-epoch 29 of 30). When the
`detect_metrics.json` artefacts were rsynced back to namek and read
properly, the per-class AP picture revealed two things:

1. The headline "test AP" we'd been reading off the eligibility manifest
   was **class-agnostic** (`nocls_measures.ap`) — not per-class. That
   explains the suspicious ~5x gap vs the in-training vali
   `coco_eval_bbox` we'd been comparing against. (Vali eval is
   per-class CocoEvaluator; "test AP" in the manifest is nocls.)
2. The per-class breakdown shows **pup AP ~0.010 in both schemes**,
   confirming pup-detection-at-hgnetv2_n+320 is essentially at floor.

## Per-class AP from the rsynced eval artifacts

**pup_v6** (`pup_vs_nonpup` scheme):

| class | AP | realpos |
|---|---|---|
| pup | 0.0104 | 1547 |
| nonpup_sealion | 0.2630 | 4610 |
| class-agnostic (nocls) | 0.1987 | — |

**lifestage_v4 best_stg2 (epoch 23)** (`lifestage_6cls` scheme):

| class | AP | realpos |
|---|---|---|
| bull | 0.4406 | 642 |
| female | 0.1790 | 1826 |
| subadult_male | 0.1167 | 317 |
| northern_fur_seal | 0.0674 | 3652 |
| juvenile | 0.0285 | 1825 |
| pup | 0.0104 | 1547 |
| class-agnostic (nocls) | 0.0943 | — |

## Decision: NFS counts as a negative

The user's operational rule (added to agent memory:
[[feedback-nfs-always-negative]] and
[[feedback-detection-ap-is-selection-criterion]]):

> NFS (northern_fur_seal) is in the training data only so the model
> can learn to *separate* fur seals from sea lions — NFS predictions
> are not a positive operational signal. When scoring detection
> ability, drop NFS from both the predictions and the ground truth
> before computing AP.

Implication for the lifestage_v4 results above: the class-agnostic AP
of 0.0943 includes NFS as a positive, which means it credits NFS
detections towards the "model finds sea lions" metric. After NFS is
excluded, the true sea-lion detection AP for lifestage_v4 will be
different (probably lower — NFS is the second-most-populous class in
that scheme so removing it reweights everything).

Recomputed scores land in a separate artefact next to this journal
entry; see the recompute script in `scripts/recompute_sealion_ap.py`.

## Bug found: `single_sealion` scheme maps NFS → sealion

[docs/class_schemes.yaml::single_sealion.mapping](../class_schemes.yaml)
currently has:

```yaml
mapping:
  B:   sealion
  S:   sealion
  F:   sealion
  J:   sealion
  P:   sealion
  NFS: sealion      # <-- bug under the new rule
  DP:  sealion
```

`single_sealion_v4` (the run that OOM'd) was trained against this
scheme, so the 15,965 NFS annotations were treated as positive sea-lion
targets. Re-running `single_sealion` (planned as v5 at batch=16) needs
this scheme fix first — either drop NFS entirely from the scheme
mapping or treat it as background. The natural choice is "drop" since
`single_sealion` has no separate-class slot to put NFS in.

The `pup_vs_nonpup` scheme already drops NFS at the scheme level
(`drop: [NFS, O, DP, DN]`) so the pup_v6 metrics above are *not*
affected by NFS — they're already NFS-clean. The class-agnostic AP of
0.199 for pup_v6 is the honest detection AP.

The `lifestage_6cls` scheme keeps NFS as a separate target class,
which is *correct training behaviour* under the new rule — the model
should learn NFS as its own class to differentiate from sea lions.
The fix is at *eval time*, not in the scheme: drop NFS from GT+pred
before AP is computed.

## What changed in the eval pipeline

`kwcoco_detector_kit/orchestration/eligibility.py::_find_eval_ap` reads
`nocls_measures.ap` directly from `detect_metrics.json`. The function
is the right shape, but the *upstream* eval step (whatever writes the
metrics json) needs to be configured to construct the GT+pred kwcocos
with NFS dropped before passing them to the scorer.

Specifically: anywhere the kit invokes `kwcoco eval ...` for a
checkpoint trained on `lifestage_6cls`, the GT and pred bundles need
NFS-class annotations filtered first. Easiest place to put this is in
the per-scheme eval step itself — read the scheme's `drop` list plus
the operational "exclude from detection AP" list (NFS, plus any future
non-target classes) and prune before scoring.

## Lessons

- **Headline metric was structurally misleading.** "Test AP 0.199"
  shown by the manifest is class-agnostic. The
  in-training `coco_eval_bbox` is per-class. They're computing
  different things on different data; comparing them as the same
  metric obscured what the model was actually learning. Future
  reporting should label which AP is which.
- **The scheme is the metric.** Once we knew the data carries NFS,
  every downstream AP number is conditional on whether NFS is in or
  out. The class scheme YAML defines training behaviour, but it
  doesn't fully define *eval behaviour* — we need an explicit
  "exclude from detection AP" list per scheme. Without it, the eval
  step silently scores NFS as a positive.
- **Pup AP=0.010 is reproducible across the two schemes**, which
  rules out "scheme-specific bug" and points squarely at
  resolution + model-capacity. Saved as
  [[project-pup-is-binding-constraint]].
- The user explicitly does **not** want to scale to a bigger
  backbone until the metric is fixed and the existing scores are
  recomputed. Smaller models are also on the table as the test bed.
