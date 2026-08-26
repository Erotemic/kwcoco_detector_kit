# Recipe extraction and DINO normalization, inbound from fish (2026-08-26)

Cross-project note. Two kit-level defects found while reviewing the fish
project change what this project's next run will do, and one of them concerns
HGNetv2-N directly.

Full account:
[the fish journal](../../../viame_fish_2026/docs/journals/2026-08-26_recipe_extraction_and_gen006.md).

## The phantom schedule was YOURS

`trainers/deimv2.py` carried `_UPSTREAM_AUG_POLICY_EPOCHS = (4, 78, 148)` over
150 epochs and scaled every variant from it. That is the **HGNetv2-N** recipe
-- this project's variant -- with its true total of **160** mis-transcribed as
150. So sea-lion runs were scaled from roughly the right schedule with a 6%
error in the denominator, while fish runs were scaled from a different backbone
family's schedule entirely.

The kit now reads the recipe out of the selected upstream config
(`trainers/_deimv2_recipe.py`) instead of carrying a copy. For hgnetv2_n that
resolves to `epochs 160, policy [4, 78, 148], mixup [4, 78], copyblend [4, 78],
stop_epoch 148, matcher_change 136, flat 78, no_aug 12`.

### An upstream typo you should know about

`deimv2_hgnetv2_n_coco.yml:68` sets `flat_epoch: 7800` against `epoches: 160`.
Under FlatCosineLR that means the LR **never anneals**. Every correctly
configured variant sets `flat_epoch` equal to the middle augmentation boundary
(dinov3_x 29/[4,**29**,50], dinov3_s 64/[4,**64**,120]), so N's intended value
is **78** and 7800 is a transcription slip.

The kit repairs it, narrowly -- only on that exact `(7800, 160)` pair, so a
future upstream error surfaces as a loud validation failure instead of being
absorbed. Reading it faithfully would have given every HGNetv2 run a constant
learning rate for its entire schedule, which the old `num_epochs // 2` did not.

## Three parameters this project has never emitted

- **`mixup_epochs` / `copyblend_epochs`.** DEIMv2 merges dicts recursively, so
  a 30- or 45-epoch sea-lion run inherited upstream's absolute `[4, 78]` and
  those augmentations never terminated. The clean final stage the recipe is
  built around has not happened in any run here.
- **`matcher_change_epoch`.** Inherited 136 from hgnetv2_n, unreachable on any
  schedule this project runs. Now scaled: 136/160 x 30 = **26** for a 30-epoch
  run.
- **`weight_decay`.** Hardcoded 1e-4. Correct for hgnetv2_n and for dinov3_s,
  but **wrong for dinov3_l and _x**, which use 1.25e-4. If this project trains
  an L or X variant, that was silently off.

## `stop_epoch` was pinned to 1

`train_policy=fixed` forced `collate_fn.stop_epoch = 1`. DEIMv2 uses that field
as the stage-1/stage-2 boundary, not merely as a multiscale switch, and
`best_stg1.pth` is only written while `epoch < stop_epoch` -- so it froze at
epoch 0 and DEIMv2's restore-the-best branch reset the **entire training state**
to epoch 0 on every non-improving eval.

The fish gen004 run took that hit at epochs 2, 7, 13 and 19 and never beat its
own epoch-1 score across 21 epochs.

**This project uses `multiscale`, not `fixed`**, so its `stop_epoch` came from
`num_epochs - 4` rather than 1 and the reload behaved. But `n - 4` is not
upstream's ratio either (148/160 = 0.925; at 30 epochs that is 28, not 26). The
recipe now supplies it.

## Normalization: NOT for this project

Upstream normalizes DINOv3 inputs with ImageNet mean/std in the config and in
every inference tool. The kit emitted `Normalize` nowhere, so every DEIMv2 run
in both projects fed raw `[0, 1]` tensors to a COCO checkpoint optimised for
normalized input.

The fix is **gated on family, deliberately**: all four `dinov3_*` configs
normalize; all eight `hgnetv2_*` do not (`base/deimv2.yml:104-105` goes straight
from `ConvertPILImage` to `ConvertBoxes`). Applying it to HGNetv2 would hand
`pup_vs_nonpup`'s checkpoint a distribution it never trained on -- the same
mistake in the other direction.

So `pup_vs_nonpup` on `deimv2_hgnetv2_n` is unaffected. A `deimv2_dinov3_s`
run WILL now normalize, and its checkpoints are not interchangeable with
pre-2026-08-26 ones. Predictor and export recover the contract from each run's
own `train.yml`, so old checkpoints keep being scored unnormalized
automatically.

## What to expect on the next run here

Requires an image rebuilt after `db1fda4`, `d97a148`, `c129329`, `449275f`.

- a genuinely terminating Mosaic/MixUp/CopyBlend phase, for the first time
- a reachable matcher change
- `flat_epoch` from upstream's ratio rather than `num_epochs // 2`
- a step **up** in loss when augmentation engages, and a step **down** when it
  terminates -- both are the recipe working, not regressions
- unchanged input normalization for hgnetv2 variants

## Also available

- **`--selection_journal`** stages every epoch for post-training reranking
  under the true-tiled protocol. The plumbing existed but `pareto_sweep` never
  passed it, so no run in either project has used it.
- **`true_tiled` is v2**: `per_window_nms` is part of the protocol identity now
  and pinned False. V1 fingerprints do not pin down how their numbers were
  produced, since the two call sites disagreed on the default -- treat old
  true-tiled fingerprints as ambiguous rather than comparable.
- **`run-health`** diagnoses a run from its slurm log; the abort-on-NaN guard
  fails a numerically dead run in minutes instead of letting it train to
  completion producing nothing.
