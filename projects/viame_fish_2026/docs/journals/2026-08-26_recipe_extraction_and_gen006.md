# The kit was scaling a schedule that does not exist (2026-08-26)

An external review of the gen005 plan turned into six rounds of code review and
found more wrong with the kit's DEIMv2 integration than the whole gen001-gen005
sequence had. gen006 is the first fish run configured from DEIMv2's actual
recipe rather than from constants the kit had accumulated.

Status at time of writing: **gen006 (slurm 498) is at epoch 8 of 14, healthy**,
no NaN aborts, no checkpoint reloads, ~13 h remaining.

## The root finding

`trainers/deimv2.py` carried `_UPSTREAM_AUG_POLICY_EPOCHS = (4, 78, 148)` over
150 epochs and scaled every run from it. That schedule is not DINOv3-X's. It is
**HGNetv2-N's** -- the sea-lion project's variant -- with its true total of 160
mis-transcribed as 150. DINOv3-X is `[4, 29, 50]` over **58**.

Each variant has its own totals, so no single constant could ever have been
right:

| variant | epochs | policy | flat | wd |
|---|---|---|---|---|
| dinov3_x | 58 | `[4, 29, 50]` | 29 | 1.25e-4 |
| dinov3_l | 68 | `[4, 34, 60]` | 34 | 1.25e-4 |
| dinov3_m | 102 | `[4, 49, 90]` | 49 | 1e-4 |
| dinov3_s | 132 | `[4, 64, 120]` | 64 | 1e-4 |
| hgnetv2_n | 160 | `[4, 78, 148]` | 78 | 1e-4 |
| atto/femto/pico | 500 | `[4, 250, 400]` | 250 | 1e-4 |

Three more parameters were flattened or missing:

- **`weight_decay` hardcoded to 1e-4.** DINOv3-L and -X use 1.25e-4.
- **`mixup_epochs` / `copyblend_epochs` never emitted.** DEIMv2 merges dicts
  recursively, so a 12-epoch run silently inherited upstream's absolute
  `[4, 29]` / `[4, 50]` and those augmentations never terminated -- the clean
  final stage the recipe is built around could not happen even when the
  transform policy said it should.
- **`matcher_change_epoch` never emitted.** Inherited 45, unreachable on every
  schedule the kit has ever run. gen006 is the first fish run that will reach a
  matcher change at all.

Fixed by reading the recipe out of the selected upstream config and rescaling
it (`trainers/_deimv2_recipe.py`), rather than adding a per-variant table --
a table fixes today's numbers and preserves the failure mode.

## Missing input normalization

Upstream normalizes DINOv3 inputs with ImageNet statistics in the config
(`deimv2_dinov3_x_coco.yml:77`) and in every inference tool
(`tools/inference/{torch,onnx,trt}_inf.py`). The kit emitted `Normalize`
**nowhere** -- train, val, and both predictor paths fed raw `[0, 1]` tensors.
Every DEIMv2 run this project has done, fish and sea lion, handed a COCO
detector optimised for normalized DINO input a different distribution.

The backbone does not compensate: the only `normalize` in
`engine/backbone/dinov3/` is `pos_embed_rope_normalize_coords`, which is
positional-embedding coordinate handling.

It is **DINOv3-only**. All four `dinov3_*` configs normalize; all eight
`hgnetv2_*` do not. An ungated fix would have broken the sea-lion
`pup_vs_nonpup` recipe by normalizing a backbone that never trained that way.

Predictor and ONNX export now RECOVER the contract from the run's own generated
`train.yml` rather than re-deriving it from the variant, which makes parity
structural: a checkpoint trained before this change is scored and exported
without normalization, automatically, with no version flag to get wrong.

## `stop_epoch = 1` was three bugs

`train_policy=fixed` pinned `collate_fn.stop_epoch = 1`, treating it as a "stop
varying input size" switch. DEIMv2 uses it as the stage-1/stage-2 boundary.

1. **The gen002 NaN.** With the policy scaled, `e0` clamped to 1 for every
   schedule under ~80 epochs, so augmentation switched on in the same epoch the
   optimizer, GradScaler and EMA were reloaded. Fixed earlier by decoupling.
2. **The gen004 plateau.** `best_stg1.pth` is only written while
   `epoch < stop_epoch`, so it was frozen at epoch 0 forever, and DEIMv2's
   restore-the-best branch reset the entire training state to epoch 0 on every
   non-improving eval. gen004 took that at epochs 2, 7, 13, 19 and never beat
   its own epoch-1 score in 21 epochs.
3. **Misleading provenance.** `policy.json` reported `stop_epoch: 1` while
   training used something else entirely.

gen006 derives it from the recipe: **12 of 14**. `best_stg1.pth` now
accumulates the genuine best of epochs 0-11, and the reload cannot fire before
epoch 12. **Zero reloads through 8 epochs**, against gen004's four in 19.

## Corrections to earlier entries

- The collision fix that forbade **any** augmentation boundary from coinciding
  with `stop_epoch` was over-broad. Upstream deliberately sets the FINAL
  boundary equal to it (both 50 for X); only the boundary that turns
  augmentation ON must stay clear. atto/femto/pico decouple them entirely
  (stop 468 against e2 400).
- The `best_stg2.pth` reload preference added to the fork was a workaround for
  `stop_epoch=1`. With the configuration fixed at source, upstream's
  `best_stg1.pth` behaviour is correct and the fork should not carry a
  divergence. Reverted; the NaN fail-fast guard stays.
- A first attempt at generic scaling **rebuilt** mixup/copyblend/stop_epoch
  from the policy boundaries. That is right for DINOv3 and wrong for four of
  twelve variants: hgnetv2_n's copyblend is `(4, 78)` against e2=148, and
  atto/femto/pico ship `(40000, 15000)` -- start after end, i.e. DISABLED --
  which rebuilding would have switched on. Every field is now scaled from its
  own value, and the disabled sentinel is preserved by NOT clamping.

## bf16, on evidence rather than theory

fp16 is DEIMv2's native precision and the kit default was restored to it after
gen003. Three fp16 runs then aborted on non-finite `pred_boxes`, all inside an
augmented epoch, at stable loss:

| run | dtype | aug on | NaN |
|---|---|---|---|
| gen001 | fp16 | e4 | e4 step 7500 |
| gen002 | fp16 | e1 | e1 step 7700 |
| gen005 | fp16 | e2 | e3 step 2874 |
| gen003 | bf16 | e2 | none, 22 augmented epochs |

gen006 pins bf16 for this run while the kit default stays fp16 -- one project's
experience is not enough to move a default. The GradScaler is disabled under
bf16, since it exists to keep fp16 gradients out of underflow.

## Two bugs the tooling caught, and one it did not

The Docker build gate (`RUN_TESTS=1`) caught two `NameError`s and one API
regression before any of them reached a GPU. It is worth the build time.

It did **not** catch that `pareto_sweep` passed `workdir/"staging"` as
`selection_journal_dpath` when `RunJournal` reads `workdir/journal/`. That one
is quiet in a specific way: `det_solver` derives staging as
`<journal_dir>.parent/'staging'`, so `staging.parent/'staging'` is the same
directory -- every checkpoint landed correctly and only the journal was
misplaced. gen006 would have staged all 14 epochs, written 11 GB, and the
reranker would have found nothing, with no error anywhere.

`dev/check_undefined_names.py` was added to catch the NameError class locally,
since this dev host has no pytest, numpy or kwconf. Making it usable took three
passes -- naive scoping flagged 46 legitimate closures, then lambda parameters
-- and it failed its own selftest once. A module-level extension caught the
target bug zero times while producing nine false positives and was removed
rather than shipped.

## gen006

```
fresh COCO DINOv3-X, normalized input, bf16, GradScaler off
495,514 native 1229px tiles -> 1024 model input
global batch 32 (upstream's LR/batch pairing), 14 epochs
lr 5e-4 / backbone 1e-5 / weight_decay 1.25e-4
policy [1, 7, 12]  mixup [1, 7]  copyblend [1, 12]
stop_epoch 12  matcher_change 11  flat 7  no_aug 2
tiled eval, 1229px source window, per-epoch staging
```

14 epochs at batch 32 reproduces upstream's budget in both senses: ~217k
optimizer updates and ~6.9M tile-views against upstream's ~214k and ~6.9M, with
every landmark within ~2k updates of its upstream position.

Everything scientifically relevant is hard-pinned rather than defaulted, and
stale exports are actively cleared -- several of these had been overridden by
hand during debugging.

### Progress

| epoch | wall | loss | vali AP |
|---|---|---|---|
| 0 | 1:53:57 | 35.28 | 0.5260 |
| 1 | 1:59:37 | 35.29 | 0.5310 |
| 2 | 2:00:16 | 34.16 | 0.5320 |
| 3 | 2:00:43 | 33.65 | 0.5320 |
| 4 | 2:01:06 | 33.20 | **0.5329** |
| 5 | 2:00:46 | 32.93 | 0.5310 |
| 6 | 2:00:46 | 32.70 | 0.5300 |
| 7 | 2:00:12 | 28.28 | 0.5260 |

Epoch 7 is the designed phase change -- flat LR ends, Mosaic and MixUp
terminate. Loss fell 4.42 in one epoch, six times any previous step, because
the training distribution got easier; vali never had Mosaic, so AP dipped. The
first fish run to reach such a transition at all.

**Vali AP here is tile-level and is NOT comparable to gen003's 0.5406**, which
was whole-image.

### The open question

gen003 needed **9 epochs after its transition** to peak (0.5410 at e21, from
e12). gen006 has **6** (e8-e13). Its cosine tail is the same fraction of the
schedule but shorter in absolute epochs, and recovery-then-climb may need a
number of epochs rather than a fraction.

Two things cut the other way: gen003's dip at +2 was a `best_stg1` reload, not
the transition, and gen006 cannot reload before epoch 12; and gen006 entered
the transition from a lower base, so there is more headroom.

Epochs 9-11 decide it. If gen006 is back above 0.533 by epoch 10, the tail is
working. If it is still under 0.530 at epoch 11, 14 epochs was too short for
this transition to pay off -- which is a result, not a failure, at 31 h.
