# Orientation: what it would take to train the best fish detector (2026-08-14)

## Context

Goal: use a weekend of wall-clock on aiq-gpu (4× RTX PRO 6000 Blackwell, 96 GB
each) to train a FishTrack23 detector. The prompt that started this was that
the parameters VIAME used for the last fish model looked suboptimal.

**Scope settled during the session:** an RF-DETR model has already been trained
through VIAME's native tooling. This run is *not* a retune of that — it is a
**complementary DEIMv2 model**, box-only, trained through the kit pipeline.
That changes how the config review below should be read: it is no longer a
to-do list of things to fix in the VIAME config, it is a map of where the
existing RF-DETR model is most likely to be weak, and therefore where the DEIM
model has room to be genuinely complementary rather than redundant.

Starting state of this project:

- `docs/training_runs.yaml` lists exactly one run, `gen001`, status **planned**.
  Nothing has trained to completion under this runbook. An earlier v0.22.6
  attempt hung and was abandoned.
- The runbook (`scripts/`) is solid on *provenance* — versioned run entry
  points, config snapshot + SHA into every attempt, immutable VIAME installs —
  and silent on *methodology*. There is no split definition, no held-out test
  set, and no scoring script.
- Nothing in this repo records the shape of the FishTrack23 corpus. Image
  dimensions, object-size distribution, category frequency, and video-vs-still
  composition are all unknown here, and every interesting hyperparameter
  depends on them.

This session was orientation only. No training was launched.

## What we did

Built the missing measurement step, because every downstream decision is
blocked on it and it costs nothing to run:

- `scripts/inventory_data.py` — stdlib-only inventory of a VIAME-style training
  directory. Walks the tree, parses VIAME CSVs (per-category counts, box
  width/height/area percentiles, track and frame counts), and reads image
  dimensions from file headers without decoding pixels. Emits
  `inventory.json` + `inventory.md`. Stdlib-only on purpose so it runs under
  whatever interpreter the training host has, including VIAME's bundled python,
  without installing anything into either.
- `tests/unit/test_inventory_data.py` + `tests/conftest.py` — synthetic
  fixtures for both parsers. The tool runs unattended against a corpus we
  cannot see from the workstation, so its parsers are tested rather than
  trusted on first contact. 10 tests, passing.
- `scripts/collect_data_manifest.sh` — runs the inventory on the training host
  and packages a small transferable manifest (inventory, depth-3 tree, disk
  usage/free, GPU list, core count; `--with-annotations` adds a tarball of the
  raw CSVs). No pixels, so it is cheap to rsync back.
- `docs/HANDOFF_aiq.md` — for a second agent running on a VM co-located with
  aiq-gpu: which directories to bind-mount, and the rule that they must appear
  at *identical absolute paths* on both sides.

## Reading the RF-DETR config: where that model is likely weak

Reviewed `configs/train_detector_rf_detr_l_720_90gb.conf` against VIAME's
`plugins/pytorch/rf_detr_trainer.py` (fetched from VIAME main). Findings split
into two very different severities.

### Methodological — these change what the run is *worth*

1. **No held-out test set, anywhere.** `scripts/_launch_viame_train.sh` invokes
   `viame_train_detector -i "$VF_INPUT_DPATH"` with no `-v`, so the trainer
   carves its own validation split out of the input using
   `default_percent_validation = 0.10`. That same split then selects the
   exported checkpoint. There is nothing left over to measure generalization
   on, so whatever number the run reports is a selection score, not a test
   score, and it cannot be compared against a future run trained on a different
   split. **This, not any learning rate, is the biggest problem.**

2. **The split is drawn by the trainer, and we do not know along what axis.**
   On tracked video data, a frame-level random split puts adjacent frames of
   the same fish track on both sides of the train/val boundary — near-duplicate
   images, so the validation metric is inflated and checkpoint selection is
   biased toward memorization. *Open item: confirm how `viame_train_detector`
   draws the split before assuming leakage.* The robust fix does not require
   the answer: build sequence-disjoint `train/ vali/ test/` directories
   ourselves and pass `-i train -v vali`, keeping `test/` away from the trainer
   entirely. This is the same lesson the sea-lion project learned the
   expensive way (see the sea-lion journals on NFS-as-negative and per-
   checkpoint rescoring: the number the trainer reports is not the number that
   decides which model is best).

3. **No explicit `--labels`.** The class list is auto-derived from whatever
   appears in the corpus, so annotation typos and singleton classes become
   model classes. An explicit labels file makes the class set reproducible and
   lets us fold or drop the tail deliberately.

### Parameterization — these change how good the model is

4. **Batch/LR mismatch under DDP.** `batch_size = 8` and `grad_accum_steps = 2`
   are *per-device*. On 4 GPUs the global batch is 8 × 2 × 4 = **64**, while
   `learning_rate = 1e-4` is the trainer's default, tuned for the default
   effective batch of 16 (`auto_batch_target_effective`). The run would train
   at a 4× larger batch with an unscaled LR. Either scale the LR or use
   `batch_size = adaptive`, which sizes the micro-batch from VRAM and sets
   `grad_accum_steps` so the *global* batch lands on
   `auto_batch_target_effective` regardless of GPU count — one config that is
   correct on any node.

5. **`small_box_area = 75` + `small_action = remove`.** Objects below ~8.7 × 8.7
   px are deleted from the training data. Deleted, not ignored — so those
   pixels become background the model is actively taught to reject. If a
   meaningful fraction of FishTrack23 targets are that small, this caps recall
   by construction. The sea-lion work found small-object AP was the binding
   floor (AP-small ≈ 0.005 vs AP-large ≈ 0.35); this parameter is exactly the
   mechanism that produces that. **The inventory's box-size percentiles decide
   whether this is a real problem or a non-issue** — it is the single most
   valuable number the inventory will produce.

6. **`resolution = 720`.** Same concern from the other direction. Chips are
   720 px with a 480 px step, and `chip_adaptive_thresh = 1600000` means images
   under ~1.6 MP are resized whole instead of chipped. Whether 720 is right
   depends entirely on the real image dimensions and object sizes.

7. **`segmentation = True`.** This trains RFDETRSegLarge — masks *and* boxes.
   It is far heavier than box-only, forces validation to batch 1, and
   constrains resolution to multiples of 24. If the deliverable is boxes, the
   entire cost is wasted and could have bought resolution or epochs instead.
   Relatedly, `max_mask_instances` (a guard against the seg mask loss OOMing on
   densely annotated chips) is left at 0/off — a real risk if fish appear in
   schools of hundreds per chip.

8. **`max_epochs = 20`, `early_stopping` off, `checkpoint_interval = 10`.**
   The trainer tracks best regular/EMA checkpoints independently of the
   interval, so selection still happens, but only 2 periodic checkpoints are
   kept — not enough to rescore a training curve after the fact, which is
   exactly what we ended up needing on sea lions.

9. **Unused levers for a long-tailed corpus.** `val_subsample`,
   `val_min_class_instances`, and `min_class_support` exist precisely to stop
   rare-class AP noise from driving checkpoint selection, and none are set. If
   FishTrack23 is long-tailed (unknown until the inventory runs), these matter.

### Checked and *not* a problem

`learning_rate_encoder = 1.5e-4` being numerically larger than
`learning_rate = 1e-4` looks backwards for fine-tuning, but it is not: the
trainer applies `lr_component_decay = 0.7` (backbone gets a further square of
it, ≈ 7.35e-5) and `lr_vit_layer_decay = 0.8` down the ViT blocks, so the
backbone does train slower than the head. Noting this so nobody else re-raises
it.

## What "complementary" implies for the DEIM run

Reading the RF-DETR config as a weakness map, the DEIM model's opportunities
are, in order of expected payoff:

1. **Small objects.** RF-DETR trained at 720 px with `small_box_area = 75` /
   `small_action = remove` — anything under ~8.7 × 8.7 px was deleted from its
   training data and taught as background. If the inventory shows a
   meaningful mass of boxes down there, that is a whole size regime the
   existing model was trained *against*, and tiled DEIM training at higher
   effective resolution addresses it directly. This is the strongest
   complementarity argument and the sea-lion project already has the tooling
   for it.
2. **Rare classes.** RF-DETR set none of `val_subsample`,
   `val_min_class_instances`, or `min_class_support`, so if the corpus is
   long-tailed its checkpoint selection was driven by noisy tail AP. The kit's
   balance machinery (`KCD_BALANCE_TARGET_JSON`, sampler mode) targets exactly
   this.
3. **Architectural diversity for ensembling.** DEIMv2 + DINOv3 is a different
   backbone family and a different training recipe from RF-DETR's ViT. Two
   models that fail on the same images are not worth ensembling; these
   plausibly do not.

## Where the plan stands

Stack decision: **kwcoco_detector_kit DEIMv2, box-only.** Sequenced so nothing
expensive happens before the cheap thing that informs it:

1. **Inventory the corpus on aiq-gpu.** Everything below is conditioned on it.
   In particular the box-size percentiles decide the resolution/tiling design,
   and the category histogram decides whether balance machinery is needed.
2. **Write a VIAME-CSV → kwcoco converter.** The kit has no reader for the
   VIAME alternating class/score format — `convert_sealions_csv_to_kwcoco.py`
   handles a *different*, headered format and explicitly says so. The parser in
   `scripts/inventory_data.py:parse_viame_csv` is a tested starting point for
   the row-level logic. Cannot be written responsibly until the inventory shows
   the actual directory layout (per-sequence CSVs? extracted frames? videos?).
3. **Freeze sequence-disjoint `train/vali/test` splits.** Whole sequences on
   one side only — frame-level splits leak across tracks on video data.
4. **Build the tile cache** at the size the inventory implies.
5. **Submit through slurm on aiq** with the kit docker image, following the
   sea-lion `submit_train_*_aiq_*.sh` pattern (`KCD_NO_SLURM=0`,
   `KCD_DOCKER_GPU_MODE=gpus`).
6. **Score both models on the same protocol** — class-agnostic detection AP on
   the held-out test split, per-class AP as diagnostic only.

### Open problem: the comparison is contaminated

The existing RF-DETR model chose its own validation split out of the full input
directory, so it very likely saw **every sequence** in the corpus during
training. Any test split we now construct is therefore held-out for DEIM but
not for RF-DETR, and a head-to-head on it is biased toward RF-DETR by an
unknown margin. Options, in order of preference:

- Find data RF-DETR genuinely never saw (a sequence outside the directory it
  was trained on, or a later delivery) and use that as the honest test set.
- Recover which sequences RF-DETR trained on from its run artifacts and
  restrict the test set to what it excluded — only possible if that run's
  split was recorded.
- Accept the bias and report it explicitly alongside every comparison number.

**Not** resolvable by anything on the DEIM side, so it needs deciding before
the comparison is presented, not after. *Open item: locate the RF-DETR run's
artifacts and determine what it trained on.*

## Lessons

- The provenance machinery in this project was built before the methodology.
  Perfect reproducibility of an unevaluatable run is not worth much: a run you
  can reproduce exactly but cannot score against a held-out set still cannot
  tell you whether it beat the last one.
- Read the trainer source before criticizing a config. Two of the parameters
  that looked wrong (the encoder LR, the sparse checkpoint interval) are
  handled by machinery elsewhere in the trainer, and one problem that is not
  visible in the config at all (no `-v`, hence no test set) is the worst one.
