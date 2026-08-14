# aiq: RF-DETR baseline audit + DEIMv2 prep runbook (2026-08-14)

Session ran on a VM co-located with aiq-gpu, with `/home/local/KHQ/jon.crall/ssd-data`
and `/data/users/jon.crall` bind-mounted at identical absolute paths per
[HANDOFF_aiq.md](../HANDOFF_aiq.md). No data prep or training was executed here;
the host has the cores and the GPUs. This session inspected, measured, and
wrote the scripts the host will run.

Reading order: this entry supersedes several conclusions in
[2026-08-14_orientation.md](2026-08-14_orientation.md), which was written
without access to the corpus or the run artifacts. Where they disagree, this
one measured and that one inferred.

## Headline: the baseline exists, completed, and is weaker than its number

`docs/training_runs.yaml` recorded `fishtrack23_rfdetr_l_seg720_4gpu_viame0227_gen001`
as **deferred**. It is not. It ran 2026-08-04 19:50 -> 2026-08-07 19:15 (~71 h),
exited 0, and produced `fish_detector.zip`. Registry corrected.

Artifacts: `/data/users/jon.crall/fish/runs/fishtrack23_rfdetr_l_seg720_4gpu_viame0227_gen001/attempt_20260804_195055/`

### Its actual metrics

Final epoch (20 of 20), from `train.log`:

| metric | value |
|---|---|
| box mAP@50:95 | **0.4429** |
| box mAP@50 | 0.7166 |
| box mAP@75 | 0.4552 |
| mAR@500 | 0.6587 |
| F1 / precision / recall | 0.6733 / 0.7441 / 0.6149 |
| segm mAP@50:95 | 0.3852 |

Best EMA checkpoint: **0.4461 at epoch 15**. The improvement curve was still
creeping upward when training stopped (0.4426 @ e11 -> 0.4461 @ e15, nothing
after), so 20 epochs was plausibly short.

Effective config from `rf_detr_mgpu_params.json`: `batch_size=4`,
`grad_accum_steps=1`, `devices=4` -> global batch **16**. The orientation
journal's finding #4 (batch/LR mismatch, global batch 64 against an LR tuned
for 16) **did not materialize** — VIAME's adaptive mode sized the micro-batch
itself and landed exactly on `auto_batch_target_effective`. Worth recording so
nobody re-raises it.

### Why that 0.4429 is not the number it looks like

Four things, in descending order of how much they matter:

1. **The model is single-class.** `Train/labels.txt` is a single line: the
   output class `fish` followed by 321 aliases. Every one of the 150 observed
   species folds into it, and `rf_detr_mgpu_params.json` confirms
   `class_names: ["fish"]`. This model does no species discrimination at all.

2. **`valid/` and `test/` are byte-identical.** Both
   `rf_detr_dataset/valid/_annotations.coco.json` and `.../test/...json` hash to
   `ab1b10d180f15df355587de574dda9ee`. VIAME's "test" split is a copy of its
   validation split. The orientation journal suspected there was no held-out
   test set; the reality is stronger than suspected.

3. **Validation is tiny and drawn from the training sequences.** 4,000 chips /
   1,532 annotations, against 650,529 training chips — a 0.6% split, carved
   frame-level from the same videos. On tracked video that means the validation
   images are near-duplicates of training images. So 0.4429 is a noisy
   selection score on contaminated data, not a generalization estimate.

4. **78% of training chips are empty.** 505,955 of 650,529 chips carry no
   annotation.

**None of this means the model is bad.** It means 0.4429 is not evidence either
way, and any DEIM-vs-RF-DETR table that quotes it as a peer of a held-out
number would be misleading.

## Two open problems from the orientation journal, resolved

### Contamination — solved by the corpus itself

The orientation journal called this "not resolvable by anything on the DEIM
side" and listed finding data RF-DETR never saw as the best case. That data
ships with the corpus: **`FishTrack23-Latest/Test/`**, 54 videos + 18 image
directories = **72 sequences**. Both the abandoned July run
(`run_fish_training.sh`) and gen001 (`run_manifest.txt`) invoke
`viame_train_detector -i .../FishTrack23-Latest/Train` and nothing else, so
RF-DETR provably never saw any of it.

That is an honest held-out test set for **both** models. The prep pipeline
converts it and never trains on it.

### Small objects — the premise was wrong

The orientation journal ranked "small objects" as the strongest
complementarity argument, on the theory that `small_box_area = 75` /
`small_action = remove` deleted a meaningful mass of targets. Box percentiles
over all 665,228 boxes say otherwise:

| percentile | width | height |
|---|---|---|
| p1 | 41.9 | 44.0 |
| p5 | 59.1 | 53.8 |
| p50 | 150.0 | 108.9 |
| p95 | 358.5 | 229.7 |

on 1920x1200 imagery. `small_box_area = 75` is ~8.7 x 8.7 px — below the 1st
percentile by a factor of five. **It deleted essentially nothing.**

Consequences: there is no small-object regime to exploit, and **the DEIM run
needs no tiling**. Whole frames resized to 1024 leave the p1 box at ~22 px and
the median at ~80 px. That removes the entire tile-cache stage the sea-lion
project needs, and with it days of prep and a large multiple of epoch cost.

The real complementarity argument is what is left: a different backbone family
(DINOv3 vs RF-DETR's ViT), whole-frame training vs 720px chips, and — for the
first time on this corpus — a score on data the model has not seen.

## Storage: the training data was on the RAID array, not the SSD

Confirmed a hypothesis raised mid-session. aiq-gpu has two filesystems:

```
/dev/md0         37T   16T used   20T avail   /data      <- RAID array
/dev/nvme0n1p2  1.8T  1.2T used  506G avail   /          <- NVMe
```

The RF-DETR run put **771 GB of extracted PNG frames** (`augmented_images/`)
and its chip cache (383 GB for the July run, 81 GB for gen001) under
`/data/users/jon.crall/fish/`, i.e. on md0, then read 650k small PNGs in random
order every epoch. The source corpus (`$HOME/ssd-data/FishTrack23-Latest`) was
on the NVMe the whole time; only the generated data went to the slow device.

Whether that was *the* binding constraint on the ~3.2 h epochs is not provable
from the logs alone — RFDETRSegLarge at 720px is genuinely heavy, and mask loss
is not free. But it is a needless constraint, so the new pipeline puts frames,
bundles, and run workspaces on the NVMe (`VF_KCD_ROOT=$HOME/ssd-data/fish_kcd`).

The budget only works because we extract **annotated frames only**: 250,753 of
~4.4M video frames (~6%). At JPEG q95 that is ~75 GB against 506 GB free.
Extracting everything as PNG, as VIAME did, would not fit at all.

## Corpus shape (measured)

From `collect_data_manifest.sh` output at `/data/users/jon.crall/fish/inventory/`:

| quantity | Train/ |
|---|---|
| sequences | 504 (420 video + 84 image dirs) |
| annotated frames | 286,651 (250,753 video + 35,898 image dir) |
| boxes | 665,228 |
| tracks | 16,867 (~39 annotated frames per track) |
| categories | 150 (all fold to `fish`) |
| malformed CSV rows | 0 |

Frame rates: 349 videos at 5 Hz, 32 at 10 Hz, 20 without an fps comment
(handled by ffprobe fallback). Image sizes are dominated by 1920x1200,
1920x1080, 1920x1088, with a 968x728 minority.

Long tail is severe: `lutjanus_campechanus` 189,253 boxes down to
`muraena_retifera` 155. Irrelevant for this single-class run; decisive if a
species model is ever attempted.

## Frame indexing — the part that had to be nailed down

A VIAME CSV addresses frames by integer index, and getting that mapping wrong
attaches every box to the wrong image with no error raised. Both conventions
were verified against the real corpus before being encoded:

* **video** — CSV column 2 is a timestamp and column 3 an index, satisfying
  `timestamp == index / fps` **exactly** (checked at both ends of multiple
  videos). VIAME extracts to `frame%06d`, 1-based, so index `i` is
  `frame{i+1:06d}`.
* **imagedir** — column 2 is empty; the index is a position in the sorted file
  listing. `PIFSC-MOUSS-Onaga1`: 541 images, indices 0..540, exact.

`extract_frames.py` does not *assume* the video mapping, it *recovers* it: the
`showinfo` filter sits downstream of `select` and reports each surviving
frame's original `pts_time`, so every output file is named from its own
timestamp rather than its position in the output sequence. A decoder hiccup
then shortens the output instead of silently shifting every subsequent frame by
one.

A handful of CSVs reference one frame past what the container decodes (e.g.
`CDFW-LakeCam-April-SpiderBlocks1`, index 10581 of 10581 frames). Those
annotations are dropped and counted, not fatal.

## What was built

All under `projects/viame_fish_2026/scripts/`:

| script | role |
|---|---|
| `paths.sh` (extended) | NVMe-first `VF_*`/`KCD_*` layout, checkpoint resolution, fish-specific `kcd_require_train_inputs` (no tile-cache check) |
| `extract_frames.py` | annotated-frames-only ffmpeg extraction to JPEG, parallel, resumable, self-validating index recovery |
| `convert_viame_to_kwcoco.py` | VIAME alternating class/score CSV -> kwcoco; both layouts; labels.txt folding; box-only; bbox clipping |
| `build_splits.py` | sequence-disjoint train/vali, deployment-grouped, annotation-balanced |
| `prep_all.sh` | host driver chaining the three above |
| `_launch_train.sh` | in-container launcher; no tiling / scheme / balance stages |
| `_submit_train.sh` | slurm submit; reuses the sea-lion `_sbatch_train.sh` for GPU hardening |
| `submit_train_fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen001.sh` | the run |

`tests/unit/test_viame_to_kwcoco.py`: 32 tests, all passing, covering both
frame-index conventions, the alternating class/score parse, label folding,
bbox clipping, and split disjointness.

### Validated against real data

Converting `PIFSC-MOUSS-Onaga1` end to end: **2,285 annotations**, matching the
independent inventory count exactly; 541 images matching the 541 PNGs on disk;
frame indices 0..540; 0 malformed rows; 7 boxes clipped, matching the 7 found
by hand.

### The deployment-grouping trap

Sequence-disjoint is necessary but insufficient. `SEFSC-SeaMap-761901231-Cam2`
and `-Cam3` are simultaneous cameras on one baited station, often showing the
same individual fish; splitting them across train/vali leaks.

The first version of `deployment_key` over-corrected — it stripped every
trailing numeric token and collapsed **295 of the 378** SEFSC-SeaMap sequences
into a single group, which makes a balanced split impossible. Anchoring the
pattern on `Cam` gives 504 sequences -> 239 deployments, station `761901361`
correctly grouping its 44 camera views. Both behaviours are now pinned by
tests, including a regression test for the over-grouping.

Simulated split at `vali_fraction=0.12`, seed 0: train 575,800 boxes / vali
89,428 boxes (13.4%), 83 vali sequences, every collection represented, zero
deployment overlap. (575,800 + 89,428 = 665,228, matching the inventory.)

## The planned run

`fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen001`

| knob | value | why |
|---|---|---|
| variant | `deimv2_dinov3_x` | 50.3M params, 57.8 COCO AP; already on the host |
| input | `[1024, 1024]` whole frame | p1 box still ~22 px; no tiling needed |
| classes | `fish` | identical folding to the baseline, via the same labels.txt |
| batch | 6/GPU, total 24 | ~49 GB of 96 GB by the sea-lion memory model, area-scaled to 1024 |
| LR | 1e-3 / backbone 5e-5 | sqrt scaling from the sea-lion reference; same pair validated by sea-lion gen007 |
| epochs | 20 | matches the baseline so length does not confound the comparison |
| eval | whole-image | matches how it trains; tiled eval would measure something else |

Estimated ~10.4k steps/epoch at stride 1, roughly 20-24 h.

`VF_FRAME_STRIDE` subsamples annotated frames when the **splits** are built,
not at extraction, so changing it never re-runs ffmpeg. Default 1 (every
annotated frame, matching what RF-DETR consumed). With ~39 annotated frames per
track, stride 2-3 would still leave 13-20 well-separated samples per track if
wall-clock gets tight.

## Open items

* **Scoring is deliberately deferred** until there are two deliverable models.
  When it happens, both get scored on `Test/` under one protocol. Note that
  RF-DETR's `fish_detector.zip` needs a VIAME inference pass and a reader for
  its output CSV, neither of which exists in the kit yet.
* **`_sbatch_train.sh` lives in the sea-lion project** but is already
  project-agnostic (`KCD_REPO_ROOT` + `KCD_LAUNCH_SCRIPT`). The fish submit
  script reaches across to it rather than duplicating ~380 lines of GPU
  pinning, zombie-container cleanup, and leak detection. Promoting it to a
  shared location is the right cleanup; not done here because it edits another
  project as a side effect.
* **20 epochs may be short for both models.** The baseline's EMA was still
  improving at epoch 15. Worth a look at the DEIM curve before calling the
  comparison done.
* **The four `non_fish_*` categories** (25,392 boxes, 3.8%) are dropped, exactly
  as VIAME dropped them. The dataset readme says they are not consistently
  annotated across the release, so this is defensible on its own merits — but
  the reason it is done here is comparability.

## Lessons

* Read the run artifacts before trusting the run registry. `training_runs.yaml`
  said `deferred`; `exit_code.txt` said `0` and there was a 134 MB model zip
  next to it. Three days of GPU time were nearly written off as not having
  happened.
* Check whether the dataset already solved your methodology problem. The
  previous session designed around an unresolvable contamination issue; the
  corpus had shipped a held-out test split all along, one `ls` away.
* Measure before designing around a weakness. Two of the three complementarity
  arguments in the orientation plan (small objects, rare-class selection noise)
  do not survive contact with the box percentiles and the single-class
  labels.txt. The tiling stage they justified would have cost days.
