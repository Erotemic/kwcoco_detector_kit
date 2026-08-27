# 2026-08-27 — the corpus is 81 effective sequences, not 495,514 samples

gen006 finished clean and still peaked at epoch 4 of 14, declining to epoch 11
while training loss fell 33.20 → 26.80. That shape — loss down, validation down
— is a model fitting something that does not transfer, so gen007 changes what
an epoch *is* rather than how long the schedule runs.

Before choosing any sampler probability, the imbalance was measured.

## The measurement

`python -m kwcoco_detector_kit.data.sequence_balance measure` over the real
train corpus (495,514 tiles, joined tiled-bundle → source bundle for sequence
identity, since the tiler stamps `tile_source_gid` but not `video_id`):

| grouping | groups | min / median / mean / max | imbalance | gini | **effective count** |
|---|---|---|---|---|---|
| **sequence** | 439 | 30 / 260 / 1129 / 13,450 | 448× | 0.747 | **81 (18.3%)** |
| **track** | 14,103 | 1 / 31 / 55 / 4,794 | 4794× | 0.573 | **2,963 (21.0%)** |
| source frame | 251,143 | 1 / 2 / 2 / 2 | 2× | 0.013 | 249,462 (99.3%) |

Read the effective counts (inverse Simpson — the number of equally-sized groups
that would produce the same concentration) first. **Uniform sampling over
495,514 tiles draws from something that behaves like 81 sequences and 2,963
tracks.** Four fifths of the nominal diversity is redundancy. The top 10% of
sequences hold 66.4% of all tiles; one fish, tracked across 4,794 tiles,
outweighs 150 median-length tracks by itself.

That is a concrete mechanism for the curve gen006 produced, and it is not
something the augmentation stack or the schedule could have fixed.

The source-frame row is why gen007 balances sequences and tracks and **not**
frames: tiles are already almost perfectly uniform per frame (2 per frame,
gini 0.013). `frame_alpha` stays 0 — it would add a knob and do nothing.

20.8% of tiles carry no annotation (111,835 negative + some positives whose
annotations fell below the keep threshold).

## The metric was wrong the first time

The first α sweep reported an effective track count of **1,454** for the
uniform baseline, against a true unweighted count of **2,963**. Cause: each
tile's weight was divided among the tracks on it (`w_i / len(tracks_i)`), so a
fish photographed beside four others counted one fifth as much as a fish
photographed alone. Every α was therefore scored against a reference that did
not match the corpus.

Correct version: each tile contributes its **full** weight to every track on
it. Uniform weights then reproduce 2,963 exactly, and that identity is now a
test — it is the only cheap check that the metric measures the corpus.

A second, separate bug lived in the test helper: it zipped 495,514 tile
weights against a **flattened** list of 780,566 (tile, track) pairs, silently
truncating 285,052 of them. It passed only because the synthetic fixture gave
every tile exactly one track. `mass_effective_count` now rejects a flattened
pair list outright rather than truncating it.

*(Credit to the external review for spotting that 1,454 ≠ 2,963. Its proposed
mechanism — zip truncation in the sweep — was not what happened there; the
sweep zipped two equal-length sequences. But the truncation it described was
real, in the test helper, which is where it would have hidden longest.)*

## Choosing alpha from the corrected data

| seq_α | track_α | cap | eff. sequences | eff. tracks | neg % | max draws/tile |
|---|---|---|---|---|---|---|
| 0.00 | 0.00 | — | 81 | 2,963 | 20.8% | 0.19 |
| 0.25 | 0.50 | 8 | 139 | 7,281 | 20.2% | 1.55 |
| **0.50** | **0.50** | **8** | **238** | **5,306** | **19.8%** | **1.55** |
| 0.50 | 0.75 | 8 | 221 | 5,969 | 19.7% | 1.55 |
| 0.75 | 0.50 | 8 | 334 | 3,725 | 19.9% | 1.55 |
| 1.00 | 0.50 | 8 | 365 | 2,893 | 20.2% | 1.55 |

1. **Full flattening still fails, and now the evidence is cleaner.** At
   `seq_alpha=1.0` effective tracks land at 2,893 — *below* the 2,963 they
   started at. Every bit of the sequence gain is paid for out of track
   diversity, because flattening pours mass into short sequences and short
   sequences are short precisely because they hold few tracks. The broken
   metric had exaggerated this (1,729); the conclusion survives, the magnitude
   does not.

2. **0.5/0.5 nearly triples sequences and nearly doubles tracks.** 0.5/0.75 is
   marginally better on tracks and marginally worse on sequences; the two are
   within the noise of any scalar that combines them, so the symmetric setting
   stands rather than tuning an asymmetry the data does not support.

3. **The cap keeps the cure from becoming the disease.** Uncapped,
   `seq_alpha=0.5` draws some single tile 7.7 times per epoch — memorisation
   of a different tile. `max_oversample=8` brings that to 1.55.

Nothing is discarded. Every tile keeps a strictly positive weight; the draw is
with replacement and redrawn each epoch, so rare sequences are seen more and
dominant ones less while the full corpus stays reachable.

## gen007

`submit_train_..._gen007_seqbalance.sh`. Not launched — awaiting review.

- **sampling** seq_α 0.5, track_α 0.5, empty_weight 1.0, cap 8, seed 0,
  **without replacement**
- **epoch** 96,000 tiles × 28 epochs / batch 32 = **84,000 updates**,
  split **20 primary (60k) + 8 tail (24k)**
- **augmentation** `tiled_light` — drops Mosaic, RandomZoomOut, RandomIoUCrop;
  disables mixup/copyblend. All five assume each sample is an independent
  scene; these samples are 1229px crops of video frames, so the crop diversity
  is already supplied by the tiler and compositing manufactures scenes the
  sensor cannot produce. Photometric distortion and horizontal flip stay.
- **lr** 2.5e-4 / 5e-6 — both halved from gen006, keeping upstream's 50:1
  head/backbone ratio, which is the part DINOv3-X was actually tuned with.
- **schedule** policy [2, 12, 20], flat 12, no_aug 8, stop 20, matcher 18,
  wd 1.25e-4, mixup/copyblend at the disabled sentinel (40000, 15000).
- expected ~11 h on 4 GPUs, versus gen006's 27.9 h.

### Why 84k updates

gen006 peaked at epoch 4 ≈ 62k updates and never beat it across the remaining
~155k. 84k gives that observed peak room plus margin at a third of gen006's
217k. Each tile is now seen ~5.4 times across the whole run instead of 14, and
*which* tiles are seen is redrawn every epoch. The 28 stages also give the LR
cosine and the augmentation state machine real resolution instead of gen006's
coarse 14.

**This is a prior, not a law.** If the stride-8 gen006 epoch ranking shows the
tiled peak materially later than epoch 4, raise `KCD_NUM_EPOCHS`.

### The tail is absolute, not proportional

Upstream DINOv3-X ends with **8 epochs** past `stop_epoch` — the stage-2 phase
where augmentation is off and EMA restarts from the stage-1 best. Scaled
proportionally that tail becomes 4 epochs at 28 and 2 at 14, which is
backwards: the phase that consolidates shrinks exactly as the schedule gets
shorter, so the runs with the least training get the least consolidation.
gen006's 14-epoch run had a 2-epoch tail.

`retarget_tail` fixes the tail at upstream's own absolute 8 and fits the
primary phase into what remains, keeping every landmark at upstream's ratio
*within* that phase. It is not hand-tuned — [2, 12, 20] falls out of the same
4/50, 29/50, 45/50 ratios upstream uses.

### The draw is without replacement

Measured on this corpus at epoch_length 96,000, world_size 4:

| mode | drawn | unique | wasted |
|---|---|---|---|
| with replacement | 96,000 | 77,766 | **19.0%** |
| without replacement | 96,000 | 96,000 | **0** |

Nearly a fifth of every epoch was being spent re-showing a tile already seen in
that same epoch. 19.0%, not the ~9% a *uniform* draw wastes: reweighting
concentrates the mass, and concentrated mass collides more. On top of that,
per-rank independent streams let two GPUs spend the same synchronised optimizer
step on the same tile.

`DistributedWeightedNoReplacementSampler` makes one global Gumbel-top-k draw
(equivalently Efraimidis–Spirakis) with the same seed on every rank, then rank
`r` takes `order[r::world_size]`. Verified on the real weights: 96,000 unique,
24,000 per rank, **zero cross-rank overlap**, 0.4 s for all four ranks.

### Negatives survive the reweighting

The negative-tile fraction of an actual 96,000-tile draw is **21.1%** against a
corpus rate of **20.8%**. So negatives are *not* distributed pathologically
across sequences, and `empty_weight` stays at 1.0 — no tuning needed. (0.7 →
14.8%, 1.3 → 24.2%, if it ever needs steering.)

## Infrastructure notes

- No new dataloader plumbing was needed. `balanced_sampler.py`'s sidecar,
  `sampler_from_weights_file`, and the patched `_solver.py` already existed and
  are used by the sea-lion project; `sequence_balance.py` only computes a
  different weight vector for the same sidecar schema. It deliberately does not
  depend on `BalancedSampleForest.index_weights()`, which the submodule has
  not shipped.
- `KCD_TILE_SOURCE_KWCOCO` added to `paths.sh`. Sequence identity exists only
  in the untiled bundle; without it, balancing silently degrades to frame-level
  grouping, which the measurement shows would do nothing.

## A correction about verification

`dev/check_undefined_names.py` called `Path(root).rglob("*.py")`, which yields
**nothing** when the argument is a file rather than a directory. Every
invocation in this session and the previous one that passed explicit file paths
checked zero files and printed "0 finding(s)". Those green results meant the
checker had not run.

It was caught when pytest rejected a `SyntaxError` in `deimv2.py` — a keyword
argument placed before positional ones — that the checker had just reported
clean. Fixed: files are checked directly, and a nonexistent path is now a
finding rather than silence. Re-run afterwards across every touched file: clean
for real this time.

## Next

1. Review gen007, then launch.
2. Stage-1 (stride-8) scoring of gen006's 14 staged epochs remains outstanding
   and is *not* a prerequisite — it refines the update budget, it does not gate
   the run.
3. Test split still untouched.
