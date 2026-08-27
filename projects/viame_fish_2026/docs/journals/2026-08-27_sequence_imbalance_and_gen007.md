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

## Choosing alpha from the data, not from taste

Sweeping the exponents and recomputing effective counts under the resulting
mass:

| seq_α | track_α | cap | eff. sequences | eff. tracks | max draws/tile/epoch |
|---|---|---|---|---|---|
| 0.00 | 0.00 | — | 81 | 1,454 | 0.19 |
| 0.25 | 0.25 | — | 137 | 3,741 | 1.41 |
| 0.50 | 0.50 | — | 243 | 4,408 | 7.69 |
| **0.50** | **0.50** | **8** | **238** | **4,473** | **1.55** |
| 0.75 | 0.50 | 8 | 334 | 3,059 | 1.55 |
| 1.00 | 0.50 | 32 | 394 | 1,729 | 6.20 |

Two findings decided the setting:

1. **Full flattening is actively harmful.** At `seq_alpha=1.0` effective
   *tracks* collapse from 4,473 to 1,729, because flattening pours mass into
   short sequences and short sequences are short precisely because they
   contain few tracks. It buys sequence diversity with track diversity.
   `seq_alpha=0.5` is the only row that improves both — roughly 3× each.

2. **The cap is what keeps the cure from becoming the disease.** Uncapped,
   `seq_alpha=0.5` draws some single tile 7.7 times per epoch — memorisation
   of a different tile. `max_oversample=8` brings that to 1.55 and effective
   track count goes *up*, not down.

Nothing is discarded. Every tile keeps a strictly positive weight; the draw is
with replacement and redrawn each epoch, so rare sequences are seen more and
dominant ones less while the full corpus stays reachable.

## gen007

`submit_train_..._gen007_seqbalance.sh`. Not launched — awaiting review.

- **sampling** seq_α 0.5, track_α 0.5, empty_weight 1.0, cap 8, seed 0
- **epoch** 96,000 tiles × 28 epochs / batch 32 = **84,000 updates**
- **augmentation** `tiled_light` — drops Mosaic, RandomZoomOut, RandomIoUCrop;
  disables mixup/copyblend. All five assume each sample is an independent
  scene; these samples are 1229px crops of video frames, so the crop diversity
  is already supplied by the tiler and compositing manufactures scenes the
  sensor cannot produce. Photometric distortion and horizontal flip stay.
- **lr** 2.5e-4 / 5e-6 — both halved from gen006, keeping upstream's 50:1
  head/backbone ratio, which is the part DINOv3-X was actually tuned with.
- **schedule** from the recipe at 28 epochs: policy [2, 14, 24], flat 14,
  no_aug 4, stop 24, matcher 22, wd 1.25e-4, mixup/copyblend at the disabled
  sentinel (40000, 15000).
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
