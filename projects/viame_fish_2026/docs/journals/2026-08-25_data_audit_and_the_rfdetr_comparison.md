# Data audit, and the state of the RF-DETR comparison (2026-08-25)

Written after four DEIMv2 configurations (gen001-gen005) all landed at vali
AP ~0.53-0.54 regardless of schedule, batch size, dtype or resolution, and the
question was raised whether something is structurally wrong with how the fish
data is prepared — since the sea-lion project, on the same kit, did not behave
this way.

Short version: **the fish data is not corrupt**, one real misconfiguration was
found, one strong pipeline contrast with sea lions was found, and the RF-DETR
comparison turns out to rest on an inference run with serious coverage
anomalies. All numbers below are reproducible from artifacts on disk.

## Corrections to earlier claims in this journal series

Recorded first because they were load-bearing and wrong.

1. **"Sea lions are independent aerial stills."** False, and I asserted it
   without looking. Sea-lion source imagery is ~22 Mpx aerial survey frames
   (5760x3840, 5,995 of them), cut into **154 tiles each**. Survey frames
   overlap along a flight line, so they are not independent either. A
   recommendation built on that claim — subsample the fish video 5-8x — is
   withdrawn. Neighbouring frames of the same fish behave as cheap
   augmentation; they are not worth discarding.

2. **Held-out test was used for design decisions**, four times, including the
   argument that the project was resolution-limited. See the correction section
   in the 2026-08-23 journal and the holdout-discipline rule now in paths.sh.

3. **"gen005 feeds tiles 1:1 with no resize."** False — see finding 1.

## Finding 1: the tiles are 1229px, not 1024px

`tile.py` computes `disk_tile_size = round(tile_size * oversize_factor)`.
With `tile_size=1024` and the `oversize_factor=1.2` I copied from the sea-lion
params, every emitted tile is **1229x1229**, and each tile record says so:

```
width/height: 1229    tile_model_input_size: [1024, 1024]    tile_oversize_factor: 1.2
```

The knob exists so a load-time crop can jitter scale and position inside an
oversized tile without hitting zero-padded borders. The module docstring calls
that "a **future** trainer-side load-time crop". It is not implemented. So the
1229 tiles are simply resized down to 1024 by the model's Resize op:
**0.833 scale, not 1.0**.

Better than the 0.533 of whole-frame training, but not what gen005 was designed
to test. `KCD_TILE_OVERSIZE_FACTOR=1.0` and a re-tile (~1.5 h) gives true
native resolution.

**Sea lions have the identical setting** (640 tile, 1.2 -> 768 on disk), so
this is a kit-wide inefficiency, not a fish-specific defect, and not the reason
the two projects behave differently.

## Finding 2: the fish bundles are geometrically clean

Full audit of every annotation in the source and tiled bundles — degenerate
boxes, inverted boxes, NaN/inf coordinates, out-of-bounds boxes, area/bbox
disagreement, missing images, normalized coordinates outside [0,1]:

| bundle | images | annotations | findings |
|---|---|---|---|
| source train | 251,143 | 561,572 | 1 box with a sub-2px side |
| tiles 1024 train | 495,514 | 780,566 | none |

Also checked: 158 boxes exceed 60% of frame area (0.028%, plausible close-ups,
largest 99.5%); 150 exact duplicate annotations; aspect ratios p99 3.94, exactly
one box above 20:1; a 400-tile sample confirmed every file exists.

**There is no data corruption.** Whatever is limiting these models, it is not
malformed annotations.

## Finding 3: the corpus is video, and much smaller than it looks

| | |
|---|---|
| adjacent video frames | 99.3% of frame pairs |
| annotations | 561,572 |
| distinct tracks | **14,103** |
| labels per fish | 39.8 |
| IoU between consecutive observations of one fish | median **0.77**, 62% >0.7 |

Sequences average 583 consecutive frames (max 6,725). The same fish is labelled
~40 times as it is tracked.

I initially read this as the cause of the plateau and proposed subsampling.
That was wrong-headed: sea lions are also non-independent, and near-duplicate
frames function as augmentation. What the measurement **is** good for is
calibrating noise:

- vali contains **2,140 tracks across 46 sequences**, not 35,111 independent
  samples.
- test contains **2,645 tracks across 69 sequences**.

So AP differences of ~0.003 between runs — the differences four experiments
were compared on — are almost certainly noise. That, not any single
hyperparameter, is why nothing separated.

## Finding 4: the real pipeline contrast with sea lions

| | sea lions | fish |
|---|---|---|
| source | ~22 Mpx aerial (5760x3840) | 2.3 Mpx video (1920x1200) |
| training input | **154 native tiles per image** | **whole frame -> 1024** |
| effective scale | 0.833 | **0.533**, plus 1.6-1.78x aspect distortion |
| tiled eval | default True | never enabled (and never forwarded — see below) |

**Sea lions were always tiled. Fish never was, through gen004.** Nobody tiled
fish because 1920x1200 looks small enough to feed whole — but a square resize
to 1024 discards 47% of horizontal resolution and squashes aspect, while
RF-DETR's VIAME config (`chip_width 720`, `chip_step 480`,
`chip_adaptive_thresh 1.6 Mpx`) cuts every fish frame at native scale.

This is the same conclusion the tiling work was based on, but reached from the
pipeline contrast rather than from peeking at the test split.

Related defect, now fixed: `_launch_train.sh` never forwarded `KCD_TILED_EVAL`
to the sweep, so every fish run's setting was inert from gen001 onwards.

## Finding 5: the RF-DETR comparison

Both prediction sets exist on disk and are set up comparably — same held-out
test bundle, same category, both floored at score >= 0.5 (RF-DETR's VIAME
plugin applies that threshold internally; DEIMv2 was truncated to match, which
is the rescore that made them comparable):

```
rfdetr_test_inference/rfdetr_test_preds.kwcoco.json   67,377 preds
headtohead/deim_preds_min05.kwcoco.json               56,231 preds
```

`score_headtohead.sh` exists but **its outputs are not on disk** — no
`detect_metrics.json` under `headtohead/deimv2/` or `headtohead/rfdetr/`, and
no slurm log mentions the run. If it was scored, the numbers went to a terminal
and were never recorded.

Scored here with a standalone AP@0.5 implementation (stdlib, greedy IoU>=0.5
matching, monotonic precision envelope, all-point interpolation) applied
identically to both sides:

| | AP@0.5 | preds | TP | FP | precision | recall |
|---|---|---|---|---|---|---|
| RF-DETR (VIAME 720) | **0.1133** | 67,377 | 22,219 | 45,158 | 0.330 | 0.262 |
| DEIMv2 gen001 @1024 | **0.5945** | 56,231 | 51,595 | 4,636 | 0.918 | 0.609 |

**Do not take this at face value.** The RF-DETR run has anomalies that a model
comparison cannot survive:

| | RF-DETR | DEIMv2 |
|---|---|---|
| test images with >=1 prediction | **14,835 (44.4%)** | 28,027 (83.8%) |
| test sequences covered | **32 of 69** | 62 of 69 |
| predictions with ZERO overlap with any GT | **53.3%** | 4.6% |

- The **37 sequences RF-DETR returned nothing for contain 19,767 ground-truth
  fish — 23.3% of all test GT.** 33 of those 37 have GT. The inference input
  list was complete (33,434 images, all 69 sequences), so this is not a missing
  input.
- One sequence, `IFREMER-DropCam-29-Bio-Lactips-21072020`, has **40 GT fish
  across 300 images and drew 7,662 RF-DETR predictions** — a ~190x
  over-firing, and 21% of all its zero-overlap detections.

Geometry is *not* the problem: RF-DETR's box centres track ground truth closely
(cx p10/p50/p90 0.13/0.51/0.85 vs GT 0.13/0.48/0.85) and only 1,139 of 67,377
boxes fall outside image bounds, so the VIAME-CSV-to-kwcoco conversion is
placing boxes correctly. The failure is coverage and calibration, not
coordinates.

Two readings are consistent with this and the artifacts cannot separate them:

1. **Real operating-point behaviour.** RF-DETR's plugin hard-thresholds at 0.5.
   If it is poorly calibrated on the IFREMER/CDFW domains it simply emits
   nothing, and total silence on 33 GT-bearing sequences is what that looks
   like at that threshold.
2. **A partially broken inference run.** Silence on 54% of sequences combined
   with 190x over-firing on another is an odd shape for a model that reportedly
   scores 0.7166 on its own data.

## Open questions for review

1. Is the RF-DETR inference run trustworthy? Specifically, why zero output on
   37 of 69 sequences when the input list covered all of them?
2. If it is trustworthy, is the right comparison at the 0.5 floor at all?
   RF-DETR cannot produce a recall curve below it, so AP is arguably the wrong
   statistic; precision/recall at the operating point (0.330/0.262 vs
   0.918/0.609) may be the honest framing.
3. The AP figures above come from a standalone implementation, not
   `kwcoco eval`. `score_headtohead.sh` should be run to get authoritative
   numbers under the same protocol as every other figure in this project.
4. Given 2,140 vali tracks, what AP difference is actually resolvable? Every
   comparison in this project so far has been made on differences smaller than
   that.

## Artifacts

- source/tiled bundle audit: reproducible from `bundle/*.kwcoco.json` and
  `tiles_1024/*/tiles.kwcoco.json`
- head-to-head inputs: `rfdetr_test_inference/`, `headtohead/`
- the 0.5-floor rescore of DEIMv2: `headtohead/deim_preds_min05.kwcoco.json`
