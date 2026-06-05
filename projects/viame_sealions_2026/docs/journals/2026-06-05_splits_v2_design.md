# 2026-06-05 — Splits v2: test-window design for class coverage

## Problem with v1 splits

The current `*_norm.kwcoco.zip` test split was generated with
`sealion_pipeline.py:build_split_lookup` using
`test_years=('2009', '2019', '2024')` and `test_frac=0.25` —
contiguous 25% chunks per year. Per-class image counts in the
resulting `test_norm`:

| | bull | sam | female | juv | pup | NFS | DN | DP | neg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 test (current) | 455 | 414 | 334 | 482 | 193 | **0** | 1 | 17 | 133 |

Two failures:

1. **No NFS in test.** All NFS clusters in 2011, 2014, 2016,
   2022, 2023 — none in the v1 test years. We cannot measure
   how distracted the model is by NFS on held-out data.
2. **1 dead_nonpup.** DN is rare (19 in whole corpus), so this
   is more or less the best achievable, but worth tracking.

## Per-year per-class data (image counts)

```
year    tot  bull  sam  female  juv  pup   NFS  DN  DP   neg
2007    371   331  240  251     285  142    1   .   4    24
2008    490   411  401  368     461  213   16   7  27   266
2008W    87    29   50   76      86    3    .   .   .    84
2009    866   831  467  584     697  460    2   7  26    44
2010    198   177  126  128     163   66    1   .   4    68
2011    643   552  435  424     511  273   21   4  61    95   ← NFS cluster
2012     26    21   23   12      19    8    .   .   .    21
2014    670   536  420  321     385  243   30   8  25   172   ← NFS cluster
2015    787   483  523  418     592  247    6   .  46   207
2016    674   531  396  330     422  237   18   .  39   200   ← NFS cluster
2019    961   591  682  489     732  292    1   5  41    17
2021    981   110  114  102     137   78    2   4  20     .
2022    380   274  224  138     229  106   32   2  27   300   ← NFS cluster (DENSE)
2023    452   393  206  174     214  145   28   3  12   385   ← NFS cluster (DENSE)
2024    632   497  396  276     426  177   11   9  32   495   ← NFS cluster
```

**Key insight**: NFS clusters tightly within years. A 15%
contiguous chunk of 2022 (57 of 380 images) captures **all 32
NFS-bearing images**. Same for 2023 (68-image chunk → 28 NFS).
This is because NFS images are taken on specific flight days at
specific sites, not scattered.

## Sliding-window best-coverage analysis

For each year, I swept all contiguous chunks of size
{15%, 20%, 25%} and scored by (NFS present > 0) × 1000 +
(distinct classes present) × 10 + (DN present > 0) × 5 +
(NFS count). Best chunks per year:

```
year   frac  size  start  ncls  NFS  DN  notes
----   ----  ----  -----  ----  ---  --  -----
2008   15%    74    261    9     7   1   second-best NFS source
2009   15%   130    664    9     1   2   keep for continuity
2009   25%   216    650    9     1   3
2011   15%    96    304    9    14   1   NFS-rich
2014   15%   100    316    9    18   1   NFS-rich + all classes
2014   20%   134    318    9    30   1   ALL NFS in year
2016   15%   101    322    8    16   0   NFS-rich, no DN
2019   15%   144    528    7     1   0   needed 25% for all-9
2019   25%   240    528    9     1   1
2022   15%    57    119    8    32   0   ALL NFS in year (DENSE)
2023   15%    68    384    8    28   0   ALL NFS in year (DENSE)
2024   15%    95    537    9    11   1   all NFS + all classes
2024   25%   158    474    9    11   2
```

## Three candidate compositions (all class-complete)

### REC_minimal — 585 imgs, 7.1% of corpus

```
2009 [start=664, size=130]   9cls  NFS= 1  DN=2
2019 [start=528, size=240]   9cls  NFS= 1  DN=1   (25% — 15% missed 2 classes)
2022 [start=119, size= 57]   8cls  NFS=32  DN=0   ← key NFS chunk
2024 [start=474, size=158]   9cls  NFS=11  DN=2
```
Test per-class image counts: bull=460 sam=366 female=305 juv=423
pup=206 NFS=**45** DN=5 DP=27 neg=167. All 9 classes ✓.

Closest in size to the broken v1 (616). Continuity with prior
gen004 test-year choices, with 2022 added for NFS.

### REC_balanced — 653 imgs, 7.9% of corpus

```
2009 [start=664, size=130]   9cls  NFS= 1  DN=2
2019 [start=528, size=240]   9cls  NFS= 1  DN=1
2022 [start=119, size= 57]   8cls  NFS=32  DN=0   ← NFS cluster #1
2023 [start=384, size= 68]   8cls  NFS=28  DN=0   ← NFS cluster #2
2024 [start=474, size=158]   9cls  NFS=11  DN=2
```
Test per-class: NFS=**73**. Adds 2023 for NFS redundancy
(2 independent NFS clusters → more robust NFS AP measurement).
73 NFS images → can compute reliable NFS-as-distractor metric.

### REC_larger — 935 imgs, 11.4% of corpus

```
2009 [start=650, size=216]   9cls  NFS= 1  DN=3   (25%)
2019 [start=528, size=240]   9cls  NFS= 1  DN=1
2011 [start=304, size= 96]   9cls  NFS=14  DN=1   ← +
2014 [start=316, size=100]   9cls  NFS=18  DN=1   ← +
2022 [start=119, size= 57]   8cls  NFS=32  DN=0
2023 [start=384, size= 68]   8cls  NFS=28  DN=0
2024 [start=474, size=158]   9cls  NFS=11  DN=2
```
Test per-class: NFS=**105**, DN=8. 7 test years = strong
generalization signal. Costs ~320 more training images vs
balanced.

## FINAL recommendation (2026-06-05 late): OPT_F_12pct

User requirements after several iterations:
* Test 10–15% of corpus (statistical power for AP).
* All classes covered in test.
* **Train must retain the BULK of rare classes** (esp. NFS).

Designed by:
1. Keeping two mini-chunks that include partial NFS clusters
   (2014 [426:441] size=15 → 5 NFS; 2022 [115:130] size=15
   → 11 NFS).
2. Growing 2009/2019/2024 chunks to 35–40% size *without
   capturing additional NFS*, since 2009/2019 have ≤1 NFS
   each and 2024's 11 NFS are already captured at 25% chunk
   size. Growing these chunks pulls common-class imagery
   only.

### OPT_F_12pct (recommended primary)

| year | start | size | size% | first image | last image |
|---|---:|---:|---:|---|---|
| 2009 | 319 | 346 | 40% | `20090627_SSLC1936_C.JPG` | `20090708_SSLC3974_C.JPG` |
| 2014 | 426 |  15 |  2% | `20140628_SSLP1914_C.jpg` | `20140628_SSLS0145_C.jpg` |
| 2019 | 193 | 336 | 35% | `20190624_SEA LION ISLANDS_SSLC0412.jpg` | `20190628_MARMOT_SSLC0507.jpg` |
| 2022 | 115 |  15 |  4% | `20220627_AMAK+ROCKS_SSLS0226.jpg` | `20220627_BOGOSLOF_SSLC0462.jpg` |
| 2024 | 379 | 253 | 40% | `20240703_HOOK POINT_SLC00144_BB.jpg` | `20240718_OTTER_DJI0356_BB.jpg` |

* Test: **965 imgs (11.7% of corpus), 56,889 annots (12.0% of
  corpus annots).**
* Per-class test img counts: bull=780, sam=578, female=530,
  juvenile=721, pup=357, **NFS=29**, dead_nonpup=10, dead_pup=33,
  negative=231.
* **NFS in train**: 140 imgs / 27,681 anns — 89.7% of corpus
  NFS annotations retained for training.
* 4 independent NFS clusters in test: 2009 (1), 2014 (5),
  2022 (11), 2024 (11).
* All 9 classes present in test ✓.

### OPT_G_14pct (alternative — adds 2021 chunk)

Adds `2021 [128:324] size=196` for a 6th test year representing
the UAS-platform transition. Test grows to 1,161 imgs (14.1%
of corpus). NFS train-retention basically unchanged
(138 imgs / 27,676 anns — 89.7%). Generalization signal
across more flight platforms.

### Rejected alternatives (kept for reasoning trail)

| name | test imgs | NFS test% | why rejected |
|---|---:|---:|---|
| REC_minimal | 585 | 27% | only 1 NFS cluster (2022) |
| REC_balanced | 653 | 43% | NFS train-starved — backwards |
| REC_larger | 935 | 62% | severely NFS train-starved |
| REC_train_max | 480 | 14% | only 1 NFS cluster in test |
| REC_train_heavy | 495 | 17% | small test pool, 2 clusters |
| REC_train_diverse (15%) | 495 | 17% | test pool too small |
| REC_train_diverse_LARGER | 644 | 17% | still under 10% test |
| OPT_F_10pct | 799 | 17% | just under 10% threshold |

## Earlier recommendation (REC_train_diverse_LARGER — superseded)

After surfacing per-class TRAIN counts, REC_balanced was
revealed to be backwards on NFS — putting 43% of NFS images
(and 53% of NFS annotations) in test, leaving train
NFS-starved. User feedback: "we need to make sure NFS and rare
classes appear in TRAIN; test just needs some."

Solution: **truncate** the 2022 / 2014 NFS chunks (the
clusters fit inside 15 contiguous images) so we hold out just
~10 NFS imgs from each year for test, keeping the rest in
train. Then add a bigger 2009 + 2024 chunk for continuity and
all-9-class coverage.

**REC_train_diverse_LARGER:**

| year | start | size | size% | cls | NFS-test | NFS-train(yr) |
|---|---:|---:|---:|---:|---:|---:|
| 2009 | 650 | 216 | 25% | 9 | 1 | 1 |
| 2014 | 426 |  15 |  2% | 9 | 5 | 25 |
| 2019 | 528 | 240 | 25% | 9 | 1 | 0 |
| 2022 | 115 |  15 |  4% | 8 | 11 | 21 |
| 2024 | 474 | 158 | 25% | 9 | 11 | 0 |

Test total: 644 imgs (7.8% of corpus), 38,867 annots.
Train NFS: **140 imgs / 27,681 annots** (89.7% of corpus NFS
annotations retained for training).
Test NFS: 29 imgs across **4 independent test-year
clusters** (2009/2014/2022/2024) — robust distractor
measurement.

### Why this composition

- **Test size matches v1** (644 vs broken v1's 616 imgs).
  Statistical power for mAP is the same.
- **4 independent NFS clusters in test**: 2009 (1 NFS),
  2014 (5 NFS, older year), 2022 (11 NFS, recent),
  2024 (11 NFS, latest). NFS measurement is robust to any
  single cluster being unrepresentative.
- **NFS in train preserved**: 11 different years contribute
  NFS to train (vs only 2 absent: 2019, 2024). 27,681
  NFS annotations is plenty for the model to learn NFS as
  distractor.
- **dead_nonpup**: 7 in test, 42 in train (best achievable
  given corpus scarcity of 49 total).
- **`negative` (background)**: 140 in test, 2,238 in train —
  reasonable.
- Bumping 2009 and 2024 to 25% chunks (vs 15%) was free —
  NFS in those years is already fully captured by 15%
  chunks, so larger chunks only add common-class imagery
  to the test pool.

### Rejected alternatives

| name | test imgs | NFS test% | why rejected |
|---|---:|---:|---|
| REC_minimal | 585 | 27% | OK but only 1 NFS cluster (2022) |
| REC_balanced | 653 | 43% | NFS train-starved — backwards |
| REC_larger | 935 | 62% | Severely NFS train-starved (7,705 ann left) |
| REC_train_max | 480 | 14% | Only 1 NFS cluster in test |
| REC_train_heavy | 495 | 17% | 2 clusters but small test pool |
| REC_train_diverse (15%) | 495 | 17% | Test pool too small for user comfort |

## Original recommendation (superseded — kept for reasoning trail)

REC_balanced was the original best fit for the user's stated
constraints ("lower test to 15%, can bump to 25% for coverage",
"NFS is important"):

* Total ~8% — close to original v1 size, doesn't gut training.
* **2 independent NFS clusters** (2022 + 2023) — measurement
  robust to one cluster being unrepresentative.
* All 9 classes present via union of chunks.
* Year-held-out structure preserved (5 distinct test years).
* DN=5 — best achievable given corpus scarcity.

Per-year test fraction breakdown:
* 2009 = 130/866 = 15%
* 2019 = 240/961 = 25% (needed for all-9 coverage)
* 2022 = 57/380 = 15% (already saturates NFS)
* 2023 = 68/452 = 15% (already saturates NFS)
* 2024 = 158/632 = 25% (needed for all-9)

So "15% with 25% bump for class coverage where needed" matches
user's framing.

## Hard-coded split manifest (proposed)

The split definition belongs in **version control** inside the
kit project. Hard-coding by **image-name boundaries** (not
sorted-list indexes) is robust to upstream image-set changes —
if a new image gets added or removed, the start/end name still
identifies the chunk, and we can assert expected_size as a
tripwire.

Proposed file:
`projects/viame_sealions_2026/data/splits_v2.yaml`

```yaml
# Sea lion split definition v2 (replaces unpacked/splits_v1.json).
#
# Hard-coded test windows chosen for class coverage (especially
# NFS, which v1 missed entirely). Derived from sliding-window
# analysis on 2026-06-05 — see
# docs/journals/2026-06-05_splits_v2_design.md.
#
# Each test chunk is a contiguous range of images sorted by
# name within a year. The boundary image names ARE the
# definition; expected_size is a tripwire that fails if the
# upstream image set changes.
#
# vali is a seeded random 15% of the learn pool.

version: 2
test_chunks:
  - year: '2009'
    first_image: '20090708_SSLC3974_C.JPG'
    last_image:  '20090714_SSLC4990_C.JPG'
    expected_size: 130
    rationale: continuity with v1; all 9 classes; 1 NFS, 2 DN
  - year: '2019'
    first_image: '20190628_MARMOT_SSLC0507.jpg'
    last_image:  '20190701_LIGHTHOUSE ROCKS_SSLC0573.jpg'
    expected_size: 240
    rationale: 25% chunk needed for all-9 class coverage
  - year: '2022'
    first_image: '20220627_BOGOSLOF_SSLC0308.jpg'
    last_image:  '20220627_UNALASKA_SPRAY CAPE_SSLP0699.jpg'
    expected_size: 57
    rationale: captures all 32 NFS-bearing images in 2022
  - year: '2023'
    first_image: '20230706_UNIMAK_CAPE SARICHEF N_SLC00013.jpg'
    last_image:  '20230709_VSEVIDOF_SLS00401.jpg'
    expected_size: 68
    rationale: captures all 28 NFS-bearing images in 2023 (2nd NFS cluster)
  - year: '2024'
    first_image: '20240703_WOODED_FISH_SLC00303_BB.jpg'
    last_image:  '20240718_OTTER_DJI0356_BB.jpg'
    expected_size: 158
    rationale: 25% chunk for all-9 classes + 11 NFS + 2 DN

# Vali split: random 15% of learn pool (images not in any test
# chunk). Seeded.
vali:
  seed: 20260605
  frac: 0.15
  stratify_by: year   # stratify so vali year-distribution matches train

# Class-coverage assertions — fail loudly if any test chunk
# regresses on these.
test_class_coverage_required:
  - northern_fur_seal: 50    # NFS in test must be >= 50 images
  - dead_nonpup:        3    # DN in test must be >= 3 images
  - all_9_classes_present: true
test_size_bounds:
  min_images: 600
  max_images: 700
  expected_images: 653
```

## Proposed new tool

`projects/viame_sealions_2026/scripts/build_splits.py`

Single-purpose script. Reads `splits_v2.yaml` + the upstream
`unpacked/all_norm.kwcoco.zip` (the canonical source produced
by `sealion_pipeline.py`). Emits:

* `<out>/train_norm_v2.kwcoco.zip`
* `<out>/vali_norm_v2.kwcoco.zip`
* `<out>/test_norm_v2.kwcoco.zip`
* `<out>/learn_norm_v2.kwcoco.zip` (= train + vali)
* `<out>/splits_v2.json` (per-image assignment, full audit
  trail)
* `<out>/splits_v2_report.json` (class coverage, year
  distribution, assertion results)

Algorithm:
1. Load `all_norm.kwcoco.zip`.
2. For each `test_chunks` entry: find the image list for that
   year (sorted by name), locate `first_image` and `last_image`,
   assert the index range size equals `expected_size`. Mark
   those images as `split=test`.
3. Remaining images → `split=learn`.
4. For `learn` pool: stratify by year, seed RNG with
   `vali.seed`, assign `vali.frac` of each year's learn images
   to vali, rest to train.
5. **Hard assertions** before writing:
   - Every test_chunks entry produced exactly `expected_size`
     images.
   - `test_class_coverage_required` all satisfied.
   - `test_size_bounds.min ≤ |test| ≤ max`.
   - No image appears in more than one split.
   - All 9 categories present in train AND vali AND test.
6. Write bundles + manifest + report. Print the per-split
   per-class histogram so a sanity check is just looking at
   the report.

Shape of `build_splits.py` (sketch):

```python
"""Build splits_v2 from all_norm.kwcoco.zip and splits_v2.yaml."""
import argparse, json, random, sys, yaml
from collections import Counter, defaultdict
from pathlib import Path
import kwcoco

def find_chunk_bounds(year_imgs, first_name, last_name, expected):
    names = [name for name, _ in year_imgs]
    assert first_name in names, f"first_image {first_name!r} not in year"
    assert last_name in names, f"last_image {last_name!r} not in year"
    i0 = names.index(first_name); i1 = names.index(last_name) + 1
    assert i1 - i0 == expected, f"size mismatch: got {i1-i0}, expected {expected}"
    return i0, i1

def build_splits(all_norm, manifest, out_dpath):
    dset = kwcoco.CocoDataset(all_norm)
    year_imgs = defaultdict(list)
    for img in dset.dataset['images']:
        year_imgs[img['year']].append((img['name'], img['id']))
    for y in year_imgs:
        year_imgs[y].sort(key=lambda x: x[0])

    test_gids = set()
    for chunk in manifest['test_chunks']:
        i0, i1 = find_chunk_bounds(
            year_imgs[chunk['year']],
            chunk['first_image'], chunk['last_image'],
            chunk['expected_size'])
        for name, gid in year_imgs[chunk['year']][i0:i1]:
            test_gids.add(gid)

    learn_by_year = defaultdict(list)
    for img in dset.dataset['images']:
        if img['id'] not in test_gids:
            learn_by_year[img['year']].append(img['id'])

    rng = random.Random(manifest['vali']['seed'])
    vali_gids = set()
    for year, gids in learn_by_year.items():
        gids = sorted(gids)
        rng.shuffle(gids)
        n_vali = max(1, round(len(gids) * manifest['vali']['frac']))
        vali_gids.update(gids[:n_vali])

    splits = {}
    for img in dset.dataset['images']:
        gid = img['id']
        if gid in test_gids: splits[gid] = 'test'
        elif gid in vali_gids: splits[gid] = 'vali'
        else: splits[gid] = 'train'

    # ... assertions, write bundles, write report ...
```

This puts the split logic under version control where any
future agent reading the kit can audit it.

## Out of scope for this design doc

* The kit-project's `class_schemes.yaml` / scheme tooling
  changes ([[2026-06-05_phase2_change_inventory]]) are
  orthogonal — schemes consume `*_norm` regardless of how it
  was split.
* The actual tool implementation lands as a follow-up commit
  once the user signs off on REC_balanced (or another
  composition).

## What still needs user input

1. **Which composition?** REC_balanced (recommended) vs
   REC_minimal vs REC_larger. Tradeoff is purely how much
   training data we give up to get more NFS measurement
   robustness.
2. **`stratify_by: year` for vali?** v1 vali was random over
   the global learn pool, not year-stratified. Stratifying is
   better practice (vali year-distribution matches train,
   so vali AP is representative). Default to yes.
3. **Output location?** `unpacked/` (alongside v1) or
   `unpacked/v2/`? Recommend the latter to avoid clobbering
   v1 while we validate.
