# 2026-06-05 — Canonical data-prep pipeline: zipfiles → unpacked → splits

The complete pipeline from VIAME Girder zipfiles to the
`train_norm` / `vali_norm` / `test_norm` kwcoco bundles is
captured in **one file**:

> `/data/Public/VIAME/viame_sealions_2026/scripts/sealion_pipeline.py`

Everything else (`build_sealion_unpacked.py`,
`convert_sealion_unpacked_to_kwcoco.py`,
`convert_sealions_csv_to_kwcoco.py`,
`project_sealion_detection_categories.py`) is a thin CLI
wrapper that delegates to functions in `sealion_pipeline.py`.

**Authoritative entrypoint:**
`python3 build_sealion_unpacked.py --repo $REPO --unpacked $UNPACKED`
or the equivalent `sealion_pipeline.run_pipeline(repo, unpacked,
seed=20260514)`.

**Where the pipeline was authored:**
`/data/Public/VIAME/viame_sealions_2026/` — a sibling project
to `kwcoco_detector_kit/projects/viame_sealions_2026/`. It has
its own `docs/`, `scripts/`, `.venv/`, `tests/`,
`training_runs.yaml`. It owns the corpus build; we (the kit
project) are downstream consumers.

**Where the design + outputs are documented:**

* `docs/data_setup_plan.md` — initial intent / open questions
* `docs/unpacking_strategy.md` — folder layout invariants
  (kwcoco at or above image paths, no `..` in `file_name`)
* `docs/data_inventory.md` — per-year zip vs CSV image counts
  before the build
* `docs/unpacked_build_report.md` — output of the build
  (split policy used, per-year chunk start/size, validation
  results)
* `notes.md` — provenance (Girder folder IDs, download
  commands, raw histograms)
* `unpacked/pipeline_report.json` — machine-readable run
  report from the last build
* `unpacked/splits_v1.json` — image-by-image split assignments

## Stage-by-stage pipeline

### Stage 0 — source inputs (immutable)

`SourceSpec` list at `sealion_pipeline.py:36-55` enumerates
the 18 redacted source years. Each spec ties:

* a **zip** under `burlynb/Public/Redacted_Imagery/<YEAR>.zip`,
* an **image dir** under
  `unpacked/burlynb/Public/Redacted_Imagery/<YEAR>/` (the
  2024 zip extracts to `2024_ForDetections/`, all others
  match the zip stem),
* one or two **CSV** annotation files. The CSV is preferred
  in this order:
  * `IncludesNewAnnotations/<YEAR>_annotations.csv` (newer,
    email-delivered, used for 2008/2009/2011/2022/2023/2024)
  * `Redundant/<YEAR>_annotations.csv` (older, used for
    2007/2008W/2010/2012/2014/2015/2016/2019/2021)
  * `Redundant/<YEAR>_detections_incomplete.csv` (flagged
    incomplete; built but EXCLUDED from combined splits)

| year | CSV used | complete? |
|---|---|---|
| 2007 | Redundant/2007_annotations.csv | ✓ |
| 2008 | IncludesNewAnnotations/2008_annotations.csv | ✓ |
| 2008W | Redundant/2008W_annotations.csv | ✓ |
| 2009 | IncludesNewAnnotations/2009_annotations.csv | ✓ |
| 2010 | Redundant/2010_annotations.csv | ✓ |
| 2011 | IncludesNewAnnotations/2011_annotations.csv | ✓ |
| 2012 | Redundant/2012_annotations.csv | ✓ |
| **2013** | Redundant/2013_detections_incomplete.csv | **× excluded** |
| 2014 | Redundant/2014_annotations.csv | ✓ |
| 2015 | Redundant/2015_annotations.csv | ✓ |
| 2016 | Redundant/2016_annotations.csv | ✓ |
| **2017** | Redundant/2017_detections_incomplete.csv | **× excluded** |
| **2018** | Redundant/2018_detections_incomplete.csv | **× excluded** |
| 2019 | Redundant/2019_annotations.csv | ✓ |
| 2021 | Redundant/2021_annotations.csv (+ incomplete CSV also built but not split) | ✓ (complete only) |
| 2022 | IncludesNewAnnotations/2022_annotations.csv | ✓ |
| 2023 | IncludesNewAnnotations/2023_annotations.csv | ✓ |
| 2024 | IncludesNewAnnotations/2024_annotations.csv | ✓ |

`is_incomplete_csv(csv_rel)` (line 427) detects incomplete
CSVs by name substring; `collect_complete_norm_sources`
(line 535) filters them out before split assignment.

### Stage 1 — unpack (`unpack_source`, line 152)

For each spec:
1. Hash the source zip (SHA256 over sorted
   filename + size + CRC — `zip_manifest_hash` at line 123).
2. If `source_manifest.json` already exists in the target
   dir with a matching hash, skip extraction (idempotent).
3. Otherwise extract into a tempdir, assert exactly one
   top-level folder, move it to the final
   `unpacked/burlynb/Public/Redacted_Imagery/<YEAR>/` path.
4. Write the manifest.

**Output**: per-year image dirs + `source_manifest.json`
recording zip path, size, mtime_ns, SHA256, and image count.

### Stage 2 — raw kwcoco (`convert_csv_to_raw_kwcoco`, line 237)

For each (image dir, CSV) pair:
1. Build two indexes of images on disk (exact name +
   normalized `[A-Z0-9]`-only stem).
2. Read CSV with `csv.DictReader`. Columns expected: `IMAGE`,
   `TL_X`, `TL_Y`, `BR_X`, `BR_Y`, `CLASS`, `ID`, `FRAME`,
   `REVIEW_D`, `REVIEW_C`, `ATTRIBUTE`, `TARGET_LENGTH`.
3. Match each row to an image:
   * `exact`: 1 hit on filename
   * `normalized`: 1 hit after case + non-alphanumeric strip
   * `ambiguous_exact` / `ambiguous_normalized`: skip, log
   * `missing`: skip, log (e.g. 40 unmatched 2019 rows)
4. Add every disk image to the kwcoco (whether annotated or
   not — **this is why 10% of train images have zero
   annotations**: they exist in the imagery zip but have no
   matching CSV row).
5. For each matched row, write an annotation. Validates
   `w>0`, `h>0`; warns on out-of-bounds bboxes but does not
   drop them.
6. Preserves the original CSV row under `ann["viame"]["raw_row"]`
   for full provenance.
7. Unreadable images (PIL.Image.open fails) are added with
   `width=height=None` and logged in `bad_images`. 3 such
   files: 2009/20090626_SSLS1294_C.JPG,
   2009/20090626_SSLS1472_C.JPG,
   2019/20190627_CAPE FAIRFIELD_SSLC0635.jpg.

**Output**: `<csv_stem>.raw.kwcoco.zip` + `.report.json`
sibling to the imagery.

### Stage 3 — normalized kwcoco (`normalize_kwcoco`, line 368)

For each raw kwcoco:
1. Look up each annotation's class via
   `raw.index.cats[ann["category_id"]]["name"]`.
2. Look up `ann["viame"]["attribute"]` (CSV `ATTRIBUTE`
   column).
3. Apply `normalize_class(raw_class, attr)` (line 350):
   * Strip whitespace, lowercase, collapse internal spaces.
   * Look up in `CLASS_ALIASES` (line 58). Falls back to
     `'negative'` if no alias matches (so unknown / empty
     classes become negatives).
   * `role`: `'positive'` iff normalized name is in
     `POSITIVE_CLASSES = {bull, subadult_male, female,
     juvenile, pup, dead_nonpup, dead_pup}` else
     `'negative'`.
   * `negative_kind` (only for negatives):
     `'water_region'` if `ATTRIBUTE` contains `"NOTE WATER"`,
     else `'northern_fur_seal'`, else `'background'`, else
     `'other'`.
4. Write under `ann["normalized"] = {class, source_class,
   role, negative_kind}` so the normalization decision is
   auditable per-annotation.

The full **CLASS_ALIASES** table (line 58–88) is the source
of truth for raw-CSV-string → normalized name:

| raw CSV strings | normalized class |
|---|---|
| `b`, `bull` | `bull` |
| `s`, `sam` | `subadult_male` |
| `f`, `fem`, `female` | `female` |
| `j`, `juv`, `juvenile` | `juvenile` |
| `p`, `pup` | `pup` |
| `dn`, `deadnp`, `dead np`, `dead non pup`, `dead non-pup` | `dead_nonpup` |
| `dp`, `deadpup`, `dead pup`, `dead-pup` | `dead_pup` |
| `nfs`, `furseal`, `fur seal` | `northern_fur_seal` |
| `o`, `background`, `unknown`, `''`, `age_sex` | `negative` |

The 9 output categories from this stage are:
`bull, subadult_male, female, juvenile, pup, dead_nonpup,
dead_pup, northern_fur_seal, negative`. **This is the
authoritative class vocabulary.**

**Output**: `<csv_stem>.norm.kwcoco.zip` + `.report.json`.

### Stage 4 — split assignment (`build_split_lookup`, line 493)

`build_combined` (line 550) drives this:

1. Collect all complete norm sources (skips incomplete
   CSVs).
2. **First pass — year-held-out test split** (`build_split_lookup`):
   * Seed: `20260514`.
   * For each source: load image names, sort.
   * If `year ∈ {2009, 2019, 2024}`:
     * `n_test = ceil(len(image_names) * 0.25)`
     * `start = rng.randint(0, len(image_names) - n_test)`
     * Test set = contiguous chunk `image_names[start:start+n_test]`.
   * Recorded chunks (from `unpacked_build_report.md`):

     | year | total | start | size |
     |---|---:|---:|---:|
     | 2009 | 866 | 156 | 217 |
     | 2019 | 961 | 184 | 241 |
     | 2024 | 632 | 228 | 158 |

   * Else (non-test year): every image gets `'learn'`.
3. **Second pass — learn → train/vali** (`build_combined`
   lines 561-572):
   * New RNG seeded with `seed + 1 = 20260515`.
   * Collect all `(source_kwcoco, image_name)` keys with
     `split == 'learn'`, sort, shuffle.
   * Take first `round(len(learn) * 0.15)` as `vali`.
   * Rest → `train`.

**Caveat — vali is NOT year-stratified.** The shuffle is
global across all learn images; the 15% vali split is
random over the combined learn pool. In practice the per-year
counts roughly mirror the year distribution, but this is not
guaranteed — see the per-year vali histogram from
`train_detection_v1` stats for the actual breakdown.

**Output**: `unpacked/splits_v1.json` — every image's
assignment recorded with `{source_kwcoco, image_name, year,
split, train_vali_split}`.

### Stage 5 — combined kwcoco bundles (`build_combined`, line 550)

For each output spec, instantiate an empty kwcoco, then
walk every source norm kwcoco and copy the images whose
split assignment matches:

| output bundle | allowed splits |
|---|---|
| `all_norm.kwcoco.zip` | all (no filter) |
| `learn_norm.kwcoco.zip` | {`train`, `vali`} |
| `train_norm.kwcoco.zip` | {`train`} |
| `vali_norm.kwcoco.zip` | {`vali`} |
| `test_norm.kwcoco.zip` | {`test`} |

When an image is copied:
* `file_name` is rewritten as `rel_prefix / source_file_name`
  where `rel_prefix = image_dpath.relative_to(unpacked)`.
  So the combined bundle stores paths like
  `burlynb/Public/Redacted_Imagery/2007/20070609_SLAP5808_C.JPG`
  relative to `unpacked/`.
* `source_kwcoco` is set to the per-year norm.kwcoco.zip
  path (this is an **absolute** namek path; it's
  provenance-only and unused at training time).
* `split` field is set.

Categories are unioned across sources via
`dst.ensure_category(name=cname)`, so the combined
bundle ends up with the union of all 9 normalized
categories. `test_norm` ends up with only 8 (no
`northern_fur_seal` — NFS not present in 2009/2019/2024
test chunks).

**Output**: 5 combined `*_norm.kwcoco.zip` files under
`unpacked/` + `combined_report.json` + `pipeline_report.json`.

### Stage 6 — detector-ready projection (`project_sealion_detection_categories.py`)

This is a **separate** script run after `sealion_pipeline.py`
completes. It collapses the 9-category `*_norm` bundles to a
3-category detector projection:

```python
ADULT_CLASSES    = {bull, subadult_male, female, juvenile, dead_nonpup}
PUP_CLASSES      = {pup, dead_pup}
NEGATIVE_CLASSES = {negative, northern_fur_seal}
```

Output: `*_detection_v1.kwcoco.zip` for each of all/learn/train/
vali/test, plus `prepare_report.json`.

**Important**: this is just one possible projection. The
kit project's class schemes (`single_sealion`,
`pup_vs_nonpup`, `lifestage_6cls`) are different
projections and should be derived directly from
`*_norm.kwcoco.zip`, NOT from `*_detection_v1.kwcoco.zip`.
The detection_v1 collapse loses the NFS distractor
information that `lifestage_6cls` needs.

## Reproducibility properties

* **Single seed** (`20260514`) controls all randomness. Test
  chunk starts use this seed; vali split uses `seed + 1`.
* **Idempotent unpack** via zip-manifest SHA256.
* **`source_zip`, `source_csv`, `source_kwcoco`,
  `viame.raw_row`** all preserved on every annotation for
  full audit trail.
* **`unpacked/splits_v1.json`** is the canonical record of
  every image's assignment — load this if you ever need to
  re-derive splits without re-running the pipeline.
* **`pipeline_report.json`** captures the full per-stage
  status and histograms from the last successful run.
* **`source_manifest.json`** in each per-year directory
  records the source zip's hash and the per-source
  annotation manifests.

## How to rebuild the corpus from scratch

```bash
# On namek (where the raw zips live):
cd /media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026
source /data/Public/VIAME/viame_sealions_2026/.venv/bin/activate
python3 /data/Public/VIAME/viame_sealions_2026/scripts/build_sealion_unpacked.py \
    --repo /media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026 \
    --unpacked /data/Public/VIAME/viame_sealions_2026/unpacked \
    --seed 20260514
# optionally:
python3 /data/Public/VIAME/viame_sealions_2026/scripts/project_sealion_detection_categories.py \
    --unpacked /data/Public/VIAME/viame_sealions_2026/unpacked
```

`--force-unpack` to re-extract zips ignoring the manifest
hash. `--force-convert` to re-run raw + norm conversions.
Neither is needed under normal circumstances.

## Known data caveats

From `unpacked_build_report.md`:
* 3 unreadable image files (2009 ×2, 2019 ×1).
* 40 unmatched 2019 CSV rows (CSV references images not in
  the redacted zip).
* All other validation pass: no missing image paths, no
  `..` in any kwcoco file_name.
* ~10% of train images have ZERO annotations — these are
  images present in the redacted zip that the CSV didn't
  annotate. They contribute to negative supervision at
  training time.

## Open data questions (from `docs/data_setup_plan.md`)

Unresolved as of 2026-05-21:
* Are `*_detections_incomplete.csv` for 2013/2017/2018
  usable, or just legitimately incomplete?
* Does `Redundant/` mean "image duplicates" or "superseded
  CSVs"?
* 2025 has annotations + RAW imagery but no redacted
  imagery on disk. Out of scope for gen005 but worth
  tracking.
* Annotation conversion spotcheck flagged on this image —
  status unknown:
  `_viz__sealions_2021_2024_sample40.kwcoco_f98c785e/loose-images/_anns/null/32_null_20240701_UNGA_ACHEREDIN POINT_SLP01717_KLS.jpg.view_ann.jpg`

## Canonical pointer

If a future agent asks "where is the pipeline documented?":

* **Code**: `/data/Public/VIAME/viame_sealions_2026/scripts/sealion_pipeline.py`
* **Design**: `/data/Public/VIAME/viame_sealions_2026/docs/unpacking_strategy.md`
* **Build output**: `/data/Public/VIAME/viame_sealions_2026/docs/unpacked_build_report.md`
* **Per-year inventory**: `/data/Public/VIAME/viame_sealions_2026/docs/data_inventory.md`
* **Last-run JSON report**: `/data/Public/VIAME/viame_sealions_2026/unpacked/pipeline_report.json`
* **Per-image split record**: `/data/Public/VIAME/viame_sealions_2026/unpacked/splits_v1.json`
* **Phase summary (this file)**: this journal entry in the kit project.

Related: [[2026-06-05_corpus_audit_wrong_bundle]],
[[2026-06-05_phase2_change_inventory]].
