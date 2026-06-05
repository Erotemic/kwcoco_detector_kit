# 2026-06-05 — Corpus audit: gen001-gen004 trained on the wrong bundle

**TL;DR.** Every kit training run to date (gen001 through gen004,
v1 through v6) used `training_ready_v1/`, a 1314-image
2021–2024-only slice with `n_cats=1`. The authoritative corpus is
`/data/Public/VIAME/viame_sealions_2026/unpacked/`, built by a
parallel sibling project at `/data/Public/VIAME/viame_sealions_2026/`
(notes, docs, scripts, training_runs registry — basically a
corpus-prep kit). It produces ~5× more training images, 14 years of
coverage, proper 3-cat kwcoco categories, NFS already folded into
`negative`, and a clean year-held-out test set. All measured gen004
"deployable detector" AP numbers (single_sealion 0.581,
pup_vs_nonpup 0.565) are SUBSET numbers and must be re-measured
against the real corpus before any claim sticks.

## What we have

### The bundles

Under `/data/Public/VIAME/viame_sealions_2026/unpacked/`:

| bundle | imgs | annots | n_cats | what |
|---|---:|---:|---:|---|
| `train_detection_v1.kwcoco.zip` | 6,462 | 370,957 | 3 | training pool |
| `vali_detection_v1.kwcoco.zip` | 1,140 | 64,821 | 3 | random-split val |
| `test_detection_v1.kwcoco.zip` | 616 | 39,873 | 3 | **year-held-out test** |
| `learn_detection_v1.kwcoco.zip` | 7,602 | 435,778 | 3 | train+vali |
| `all_detection_v1.kwcoco.zip` | 8,218 | 475,651 | 3 | full |
| `*_norm.kwcoco.zip` (paired) | same | same | 9 | pre-collapse with fine labels |

**Detector categories (`*_detection_v1`):** `sealion_adult`,
`sealion_pup`, `negative` (in that order in `all`/`train`/`vali`,
swapped pup↔adult in `test` — must verify the trained model's class
index is keyed off `target_order`, NOT first-seen order).

**Norm categories (`*_norm`, 9-class):** `bull`, `dead_nonpup`,
`dead_pup`, `female`, `juvenile`, `negative`, `northern_fur_seal`,
`pup`, `subadult_male`. `test_norm` has 8 (no `northern_fur_seal` —
no NFS in the test years).

### The collapse policy (from `prepare_report.json`)

```
sealion_adult ← bull, dead_nonpup, female, juvenile, subadult_male
sealion_pup   ← dead_pup, pup
negative      ← negative, northern_fur_seal
```

NFS-folded-into-negative is already baked. This is consistent with
the kit's `feedback_nfs_always_negative` memory, but means the
detector emits a `negative` foreground class — kit-side eval must
either drop `negative` predictions before AP, or use the `_norm`
splits and re-collapse to the kit's existing schemes.

### The splits (from `unpacked_build_report.md`)

- Seed: `20260514`
- Incomplete years (2013, 2017, 2018, 2021_incomplete) excluded.
- **Test = 25% contiguous chunk** from years `2009, 2019, 2024`,
  RNG-picked start.

  | year | total | test start | test size |
  |---|---:|---:|---:|
  | 2009 | 866 | 156 | 217 |
  | 2019 | 961 | 184 | 241 |
  | 2024 | 632 | 228 | 158 |

- Remaining images → `learn`, then 87/13 random split into
  `train`/`vali`.
- Train years: 2007, 2008, 2008W, 2009 (75%), 2010, 2011, 2012,
  2014, 2015, 2016, 2019 (75%), 2021, 2022, 2023, 2024 (75%).
- **Excluded from training: 2013, 2017, 2018, 2025** (incomplete
  annotations or no imagery — see `docs/data_inventory.md`).

### Path handling

- `file_name` is **relative** to bundle dir: e.g.
  `burlynb/Public/Redacted_Imagery/2007/20070609_SLAP5808_C.JPG`.
- `bundle_dpath` resolves on both arisia and namek
  (`/data/Public/VIAME/viame_sealions_2026/unpacked/`) — verified
  100/100 random sample images on namek.
- The unpacked tree carries images under
  `unpacked/burlynb/Public/Redacted_Imagery/<year>/...`. Same path
  works on arisia and namek.
- Per-image `source_zip`, `source_csv`, `source_kwcoco`
  fields are namek-absolute (`/media/joncrall/raid/...`). **Not
  used at training time** — provenance only. Safe to ignore.
- 3 known unreadable images noted in the build report (2 in 2009,
  1 in 2019); 40 unmatched 2019 CSV rows pointing at missing
  redacted images. Captured in the corpus build report — won't
  affect us at training time.

### The training_ready_v1 mistake

`training_ready_v1/` was built BEFORE the unpacked tree existed
(2021–2024 only). It survived as the kit project's `paths.sh`
default after the proper corpus landed, and every kit submit script
inherits from `paths.sh`. Stats:

| field | training_ready_v1 | *_detection_v1 |
|---|---:|---:|
| train images | 1,314 | **6,462** (4.9×) |
| years | 2021–2024 | **2007–2024** |
| n_cats | 1 (`sealion`) | 3 |
| fine labels | hidden in `source_category` (B/S/F/J/P/NFS/O/DP/DN) | first-class in `*_norm` |
| test split | random | year-held-out (2009/2019/2024) |
| NFS handling | dropped in `pup_vs_nonpup` scheme | folded to `negative` |

## Sibling corpus-prep kit

`/data/Public/VIAME/viame_sealions_2026/` is a full project tree
sibling to `kwcoco_detector_kit/projects/viame_sealions_2026/`:

```
notes.md
docs/{class_schemes.yaml,data_inventory.md,data_setup_plan.md,
      research_plan.md,training_runs.yaml,
      unpacked_build_report.md,unpacking_strategy.md}
scripts/{build_sealion_unpacked.py,build_scheme_kwcoco.py,
         convert_sealions_csv_to_kwcoco.py,
         project_sealion_detection_categories.py,
         sealion_pipeline.py,inventory_sealion_data.py,...}
```

It owns the corpus build and bundle definitions. Its `scripts/` even
has parallel-named launch/submit shells — but those were likely
templates that got forked into the kit project.

### Class-scheme divergence vs the kit project

Both projects define `single_sealion`, `pup_vs_nonpup`,
`lifestage_6cls` in their own `docs/class_schemes.yaml`. **They
disagree on NFS treatment:**

| scheme | sibling corpus kit | kit project |
|---|---|---|
| `single_sealion` | NFS → sealion (positive) | NFS dropped |
| `pup_vs_nonpup` | NFS dropped | NFS dropped |
| `lifestage_6cls` | NFS → northern_fur_seal | NFS → northern_fur_seal with distractor_classes |

The kit project's behaviour matches our
`feedback_nfs_always_negative` memory — sibling's `single_sealion`
is wrong by that policy. The detection_v1 bundle uses neither: it
makes NFS a third `negative` category, which is closer to the kit's
intent but in a 3-cat form the kit's schemes don't have.

The kit project's `class_schemes.yaml` is the authoritative one
going forward. The sibling kit's NFS-as-positive in `single_sealion`
should be considered stale.

### Scheme-mapping mechanics

Both projects' `build_scheme_kwcoco.py` and the kit project's
`apply_scheme_to_kwcoco.py` read `source_category` (letter code)
from each annotation. The `*_norm.kwcoco.zip` bundles have NO
`source_category` field — the class lives in `categories[].name` as
the full word (`bull`, etc.). **We need to either:**

1. Add a category-name → letter-code translation step in
   `apply_scheme_to_kwcoco.py` (cleanest), so it can consume
   `*_norm` directly.
2. Or build a "norm → letter-code-on-source_category" preprocess
   pass that produces an equivalent kwcoco with the letter codes
   baked in (more work, more state).
3. Or rewrite the kit's `class_schemes.yaml` to map full names
   (drop letter codes) and update the scheme tooling. Cleaner
   long-term but more breaking changes.

**Recommended:** option 1 — add a name↔code alias table in
`class_schemes.yaml` and update `build_scheme_kwcoco.py` to fall
back to category name when `source_category` is absent. Keeps every
existing letter-code-based artifact working.

## Action plan (proposed)

Before any training:

1. **Stop ongoing runs (2580, 2581).** They're training on the
   subset, no point continuing. The trajectories are still useful
   for hyperparameter selection — keep the journals — but don't
   burn more GPU.
2. **Repoint `projects/viame_sealions_2026/scripts/paths.sh`**
   (or specifically the `KCD_TRAIN_KWCOCO` / `KCD_VALI_KWCOCO` /
   `KCD_TEST_KWCOCO` variables) at the `*_norm.kwcoco.zip` bundles.
   Use `_norm`, NOT `_detection_v1`, so the kit's existing
   scheme tooling can collapse to whichever target scheme we want.
3. **Update `apply_scheme_to_kwcoco.py`** to read the class from
   `categories[ann.category_id].name` when `source_category` is
   absent. Add a name↔code alias in `class_schemes.yaml`
   (`bull→B`, `subadult_male→S`, ..., `northern_fur_seal→NFS`,
   `negative→O`, `dead_pup→DP`, `dead_nonpup→DN`).
4. **Update the kit's class_schemes.yaml** to reflect the
   corpus's reality (drop the "2021-2024 89955 annot" header
   comment; replace with the real 475651-annot, 2007-2024
   numbers).
5. **Add a corpus-validation pytest** to the kit project that
   refuses to launch a run unless `KCD_TRAIN_KWCOCO` points at
   one of the known-good bundle names. This is the trip-wire that
   would have caught this months ago.
6. **Run scheme projection** to produce
   `unpacked/by_scheme/{single_sealion,pup_vs_nonpup,lifestage_6cls}/{train,vali,test}.kwcoco.zip`
   from `*_norm.kwcoco.zip`. Commit those bundle paths into the
   kit project's training_runs registry as the new defaults.
7. **Restart training (gen005)** on the real corpus. Use the same
   gen004 recipe (dinov3_s + balance + fixed 640 + AMP +
   per_gpu_batch=8 + max_oversample=1) as a head-start — the
   recipe choice is informed by gen004 even though the AP numbers
   were measured on the subset.

## Things to validate AFTER pipeline fix

- Does `test_detection_v1` (year-held-out 2009/2019/2024) put a
  meaningful generalization gap on dinov3_s+balance? gen004 random
  split AP isn't comparable.
- Are the per-year AP gaps wide enough to need year-specific
  balancing or curriculum?
- 2025 RAW imagery — open question per `notes.md`. Out of scope
  for gen005 but worth tracking.
- Spotcheck flag from corpus notes:
  `_viz__sealions_2021_2024_sample40.kwcoco_f98c785e/loose-images/_anns/null/32_null_20240701_UNGA_ACHEREDIN POINT_SLP01717_KLS.jpg.view_ann.jpg`
  — per-class numbers were suspect until that was resolved. Status
  unknown.

## Why this slipped

`training_ready_v1/` was promoted to the kit project's default in
`paths.sh` when the kit project itself was forked from the sibling
corpus kit. The unpacked detection_v1 bundles landed later
(2026-05-14 mtime) but nobody re-pointed the defaults. Every
subsequent submit script inherited from `paths.sh` and was
internally consistent — there was no error signal to look for. The
trip-wire pytest in step 5 is meant to make this kind of drift
loud the next time it happens.

## Related

- [[2026-06-04_gen004_forensic_and_resume]] — the subset-AP
  measurements that now need redoing on the real corpus.
- [[project-correct-bundle-is-detection-v1]] — memory file
  capturing this finding.
- [[project-dinov3-s-is-pup-winner]] — recipe choice is still
  good; AP numbers there need an asterisk.
- `/data/Public/VIAME/viame_sealions_2026/docs/research_plan.md`
  — the sibling kit's phase plan. Worth re-reading; it predates
  our gen004 ablations.
