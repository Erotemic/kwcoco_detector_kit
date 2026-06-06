# 2026-06-05 — Phase 2: what needs to change to consume `*_norm` (READ-ONLY audit)

Companion to
[[2026-06-05_corpus_audit_wrong_bundle]]. Goal: enumerate every
location that needs to change before we can train on
`unpacked/*_norm.kwcoco.zip`, without changing anything yet.

The kit (`kwcoco_detector_kit/`) is a generic toolkit and is **not**
modified. All changes are in
`projects/viame_sealions_2026/`. The kit's data-layer modules
(`tile.py`, `coco_export.py`, `merge.py`, `balance_mscoco.py`)
already speak category **names** — no kit changes are needed.

## What stays correct as-is

* **`kwcoco_detector_kit/data/tile.py`** (line 295: passthrough
  whitelist includes `source_category`; lines 304–313: fallback
  reads `src_dset.cats.get(cid).name`). Category-name-based;
  unchanged.
* **`kwcoco_detector_kit/data/coco_export.py`** (filters by
  `src_dset.cats.get(src_cid).name`). Unchanged.
* **`kwcoco_detector_kit/data/merge.py`** (filters by
  `src_cat["name"] in target_name_to_new_cid`). Unchanged.
* **`kwcoco_detector_kit/data/balance_mscoco.py`** (bucket
  names = category names). Unchanged.
* **`kwcoco_detector_kit/eval/kwcoco_eval.py`** (distractor
  pruning at the kwcoco data layer, by category name).
  Unchanged.
* **The distractor mechanism overall.** `distractor_classes`
  in `class_schemes.yaml` → `_launch_train.sh` env var →
  passed to kit eval → kit prunes by category name → sidecar
  `detect_metrics.<suffix>.json`. Works regardless of source
  bundle. Confirmed for `lifestage_6cls.distractor_classes:
  [northern_fur_seal]`.

## What needs to change (project-side only)

### 1. `docs/class_schemes.yaml`
Currently keys mappings by letter codes (`B`, `S`, `F`, `J`,
`P`, `NFS`, `O`, `DP`, `DN`). The `*_norm` bundles use full
names. Need either:

* **Option A — replace keys with full names**:
  ```yaml
  pup_vs_nonpup:
    mapping:
      pup: pup
      bull: nonpup_sealion
      subadult_male: nonpup_sealion
      female: nonpup_sealion
      juvenile: nonpup_sealion
    drop: [northern_fur_seal, negative, dead_pup, dead_nonpup]
  ```
  Cleaner; long-term right answer. Letter-code source is dead.
* **Option B — add `aliases:` block** at top of YAML
  (`bull: B`, etc.) and have the scheme tool consult it as a
  fallback. Less code change, but keeps two naming systems.

**Recommend A.** Letter codes were a `training_ready_v1`
artifact; the corpus authoritative form uses full names.

Also: `single_sealion.drop` currently lists `[NFS, O, DN]`.
Translated: `[northern_fur_seal, negative, dead_nonpup]`. **DP
is currently MAPPED → sealion** (line 49). Verify intent:
should `dead_pup` count as a positive `sealion` instance? In
the corpus's `*_detection_v1` policy `dead_pup → sealion_pup`,
so the project's intent matches. Keep.

Header comment at top of YAML (lines 25–27) lists the
2021–2024 89,955-annot histogram; **replace with the
2007–2024 475,651-annot histogram from the real corpus** (use
the `*_norm` numbers from `unpacked_build_report.md`).

### 2. `scripts/build_scheme_kwcoco.py`
Line 131: `src_cat = ann.get("source_category")`. Replace with:
```python
src_cat = src_dset.cats[ann["category_id"]]["name"]
```
Drop the `source_category` passthrough block (line 144) — the
new source bundles already have the correct first-class
category. Keep emitting it on output for downstream tooling
that expects it, but populate from the looked-up name.

### 3. `scripts/apply_scheme_to_kwcoco.py`
No direct code change; it re-exports `load_scheme` and
`remap_split` from `build_scheme_kwcoco.py`. Will pick up the
fix automatically. The CLI docstring (line 8) and
`--src` help text (line 48) reference `source_category` — update
the text to "annotation category name (from
`src_dset.cats[ann.category_id].name`)" so future agents don't
look for a non-existent field.

### 4. `scripts/paths.sh`
Lines 79–80:
```bash
export KCD_TRAINING_READY_DIR="${KCD_TRAINING_READY_DIR:-$KCD_DATA_DPATH/training_ready_v1}"
export KCD_SCHEMES_DIR="${KCD_SCHEMES_DIR:-$KCD_TRAINING_READY_DIR/by_scheme}"
```

Repoint to:
```bash
export KCD_UNPACKED_DIR="${KCD_UNPACKED_DIR:-$KCD_DATA_DPATH/unpacked}"
export KCD_TRAIN_NORM="${KCD_TRAIN_NORM:-$KCD_UNPACKED_DIR/train_norm.kwcoco.zip}"
export KCD_VALI_NORM="${KCD_VALI_NORM:-$KCD_UNPACKED_DIR/vali_norm.kwcoco.zip}"
export KCD_TEST_NORM="${KCD_TEST_NORM:-$KCD_UNPACKED_DIR/test_norm.kwcoco.zip}"
export KCD_SCHEMES_DIR="${KCD_SCHEMES_DIR:-$KCD_UNPACKED_DIR/by_scheme}"
```

Open question: keep `KCD_TRAINING_READY_DIR` as a legacy
read-only pointer (so historic scripts still resolve), or
delete it? **Recommend keeping it for now**, retiring after
gen005 is shaken down.

### 5. `scripts/_launch_train.sh`
Multiple touchpoints:

| Line | What | Change |
|---|---|---|
| 42–48 | Reads `train.kwcoco.zip` etc. from `$KCD_TRAINING_READY_DIR` | Point at the new `KCD_TRAIN_NORM` / `KCD_VALI_NORM` / `KCD_TEST_NORM` vars (or build a temporary universal-collapsed view) |
| 251–261 | Tile step: `--category_names sealion` (hard-coded single class) | Pass the union of `*_norm` category names so all positives survive: `bull,subadult_male,female,juvenile,pup,northern_fur_seal,negative,dead_pup,dead_nonpup`. Apply-scheme then collapses to the scheme's `target_order`. |
| 316–332 | `apply_scheme_to_kwcoco.py` invocation | No change; tool will work once #2 above lands. |

The tile-cache key (line 191) is keyed by tile geometry +
writer_fingerprint and is scheme-agnostic, so the tile cache
is invalidated automatically when the input bundle name
changes. Good — no stale-cache risk.

### 6. Submit scripts (40+)
The `submit_train_*.sh` scripts that hard-code `KCD_SCHEME`
and `KCD_CATEGORY_NAMES` are fine — those are scheme-derived
and unchanged. The implicit `KCD_TRAINING_READY_DIR` defaulting
via `paths.sh` is fixed by #4. **No per-script edits needed**.
New submit scripts for gen005 should use `KCD_TRAIN_NORM` etc.
as explicit overrides if needed.

### 7. `scripts/rescore_per_checkpoint.py`,
`recompute_sealion_ap.py`, `rescore_all_runs.py`

* `recompute_sealion_ap.py:36`:
  `DEFAULT_DISTRACTORS = ("northern_fur_seal",)` — **keep**.
  This matches the corpus and is the right default.
* All three read the scheme YAML's `distractor_classes` —
  works as soon as #1 lands (scheme yaml uses correct target
  names, which match what eval prunes by).
* `rescore_per_checkpoint.py` resolves the eval kwcoco from
  the project's `paths.sh` defaults. After #4, it will
  automatically use the new test bundle. No code change
  needed.

### 8. Tests
`tests/unit/` and `tests/expensive/` reference
`training_ready_v1` paths in fixtures and integration tests.
Need to either:
* Repoint fixtures at `*_norm`-shaped synthetic bundles
  (tests are supposed to be synthetic; the contract is
  category-name-driven so this should just work).
* Skip-condition `tests/expensive/` until `*_norm` is
  available in the docker image's bake step.

Detailed test inventory pending — separate audit before any
edits there. Note for future agent.

### 9. Docs
* `projects/viame_sealions_2026/README.md` — paths table and
  workflow §1 reference `training_ready_v1` and
  `by_scheme/`. Replace with the new tree once the code
  changes land.
* `projects/viame_sealions_2026/AGENT.md` — same.
* `projects/viame_sealions_2026/docs/class_schemes.yaml`
  header (see #1).

## Architectural choice to make BEFORE editing

There are two clean ways to integrate `*_norm`:

### Path A — teach the pipeline full-name categories (preferred)

* Scheme yaml uses full names (#1 Option A).
* `_launch_train.sh` tiles `*_norm` directly with the full
  category-name union.
* Apply-scheme collapses to scheme target_order.
* Eliminates the letter-code intermediate. Less code, less
  state, less drift.
* **Cost:** every scheme yaml entry needs rewriting; if other
  scripts (not surveyed) read letter codes from yaml, they
  break.

### Path B — adapt `*_norm` to the existing `training_ready_v1` interface

* Add a pre-step: build
  `unpacked/training_ready_v2/{train,vali,test}.kwcoco.zip`
  with:
  * a single `sealion` category,
  * every annotation's `source_category` populated with the
    LETTER CODE corresponding to its original
    full-name category (bull→B, etc.).
* Existing scheme yaml, apply-scheme, launch-script all
  unchanged.
* **Cost:** one extra build artifact, one extra build script,
  the letter-code intermediate persists indefinitely. Plus
  re-translation work that's purely cosmetic.

**Recommend Path A.** Cleaner, less ceremony, gets rid of an
encoding (letter codes) that only ever existed because
`training_ready_v1` collapsed too aggressively.

## Sanity checks BEFORE running gen005

1. Run `apply_scheme_to_kwcoco.py --scheme pup_vs_nonpup` on
   `vali_norm.kwcoco.zip` and assert:
   - Output has exactly 2 categories (`pup`, `nonpup_sealion`).
   - Annotation count = (pup) + (bull+subadult_male+female+juvenile)
     from `vali_norm` source counts = 14,051 + 4,328 + 2,679 +
     17,606 + 19,874 = **58,538**. Cross-check against actual.
   - No annotation with category in `{northern_fur_seal,
     negative, dead_pup, dead_nonpup}` survives.
2. Same for `single_sealion` (expect 1 category, count =
   pup+dead_pup+bull+subadult_male+female+juvenile = 58,718).
3. Same for `lifestage_6cls` (expect 6 categories, NFS
   present as distractor; counts per class match source).
4. Tile + apply on a single image and visually spot-check the
   tiles + annotations match the corpus image.
5. Run kit eval with `distractor_classes=northern_fur_seal`
   on a dummy 6-class detection result; verify the sidecar
   `detect_metrics.<suffix>.json` is produced and the
   distractor class IS in the full metrics but NOT in the
   nocls metrics.
6. Smoke-test the WDS path end-to-end on CPU per
   `feedback_wds_path_smoke_first` before any slurm submit.
7. Confirm test split is year-held-out by checking image
   `year` field distribution after tile + apply.

Only after all 7 pass: queue gen005.

## Open questions for the user

1. **Path A vs Path B?** Default to A unless there's a
   reason to keep the letter-code interface.
2. **Should NFS be a distractor in `pup_vs_nonpup` /
   `single_sealion`?** Currently scheme yaml drops NFS at
   the data layer (so it acts as implicit hard negative).
   Could promote it to a learnt distractor class as an
   ablation later, but not in gen005.
3. **What about the corpus's `negative` (background hard
   negative) category?** Currently scheme yaml drops it.
   The kit's mine.py picks negatives separately, so this
   is consistent. But the corpus already curated ~12k
   background boxes; we may be discarding useful
   information. Worth a future ablation.
4. **Scancel 2580 / 2581 now?** They're training on the
   subset. Recipe info is logged. No reason to keep
   burning GPU, but per memory rule we need explicit
   permission before scancel.
5. **gen005 scheme order**: research_plan.md proposed
   `single_sealion → pup_vs_nonpup → lifestage_6cls`. With
   gen004 showing pup_vs_nonpup works with dinov3_s+balance,
   should gen005 lead with pup_vs_nonpup (operational
   target) or go back to single_sealion baseline first to
   establish the corpus's headline AP?

Nothing in this document changes code. All edits enumerated
above are deferred until the architectural choice (A vs B)
is confirmed.
