# 2026-05-26 — passthrough whitelist wasn't enough: tile writer needed to *stamp* source_category
> [!WARNING]
> **2026-06-05 retroactive correction.** Every training run referenced
> below used `training_ready_v1/` — a 1,314-image, 2021–2024-only,
> n_cats=1 subset of the actual corpus. The authoritative bundles at
> `/data/Public/VIAME/viame_sealions_2026/unpacked/*_norm.kwcoco.zip`
> have 6,462 train images across 14 years (2007–2024) with 9 proper
> kwcoco categories. All AP numbers in this journal are SUBSET-only
> and not comparable to gen005+ runs trained on the real corpus. The
> recipe-level findings (dinov3_s + balance beats hgnetv2_n, EMA
> beats best_stg1, multiscale OOMs at 2-GPU 512–768) likely
> generalize; the absolute AP numbers (kit AP 0.5xx, in-train mAP
> 0.1xx) do not. See `2026-06-05_corpus_audit_wrong_bundle.md`.


## Context

Follow-up to [2026-05-26_first_baseline_attempt.md](2026-05-26_first_baseline_attempt.md).
After shipping the fail-fast guard + writer-fingerprint (kit `a1aa45b`),
nuking `/data/users/jon.crall/kcd_sealion/tile_cache/_universal/` on
arisia, and resubmitting `pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_v2`
as slurm job 2494, the rebuilt universal tile bundle *still* produced
0 annotations after apply_scheme — same symptom, same class of bug.
The fail-fast guard correctly caught it in seconds rather than 48 h.

## Symptom

From `slurm_logs/pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_v2-2494.out`:

```
tile.multiscale: wrote 229617 tiles (pos=65614, neg=164003, dropped_neg=0)
  annotations: 903603
  -> /data/users/jon.crall/kcd_sealion/tile_cache/_universal/518d4bd0/tiles.kwcoco.zip

apply pup_vs_nonpup to train: ... -> ...
{
  "n_images": 229617,
  "n_annotations": 0,
  "per_target_class": {},
  "dropped_by_source": {},
  "n_unknown_source_categories": 903603
}
ERROR: scheme_applied/train.kwcoco.zip has 0 annotations.
  Refusing to start a 48h training run with empty targets.
```

The tile step emitted 903,603 annotations; every single one came back
as `n_unknown_source_categories` in apply_scheme.

## Diagnosis

- Docker image was fresh (built 2026-05-26T15:50) and contained the
  patched code — verified by
  `docker run --rm kwcoco-detector-kit:ogdino-cu132-arisia python -c
  "from kwcoco_detector_kit.data import tile; print(tile._PASSTHROUGH_ANN_FIELDS)"`
  which returned `('source_category', 'track_id', 'caption', 'score')`.
- So the universal tile *was* built with the patched writer. But the
  output annotations had no `source_category`.

Root cause: `_passthrough_fields(ann)` only **copies** fields that
already exist on the source annotation. The raw VIAME bundles at
`/data/Public/VIAME/viame_sealions_2026/training_ready_v1/` carry
class info via `category_id` only — there is no pre-set
`source_category` field on those annotations. So the passthrough
whitelist did nothing for the first-time tile of raw data.

The 5d99545 / `a1aa45b` fix only solves the second-tile-onward case;
it doesn't solve the raw-input case, which is the only case that
actually matters.

## Fix (kit `852df64`)

1. `_passthrough_fields(src_ann, src_dset=None)` — when
   `source_category` isn't already on the source ann, stamp it from
   `src_dset.cats[ann['category_id']]['name']`.
2. Thread `src_dset` through all four emit sites
   (`_run_full_only`, `_run_quadrant`, two paths in `_run_multiscale`).
3. Multiscale specifically: the intermediate `kept_anns` dict was
   dropping the `"passthrough"` key on its way from `anns_scaled` →
   `kept_anns`; restore it.
4. New constant `_TILE_WRITER_VERSION = 2`. `_launch_train.sh` now
   mixes `v{VERSION}:{fields}` into the universal-tile fingerprint so
   bumping `_TILE_WRITER_VERSION` busts the cache automatically even
   when no field names change.
5. New regression test
   `tests/unit/test_tile.py::test_source_category_is_stamped_from_src_dset`
   parametrised over all three tile modes. The synthetic fixture
   mirrors the raw-VIAME scenario (category_id only, no pre-set
   source_category) — the test fails without the stamping logic and
   passes with it. 20/20 tile tests pass.

## Lessons

- "Add a passthrough whitelist" is the wrong mental model for the
  first encoding of raw input. Passthrough = preserve what's already
  there. Stamping = derive missing structure from what we do know
  (category_id + cats lookup). The tile writer is the first hop where
  we have the source dataset in scope, so it's the right place to
  stamp.
- The fail-fast guard paid for itself within 24 h. The same bug class
  recurred and the guard saved another 48 h × N-jobs of wasted
  compute. Keep adding hard preconditions even when the symptom feels
  unique — they're cheap insurance.
- The cache-fingerprint mechanism worked: bumping
  `_TILE_WRITER_VERSION` (or, last cycle, adding a passthrough field)
  changes the hash and forces a rebuild. Without this, the next agent
  to hit a similar bug would have spent hours puzzling over a "stale
  cache vs stale code" question.
- When a single diagnostic command (`docker run ... print(...)`)
  cleanly answers "did the patch make it into the image", spend the
  10 seconds to run it before going deeper. It eliminated half the
  hypothesis space here.

## Code refs

- Fix: `kwcoco_detector_kit/data/tile.py:302-322` (passthrough + stamp +
  version constant), `kwcoco_detector_kit/data/tile.py:625` (kept_anns
  passthrough propagation in multiscale).
- Cache busting: `projects/viame_sealions_2026/scripts/_launch_train.sh:127-135`.
- Test: `tests/unit/test_tile.py::test_source_category_is_stamped_from_src_dset`.
- Commit: kit `852df64`.

## Next

1. Rebuild arisia docker image (or resubmit with
   `KCD_DEV_MOUNT_KIT=1` to overlay host kit).
2. The `_TILE_WRITER_VERSION=2` bump will produce a new universal hash
   automatically — no manual nuke needed, the old `_universal/518d4bd0/`
   will just be orphaned (safe to delete later).
3. Resubmit `pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_v2`. Expect:
   apply_scheme reports nonzero per_target_class counts for `pup` and
   `nonpup_sealion` on train; fail-fast guard stays quiet; training
   proceeds.
4. After the run finishes (success or not), append a third journal
   entry recording the result.
