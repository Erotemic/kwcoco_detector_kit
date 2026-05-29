# 2026-05-26 — first 1-GPU baseline cycle: 48h spent training empty targets

## Context

This was the first cycle of attempted detector training on arisia after
the project moved into the kit (`projects/viame_sealions_2026/`).
Three target experiments lined up as 1-GPU baselines on
`deimv2_hgnetv2_n` (mobile, COCO-init):

- `submit_train_pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_v2.sh` (P1)
- `submit_train_single_sealion_deimv2_hgnetv2_n_1gpu_arisia_v1.sh` (P0)
- `submit_train_lifestage_6cls_deimv2_hgnetv2_n_1gpu_arisia_v1.sh` (P2)

All three submitted around May 22 ~00:54 UTC on arisia, hit slurm's
48h walltime around May 24 ~20:51-20:55 UTC.

**The outcome: zero useful training.** All three models collapsed to
"predict nothing" from epoch 0 onward, ran the full 48 hours, then
got killed by slurm. DEIMv2's own per-epoch COCO eval reported
`AP=0.000` from the start — but no part of the launcher noticed.

## What we built leading up to the cycle

This was a productive build cycle separate from the training failure:

### Universal-tile + apply-scheme architecture (kit commit 5d99545)

Refactored the pipeline so the tile step happens ONCE per
`(geometry-hash)` and the scheme's class-collapse is applied as a
fast post-step into `$KCD_ROOT/scheme_applied/`. Lets one tar shard
set serve every scheme; saves ~25 GB of tile bundles × N schemes.

- `data/tile.py` gained `_PASSTHROUGH_ANN_FIELDS` so
  `source_category` survives tiling.
- `projects/.../scripts/apply_scheme_to_kwcoco.py` (new) reuses
  `build_scheme_kwcoco.remap_split` to apply the scheme to a
  pre-tiled bundle.
- `_launch_train.sh` rewired to tile-universal → apply-scheme → sweep.

### Descriptive submit-script convention (kit commits 28f7694, 8f506c7, 7040871)

`submit_train_<scheme>_<variant>_<ngpus>_<host>_v<N>.sh` for every
training run. Each entry-point sets every hyperparameter as
`export KCD_*=...` so it's grep-visible; boilerplate (sbatch + docker
flags) lives in shared `_submit_train.sh` / `_sbatch_train.sh` /
`_launch_train.sh`. Hyperparam tweaks = new `_v<N+1>.sh` file.

### File-based env handoff to sbatch (kit commit 057ed07)

`sbatch --export=ALL,KEY=VAL,...` uses commas as the entry
separator. Values containing commas (like `KCD_CATEGORY_NAMES=
pup,nonpup_sealion` or `KCD_INPUT_HW='[320, 320]'`) silently truncate
on the way through. Fixed by writing all `KCD_*` env to a sourceable
file in `$KCD_SLURM_LOG_DPATH/<run_name>.env` and passing ONLY the
file path through `--export`.

This was a real failure mode in the early attempts (job 2483 onward)
where `categories: pup` would appear in the log instead of
`pup,nonpup_sealion`.

### kwcoco_dataloader writer + reader (kwcoco_dataloader commits 460db5d, d5cb03c, 8e6d3a5)

Phase 1 + 2 of the HDD-friendly fast-IO replacement for the
many-small-files dataloader:

- **Writer** (`cli/build_detection_webdataset.py`): kwcoco bundle →
  WebDataset tar shards, bucketed by `dominant_raw_class`, with rich
  per-shard footer manifests (raw class histograms + ann counts).
- **Reader** (`readers/detection.py`): four composable pieces —
  `SchemeMapping` + `relabel_detection_sample` + `WebDatasetStream`
  + `WeightedChunkMix` + `load_bucket_streams`. HDD-friendly
  sequential streaming, with chunk-K mixing for class-weighted
  sampling.
- **Benchmark** (`benchmarks/bench_detection_throughput.py`):
  apples-to-apples throughput comparison vs the kwcoco/CocoDetection
  baseline. `line_profiler.profile` decorators on the hot path
  (no-op unless `LINE_PROFILE=1`).

These didn't run on arisia during this cycle — they're scaffolding
for Phase 3 (DEIMv2 integration) and the eventual Rust port.

## The slurm timeline (from /data/users/jon.crall/slurm_logs/)

| Job | Window | End state |
|---|---|---|
| 2479–2488 | May 22 11:37–20:00 | Script/submission-debug iteration. Various `Error`/`No such file` failures as we wired `KCD_DEV_MOUNT_KIT`, sbatch `--export`, missing `kwutil`, `torch_data` import order, `safetensors` host install, `input_hw` smartcast parse, `category_names` truncation through `--export` |
| 2489 (pup_v1) | May 22 00:36 → 20:51 | **User scancel** after ~20h |
| 2491 (pup_v2 first attempt) | May 22 00:54 → 20:54 | **User scancel** 4 min later — duplicate or accidental dual-submit |
| **2490 (pup_v2)** | May 22 00:52 → **May 24 20:51** | **slurm CANCELLED DUE TO TIME LIMIT** after 48h |
| **2492 (single_sealion)** | May 22 00:54 → **May 24 20:54** | **slurm CANCELLED DUE TO TIME LIMIT** after 48h |
| **2493 (lifestage_6cls)** | May 22 00:56 → **May 24 20:55** | **slurm CANCELLED DUE TO TIME LIMIT** after 48h |

The "did I cancel this or did slurm?" question got answered: the
user scancels were 2489 + 2491 on May 22. The three big-walltime
hits were slurm's own — and they happened on jobs that should have
finished cleanly in <12h but had no signal to bail early on.

## What broke (root-causes, in causal order)

### 1. Stale universal-tile cache predating tile.py's passthrough patch

`tile_cache/_universal/1212f603/tiles.kwcoco.zip` (229,617 images,
903,603 annotations) was built by a tile run from BEFORE
[kit commit 5d99545](https://github.com/Erotemic/kwcoco_detector_kit/commit/5d99545)
landed (the universal-tile + apply-scheme architecture, which added
`_PASSTHROUGH_ANN_FIELDS` to `data/tile.py`). On disk, the bundle
has 0/903603 annotations carrying `source_category`.

Most likely cause: an early submission ran with the docker image's
baked-in `data/tile.py` (which predated the patch) before
`KCD_DEV_MOUNT_KIT=1` reliably overlaid the host's patched copy.
Once that cache existed, the cache key (hashed from tile *geometry*
params only) matched on every subsequent run, so newer runs silently
reused the poisoned bundle without re-tiling.

The kit code on disk at run-launch time HAD the passthrough. The
cache predated it. The cache-key hash didn't notice.

### 2. `apply_scheme_to_kwcoco.py` silent-dropped every annotation

`build_scheme_kwcoco.remap_split` (which `apply_scheme_to_kwcoco.py`
calls) treats annotations with `source_category` unmapped AND not
in `drop` as "unknown" — silently incremented in stats but otherwise
dropped. With `source_category=None` on every annotation,
`scheme_applied/train.kwcoco.zip` ended up with 0 annotations.

The `n_unknown_source_categories` counter in `scheme_report.json`
would have shown ~903k unknown if we'd looked. Nothing in the
launcher looked.

### 3. DEIMv2 trained on empty targets → collapsed to "no object"

With `train.mscoco.json` listing 229,617 images and 0 annotations,
DEIMv2's matcher had nothing to assign on any batch. Hungarian
matcher returned empty indices everywhere. The classification head
learned to predict "no object" for every query (which is correct on
this data — there ARE no objects in the targets the model sees).
After 30 epochs the model perfectly produces zero confident
detections.

DEIMv2's `det_solver.py` ran its end-of-epoch COCO eval and emitted
`AP=0.000` at every epoch print. Logs are explicit:

```
[2026-05-25T00:15:33]  Average Precision  (AP) @[ IoU=0.50:0.95 | area=all ] = 0.000
[2026-05-25T00:15:33] best_stat: {'epoch': -1, 'coco_eval_bbox': 0.0}
[2026-05-25T00:15:53] Epoch: [25] [0/1793] ... loss: 0.0000 ...
```

Nothing in the kit's launcher noticed.

### 4. (Unrelated) DDP loss-key divergence — fixed earlier in the cycle

Separate root cause from the above; fixed during the script-debug
phase. Detail: when a per-rank batch has zero GT boxes, DEIMv2's
model omits `dn_outputs` for that rank, the criterion's loss_dict
ends up missing ~22 dn-related keys, `dist_utils.reduce_dict` then
stacks mismatched tensors across ranks and the next NCCL all_reduce
deadlocks. Fixed in [DEIMv2 fork commit 3a6ad01] (`setup_print:
prefix every emitted line with timestamp` is `aeabc7e`; the DDP fix
is its parent).

Diagnosis trace was fun: flight recorder dump showed rank 0 doing
`AllReduce count=25` while ranks 1/2/3 did `AllReduce count=47` at
the same `collective_seq_id=3570`. The 22-key delta was exactly the
dn-related loss block.

## Fixes shipped after the failure (kit commit a1aa45b)

1. **Fail-fast guard in `_launch_train.sh`**: after `apply_scheme`
   completes, count `n_annotations` on `scheme_applied/train.kwcoco.zip`.
   If 0, refuse to start training with an explicit error message
   pointing at the most-likely cause (stale cache) and the exact
   `rm -rf` command. Bails in ~2 seconds instead of 48 hours.

2. **Writer fingerprint in tile-cache hash**: the hash now mixes in
   a fingerprint of `tile._PASSTHROUGH_ANN_FIELDS` so kit code
   changes that add/remove preserved fields auto-invalidate the
   cache. Future "stale cache predating a passthrough change"
   failures impossible without a deliberate override.

3. **Provenance capture** (came in via upstream merge,
   [kit commit 82e079e]): every `policy.json` + `detect_metrics.json`
   now carries `kit_sha`, `DEIMv2_sha`, `OGDino_sha`. We'd have
   spent ~10 seconds diagnosing this if it had existed during the
   cycle ("the run's policy.json says kit_sha < 5d99545 → predates
   the passthrough → cache is stale").

4. **Git `safe.directory=*`** for bind-mount provenance reads
   ([kit commit 49c7151]): lets the provenance dict's `git rev-parse`
   work when the kit checkout is bind-mounted into docker (the
   host UID/GID doesn't match the container's root). Without this,
   `KCD_DEV_MOUNT_KIT=1` runs would land `<unknown>` SHAs.

## Lessons

- **A 48h walltime job training on 0 annotations is not just a bug —
  it's an entire category of pipeline failure that needs structural
  defense, not vigilance.** Three jobs ate the full walltime
  producing no learning. Add cheap fail-fast assertions at every
  stage boundary; don't trust downstream stages to notice. The
  apply-scheme guard from `a1aa45b` is the prototype — extend the
  pattern to every "this should produce non-empty output" step.

- **Cache keys must include both *inputs* (params) and *code*
  (writer fingerprint).** A param-only hash silently rots when the
  writer changes the schema. The May-24 episode is the canonical
  example. When in doubt, hash more, not less.

- **Provenance stamping pays for itself the first time it gets used.**
  10 seconds with `cat policy.json` vs the half-hour we spent
  forensically tracing through tile caches. Future runs:
  `cat /data/users/jon.crall/kcd_sealion/runs/<run>/runs/<candidate>/policy.json`
  is now the first move on any failure.

- **DEIMv2's own AP=0 print is a strong tripwire.** A future
  improvement: parse the per-epoch eval output during training; if
  AP stays 0.0 for the first ~3 epochs of fine-tuning from a
  COCO-pretrained init, refuse to continue. Cheap, would have
  caught this on epoch 4 instead of epoch 30. Filed as a follow-up.

- **`--export=ALL,K=V,...` with commas in V is broken by design.**
  Use env-files. Easy to miss this if you only test with scalar
  values.

## Next cycle (post-fixup)

User is invalidating the stale caches now (`rm -rf
$KCD_TILE_CACHE_DPATH/_universal` + the failed `runs/` dirs) and
resubmitting from a freshly-rebuilt image. The new fail-fast guard
will trip in ~2 seconds if anything's still wrong; otherwise the
job runs to a real result.

If the next cycle succeeds, the v2 hgnetv2_n × pup_vs_nonpup
result will be our first real datapoint for the project's
research_plan phases. We expect non-zero AP (the actual question:
is it competitive with the dinov3_s 4-GPU run on the same scheme?).

## Code references (commit SHAs at time of writing)

| What | Where |
|---|---|
| Universal-tile + apply-scheme | kit `5d99545` |
| Passthrough whitelist | `kwcoco_detector_kit/data/tile.py:294` |
| Fail-fast empty-anns guard | `_launch_train.sh:227-258` (kit `a1aa45b`) |
| Writer fingerprint in cache key | `_launch_train.sh:125-141` (kit `a1aa45b`) |
| Provenance capture | kit `82e079e` |
| DEIMv2 DDP loss-key alignment | DEIMv2 fork `3a6ad01` |
| Stub WebDataset writer | kwcoco_dataloader `460db5d` |
| Stub WebDataset reader | kwcoco_dataloader `d5cb03c` |
| Throughput benchmark | kwcoco_dataloader `8e6d3a5` |
| File-based env handoff to sbatch | kit `057ed07` |
