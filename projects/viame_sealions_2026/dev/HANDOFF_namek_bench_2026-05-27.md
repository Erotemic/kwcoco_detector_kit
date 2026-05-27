# Handoff: dataloader-throughput benchmarking on namek (2026-05-27)

## Why you're here

The sea-lion training runs on **arisia** are dataloader-bound (GPU
util oscillates 10-99%). We have a new HDD-friendly WebDataset
reader in `~/code/kwcoco_dataloader/` and a throughput benchmark
script in this repo. Your job: run the benchmark on **namek**
against the universal tile bundle that exists there, gather
samples/sec + profile data, and report back what's slow.

The active training runs on arisia are NOT your problem — they are
running independently and are unrelated to this benchmark. Don't
touch arisia.

## What's already been built

- **Writer** (`kwcoco_dataloader/cli/build_detection_webdataset.py`,
  commit `460db5d`): consumes a kwcoco bundle, writes
  `<key>.jpg + <key>.json` pairs bucketed by `dominant_raw_class`,
  emits rich footer manifests for chunk-K weighted mixing.
- **Reader** (`kwcoco_dataloader/readers/detection.py`, commits
  `d5cb03c` + `8e6d3a5`):
  - `Sample`, `SchemeMapping`, `relabel_detection_sample`
  - `WebDatasetStream`, `WeightedChunkMix`, `load_bucket_streams`
  - `line_profiler.profile` decorators on the hot paths
    (`_sample_from_wds_raw`, `relabel_detection_sample`,
    `WebDatasetStream.__iter__`, `WeightedChunkMix.__iter__`).
- **Benchmark CLI**
  (`kwcoco_dataloader/benchmarks/bench_detection_throughput.py`):
  baseline (kwcoco CocoDetection-style) vs WebDataset stream
  reader; `--no_loader` torch-optional fallback.
- **Driver script** (project-agnostic, lives in kwcoco_dataloader):
  `~/code/kwcoco_dataloader/dev/bench_dataloader_throughput.sh`.
  Runs 3 variants in sequence:
  1. baseline (kwcoco / PIL)
  2. webdataset (kwcoco_dataloader reader)
  3. webdataset + `LINE_PROFILE=1`

  Data is synthesized via `kwcoco toydata` (vidshapes flavor) so the
  benchmark is self-contained — no dependency on sealion tile caches.
  Each variant writes a log under `$BENCH_OUT_DPATH/<variant>.log`;
  the third also drops `profile_output.{lprof,txt}`.

## How to run

```bash
bash ~/code/kwcoco_dataloader/dev/bench_dataloader_throughput.sh

# Larger sample count + more workers:
BENCH_N_SAMPLES=2000 BENCH_WORKERS=8 \
    bash ~/code/kwcoco_dataloader/dev/bench_dataloader_throughput.sh

# Benchmark against a real kwcoco bundle instead of toydata:
BENCH_KWCOCO=/path/to/data.kwcoco.zip \
    bash ~/code/kwcoco_dataloader/dev/bench_dataloader_throughput.sh
```

Defaults: data under `$HOME/data/kwcoco_dataloader_bench/data/`,
outputs under `$HOME/data/kwcoco_dataloader_bench/runs/<ts>/`. See
the script header for the full env-var menu.

## What the benchmark should tell us

We expect WebDataset to win on:

- **Random read latency**: arisia/namek HDDs hate `O(N)` PIL.open
  scattered across a huge tile dir; WebDataset's sequential tar
  reads should be ~3-5x faster on cold cache.
- **Worker scaling**: WebDataset chunk streams parallelise cleanly;
  the kwcoco baseline tends to plateau around 4 workers because of
  json index contention.

Things to watch for in the line-profiler output:

- `relabel_detection_sample` should be cheap (<5% time). If not,
  the scheme-collapse path is doing more work than necessary.
- `WeightedChunkMix.__iter__` cost should be dominated by the
  underlying `WebDatasetStream`s, not the weighting bookkeeping.
- `_sample_from_wds_raw` is the JPEG-decode + tensor-convert. If
  this is >40% of time even on warm cache, we need to investigate
  hardware-accelerated decode (nvjpeg / torchvision.io).

Headline metric to capture: **samples/sec per worker** for each
variant.

## Hard constraints

- **Do not run on arisia.** It is currently training three
  overnight sea-lion detectors (pup_v6, single_sealion_v4,
  lifestage_6cls_v4, jobs starting ~2026-05-27 evening). The
  benchmark needs a CPU-heavy machine, not a GPU-heavy one.
- **Do not modify the kit's `tile.py` or anything in
  `kwcoco_detector_kit/` core packages** as part of this
  benchmark. If you find a real bug in the reader/writer, commit
  to `~/code/kwcoco_dataloader/` and note it in the report.
- **Group-writable outputs only**: anything you create under
  `/data/users/jon.crall/` should be `chmod 664/775`. The bench
  script already does `umask 002`; just don't subvert it.
- **No memory edits**: the agent memory at
  `/home/agent/.claude/projects/-home-joncrall-code-kwcoco-detector-kit/memory/`
  is canonical — read but don't add benchmarking-task scratch
  there. Use scratch files in this dev/ dir.

## Recent context the new agent needs

Recent kit commits:

```
0bc3c47 sealions: pup v6 + single_v4 + 6cls_v4 — sized for the Mosaic cliff
0ad3e03 sealions: v5/v3 — batch=32 after v4 box-count-variance OOM
8be8d61 sealions: single_sealion + lifestage_6cls v2 (mirror of pup v4)
163c7c6 sealions: temporarily drop --train_num_workers flag (image not rebuilt)
3bca71e sealions: configurable dataloader workers + v4 (batch 48, workers 8/4)
1ac44bf sealions: v3 submit script — drop per-GPU batch to 64 after v2 OOM
1858d93 tile: umask 002 so cache outputs are group-writable
0c1be23 journal: 2026-05-26 second entry — passthrough wasn't enough
852df64 tile: stamp source_category from src_dset when absent on input
89f221e sealions: research journal — inaugural entry covering May 22-25 cycle
```

The two most relevant for benchmark interpretation:

- **`852df64` (tile: stamp source_category)**: the universal tile
  bundles now carry `source_category` on every annotation. The
  WebDataset writer needs that field for scheme-agnostic shard
  bucketing — verify it's present in the tile bundle you're
  benchmarking against (`kwcoco stats --src tiles.kwcoco.zip`
  and look at annotation key coverage).
- **`0bc3c47` (Mosaic cliff)**: DEIMv2's augmentations multiply
  effective annotation density 4x at epoch 4. Not directly
  relevant to dataloader throughput, but if you're measuring with
  `LINE_PROFILE=1` and see weirdly high per-sample variance,
  that's why training-side measurements look the way they do.

Relevant prior journals (in `projects/viame_sealions_2026/docs/journals/`):

- `2026-05-26_first_baseline_attempt.md` — 48h spent training on
  empty targets due to stale tile cache. Background for why we
  care about source_category provenance.
- `2026-05-26_passthrough_was_not_enough.md` — the follow-on
  tile writer stamping fix.

## Deliverable

A short report in
`projects/viame_sealions_2026/dev/bench_results_namek_2026-05-27.md`
(or similar date) containing:

1. The three samples/sec numbers (baseline, webdataset, webdataset+profile).
2. Top 5 hot lines from the line-profiler output, with a one-line
   commentary on each.
3. Recommendation on whether to wire WebDataset shards into the
   sea-lion training pipeline before the next training cycle, with
   a rough estimate of the GPU-util improvement.
4. Any reader/writer bugs found (file an issue or commit to
   kwcoco_dataloader directly).

## What NOT to do

- Don't write a 2,000-line "comprehensive benchmark suite". One
  driver script + one report is enough.
- Don't redesign the reader. If you find a bottleneck, file a
  follow-up; don't try to fix everything in this session.
- Don't add a new submit script under `scripts/` — those are for
  training launches, not benchmarks.
- Don't journal benchmark results in the project's
  `docs/journals/` dir; that's for training cycles. The
  `dev/bench_results_*` file is the right place.

## If something goes wrong

Most likely failure: the WebDataset reader can't find the bucketed
shards because they don't exist on namek yet. The bench script will
attempt to call `build_detection_webdataset` to write them under
`$KCD_BENCH_ROOT/shards/` — that needs ~10-20 GB free; check
beforehand with `df -h $KCD_TRAINING_ROOT`.

If the line_profiler import fails (`pip install line_profiler`
might not be in the venv), the third variant degrades gracefully;
the first two still run. Note it in the report and move on.
