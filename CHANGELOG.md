# Changelog

All notable changes to `kwcoco-detector-kit` are recorded here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **KCD-EVAL-01 — first-class tiled eval recipe mode.** `recipe.v1` now supports an
  `eval:` block (`mode: whole_image | tiled | auto`, default `auto`) plus `data.tiled`.
  `auto` resolves to tiled when training is tiled (a `data.tiled` flag or a `multiscale*`
  matrix `train_policy`), and the resolution is logged loudly. Fixes the silent bug where
  `sweep.tiled_eval` was dropped by `recipe._build_sweep_data` passthrough (the cause of
  shitspotter v13's deferred verdict — it set `tiled_eval: true` but eval ran whole-image).
  The legacy `sweep.tiled_eval*` keys are now also passed through for back-compat.
  `orchestration/recipe.py`.
- **KCD-EVAL-02 — eval is self-describing.** `detect_metrics.json` `eval_inputs` now stamps
  `eval_mode` (`tiled`/`whole_image`), `eval_device`, and the tile params, so a tiled AP can
  never be compared to a whole-image AP by accident. `eval/kwcoco_eval.py`.
- **KCD-CFG-01 — `${VAR}` / `${VAR:-default}` interpolation** in recipes, tile-corpus specs,
  and matrix files. New `configs.expand_env_vars()` (fails loudly on an undefined `${VAR}`
  without a default); applied in `recipe._load_recipe`, `tile_corpus._load_spec`,
  `pareto_sweep._load_matrix`. Lets recipes be host-portable and retires per-project render
  shims.
- **KCD-DATA-01 — `data-manifest` op + `data.expect` guard.** New `data/manifest.py`
  (`compute_manifest`/`assert_expected`) + `data-manifest` CLI records true
  image/annotation/category counts + a content hash. `recipe-run` asserts a recipe's optional
  `data.expect:` block before any GPU time (`expect_mode: fail|warn`). Structural antidote to
  the "filenames lie" class of bug (2564-vs-7350 images).
- **KCD-DOC-01 / KCD-DOC-02** — banners disambiguating superseded `examples/{sealion_aerial,
  viame_sealions_2026}` from the live `projects/viame_sealions_2026/`; an "examples vs
  projects" section in the README; and `docs/phase3_status.md` (shipped-vs-deferred Phase-3
  surface + cross-repo `KCD-*` status).
- Tests: `tests/unit/test_recipe_eval_and_manifest.py` (env interpolation, eval-mode
  resolution incl. auto, manifest counts + expectation guard).
- Phase 1 scaffold: `pyproject.toml`, package layout, tests/, examples/, docs/.
- `data/tile.py` — unified tile extractor with three modes (`full_only`, `quadrant`, `multiscale`). Lifted from prior project's v4 `tile_kwcoco.py` + v5 `v5_tile.py`. Adds `oversize_factor` knob (defaults to 1.0; raise to ~1.4 for load-time crop-augmentation friendly tiles).
- `data/merge.py` — positive + negative tile merger for round-based training.
- `data/mine.py` — offline hard-negative miner (predictor-adapter-aware).
- `data/coco_export.py` — kwcoco → MSCOCO JSON exporter for DEIMv2 / OGDino consumers.
- `trainers/mock_tiny.py` — CPU-only smoke detector (renamed from `v4_mock` in prior project).
- `trainers/deimv2.py` — Python YAML-config generator covering all 12 DEIMv2 variants. Replaces the prior project's bash heredoc generator that triggered failure #13.
- `trainers/_interface.py`, `_registry.py`, `_tier.py` — plugin protocol + auto-tier detection.
- `orchestration/eligibility.py` — eligibility manifest state machine (renamed `PHONE_*` → `DEPLOY_*`; preserved `candidate_kind=smoke|real` filter).
- `orchestration/pareto_sweep.py`, `round_loop.py` — Python rewrites of the prior bash sweep + round drivers.
- `orchestration/setup_audit.py` — `--check-env` probe-and-install for transitive runtime deps (failure #11).
- `examples/kwcoco_demo/` — always-runnable smoke example.
- `dev/` — engineering memory (seeded from prior project; new entries land above the `Seed #19` divider).

### Changed
- **scriptconfig -> kwconf.** `kwcoco_detector_kit/data/manifest.py` was the last
  first-party user of `scriptconfig`; `DataManifestConfig` now subclasses
  `kwconf.Config` with `kwconf.Value`. Drop-in: `position=`, `required=`, `help=`
  and `.cli(argv=, data=, strict=)` all behave identically.
- The `pyproject.toml` pin moves to `kwconf>=0.11.0`, and the Dockerfile's
  hand-maintained dependency list now installs `kwconf` instead of
  `scriptconfig`. That list had **drifted** from `pyproject.toml`: it installed
  `scriptconfig` (never declared as a dependency) and omitted `kwconf` (declared
  since the pin was added), so any first-party `import kwconf` would have failed
  inside the image.
- `scriptconfig` is still installed in the image, but only as a transitive
  dependency of the vendored `tpl/kwcoco_dataloader` submodule, whose `setup.py`
  builds `install_requires` from its own `requirements/runtime.txt`. No
  first-party code imports it any more. **Requires a docker image rebuild** to
  take effect in containers.

### Notes
- Phase 1 test bar: structural YAML invariants via `yaml.safe_load` + dict-shape assertions. The DEIMv2 `engine.core.YAMLConfig` drive-through is gated behind `pytest -m requires_deimv2`.
- Storage format for intermediate training data is intentionally JPEG-on-disk + kwcoco manifest in Phase 1. Webdataset is deferred to Phase 3 and treated as one possible backend behind a `TileStore` interface (user's design constraint: oversized tiles for crop-aug, streamable from spinning disk / slow network).

## Phase 3 additions — TileStore abstraction (additive; Phase 2 path unchanged)
- `data/tile_store.py` — `TileStore` Protocol + two backends:
  - `KwcocoJpegStore` (default) — wraps the Phase 1 `.kwcoco.zip` output unchanged.
  - `WebdatasetStore` — converts a kwcoco bundle to `shard-NNNNNN.tar` + `_bundle_meta.json`. Sequential-read friendly on spinning disks; streamable from NFS/S3 via fsspec.
- `data/tile_loader.py` — `TileLoader` torch `IterableDataset` over any backend; applies load-time random / center crop using the `tile_model_input_size` metadata (the on-disk side of the `oversize_factor` story).
- `data/stats.py` — `compute_per_channel_stats` Welford mean/std probe over any TileStore (multispectral normalization).
- `cli/__main__.py` — new subcommands `convert-store` + `stats`.
- `[project.optional-dependencies.webdataset]` — `webdataset` + `braceexpand`; `setup_audit` probes for both under the new `webdataset` group.
- `docs/storage.md` — design + backends + load-time-crop discussion + CLIs + what's deferred.
- Tests: `test_tile_store.py` (round-trip kwcoco ↔ wds), `test_tile_loader.py` (crop/flip/normalize/tensors), `test_stats.py` (Welford correctness).

Phase 2 acceptance: 292 tests pass in 20s on CPU. `examples/kwcoco_demo/run_smoke.sh` still completes in 9.3s producing `HOST_PROMISING + AP=...` — kwcoco_jpeg is the default and unchanged.

## Phase 2 additions
- `trainers/opengroundingdino.py` — OpenGroundingDINO trainer plugin covering `opengroundingdino_swint` (Swin-Tiny @ tier L) + `opengroundingdino_swinb` (Swin-Base @ tier XL). Python rewrite of the v9 shell pipeline: kwcoco → MSCOCO → ODVG → `train_dist.sh` subprocess. SAM2 segmenter co-training is deferred to v1.1.
- `orchestration/pareto_sweep.py` — added `--distributed` flag for opt-in DDP via the trainer plugin's `launch(num_gpus=N, distributed=True)`.
- `examples/sealion_aerial/` — scaffold (`README.md`, `prepare_kwcoco.py` for NOAA Steller dataset conversion, `config.yaml`, `run_all.sh`). NOAA-side validation deferred to host with GPU.
- `docs/multi_gpu.md` — CLI + tier auto-detect + PCIe-link-width caveat + SLURM submit pattern.
- `dev/journals/lessons_learned.md` — Lessons #20 (scriptconfig smartcast comma-string surprise) + #21 (kwcoco-eval confusion-sidecar crash recovery).
