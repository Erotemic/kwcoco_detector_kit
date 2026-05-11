# Changelog

All notable changes to `kwcoco-detector-kit` are recorded here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
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
