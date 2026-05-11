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
