# CLAUDE.md — kwcoco_detector_kit

Domain-agnostic Python package for training object detectors on kwcoco
datasets. This file is loaded automatically by Claude Code agents
working in this repo; it captures the conventions and recent refactors
that aren't obvious from grepping the source.

## Conventions

- **`scripts/paths.sh` source-of-truth pattern** (per-project): every
  bash script sources its project's `paths.sh` and reads `KCD_*`
  variables from it; never derive paths from `$PWD` or filesystem
  globbing. Override variables in your shell rc instead of editing
  scripts. Example: `projects/viame_sealions_2026/scripts/paths.sh`.
- **Per-project test layout**: each project under `projects/` has its
  own `tests/{unit,expensive}/` with a local `conftest.py`. Unit tests
  use synthetic fixtures; expensive tests `pytest.skip` when the real
  data isn't on disk. The kit's own pytest (`testpaths = ["tests"]` in
  `pyproject.toml`) does not collect project tests automatically — run
  them from inside the project subtree.
- **Docker image is the reproducibility unit**: training jobs run via
  `kwcoco-detector-kit:ogdino-cu132-arisia` (built on arisia by
  `docker/opengroundingdino/build_arisia_cuda132.sh`; other hosts use
  `build_auto.sh` to pick a CUDA profile). The image bakes the DEIMv2
  submodule, runtime deps, kwutil, the `tests/` tree, and runs pytest
  at build time (`RUN_TESTS=1` default) so regressions don't ship.

## Recent refactors (don't undo)

- **Multi-class detection support** (`8aa6b51..0f66858`): every
  `category_name` (singular) became `category_names` (plural list)
  across `data/coco_export.py`, `data/tile.py`, `data/merge.py`,
  the configs/CLI, the predictor, evaluator, sweep, and mining. The
  invariant is "CLI category_names order == output category_id order
  == train-time class index" — see
  `tests/unit/test_multiclass_pipeline.py` for the contract.
- **Docker bakes tests + DEIMv2**: `541102b`, `2e20145`, `e348abf`
  copy `smoketests/`, `tests/`, and the DEIMv2 submodule into the
  image; the build step runs pytest before tagging. The point is to
  fail the build, not the training job, when a regression lands.

## Active project

[projects/viame_sealions_2026/](projects/viame_sealions_2026/) — NOAA
Steller sea-lion detector training. Active operational run is
`pup_vs_nonpup` (P1). See its [README](projects/viame_sealions_2026/README.md)
and [AGENT.md](projects/viame_sealions_2026/AGENT.md).

[examples/viame_sealions_2026/](examples/viame_sealions_2026/) is the
previous-generation OpenGroundingDINO recipe — historical reference
only, not the active path.

## What NOT to change without explicit request

- **No singular-`category_name` shims**: the multi-class API is
  intentional. Don't add backwards-compat aliases. (See user's
  durable preference in agent memory:
  [[feedback-no-compat-in-kit]].)
- **`category_id_start=0`** in
  [kwcoco_detector_kit/data/coco_export.py](kwcoco_detector_kit/data/coco_export.py)
  matches DEIMv2's 0-indexed class labels. Changing this silently
  breaks every trained checkpoint.
- **`--gres=gpu:4`** in
  [projects/viame_sealions_2026/scripts/sbatch_pup_vs_nonpup.sh](projects/viame_sealions_2026/scripts/sbatch_pup_vs_nonpup.sh)
  is the arisia default; per-call overrides go through
  `scripts/submit_pup_vs_nonpup.sh` env vars, not by editing the file.
- **Docker `RUN_TESTS=1` default**: tests run at build time as a
  regression gate. Don't disable.
- **`target_order` lists in `docs/class_schemes.yaml`**: define the
  trained model's class index assignment. Reordering invalidates every
  already-trained checkpoint for that scheme.
- **Running slurm jobs**: never `scancel` a job without explicit user
  permission.

## Memory

Agents working in this repo persist conventions, preferences, and
project state to
`/home/agent/.claude/projects/-home-joncrall-code-kwcoco-detector-kit/memory/`.
Read `MEMORY.md` there for the index of durable rules (e.g. "always
commit changes," "no backwards-compat in this repo").
