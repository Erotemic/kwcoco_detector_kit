# CI

## Smoke matrix

| Python | Platform | What runs |
|---|---|---|
| 3.10 | linux | unit tests + kwcoco_demo smoke |
| 3.11 | linux | unit tests + kwcoco_demo smoke |
| 3.12 | linux | unit tests + kwcoco_demo smoke |
| 3.13 | linux | unit tests + kwcoco_demo smoke |
| 3.13 | macOS | unit tests (no smoke — onnxruntime macOS wheel matrix is unstable) |

## What's gated by marker

- `pytest -q` runs everything **except** `requires_gpu` and `requires_deimv2`.
- `pytest -m requires_deimv2` runs the Level-B "drive through `engine.core.YAMLConfig`" tests; CI gates this on a separate job that clones the DEIMv2 submodule.
- `pytest -m requires_gpu` runs the few CUDA-dependent paths; only enabled on the GPU shard.

## Acceptance gate

The CI green bar is:

1. `pytest tests/ -q` ≥ 80 tests, under 60 s on a 1-CPU runner.
2. `bash examples/kwcoco_demo/run_smoke.sh` produces a populated `manifest.tsv` with one `HOST_PROMISING` candidate of `candidate_kind=smoke`, in < 90 s.
3. No kit source file contains `poop`, `shitspotter`, `mobile_app_training`, `v9 baseline`, `Pixel 5`, `tpl/poop_models`, or `tpl/Open-GroundingDino` (the substring scrub guards against accidental name leaks from the prior project).
