# Lessons learned — the 19 failure modes the kit defends against

Each entry maps to one or more probes / tests / asserts inside the kit. The canonical narrative of each failure lives in [`dev/journals/lessons_learned.md`](../dev/journals/lessons_learned.md); this doc is the dispatch table.

| # | failure (symptom) | mitigation in the kit |
|---|---|---|
| 1 | `gdown 6.x` dropped `--fuzzy` | All Google-Drive fetches use bare file IDs (no `--fuzzy`). |
| 2 | `kwimage.imresize(img, (W, H))` interprets the tuple as `scale=`, not `dsize=` — leads to multi-TB allocations | Every `imresize` call in the kit passes `dsize=` explicitly. `tests/unit/test_tile.py` covers the resize path. |
| 3 | `kwimage.imwrite(..., imwrite_params=...)` is not a cv2 kwarg | `_imwrite()` helper uses cv2's flat `params=[FLAG, val]` form. |
| 4 | gdown silently writes HTML quota-error pages | Post-download size guard (≥ 1 MiB) in the asset-download paths. |
| 5 | `kwimage.imresize(interp='area')` fails on skimage backend without cv2 | Every call is wrapped in `try/except NotImplementedError` with `'linear'` fallback. cv2 is in `install_requires`. |
| 6 | `geowatch.__init__` hard-imports `osgeo` | The kit does not depend on geowatch. Optional `merge_nearby` preprocess soft-imports `find_low_overlap_covering_boxes` if available. |
| 7 | pip 25+ rejects girder index (PEP 700) | Document the direct-wheel URL pattern for any optional dep that hasn't migrated. |
| 8 | torch / torchvision ABI mismatch | `pyproject.toml` declares `torch>=2.5` + `torchvision>=0.20` together. Document the pin recipe in `docs/install.md`. |
| 9 | `torch.onnx.export` on torch ≥ 2.5 imports `onnxscript` at function-call time | `onnxscript` is in `install_requires`. |
| 10 | DEIMv2's exporter needs `onnxsim`; opset 17 incompatible with torch ≥ 2.11 (Pad has no adapter) | `onnxsim` is in `install_requires`. Default opset 18 in `export/onnx.py`. The exporter recovers the `.onnx` artifact if the `--simplify` step crashed. |
| 11 | DEIMv2 trainer needs `faster_coco_eval`, `calflops`, `transformers`, `tensorboard`, `scipy` (undeclared) | `[project.optional-dependencies.deimv2]` declares them. `orchestration/setup_audit.py --check-env` probes for each. |
| 12 | sweep cell can silently record `status=ok` after a failed stage | `orchestration/pareto_sweep.py` records `status=fail_<stage>` on the first failure and never defaults to `ok`. |
| 13 | YAML `collate_fn` indent-leak makes DEIMv2 pass it to `CocoDetection.__init__` | The kit's `trainers/deimv2.py` generates YAML in Python with `yaml.safe_dump`, then round-trips through `yaml.safe_load` + structural-invariant assertions. `tests/unit/test_train_config_gen.py` exercises all 12 variants. |
| 14 | HGNetv2 hybrid encoder doesn't support multi-scale (pre-bakes `pos_embed`) | Trainer plugin exposes `supports_dynamic_input(variant)`. `orchestration/round_loop.py` coerces `multiscale → fixed` when False. Tested in `tests/unit/test_supports_dynamic.py`. |
| 15 | non-root users can't raise `RLIMIT_NOFILE` above `ulimit -Hn` | `_env.raise_nofile_limit()` clamps to the hard cap and returns a status. Document `KCD_TORCH_MP_SHARING=file_system` fallback. |
| 16 | DEIMv2 OOMs on 24 GB GPU with naïve upstream batch sizes | Per-`(variant, input_hw, tier)` table in `trainers/deimv2.py:_BATCH_TABLE`. Auto-shrinks with input area. |
| 17 | Multi-GPU all-reduce bottlenecks on mismatched PCIe lanes | `_env.default_cuda_visible_devices()` defaults to `"0"` on single-host non-cluster setups. `_tier.py` warns on PCIe width mismatch when `num_gpus > 1`. |
| 18 | DEIMv2 fixed-input encoders need `eval_spatial_size` = train Resize = val Resize = collate `base_size` = Mosaic `output_size` | One input-size knob in `generate_config()` drives all five values. `test_train_config_gen.py` asserts the five-sizes-match invariant for every (variant × input_hw). |
| 19 | `kwcoco subset --select_images "..."` requires undeclared `jq` Python pkg | Document the `--gids 1,2,3` form as the kit's canonical subset CLI in `docs/install.md`. |
