# Lessons learned

Postmortems of bugs that took >1 hour to diagnose, newest-first. Format: Symptom / Root cause / Fix / Takeaway.

The grep target on this file is the **symptom**, not the technically correct vocabulary that comes after diagnosis. Write entries in the language a future debugger would use *from inside the bug*.

The bar is "took >1 hour" or "would have been ≥10× faster with this entry on file."

---

### Lesson #30 — OpenGroundingDINO needs a narrow Transformers 4.x band, not just `<5`

**Symptom:** The runtime patch downgraded `transformers` from `5.8.0` to `4.57.6` via `pip install 'transformers<5'`, but OpenGroundingDINO still crashed while building BERT:

```
AttributeError: 'BertModel' object has no attribute 'get_head_mask'
```

**Root cause:** This OpenGroundingDINO fork expects older Transformers BERT internals exposed on `BertModel`, including `get_head_mask`. The requirement `transformers<5` is too loose: late 4.x releases can still remove or move the helper APIs this fork uses.

**Fix:** Pin the OpenGroundingDINO dependency band to a known-old 4.x range:

```text
transformers>=4.35,<4.47
```

Apply the same spec in `pyproject.toml`, the Dockerfile, `setup_audit`, and the temporary `KCD_RUNTIME_PIP_DEPS` patch path. Let pip resolve `huggingface-hub<1.0` through the selected Transformers version.

**Takeaway:** For upstream research repos, broad major-version caps are often not enough. Pin to a tested minor-version band and make `check-env` validate the version spec, not just importability.

---

### Lesson #29 — Docker `--gpus device=0,1,2,3` needs nested quotes

**Symptom:** A 4-GPU Slurm smoke job reaches the Docker launch line and fails immediately:

```
docker: Error response from daemon: cannot set both Count and DeviceIDs on device request
```

The printed command appears reasonable at first glance:

```
docker run ... --gpus device=0,1,2,3 ...
```

**Root cause:** Docker's `--gpus` option has its own CSV-style parser. A comma-separated device list must be nested-quoted as `"device=0,1,2,3"` so Docker treats the commas as part of the `device=` value rather than additional device-request fields. This is easy to miss in bash arrays because ordinary shell quoting is stripped before Docker receives argv.

**Fix:** When Slurm sets `CUDA_VISIBLE_DEVICES=0,1,2,3`, pass the argument with literal inner quotes:

```bash
docker_args+=(--gpus "\"device=$CUDA_VISIBLE_DEVICES\"")
```

Do not pass `--gpus "device=$CUDA_VISIBLE_DEVICES"` for comma-separated lists.

**Takeaway:** Container wrapper logs should print the exact `docker run` argv, and multi-GPU smoke tests should exercise the same `--gpus` path as production. Single-GPU success does not validate Docker's comma-list parsing.

---

### Lesson #28 — A foreground Slurm follower needs a Ctrl-C policy

**Symptom:** `submit_stage.sh` follows Slurm stdout like a foreground command, but Ctrl-C has ambiguous meaning: did the user want to stop watching, or cancel the Slurm allocation?

**Root cause:** `sbatch` and a foreground tail are two separate processes. Killing the follower does not necessarily kill the Slurm job. Treating Ctrl-C as a plain Python `KeyboardInterrupt` leaves users unsure whether GPUs are still allocated.

**Fix:** The follower catches Ctrl-C and asks:

```
[slurm-follow] Ctrl-C: cancel Slurm job <jobid>? [y/N]
```

Enter detaches and leaves the job running; `y` runs `scancel <jobid>`.

**Takeaway:** Any tool that makes Slurm feel foreground must make detach-vs-cancel explicit. The default should preserve running work; cancellation should be a deliberate confirmation.

---

### Lesson #27 — DEIMv2's `export_onnx.py` has no `-o`/`--output` flag — output path is derived from `--resume`

**Symptom:** Real-host training reached evaluation (post-lesson #26 fix), saved `best_stg1.pth`, then ONNX export failed with:

```
usage: export_onnx.py [-h] [--config CONFIG] [--resume RESUME] [--opset OPSET] [--check] [--simplify]
export_onnx.py: error: unrecognized arguments: -o /tmp/.../export/deimv2_h256_w256.onnx
```

The subprocess exited with code 2 (argparse failure) before writing anything.

**Root cause:** DEIMv2's `tools/deployment/export_onnx.py:103-107` registers only `--config`/`-c`, `--resume`/`-r`, `--opset`, `--check`, `--simplify`. The output path is hardcoded at line 68:

```python
output_file = args.resume.replace('.pth', '.onnx') if args.resume else 'model.onnx'
```

So a `-r /workdir/best_stg2.pth` invocation writes `/workdir/best_stg2.onnx`. The kit's wrapper was passing `-o <kit_path>` which argparse rejects outright. The kit's recovery path (catch `CalledProcessError` + recover existing `.onnx`) didn't trigger because nothing was on disk — the script crashed at argparse, before reaching `torch.onnx.export()`.

**Fix:** Drop `-o` from the subprocess args. After the upstream subprocess succeeds (or partially succeeds with `.onnx` on disk despite a `--simplify` crash), `shutil.move()` the derived artifact (`<resume>.replace('.pth', '.onnx')`) to the kit's canonical `<workdir>/export/<name>.onnx` slot. The kit's modelspec sidecar writer keeps working unchanged.

Regression: `tests/unit/test_deimv2_export_args.py` (3 tests):
1. captured subprocess args do not contain `-o` or `--output`,
2. derived artifact gets moved to the kit's canonical path (and the derived path no longer exists after success),
3. the `--simplify`-crash recovery still works when upstream wrote the unsimplified `.onnx`.

**Takeaway:** Inspect upstream CLI surfaces before assuming `-o` exists. The kit's pattern for subprocess-driven trainers is to (a) call the upstream tool with its native arg shape, (b) compute the upstream-written artifact path, (c) move/rename to the kit's canonical layout. This separates "where upstream wants to write" from "where the kit's manifest expects to find it".

---

### Lesson #26 — DEIMv2 `PostProcessor.num_top_queries` must shrink when `num_classes` shrinks

**Symptom:** First training epoch finishes (loss curves printed), then evaluation crashes inside DEIMv2's `engine/deim/postprocessor.py:59`:

```python
scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
RuntimeError: selected index k out of range
```

A true post-training failure, not a config bug surfaced at init. Easy to mistake for a model-output shape bug.

**Root cause:** DEIMv2's PostProcessor selects topk over `scores.flatten(1)` whose shape is `[batch, num_queries * num_classes]` — one logit per (query, class) pair. Upstream's `configs/base/deimv2.yml` sets `num_top_queries: 300` because COCO has `num_classes=91` and `num_queries=100..300` so 100*91=9100 ≫ 300. The kit's `num_classes=1` override collapses the flattened axis to `num_queries` alone. For HGNetv2 atto with `num_queries=100`, the topk asks for k=300 from a 100-element tensor → crash. HGNetv2 femto (`num_queries=150`) and pico (`num_queries=200`) crash similarly; n/s/m/l/x and all DINOv3 variants inherit the upstream `num_queries=300` so single-class topk-300 is *exactly* the boundary case (works by chance).

**Fix:** The kit's YAML generator overrides `PostProcessor.num_top_queries` to `min(300, num_queries * num_classes)`. Per-variant `num_queries` lives in `VARIANTS[name]["num_queries"]`, populated from the upstream configs (atto=100, femto=150, pico=200, all others=300). At `num_classes=91` (COCO) the override evaluates to 300 — same as upstream. At `num_classes=1` (single-class kit usage) it shrinks to `num_queries`, which is always a valid topk size. Regression test: `tests/unit/test_deimv2_topk_invariant.py` parametrizes over 12 variants × {1, 5, 20, 91} classes and asserts `num_top_queries ≤ num_queries * num_classes`.

**Takeaway:** When overriding a config field on an upstream framework, also override every other field whose validity depended on the original value. The kit's "five-sizes-match" invariant (lesson #18) catches the spatial-axis case; this is the per-batch-axis analogue. A future agent extending the kit with a new trainer plugin that supports class-count-dependent operations must reproduce this defense.

---

### Lesson #25 — OpenGroundingDino's `tools/coco2odvg.py` needs `jsonlines`; trainer must fall back gracefully

**Symptom:** Pytest tests for the OpenGroundingDINO trainer's `generate_config()` failed with `subprocess.CalledProcessError` from `tools/coco2odvg.py`. The captured stderr was `ModuleNotFoundError: No module named 'jsonlines'`.

**Root cause:** The submodule's `tools/coco2odvg.py` reads/writes JSON Lines via the `jsonlines` package, which isn't a declared dep of `kwcoco_detector_kit` or the OpenGroundingDINO submodule's own `requirements.txt`. Whenever the kit's resolver finds `tpl/Open-GroundingDino` on disk (e.g. after `git submodule update --init`), `generate_config` invokes the upstream conversion tool — and dies in a CI env that doesn't install the optional extras.

**Fix:** Two layers.
1. The trainer's `generate_config` now wraps `_coco_to_odvg()` in `try/except subprocess.CalledProcessError`. On failure it emits a warning pointing at the `[opengroundingdino]` extras (which now include `jsonlines`) and leaves the stub Python config intact. The user can train against the stub config once they install the extras; the kit's pipeline doesn't abort at config-gen time.
2. `setup_audit` now probes for `jsonlines` under the `opengroundingdino` group so `kwcoco-detector-kit check-env --groups opengroundingdino --strict_import` flags the gap before training is attempted.

**Takeaway:** When a trainer plugin invokes upstream tools that have their own hidden runtime deps, the plugin's `generate_config` should treat the conversion as best-effort and degrade to a stub when the env isn't fully provisioned. Hard-failure at config-gen time blocks pipelines that don't actually need the conversion (the kit's own tests, smoke pipelines, dry-run sweeps). Hard-failure belongs at `launch()` time, not at `generate_config()` time.

---

### Lesson #24 — DEIMv2 single-GPU launches must use `torch.distributed.run` on torch ≥ 2.10

**Symptom:** A bare `python tpl/DEIMv2/train.py -c train.yml` invocation dies at backbone init with `ValueError: Default process group has not been initialized, please make sure to call init_process_group.` The trace points at `engine/backbone/hgnetv2.py:562`:

```python
if torch.distributed.get_rank() == 0:
```

**Root cause:** DEIMv2's `setup_distributed()` calls `torch.distributed.init_process_group(init_method='env://')` inside a try/except — when `RANK` / `LOCAL_RANK` / `WORLD_SIZE` aren't set, the call fails, `enabled_dist = False`, and DEIMv2 prints "Not init distributed mode." Several modules downstream (notably `hgnetv2.py`'s rank-gated stem-print) call `torch.distributed.get_rank()` unconditionally. On torch ≤ 2.9 this returned 0 silently when no process group was initialized; **torch 2.10+ raises**. The kit's `launch()` used to invoke a bare `python train.py` for the single-GPU path (matching what the prior project's bash did) — that worked on the prior project's older torch but breaks the moment the user updates.

**Fix:** Always launch DEIMv2 under `python -m torch.distributed.run --nproc_per_node N`, even for `num_gpus=1`. torchrun sets `RANK=0`, `LOCAL_RANK=0`, `WORLD_SIZE=N` so `init_process_group(env://)` succeeds and downstream `get_rank()` works on every torch version. The `--master_port` defaults to 29500, overridable via `$KCD_MASTER_PORT` for parallel sweeps.

**Takeaway:** When an upstream trainer's distributed-init code path catches its own exceptions, that doesn't mean the rest of the trainer is single-GPU safe. The trainer's API contract is "you must launch under torchrun"; the kit's launcher should respect that contract unconditionally rather than emulating the bare-python path that used to work on older torch.

---

### Lesson #23 — torch 2.10's optimizer constructor lazily imports `torch._dynamo` → ~20-30s cold-cache hang

**Symptom:** A second invocation of an in-process trainer in the same shell session appears to hang for 20-30s on `torch.optim.Adam(model.parameters(), ...)`. Ctrl-C shows the stack inside `torch.fx.experimental.symbolic_shapes`'s sympy import chain.

**Root cause:** torch 2.10+ wraps `Optimizer.add_param_group` with a `_compile.py:inner` decorator that lazily imports `torch._dynamo` on first call. `_dynamo` imports sympy, which itself has thousands of files. On a cold filesystem cache the import chain takes 20-30s. The first invocation in a shell session warms the cache; later subprocesses (e.g. the round_loop following the sweep) hit cold pages again because the kernel evicted them under other workload.

**Fix:** Print a visible "loading torch optimizer machinery (one-time)" line right before the Adam constructor, with the elapsed seconds when done. Users no longer mistake the wait for a hang. Eagerly importing `torch._dynamo` doesn't speed up the cold-cache case — sympy is the time sink — but the visible message removes the diagnostic ambiguity.

**Takeaway:** Long lazy imports on user-visible critical paths need a "loading X (one-time)" line. Silent waits look like hangs. A 30s wait the user knew about is fine; a 30s wait with no output is a Ctrl-C waiting to happen.

---

### Lesson #22 — `transformers` ≤ 4.46 + `huggingface-hub` 1.x silently incompatible

**Symptom:** Running DEIMv2's `train.py` subprocess crashes at `from transformers import AutoTokenizer` (called transitively via `calflops`) with `ImportError: huggingface-hub>=0.34.0,<1.0 is required for a normal functioning of this module, but found huggingface-hub==1.14.0.`

**Root cause:** `transformers` versions through ~4.46 pin `huggingface-hub<1.0`. HF Hub released 1.0 in early 2026. A user who has `huggingface-hub==1.14.0` installed (a transitive of some other package) and an older `transformers` gets the conflict at `transformers/__init__.py` import time. `importlib.util.find_spec("transformers")` returns the spec — module is "present" — so the kit's original `setup_audit` find_spec check reported "all probes ok".

**Fix:** Two-tier env probe. For groups where transitive version conflicts are common (`deimv2`, `opengroundingdino`), the kit's `setup_audit` now does a real `__import__` and reports the ImportError text. A per-module heuristic table maps known error patterns to actionable fix lines — for transformers/HF-Hub the hint is `pip install -U transformers` or `pip install 'huggingface-hub<1.0'`.

**Takeaway:** `find_spec` checks "is the module present"; the user needs "does it actually import." For dep groups where the upstream stack has tight transitive pins, do the real import. Add a `--strict_import` flag for users who want to force the slower check across all groups.

---

### Lesson #21.5 — Smoke driver `[PASS]` false-positive: subprocess rc=0 ≠ "all sweep cells succeeded"

**Symptom:** A smoke script's status checker shows `[PASS]` for a stage that contains failing cells. Drilling into the log shows `[sweep] <cell_id> FAILED at <stage>: <error>` but the sweep CLI returned exit code 0 and the smoke driver trusted the rc.

**Root cause:** The kit's `pareto_sweep` defaults to `--keep_going True` — useful for catching all failures in one pass — but the CLI used to exit 0 regardless of cell outcomes. A smoke or CI driver that checks subprocess rc alone misses cell-level failures.

**Fix:** `pareto_sweep` now writes the index TSV first, then exits with `min(255, N_FAILED)`. Drivers using `--keep_going True` still see all cell results in the TSV; the process exit code surfaces the aggregate.

**Takeaway:** "Continue past failures" and "report the aggregate exit code" are independent knobs. The kit's design now separates them: `--keep_going` controls in-process behaviour, but the rc always reflects pass/fail across all cells.

---

### Lesson #20 — scriptconfig's smartcast auto-splits comma-strings into lists

**Symptom:** Passing `--source_scales "1.0,0.66"` to a CLI built on scriptconfig delivered `[1.0, 0.66]` (list of floats parsed as strings) to the receiving function — but the function called `str(scales).split(',')` and saw `'[1.0, 0.66]'`, hitting `ValueError: could not convert string to float: '[1.0'`.

**Root cause:** scriptconfig 0.9.1's `Value` uses smartcast with `allow_split="auto"` — strings containing commas are split into lists. The warning surfaces at parse-time but the cast happens silently. A field declared as `scfg.Value("1.0,0.66,0.40,0.25", help="comma-separated factors")` looks like a string but ends up as a list at the receiver.

**Fix:** Make the consumer tolerant of both shapes — `isinstance(scales, (list, tuple)) and ...` branches before string-splitting. Don't trust the Value's declared default type to match the runtime type.

**Takeaway:** When using scriptconfig, comma-string config fields are not stable as strings. Either declare `type=str` + `allow_split=False`, or make the consumer accept both list-of-X and comma-string-of-X. The kit's `_parse_scales` shows the consumer-side pattern.

---

### Lesson #21 — `kwcoco eval` confusion-sidecar dump can crash on cross-bundle relative paths

**Symptom:** `python -m kwcoco eval --true_dataset T.kwcoco.zip --pred_dataset P.kwcoco.zip --out_dpath ...` printed the computed `nocls_measures.ap` to stdout but exited non-zero. The trace showed "Failed to reroot gid=1 with fpaths=[...]" deep inside the confusion-sidecar dump.

**Root cause:** When `pred.kwcoco.zip` was written to a different bundle directory from `true.kwcoco.zip`, the pred bundle's `file_name="raw_assets/foo.jpg"` got resolved relative to the pred bundle's parent — pointing at a path that didn't exist (the actual JPEGs live next to the true bundle). The eval computed metrics fine because the image data isn't loaded for the AP calculation, but the confusion-sidecar writer tried to load the actual pixels to draw the heatmap and failed.

**Fix:** Two-layer defense.
1. When writing the pred kwcoco, rewrite `file_name` to the absolute path of the source asset (`true.get_image_fpath(gid)`).
2. When the eval subprocess returns non-zero but `detect_metrics.json` is on disk, recover the metrics and continue — same pattern as the ONNX export's `--simplify` crash recovery.

**Takeaway:** Don't trust subprocesses' all-or-nothing exit codes — check for the artifact too. Don't trust that a pred kwcoco's relative file_name resolves the same in a new bundle directory; rewrite to absolute when copying image rows.

---

## SEED ENTRIES — from `/home/joncrall/code/shitspotter/dev/journals/lessons_learned.md`

The following entries are scrubbed and ported from the prior project. They capture the 19 failure modes that cost real time in the v4/v5 prototype. They are pre-seeded here so the kit doesn't have to re-encounter them.

When you (the next agent) encounter a NEW failure mode worth >1 hour of someone's time, add it above the SEED ENTRIES block. Newest-first.

---

### Seed #19 — `kwcoco subset --select_images` requires the `jq` Python package

**Symptom:** `ModuleNotFoundError: No module named 'jq'` when running `python -m kwcoco subset --select_images ".id <= 8"`.

**Root cause:** kwcoco's `--select_images` flag uses jq syntax under the hood and requires the `jq` Python binding, which isn't a declared kwcoco dependency.

**Fix:** Use the deprecated `--gids 1,2,3,4` form (which doesn't need jq) OR `pip install jq` first.

**Takeaway:** Document the `--gids` form as the kit's canonical subset CLI. If anyone reaches for `--select_images`, the kit's check-env should remind them to install `jq`.

---

### Seed #18 — DEIMv2 HGNetv2 fixed-input invariant requires FIVE sizes to match

**Symptom:** `RuntimeError: The size of tensor a (400) must match the size of tensor b (100) at non-singleton dimension 1` deep inside `with_pos_embed` during eval, even though training appeared to work.

**Root cause:** A DEIMv2 HGNetv2 encoder pre-bakes its positional embedding at `eval_spatial_size`. For a generated config with `eval_spatial_size=[320,320]` but inherited upstream `Resize: [640, 640]` in train + val transforms (from `tpl/DEIMv2/configs/base/dataloader.yml`), the eval batch tensor was 20×20=400 feature elements but pos_embed was 10×10=100. Training had worked because the encoder dynamically resizes pos_embed during training, but eval uses the pre-baked one.

**Fix:** Override ALL FIVE of these together in the generated YAML:
- `eval_spatial_size`
- `train_dataloader.dataset.transforms[Resize].size`
- `val_dataloader.dataset.transforms[Resize].size`
- `train_dataloader.collate_fn.base_size`
- `train_dataloader.dataset.transforms[Mosaic].output_size` (conventionally INPUT/2)

**Takeaway:** When the kit's config generator changes one size knob, it MUST also propagate to all five. The benchmark-candidate Q3 enforces this. Write a unit test that loops over (variant, input_hw) pairs and asserts the five values match.

---

### Seed #17 — Multi-GPU all-reduce bottlenecked by slowest PCIe peer

**Symptom:** A 2× RTX 3090 host trained *slower* in DDP mode than single-GPU on GPU 0.

**Root cause:** GPU 1 was on a 2× PCIe link (vs. 16× for GPU 0). Gradient all-reduce runs at the slowest peer's bandwidth.

**Fix:** Default `CUDA_VISIBLE_DEVICES=0` for single-host non-cluster configs. Probe PCIe link width via `nvidia-smi --query-gpu=pcie.link.width.current` at setup; warn when `num_gpus > 1` and any active GPU is below 8× PCIe.

**Takeaway:** Multi-GPU is not unconditionally a speedup. The kit's `--tier` flag and PCIe probe must opt into multi-GPU only when it actually helps.

---

### Seed #16 — DEIMv2 OOMs on 24 GB GPU with naïve upstream batch sizes

**Symptom:** `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 MiB. GPU 0 has a total capacity of 23.54 GiB of which 117.62 MiB is free.` during deformable-attention `sampling_locations` allocation.

**Root cause:** Upstream's `tpl/DEIMv2/configs/base/dataloader.yml` sets `total_batch_size: 32` assuming 8-GPU training (per-GPU 4). On a single 24 GB GPU at batch=128 per the prototype's initial default, deformable-attention intermediate tensors pushed the model past 24 GB even at 320×320 input.

**Fix:** Per-`(variant, input_hw, tier)` memory table. For deimv2_n: batch 32 at input ≤ 320, 24 at ≤ 512, 16 at > 512. Scale auto-shrinks with input area.

**Takeaway:** A flat per-variant batch default fails at larger input sizes. The lookup must be two-dimensional. Document the empirical anchor point: `deimv2_n @ 320×320 batch=32 → 8.1 GB training footprint` on a 3090.

---

### Seed #15 — Non-root users can't raise RLIMIT_NOFILE above `ulimit -Hn`

**Symptom:** Wrapper script tried to raise the soft FD limit to 65536 via `ulimit -n 65536`, got "Operation not permitted", and printed "WARNING: failed to raise" without actually changing anything safer.

**Root cause:** Non-root users can only raise the soft limit up to the hard limit (`ulimit -Hn`); going above requires CAP_SYS_RESOURCE. On hosts where the kernel/systemd cap is < 65536, the wrapper would fail without a useful fallback.

**Fix:** Clamp the requested limit to `ulimit -Hn` before raising. Below 16384, direct the user to set `*_TORCH_MP_SHARING=file_system` as a kernel-cap-independent fallback (switches torch IPC from FD-per-tensor to filesystem-backed shared memory).

**Takeaway:** Don't assume your wrapper can raise resource limits arbitrarily. Always clamp to the hard cap; always have an alternative path.

---

### Seed #14 — HGNetv2 hybrid encoder doesn't support multi-scale (pos_embed mismatch)

**Symptom:** Same as Seed #18 but on the first training cell, not eval.

**Root cause:** A v4 sweep cell enabled `multiscale` collate for a HGNetv2 (DEIMv2-N) variant. The encoder pre-bakes pos_embed at `eval_spatial_size` and doesn't dynamically interpolate during training either. Upstream DEIMv2 explicitly sets `base_size_repeat: ~` for every HGNetv2 variant in its configs; only DINOv3-backed variants support multi-scale.

**Fix:** Per-variant `supports_dynamic_input()` method on the trainer plugin. The round-loop driver coerces `train_policy=multiscale → fixed` when False.

**Takeaway:** Architectural constraints leak through config. The kit's trainer plugin must expose them, and the orchestration layer must respect them. Benchmark candidate Q3 codifies this.

---

### Seed #13 — YAML `collate_fn` indent-leak (config generator bug)

**Symptom:** `TypeError: CocoDetection.__init__() got an unexpected keyword argument 'collate_fn'` deep inside DEIMv2's `workspace.create()`.

**Root cause:** The bash heredoc generator emitted `collate_fn:` with 4 spaces of indent, intending to land it as a sibling of `dataset` under `train_dataloader`. With the surrounding heredoc's indentation, it instead landed as a CHILD of `dataset`. `workspace.create()` then forwarded `collate_fn` as a kwarg to `CocoDetection.__init__`, which has no such parameter.

The trainer SURVIVED `bash -n`, SURVIVED `yaml.safe_load` (the document is *valid* YAML, just wrong), and only failed deep inside the framework at model-build time.

**Fix:** Dedent to 2 spaces. **Larger fix:** port YAML generation to Python with a `yaml.safe_load` round-trip + structural-invariant assertion before launching the trainer.

**Takeaway:** Bash-heredoc-templated YAML is fragile. Any structured-config generation by string templating needs a parse-validate step. The kit ports YAML-gen to Python explicitly to avoid this whole bug class. Benchmark candidate Q1 codifies this.

---

### Seed #12 — Sweep records `status=ok` after failed stage

**Symptom:** A sweep cell's training stage crashed but the sweep summary printed `status=ok` and proceeded to export against a missing checkpoint.

**Root cause:** The sweep's `run_cell` function was called inside `if run_cell ...; then` — bash's `errexit` behavior is surprising in `if` conditions, and a `printf ... status=ok` at the end of the function could still run after an inner pipeline failed (the pipeline's exit was hidden by `tee`).

**Fix:** Each stage gets its own explicit exit-code check; status is `fail_<stage>` on the first failure and never `ok` by default. Use `set +e` around the call, then read `$?` explicitly, so the parent script doesn't abort under `set -e`.

**Takeaway:** Never trust a default `ok` in shell error handling. Every stage that can fail must record its own status; the aggregate is the worst of the parts.

---

### Seed #11 — DEIMv2 has hidden runtime deps (faster_coco_eval, calflops, etc.)

**Symptom:** `ModuleNotFoundError: No module named 'faster_coco_eval'` 30 seconds into the first training cell, AFTER model construction.

**Root cause:** DEIMv2's `tpl/DEIMv2/engine/data/dataset/coco_dataset.py` imports `faster_coco_eval` at module top-level. The `tpl/DEIMv2/requirements.txt` declares it (and `calflops`, `transformers`, `tensorboard`, `scipy`, `PyYAML`), but no parent package installs them transitively.

**Fix:** Setup-time probe-and-install pulls all of `tpl/DEIMv2/requirements.txt`. Loop over each line, probe via `importlib.util.find_spec`, batch the missing ones into a single `pip install`.

**Takeaway:** Hidden transitive deps in submoduled trainers must be discovered at setup time, not at first run. Add a `--check-env` CLI that does this for every trainer plugin.

---

### Seed #10 — DEIMv2 ONNX export needs `onnxsim` and opset 18 on torch ≥ 2.5

**Symptom:** `ModuleNotFoundError: No module named 'onnxsim'` at end of export. Plus a `RuntimeError: No Adapter To Version $17 for Pad` warning during version conversion.

**Root cause:** DEIMv2's `tools/deployment/export_onnx.py --simplify` imports `onnxsim` after the ONNX is already written (the .onnx file is on disk; only the optional simplify step failed). The opset 17 → 18 conversion warning is because torch ≥ 2.5's dynamo exporter targets opset 18 internally and the Pad op has no opset-17 adapter.

**Fix:**
- Install `onnxsim` at setup time (added to ONNX trio).
- Default opset 18, not 17.
- The export wrapper detects `onnxsim` availability before passing `--simplify`; if missing, skip the flag and use the unsimplified ONNX.
- Recover the .onnx from the staging dir if the subprocess crashed during `--simplify` — the file is valid even without simplification.

**Takeaway:** Optional post-export steps can crash a subprocess that *already produced its main output*. Detect availability before passing optional flags; recover the artifact if it exists despite the crash.

---

### Seed #9 — `torch.onnx.export` on torch ≥ 2.5 imports `onnxscript` at function-call time

**Symptom:** `ModuleNotFoundError: No module named 'onnxscript'` from inside `torch.onnx.__init__` when calling `torch.onnx.export(...)`, even with `dynamo=False`.

**Root cause:** Modern torch's `torch.onnx.export` imports `_compat` → `_core` → `onnxscript` at function-call time, not module-import time. Passing `dynamo=False` doesn't avoid the import.

**Fix:** Declare `onnxscript` as `install_requires`. Setup-time probe adds it to the missing-deps list.

**Takeaway:** Top-level function-call-time imports are real and not preventable from the caller's side. The kit must declare every dep that *any code path* needs.

---

### Seed #8 — torch / torchvision ABI mismatch

**Symptom:** `RuntimeError: operator torchvision::nms does not exist` when importing torchvision.

**Root cause:** Independently-installed torch + torchvision with mismatched compiled-against versions. The torchvision C++ NMS op isn't registered for the installed torch's dispatcher.

**Fix:** Pin a matching pair in `pyproject.toml`. For torch 2.11, torchvision 0.26.

**Takeaway:** torchvision must always be installed alongside torch from the same matched pair. Use `pip install torch==X.Y torchvision==Z.W` from PyTorch's index, not separate installs.

---

### Seed #7 — pip 25+ rejects girder index due to missing PEP 700 metadata

**Symptom:** `ERROR: Index https://girder.github.io/large_image_wheels/ does not provide upload-time metadata.`

**Root cause:** pip 25.0+ enforces PEP 700 (upload-time metadata). Static-HTML wheel indexes (like the girder one) don't provide it.

**Fix:** Install from a direct wheel URL instead of via `--find-links`. Document the pattern in `docs/install.md`.

**Takeaway:** pip's index-discovery rules tighten over time. Direct wheel URLs always work; relying on a third-party index doesn't.

---

### Seed #6 — `geowatch.__init__` hard-imports `osgeo` at module load

**Symptom:** `ModuleNotFoundError: No module named 'osgeo'` from `geowatch/__init__.py` line 178 when importing any geowatch utility.

**Root cause:** geowatch has a `_execute_ordered_preimports()` that hard-imports `osgeo.gdal` at package init time. GDAL Python bindings need `libgdal` system libs and can't be installed cleanly via pip alone.

**Fix:** Make geowatch an OPTIONAL dep; soft-import via `try/except ImportError`. Document the manual GDAL install path (direct wheel from girder index) for users who need geowatch features.

**Takeaway:** Hard-importing optional system-lib-dependent packages at `__init__` time forces a dep onto every downstream user. The kit must NOT do this for any optional component.

---

### Seed #5 — `kwimage.imresize(interp='area')` fails on skimage backend without cv2

**Symptom:** `NotImplementedError: area` from `kwimage.im_transform._coerce_skimage_interpolation_order`.

**Root cause:** `kwimage.imresize` falls back to skimage when cv2 isn't installed. The skimage codepath doesn't recognise `'area'` interpolation.

**Fix:** Either install `opencv-python-headless` (preferred for headless servers) or catch `NotImplementedError` and retry with `'linear'`.

**Takeaway:** Library API "consistent" calls aren't really consistent when backends differ. Wrap such calls with try/except + a fallback that's known to work in all backends.

---

### Seed #4 — gdown silently writes HTML error pages

**Symptom:** Subsequent run sees a 4 KB `.pth` file at the expected path, treats it as cached, and downstream PyTorch crashes with a pickle error.

**Root cause:** When Google Drive throttles or requires confirmation, gdown writes an HTML error page to the destination path with exit code 0. The next run's `[ -f "$dst" ]` cache check accepts it.

**Fix:** Post-download size guard. Require ≥ 1 MiB for any checkpoint (well below the smallest legitimate one, well above any HTML stub). Re-download if too small. Hard error if still bogus after re-download.

**Takeaway:** Never trust `[ -f $dst ]` as "successfully downloaded." Always verify with a size threshold or a hash.

---

### Seed #3 — `kwimage.imwrite(..., imwrite_params=...)` is not a valid cv2 kwarg

**Symptom:** `cv2.error: imwrite() got an unexpected keyword argument 'imwrite_params'`.

**Root cause:** `kwimage.imwrite` forwards `**kwargs` straight to `cv2.imwrite`. cv2's JPEG-quality knob is a flat `params=[FLAG_INT, VALUE_INT, ...]` list, not a name=value kwarg.

**Fix:** Pass `params=[int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]`.

**Takeaway:** When a wrapper forwards `**kwargs`, the wrapped library's API is what matters, not the wrapper's friendlier-looking surface. Read the wrapped library's docs, not the wrapper's.

---

### Seed #2 — `kwimage.imresize(image, (W, H))` 2nd positional is `scale=`, not `dsize=`

**Symptom:** `cv2.error: Failed to allocate 44947419955200 bytes` (45 TB!) on the first image processed.

**Root cause:** Called `kwimage.imresize(image, (1280, 960), interpolation='area')` thinking the tuple was the target size. The 2nd positional argument is `scale`, so kwimage interpreted it as "scale by (1280, 960)" — a 4032×3024 phone image was asked to become ~5 megapixels wide.

**Fix:** Always pass `dsize=` keyword argument explicitly: `kwimage.imresize(image, dsize=(W, H), interpolation='area')`.

**Takeaway:** Library function positional arguments are not stable across versions or across "what the call would naturally read as." Pin every wrapper call to keyword arguments. This is a benchmark-worthy lesson: enforcing `dsize=` everywhere prevents this whole class.

---

### Seed #1 — gdown 6.x dropped `--fuzzy`

**Symptom:** `__main__.py: error: unrecognized arguments: --fuzzy`.

**Root cause:** gdown 6.0.0 removed the `--fuzzy` flag; the URL parser now handles every form natively, but the breaking CLI change wasn't surfaced clearly.

**Fix:** Pass the bare file ID as the first positional argument. Works in every gdown version.

**Takeaway:** Third-party CLI tools change incompatibly. Prefer the most stable form of the call (bare positional ID) over flags that may not exist.

---

## Empty section — first NEW lesson lands above this divider

When the kit's first >1hr-debug bug surfaces, write the entry above this line in the same Symptom / Root cause / Fix / Takeaway format. The seed entries below are reference material; new entries should reflect the kit's own history.
