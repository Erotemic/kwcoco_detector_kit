# Benchmark candidates — pipeline bootstrap

Hard-problem invariants discovered during the prior project (`/home/joncrall/code/shitspotter/experiments/mobile_app_training_v{4,5}/`) that the `kwcoco_detector_kit` MUST preserve. Each candidate is the distillation of a *real* failure into a question a future agent could plausibly mis-answer.

The thread tying these together: a training cell in the v4/v5 prototype cost minutes-to-hours of GPU time *just to surface the next missing dep or shape mismatch*. Front-loading discovery into a 30-second `--check-env` pre-flight + a CPU-only smoke test changed iteration cost from "next training restart" to "next probe call."

---

## Q1 — YAML composition by string templating + indent arithmetic

Status: draft
Level: B
Tags: config-generation, yaml, parser-divergence, train-eval-export-parity
Pre-error context: an agent extending the kit adds a new optional knob to the DEIMv2 trainer's generated config (e.g. a new mixup parameter). They templated the block as a Python f-string (or a bash heredoc, in the prior project) and dropped it into the right section.

### Source context

The kit's `trainers/deimv2.py:generate_config()` generates YAML by composition of named blocks. The block for `collate_fn` is intended to be a *sibling* of `dataset` under `train_dataloader`. With 4 spaces of indent it would land as a *child* of `dataset`, and DEIMv2's `workspace.create()` would forward it as a kwarg to `CocoDetection.__init__`:

```text
TypeError: CocoDetection.__init__() got an unexpected keyword argument 'collate_fn'
```

The trainer would survive any "did the YAML parse" smoke test (the document is *valid* YAML — just structurally wrong), and only fail deep inside `engine/core/workspace.py` at model-build time.

### The hard question

> When generating structured config files (YAML, JSON, TOML) by string composition, what is the cheapest way to verify the *parsed structure* matches the intent — not just that the file parses?

### Invariant to preserve

For `train_dataloader` in any kit-generated DEIMv2 config:

```python
import yaml
cfg = yaml.safe_load(open('train.yml'))
td = cfg['train_dataloader']
# Required:
assert set(td.keys()) >= {'total_batch_size', 'num_workers', 'dataset', 'collate_fn'}
assert 'collate_fn' not in td['dataset']    # sibling, not child
```

### Acceptance criteria

A pytest fixture that:

1. Invokes `generate_config()` for every (variant, train_policy, input_size) combination in the sweep matrix.
2. `yaml.safe_load`s the resulting `train.yml`.
3. Asserts the structural invariants above.
4. Plus: launches DEIMv2's `engine.core.YAMLConfig` against the same file and reaches the model-build step without raising.

The test should fail clearly when an interpolation indent is wrong, *before* the trainer is ever launched on a GPU.

### Why this is a benchmark question

A capable agent could:

* Get the YAML wrong by an off-by-one indent.
* Add a new heredoc-/f-string-interpolated block in a future patch and reproduce the same indent confusion.
* Add a sibling key inside `dataset:` but visually think it's under `train_dataloader:`.

Catching it requires either (a) testing the parsed structure, not just the byte stream, or (b) running through the framework that consumes it. Both are easy; neither is reflex behaviour.

---

## Q2 — Cross-library transitive runtime deps

Status: draft
Level: A
Tags: env-bootstrap, undeclared-deps, fail-fast, pipeline-orchestration

### Source context

The kit composes 6+ third-party libraries (DEIMv2, kwcoco, kwimage, geowatch, torch, gdown, kwcoco_dataloader, optional onnx-stack). Each has its own undeclared / lazy runtime deps:

* `gdown 6.x` dropped `--fuzzy` (CLI surface change).
* `torch.onnx.export` on torch ≥ 2.5 imports `onnxscript` at function-call time, even when `dynamo=False`.
* `geowatch.__init__` hard-imports `osgeo` (GDAL Python bindings) at package init.
* `shitspotter.cli.simplify_kwcoco` (in the prior project) hard-imports `geowatch.utils.util_kwimage` at top of `main()`.
* `kwimage.imresize(interp='area')` only works with cv2; skimage fallback raises NotImplementedError.
* DEIMv2's `tpl/DEIMv2/requirements.txt` (faster_coco_eval, calflops, transformers, tensorboard, scipy) is not declared by the wrapping package.

Each missing piece killed a training cell ~30s in, after model construction but before any meaningful work. Each fix surfaced the next.

### The hard question

> When orchestrating a pipeline that spans N third-party packages, what is the right place + shape for a pre-flight environment audit that catches all O(N) categories of missing-dep bugs *before* the first GPU minute is spent?

### Invariant to preserve

For any clean host that has the kit's *direct* deps installed, running `bash examples/<example>/run_all.sh` to completion should not surface any `ModuleNotFoundError` from any *transitive* library the pipeline composes.

### Acceptance criteria

A pre-flight check (the kit's `orchestration/setup_audit.py` or `cli/__main__.py --check-env`) that, given a fresh host with only the kwcoco_detector_kit package + DEIMv2 submodule installed:

1. Probes for each of: gdown, onnxscript, onnx, onnxruntime, onnxsim, every line of `tpl/DEIMv2/requirements.txt`.
2. Triggers the following imports inside a subprocess and reports each as "OK" or "missing X":
   - `from torch.onnx import export`
   - `import faster_coco_eval`
   - the trainer-plugin's `_load_model` codepath on a known-good checkpoint
3. Loops with `pip install` for each missing module that has a declared install path.
4. Exits non-zero if any required dep can't be acquired automatically, and prints the manual install command.

A pytest equivalent:

```python
def test_pre_flight_check_finds_all_required_deps():
    from kwcoco_detector_kit.orchestration.setup_audit import probe_env
    missing = probe_env()
    assert missing == [], (
        f"required deps missing: {missing}; "
        f"run `kwcoco-detector-kit check-env --install`"
    )
```

### Why this is a benchmark question

Adding a new variant, trainer plugin, or third-party submodule trivially regresses this. The fix is mechanical (`pip install`), the discovery is wasteful (one whole training cell per missing dep). Front-loading it into the audit is the lesson.

---

## Q3 — Upstream architectural constraints leak through config

Status: draft
Level: B
Tags: model-architecture, config-defaults, train-eval-export-parity

### Source context

The kit's per-cell config generator may default `train_policy=multiscale` across all variants. For DINOv3-backed variants this is fine — the encoder dynamically interpolates positional embeddings per batch. For HGNetv2 hybrid encoders (DEIMv2 Atto / Femto / Pico / N) it is a hard architectural mismatch: pos_embed is pre-baked at `eval_spatial_size` and doesn't interpolate, so a multi-scale collate produces:

```text
RuntimeError: The size of tensor a (121) must match the size of tensor b (100)
at non-singleton dimension 1
```

mid-encoder, at first batch.

Upstream DEIMv2 knows this — every HGNetv2 variant ships with `base_size_repeat: ~` in its config. Multi-scale is opt-in only for DINOv3-backed variants.

### The hard question

> When `__include__`-ing or programmatically extending an upstream framework config, which fields encode *architectural* constraints (must not be changed without code work) vs *training hyperparameters* (free to tune)?

### Invariant to preserve

For any kit-generated config using a HGNetv2-backed variant, the generated YAML must set `base_size_repeat: ~` (i.e. `train_policy=fixed`). For DINOv3-backed variants, multi-scale is permitted.

### Acceptance criteria

A pytest fixture that:

1. Generates a per-cell config for every (variant, train_policy) pair in the kit's default sweep matrix.
2. Asserts `base_size_repeat == None` for HGNetv2 variants regardless of requested train policy.
3. Asserts the train policy override is honoured for DINOv3-backed variants.

Bonus: launch each generated config through `engine.core.YAMLConfig` + `cfg.model.deploy()` on CPU with a tiny dummy batch to confirm the encoder accepts the configured shape. This catches the same class of bug for *any* future architectural mismatch.

### Why this is a benchmark question

Adding a new variant family (e.g. a future `convnext-v3-x` with a totally different encoder) trivially reintroduces this. The lesson is *always check upstream configs for which fields are non-default in the upstream's own per-variant overrides* — those are the fields that encode architectural constraints.

---

## Q4 — RLIMIT_NOFILE for torch dataloader IPC

Status: draft
Level: B
Tags: env-bootstrap, torch-multiprocessing, fail-late, pipeline-orchestration

### Source context

torch dataloader workers pass tensors back to the main process via `torch.multiprocessing.reduce_storage`, which opens a unix-domain socket per shared tensor (the `file_descriptor` sharing strategy). With 4 workers × batch=128 × O(many) shared tensors per batch, FD usage climbs past the default 1024 soft limit a few iterations into training:

```text
OSError: [Errno 24] Too many open files
  File ".../torch/multiprocessing/reductions.py", line 616, in reduce_storage
  File ".../multiprocessing/reduction.py", line 198, in DupFd
```

The crash happens *deep inside multiprocessing*, not where the trainer lives. Stack trace points to `DupFd` rather than to anything the user's code touches, so the root cause is non-obvious. Non-root users can't raise above `ulimit -Hn`, so a naïve `ulimit -n 65536` fails silently on hosts with low hard caps.

### The hard question

> Which torch-runtime-level resource limits (FDs, shared-memory bytes, NCCL timeouts, CUDA allocator behaviour) are the trainer's wrapper script responsible for setting before launching the framework? Which are the framework's job?

### Invariant to preserve

Any sweep cell launched on a fresh shell must not crash with `OSError: [Errno 24]` from inside torch.multiprocessing. The wrapper must raise `RLIMIT_NOFILE` to a value safe for the configured (workers × batch × tensor count) at the upper end of the sweep matrix, clamped to `ulimit -Hn`.

### Acceptance criteria

A pytest fixture that:

1. Captures the soft FD limit before invoking the trainer in a subshell.
2. Confirms the script raises the soft limit to ≥ V*_FD_LIMIT before the trainer subprocess is spawned.
3. Falls back gracefully (clear warning, no abort) when the shell hard limit is below V*_FD_LIMIT.
4. Documents the `*_TORCH_MP_SHARING=file_system` fallback for hosts where even the hard cap is too low.

Bonus: smoke-train a `mock_tiny` cell with `num_workers=8, batch=64` to verify no FD storm under the configured limit.

### Why this is a benchmark question

A future agent could:

* Add a new sweep cell with bigger batch / more workers and not re-check the FD math.
* Refactor the wrapper scripts and forget to re-raise the limit.
* Move the trainer launch into a helper that doesn't inherit the ulimit setting.

Each regression silently produces a "trains for a few iterations then dies in DupFd" failure mode that's expensive to debug from the stack trace alone. The lesson: **treat `RLIMIT_NOFILE` as a contract between the wrapper script and the framework**, declared once near the launcher and never assumed to come from the user's shell.

The fallback knob (`*_TORCH_MP_SHARING=file_system`) is a useful escape valve for genuinely huge batch × worker configs where even 65536 isn't enough — but it costs throughput, so default off.

---

## Composition note

Q1, Q2, Q3, Q4 chain. An agent who skips the pre-flight (Q2) won't discover the YAML structural bug (Q1) until the trainer dies inside the framework — at which point they may *also* trip the architectural constraint (Q3) because they're rapidly iterating on the wrong hypothesis. Q4 then hits at the first multi-worker batch. The cheapest defense is to run all four checks before the first GPU minute.

Future compositional questions worth catalogueing here:
- Q1 + Q3 cross-product: the agent generates a HGNetv2 config with multi-scale enabled AND a YAML indent bug, gets a `with_pos_embed` error that *looks* like Q3 but is actually Q1.
- Q2 + Q4: the agent's pre-flight passes but didn't probe FD limits; first multi-worker run hits Q4 at training time.

---

This file was seeded from `/home/joncrall/code/shitspotter/dev/benchmark-candidates/pipeline-bootstrap-questions.md` and scrubbed of project-specific names (`v4`/`v5`/`mobile_app_training`/`shitspotter`). The four candidates above are the ones the kit must preserve from day one.
