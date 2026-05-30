# Benchmark candidates — WebDataset integration into a registered Dataset

Theme: a project (`kwcoco_detector_kit`) integrates a streaming
`webdataset`-backed reader (`kwcoco_dataloader`) as a Dataset under a
third-party trainer (DEIMv2) whose data layer uses YAML composition
+ a class-name registry. The "boundary surface" between these three
systems is where every gen002 bug landed.

Each candidate is the distillation of a real failure during the
2026-05-30 gen002 cycle into a question a future agent could
plausibly mis-answer.

The thread tying these together: the kit had unit tests for the
*reader* in isolation, unit tests for the *YAML generation* in
isolation, and unit tests for the *trainer config build* in
isolation. But there was no single test that crossed all three —
"start from a synthetic kwcoco; write shards; instantiate
`WebDatasetCocoDetection` via DEIMv2's `YAMLConfig.create()`; iterate
one batch." Each bug below crossed at least two of those boundaries.

---

## Q1 — YAML-merger leak from `__include__` parent into child class kwargs

Status: draft
Level: A
Tags: config-generation, yaml-merge, third-party-trainer, register-pattern, train-dataloader

**Pre-error context.** An agent extends a kit-generated trainer config
to swap the `train_dataloader.dataset` type from `CocoDetection` (a
random-access `torchvision.datasets.CocoDetection` subclass) to a new
`WebDatasetCocoDetection` (an `IterableDataset` over `.tar` shards).
They build the new class in a third-party submodule's `engine/data/
dataset/` directory, register it via `@register()`, and update the
kit's `_build_train_yml` to emit:

```python
"dataset": {
    "type": "WebDatasetCocoDetection",
    "shards_dpath": str(shards_dpath),
    "category_names": [...],
    "transforms": _train_transforms_block(input_hw),
}
```

The kit's generated YAML uses `__include__:` to inherit a base config
that defines `train_dataloader.dataset.{img_folder, ann_file,
return_masks}` (intended for the parent `CocoDetection`).

Smoke-tested: the file parses, the class is registered, `import`
succeeds. `pytest tests/unit/` is green.

Launched, the run dies at first call to `fit()` deep inside the
YAMLConfig's `workspace.create()`:

```text
TypeError: WebDatasetCocoDetection.__init__() got an unexpected
keyword argument 'img_folder'
```

### The hard question

> A trainer's YAML config system composes `train_dataloader.dataset`
> from multiple included files using a merge-by-key strategy, and
> instantiates the class by passing every key in the merged dict as
> `**kwargs`. You override `type:` to a new class with a different
> `__init__` signature. How do you write the integration so a parent
> config you don't control can't leak orphan keys into your new
> class — and what static check would you run against the *merged*
> config before launching?

### Invariant to preserve

For any registered Dataset class instantiated by `YAMLConfig.create()`:

```python
import inspect
sig = inspect.signature(NewDatasetCls.__init__)
allowed = set(sig.parameters) | {"self"}
merged_block = yaml_config.train_dataloader.dataset
assert set(merged_block) - {"type"} <= allowed, (
    f"YAML merger leaked unknown kwargs into {NewDatasetCls.__name__}: "
    f"{set(merged_block) - {'type'} - allowed}"
)
```

### Acceptance criteria

A pytest fixture that:

1. Generates the trainer's YAML for the new dataset type.
2. Resolves the `__include__` chain (or whatever the merger does).
3. Reads `NewDatasetCls.__init__`'s signature.
4. Asserts every merged key is in the signature OR is explicitly
   accepted-and-ignored (e.g. by a comment-annotated `Optional`
   field defaulting to `None`).

The test should fail BEFORE launching the trainer, on every variant
in the sweep matrix.

### Why this is a benchmark question

The "I'll just inspect my new class's signature" instinct is
correct but agents reliably forget to do it because:

- The YAML generation code path looks clean (no `img_folder` written).
- The smoke test ("does it parse?") passes.
- The error fires inside the third-party trainer, several frames
  into its `create()` machinery, after a lot of unrelated setup
  output.
- The fix is two lines (accept-and-ignore), which lulls the agent
  into thinking the bug was trivial; the *invariant* is what matters.

### Real-world reference

`tpl/DEIMv2/engine/data/dataset/wds_coco_dataset.py` commit `6b5a2ef`
on Erotemic/DEIMv2 fork.

---

## Q2 — scriptconfig smartcast on a JSON-object string

Status: draft
Level: B
Tags: scriptconfig, cli, json, comma-split, config-generation

**Pre-error context.** An agent adds a new sweep CLI flag for a
runtime mapping (raw source category → target class name). They
pass it as a JSON-encoded object string from the launcher:

```bash
KCD_WDS_SOURCE_TO_TARGET='{"B":"nonpup_sealion","S":"nonpup_sealion","F":"nonpup_sealion","P":"pup"}'
"$PYTHON_BIN" -m kwcoco_detector_kit sweep \
    --train_wds_source_to_target "$KCD_WDS_SOURCE_TO_TARGET" \
    ...
```

In the sweep's scfg field they leave the default `type=` (which is
"smart" and inspects the value). At parse time scriptconfig sees a
string with commas, autosplits it into a set of fragments, and the
downstream `json.loads(config.train_wds_source_to_target)` raises:

```text
TypeError: the JSON object must be str, bytes or bytearray, not set
```

### The hard question

> You define a scriptconfig field that accepts a JSON-encoded value
> on the CLI. The value is sometimes a flat list (`a,b,c`), sometimes
> an object (`{"a":"x"}`), sometimes a path. Pick the field
> declaration that survives all three input shapes without changing
> the consumer code, AND explain when smartcast is the wrong default.

### Invariant to preserve

For any kit `scfg.Value` field whose value can be a JSON-encoded
object (object/array/string), declare `type=str` and parse via
`json.loads` in the consumer. The autosplit behaviour is a footgun
for any JSON value with commas.

```python
opaque_json_blob = scfg.Value(
    None,
    type=str,                  # disables comma-autosplit
    help="JSON string of mapping {source: target}.",
)
# consumer:
import json
mapping = json.loads(config.opaque_json_blob) if config.opaque_json_blob else {}
```

### Acceptance criteria

Test fixture that round-trips `--field '{"x":"y"}'` through the CLI
parser and asserts:

1. `type(config.field) is str`
2. `json.loads(config.field) == {"x": "y"}`

Same fixture parameterised over `--field "a,b,c"` (legitimate set)
and `--field "/path/to/file"` (string), verifying each ends up the
shape the consumer expects.

### Real-world reference

`kwcoco_detector_kit/orchestration/pareto_sweep.py` commit `ec758ca`
(`type=str` added to `train_wds_source_to_target`); same class of
bug fixed earlier in `KCD_INPUT_HW` (commit `03a8d9f`).

---

## Q3 — Dataclass-API drift across two co-developed packages

Status: draft
Level: B
Tags: api-stability, dataclass, refactor, kwargs-mismatch, integration-test

**Pre-error context.** An agent adds a new `WebDatasetCocoDetection`
class in repo A that internally constructs a `SchemeMapping` from
repo B's reader. They write the constructor call from memory or
docstring:

```python
self.scheme = SchemeMapping(
    source_to_target=dict(source_to_target),
    target_names=list(self.category_names),
)
```

Both `source_to_target` and `target_names` *sound* like the right
names for the fields they want. They are not. The dataclass's actual
fields are `target_order` and `mapping`:

```python
@dataclass
class SchemeMapping:
    target_order: List[str]
    mapping: Dict[str, str]
    drop: Set[str] = field(default_factory=set)
    unmapped_policy: str = "drop"
    name: str = ""
```

The kit's `pytest tests/unit/` is green because the constructor is
called only inside `__iter__`, which the kit's tests don't drive.
Found at *first sample yield* on the real trainer:

```text
TypeError: __init__() got an unexpected keyword argument 'source_to_target'
```

### The hard question

> You're integrating an external dataclass-based API where the field
> names sound similar but aren't the ones you imagine. Your code is
> already in production. What is the cheapest test that would have
> caught this BEFORE the run reached `fit()`, and what static rule
> would prevent the same mistake in a sibling integration?

### Invariant to preserve

Any kit-side adapter that constructs a third-party dataclass should
build at least one instance during *module import* (or in a CPU-only
smoke test) so the dataclass's `__init__` keyword-mismatch raises
the moment the adapter is imported, not at first yield.

```python
# at bottom of wds_coco_dataset.py, optional:
if __debug__:
    _ = SchemeMapping(target_order=["x"], mapping={"a": "x"})
```

Or, more robustly: a pytest that instantiates
`WebDatasetCocoDetection(shards_dpath=tiny_shards, ...)` with a
4-image synthetic dataset and pulls one sample.

### Acceptance criteria

A smoke test in `tests/integration/` that:

1. Builds a 4-image synthetic kwcoco with raw-class annotations.
2. Runs `build_detection_webdataset` → tiny shards.
3. Instantiates `WebDatasetCocoDetection(shards_dpath=...,
   category_names=['widget'], source_to_target={'widget':'widget'})`.
4. Iterates one sample.
5. Asserts `target['boxes'].shape == (N, 4)` and
   `target['labels'].dtype == torch.int64`.

This single test would have caught Q1, Q2, Q3 and at least one prior
bug (the `task_type` API removal). Total runtime: a few seconds.

### Real-world reference

`tpl/DEIMv2/engine/data/dataset/wds_coco_dataset.py` commit `852479f`
on Erotemic/DEIMv2 fork.

---

## Q4 — `python -m <package>` vs `python -m <package>.cli.<entry>`

Status: draft
Level: C
Tags: cli, packaging, __main__, env-bootstrap

**Pre-error context.** A launcher script invokes a Python package's
CLI from bash:

```bash
"$PYTHON_BIN" -m kwcoco_dataloader build_detection_webdataset \
    --in_fpath ... --out_dpath ...
```

The package ships a `cli/` directory with one module per command,
each exposing a scriptconfig `Cli`, but no top-level `__main__.py`
and no `[project.scripts]` console-script in `pyproject.toml`. The
bash invocation fails with:

```text
/opt/venv/bin/python: No module named kwcoco_dataloader.__main__;
'kwcoco_dataloader' is a package and cannot be directly executed
```

### The hard question

> Given a Python package that has `pkg/cli/foo.py` exposing
> `FooCLI.main` but no `pkg/__main__.py` and no console-script
> entry, which of these invocations work and why?
>
> 1. `python -m pkg foo ...`
> 2. `python -m pkg.cli foo ...`
> 3. `python -m pkg.cli.foo ...`
> 4. `python -c "from pkg.cli.foo import FooCLI; FooCLI.main()"`
>
> What's the minimal Python-side change that would let (1) work?

### Invariant to preserve

When wiring a launcher script to a third-party package's CLI, prefer
the module-path form (`python -m pkg.cli.foo`) over
"`python -m pkg <subcommand>`" because the latter requires a
`__main__.py` that dispatches on `sys.argv[1]`. If the upstream
package doesn't ship one, don't fake it from your launcher;
that's an upstream-API stability bet you don't want to be making.

### Real-world reference

`projects/viame_sealions_2026/scripts/_launch_train.sh` commit
`79e28dd` (changed `python -m kwcoco_dataloader build_detection_webdataset`
to `python -m kwcoco_dataloader.cli.build_detection_webdataset`).

---

## Q5 — Module-level import of an "optional" dep

Status: draft
Level: C
Tags: import-time, optional-dependency, packaging

**Pre-error context.** `kwcoco_dataloader/readers/detection.py`
decorates hot-path functions with `@line_profiler.profile` (which is
a no-op when line_profiler isn't running). The module imports
`line_profiler` at top level:

```python
import line_profiler

...

@line_profiler.profile
def _sample_from_wds_raw(raw): ...
```

The package's `pyproject.toml` declares `line_profiler` under an
optional `profile` extras group. The kit's docker image
`pip install -e tpl/kwcoco_dataloader` (no extras flag) doesn't pull
it. Importing the readers package crashes:

```text
ModuleNotFoundError: No module named 'line_profiler'
```

### The hard question

> A library wants to use `@line_profiler.profile` decorators that are
> no-ops when line_profiler isn't installed. Pick the import shape
> that keeps `import lib.readers` working when the user installs the
> library without the `profile` extras, AND keeps the decorators
> active when they do install it.

### Invariant to preserve

Any module-level `import optional_pkg` that's referenced only by
decorators or runtime-conditional code paths should be guarded:

```python
try:
    import line_profiler
    _profile = line_profiler.profile
except ImportError:
    def _profile(fn):
        return fn

@_profile
def hot_path(...): ...
```

Equivalent: a one-liner conditional `profile = getattr(...,
'profile', lambda f: f)`.

### Real-world reference

`docker/opengroundingdino/Dockerfile` commit `4b23c65` adds
`line_profiler` to the runtime pip install as the operational
workaround; the upstream fix (graceful import) is tracked as a
follow-up PR to `kwcoco_dataloader`.

---

## Q6 — Multiple env-var passthrough whitelists across shell stages

Status: draft
Level: A
Tags: bash, env-vars, slurm, docker, whitelist-vs-wildcard

**Pre-error context.** A launcher submits training jobs through a
chain of bash scripts:

```
submit_train_*.sh  (host, sets KCD_* env)
  → exec _submit_train.sh  (host; writes env to file; sbatch)
    → _sbatch_train.sh  (slurm; sources env file; docker run -e ...)
      → _launch_train.sh  (in container; uses KCD_* env)
```

The agent adds a new env knob (say, `KCD_RESUME_CKPT`). They:

1. `export KCD_RESUME_CKPT=...` in the submit script ✓
2. Add a `${KCD_RESUME_CKPT:+--resume "$KCD_RESUME_CKPT"}` conditional
   in `_launch_train.sh` ✓

They submit. The slurm job runs, `_launch_train.sh` echoes
`KCD_RESUME_CKPT=<unset>`, and the resume silently no-ops. Tracing
through: each layer between submit and launch has its OWN hand-
maintained list of KCD_* vars to forward (one whitelists for the env
file write, one whitelists for `docker run -e`). Both were forgotten.

### The hard question

> You have an env-var passthrough chain with three or more
> hand-maintained whitelists between the user shell and the
> innermost container. Each layer in the chain re-declares the list
> of vars to pass. The vars share a common prefix (`KCD_`). Pick a
> single one-line change at each layer that prevents future
> whitelist drift, and explain why it's safe to use a glob.

### Invariant to preserve

When the env namespace is reserved (`KCD_*`), replace every hand-
maintained whitelist with a glob enumeration:

```bash
# bash 4+ (works in container shells and host shells)
while IFS= read -r v; do
    val="${!v:-}"
    [ -n "$val" ] && printf 'export %s=%q\n' "$v" "$val" >> "$ENV_FPATH"
done < <(compgen -v | grep -E '^KCD_' | sort -u)
```

`compgen -v` (not `compgen -e`) picks up vars set without `export`
too — caught a latent regression where `paths.sh` set
`KCD_REPO_ROOT` without `export` and it silently disappeared.

### Acceptance criteria

For every new `KCD_*` env var introduced to the launcher, a
single-line addition in the shell stages should be sufficient. A
smoke test that submits a slurm job with `KCD_FOO=bar` set and
asserts the container's `printenv KCD_FOO == bar` would catch any
whitelist regression.

### Real-world reference

`projects/viame_sealions_2026/scripts/_submit_train.sh` commit
`c29dba9` and `_sbatch_train.sh` commit `9ea00cf` (two separate
whitelists, two separate fix commits). Caught only by a debug
`echo "$KCD_RESUME_CKPT"` added in `_launch_train.sh` (commit
`c29dba9`); the original symptom was "sweep `ok_resumed` instantly
with no work done".

---

## Q7 — Stale tests pinned to a removed upstream API

Status: draft
Level: B
Tags: testing, dep-upgrade, deleted-api, skip-if-missing

**Pre-error context.** The kit's `tests/unit/test_build_webdataset_detection.py`
calls `BuildWebdatasetCLI.custom_subset_detection(...)` and
`BuildWebdatasetCLI.main(..., task_type='detection', ...)`. Both
were removed when the upstream `kwcoco_dataloader` package
refactored detection-mode out of the shared `BuildWebdatasetCLI`
into its own `BuildDetectionWebdatasetCLI`.

The tests skip with `pytest.importorskip("kwcoco_dataloader")` when
the dep isn't installed. The kit's docker image previously didn't
install `kwcoco_dataloader`, so the tests silently skipped for
months. When the image was updated to bake the dep in, all 4 tests
suddenly ran and immediately failed the in-image pytest gate, which
killed the docker build.

### The hard question

> You have a kit-side test file that exercises a third-party
> package's API. The test uses `pytest.importorskip` for the dep.
> Six months later the dep is restructured and your API call goes
> away. When does your CI catch this — and what would change so the
> CI would catch it the day the dep upgrade lands, not the day the
> dep becomes part of the install plan?

### Invariant to preserve

A `pytest.importorskip("dep")` call hides API drift. Either:

1. Pin the dep in the dev/test extras so CI always has it
   (`importorskip` then becomes a one-time guard for users not
   running tests), or
2. Move kit-side integration tests for the dep INTO the dep's own
   repo, so they live with the API they exercise, or
3. Add a `skip + warn` flavour: still skip when missing, but emit
   a `pytest.warns(UserWarning, ...)` so CI dashboards flag the
   skip cluster.

### Acceptance criteria

CI configuration that fails (or visibly warns) when a test file
contains `importorskip` AND that file lives in the kit AND the dep
is declared in any extras group. Tests that exercise a third-party
API should either always run (dep pinned in dev extras) or live in
the third party's repo.

### Real-world reference

Test file `tests/unit/test_build_webdataset_detection.py` deleted
in commit `8864a80` (the kit's tests went stale against the
upstream API refactor; replacement coverage lives at
`tpl/kwcoco_dataloader/tests/test_build_detection_webdataset.py`,
which is part of the dep's own test suite).

---

## Cross-references

- The `compositions.md` candidate "WebDataset reader + register()
  Dataset + slurm pipeline" composes Q1, Q3, Q6 — agents who pass
  the components individually don't necessarily catch the case
  where all three invariants must hold simultaneously across a
  shell-script chain, a YAML composition, a dataclass constructor,
  and a third-party trainer.
- Q2 generalises the `KCD_INPUT_HW` scriptconfig smartcast question
  from `pipeline-bootstrap-questions.md::Q2` (Cross-library
  transitive runtime deps).
- Q6 cross-references the "RLIMIT_NOFILE for torch dataloader IPC"
  candidate in `pipeline-bootstrap-questions.md::Q4`: both are
  "the wrong invariant was preserved as one of several
  hand-maintained lists."
