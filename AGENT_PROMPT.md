# Next-agent first prompt — kwcoco-detector-kit

Paste this verbatim as the first message to a fresh agent.

---

## ⚠️ STOP — READ THIS FIRST: EMFILE / virtiofs storm protocol

The host VM uses **virtiofs**, which periodically exhausts kernel file descriptors and causes `OSError: [Errno 24] Too many open files` across many unrelated syscalls — `pip install`, `python -c`, `ls`, even reading source files. **This is a kernel-level limit, not a per-process `ulimit -n` problem.** It cannot be worked around in user space.

**When you see EMFILE / "too many open files" during install, test, or import:**

1. **STOP immediately.** Do not retry. Do not "just install one more thing." Do not try to work around it by closing files, batching, sleeping, or using `ulimit -n`.
2. **Tell the user**: "I'm hitting EMFILE from the virtiofs filesystem. Need a VM reboot to clear it. Continuing without a reset will produce unreliable results."
3. **Wait.** The user will reboot the VM and tell you to continue.
4. **After the reboot**, retry the failed operation once. If it works, proceed. If it still fails, ask the user — the dataset or path may be the issue, not virtiofs.

This rule supersedes any "try harder" or "be persistent" instinct. The prior agent on this project burned hours fighting this; you will too if you don't follow the protocol. **Static work (reading source files, designing, writing markdown plans) is fine during an EMFILE storm; live execution (pip install, pytest, training) is not.**

---

## Mission

You are tasked with building **`kwcoco-detector-kit`** — a clean, domain-agnostic Python package for training object detectors on kwcoco datasets. The package must scale from a single 12 GB GTX 1080 Ti up through 4× 96 GB Blackwell / A100 / H100 clusters, and from small mobile detectors (DEIMv2 HGNetv2 Atto, ~0.5M params) to large DINOv2/DINOv3-backed transformer detectors (OpenGroundingDINO, ~200M params). It will eventually support multispectral remote-sensing data and efficient webdataset-format batch storage. Target domains: land photos, aerial mosaics, underwater, satellite — anywhere a kwcoco file with bounding-box annotations is a reasonable input.

**Working directory**: `/home/joncrall/code/kwcoco_detector_kit/` (already git-initialised; this is where you ship).

## Step 1 — Read these files in full before writing any code

The handoff plan, the reference implementations, and the lessons-learned doc are all you need to plan Phase 1. **Do not skip these.** Reading time: ~90 minutes. Failure to read will cost much more than that in re-discovery.

> **Note**: the kit's own `dev/` subtree is **pre-seeded**. Items 2, 6, and 8 below also exist as scrubbed copies under `/home/joncrall/code/kwcoco_detector_kit/dev/` (see Step 3 item #11). You can read either the original source files (those in `shitspotter/dev/` and `ambition/dev/`) or the seeded copies in `kit/dev/` — same engineering content, no shitspotter-specific names in the seeded copies. The seeded copies double as the canonical kit-style scrubbed form.

1. **`/home/joncrall/code/kwcoco_detector_kit/PLAN.md`** — the full handoff plan. 740 lines. The authoritative reference for everything below. Read it cover-to-cover.

2. **`/home/joncrall/code/shitspotter/dev/journals/lessons_learned.md`** — 19 documented failure modes from the prototype. Specifically read both 2026-05-10 and 2026-05-11 entries. Most of these failures are landmines you will step on if you don't front-load defenses. *(Scrubbed copy: `dev/journals/lessons_learned.md`.)*

3. **`/home/joncrall/code/shitspotter/experiments/mobile_app_training_v4/DESIGN.md`** + **`README.md`** + **`AUDIT.md`** — the small-detector lineage rationale + an audit doc covering the v4 pipeline.

4. **`/home/joncrall/code/shitspotter/experiments/mobile_app_training_v5/DESIGN.md`** + **`README.md`** — multi-scale tile + hard-negative-mining rationale.

5. **`/home/joncrall/code/shitspotter/experiments/foundation_detseg_v3/README.md`** + **`v9_train_eval_opengroundingdino_sam2.sh`** — the big-DINO lineage. The shell script is ~900 lines and contains the canonical OpenGroundingDINO + SAM2 training pipeline. Read it in full; it is the second of two reference implementations.

6. **`/home/joncrall/code/shitspotter/dev/benchmark-candidates/pipeline-bootstrap-questions.md`** — four durable invariants the kit must preserve. *(Scrubbed copy: `dev/benchmark-candidates/pipeline-bootstrap-questions.md`.)*

7. **`/home/agent/code/kwcoco_dataloader/kwcoco_dataloader/cli/build_webdataset.py`** — at least read the header docstring (~70 lines). The existing kwcoco → webdataset pipeline. The kit depends on `kwcoco_dataloader`; it does not fork it.

8. **`/home/joncrall/code/ambition/dev/README.md`** + **`/home/joncrall/code/ambition/dev/benchmark-candidates/README.md`** — the engineering-memory template. The kit's `dev/` subtree was already ported from this. Reading both is still useful — sub-200 lines combined, and tells you the quality bar for a good entry. *(Scrubbed copies: `dev/README.md` + `dev/benchmark-candidates/README.md`.)*

After reading, summarise back to the user in ≤ 300 words what Phase 1 entails, what the highest-risk task is, and what one open question you would most want answered first.

## Step 2 — Confirm assumptions with the user before starting

Before writing any code, get answers to these (the plan's §14 lists them):

1. **Package owner / git host**: github org? gitlab.kitware? The kit needs a home.
2. **Python version range**: 3.10–3.13 default; confirm or restrict.
3. **NOAA Steller dataset location** (for the sealion example in Phase 2): is there a kwcoco bundle on disk already, or do you need to build one from scratch?
4. **Trainer plugins beyond DEIMv2 + OpenGroundingDINO**: ship YOLOX or RT-DETR in v1, or defer?
5. **SAM2 segmenter co-training**: required in v1 (the shitspotter v9 AP=0.766 result depended on it), or v1.1+?

If the user says "use defaults, start Phase 1", proceed with: github.com/Erotemic, Python 3.10–3.13, build sealion kwcoco in Phase 2, DEIMv2 + OGDino only in v1, SAM2 deferred to v1.1.

## Step 3 — Phase 1 scope (only do this; confirm before Phase 2)

The plan has three phases. Phase 1 is your initial scope:

1. Scaffold the package (`pyproject.toml`, module layout, README skeleton).
2. Lift `v5_tile.py` → `kwcoco_detector_kit/data/tile.py` with three modes (`quadrant`, `multiscale`, `full_only`).
3. Lift `v5_merge.py` + `v5_mine.py` → `data/`.
4. Lift `coco_adapter.py:_build_coco_export` → `data/coco_export.py`.
5. **THE highest-risk task**: port `_train_deimv2_variant.sh` → `kwcoco_detector_kit/trainers/deimv2.py` in **Python**, covering all 12 DEIMv2 variants (not just the four mobile ones). The bash heredoc YAML generator is fragile (failure #13 in the lessons doc). **Write its tests first.** The tests:
   - Generate the YAML in Python.
   - `yaml.safe_load` the result.
   - Assert structural invariants: `train_dataloader.collate_fn` is a sibling of `dataset`, not a child; `eval_spatial_size` equals all five matching size knobs (Resize sizes in both dataloaders, collate base_size, Mosaic output_size — see failure #18); the `num_classes` field is derived from the kwcoco categories table, not hardcoded.
   - Drive every variant × policy combination through the generator.
   - Only when ALL of these pass do you wire the generator to subprocess DEIMv2's train.py.
6. Lift `eligibility_manifest.py` → `orchestration/eligibility.py`. Preserve the four-class state machine (`NOT_READY` / `HOST_PROMISING` / `PHONE_ELIGIBLE` / `PHONE_INELIGIBLE`) and the `candidate_kind=real|smoke` filter. The "phone" terminology should be replaced with something domain-neutral like `DEPLOY_ELIGIBLE` / `HOST_PROMISING` — but keep the same gate semantics.
7. Port `02_sweep.sh` → `orchestration/pareto_sweep.py`.
8. Port `run_round_loop.sh` → `orchestration/round_loop.py`.
9. Build `examples/kwcoco_demo/` with a `mock_tiny` trainer (lifted + renamed from `v4_mock.py`) so CI runs end-to-end on CPU.
10. Port the v4 + v5 pytest tests (drop `test_simplify_status.py` — shitspotter-specific). Add the new tests listed in plan §10.

11. **Maintain the engineering memory subtree** at `/home/joncrall/code/kwcoco_detector_kit/dev/` (already pre-seeded for you):

    ```text
    dev/
    ├── README.md                              ← engineering-memory intro (ported from ambition/dev/)
    ├── benchmark-candidates/
    │   ├── README.md                          ← workflow + quality bar
    │   ├── pipeline-bootstrap-questions.md    ← 4 invariants Q1–Q4 (scrubbed)
    │   └── compositions.md                    ← multi-invariant questions (placeholder)
    └── journals/
        └── lessons_learned.md                 ← 19 seed entries Seed #1–#19 (scrubbed)
    ```

    The subtree was lifted from `/home/joncrall/code/ambition/dev/` and seeded
    with scrubbed entries from `/home/joncrall/code/shitspotter/dev/`. You do
    **not** need to create it — but you **must** maintain it as you work.

    The kit's `dev/` is **agent-readable engineering memory** — postmortems on
    >1hr-debug bugs go in `journals/lessons_learned.md`; distilled hard-problem
    questions go in `benchmark-candidates/`. The discipline is:
    - **Read** these BEFORE tackling a task that resembles a catalogued one.
      The seeded entries describe failure modes the kit *must* preserve
      defenses against — they are the canonical reference for plan §7.
    - **Write** here when YOU cause / watch / fix a mistake whose root cause is
      a transferable invariant. Add the new entry **above** the
      `Seed #19` line so newest-first ordering is preserved.

    Every new lesson learned during the kit's development must land in
    `dev/journals/lessons_learned.md`. Every distilled hard-problem question
    must land in `dev/benchmark-candidates/`. This is non-negotiable — the
    failure-mode list in plan §7 only exists because the prior project
    maintained this discipline.

    Re-read `dev/README.md`, `dev/benchmark-candidates/README.md`, and the
    first ~20 lessons in `dev/journals/lessons_learned.md` *before* writing
    any Phase 1 code. The seeded content is your most concentrated source of
    project-specific invariants.

**Phase 1 acceptance** (also plan §11):

- `pip install -e . && pytest tests/ -q` passes (Python 3.10–3.13) — target ≥ 80 tests.
- `bash examples/kwcoco_demo/run_smoke.sh` produces a `.onnx` + populated eligibility manifest with one `HOST_PROMISING` candidate in <90s on a 1-CPU laptop.
- Each of the 19 failure modes in plan §7 is either caught by a setup-time probe or by a pytest test that would fail under the bug.
- No kit source file contains `poop`, `shitspotter`, `mobile_app_training`, `v9 baseline`, `Pixel 5`, `tpl/poop_models`, or `tpl/Open-GroundingDino`.

Stop and confirm with the user before starting Phase 2 (big-DINO + multi-GPU + sealion) or Phase 3 (webdataset + multispectral + cloud).

## Step 4 — Hard constraints

**Source paths are READ-ONLY**. Do not modify anything under:

- `/home/joncrall/code/shitspotter/`
- `/home/agent/code/kwcoco_dataloader/`
- `/home/joncrall/code/shitspotter/tpl/DEIMv2/` (and other tpl/ submodules)

You harvest patterns and copy code OUT of these directories into the new kit. The kit is a new, separate package.

**Do not bring these into the kit** (plan §13):

- `simplify_kwcoco` (shitspotter-specific cluster-merge convention) — `merge_nearby_anns.py` is OK as an *optional* preprocess.
- Any reference to "v9 baseline = 0.766" or `foundation_detseg_v3` by name in code (lift patterns, drop names).
- Phone-app deploy contract (`PostprocessType.DEIMV2`, KMP+Compose, modelspec.json fields specific to the Kotlin schema).
- Names: `Pixel 5`, `mobile_app_training_v4`, `mobile_app_training_v5`, `foundation_detseg_v3`.
- `v4_mock_tiny` → rename to `mock_tiny`; keep `candidate_kind="smoke"`.
- Any reference to `github.com/Erotemic/shitspotter`.

**Required conventions** (from the prototype, lift verbatim):

- Idempotent `PYTHONPATH` prepend in any env-setup script — sourcing twice must not duplicate paths.
- Per-stage exit-code check in sweep / round drivers — no default `status=ok`.
- `RLIMIT_NOFILE` raised before launching torch trainers; clamp to `ulimit -Hn` (non-root users can't go higher).
- Default `CUDA_VISIBLE_DEVICES=0` for single-host non-cluster setups; warn on PCIe-link-width mismatch.
- DEIMv2 export defaults to ONNX opset 18 (torch ≥ 2.5 incompatible with 17).
- `install_requires` includes `onnxscript`, `onnx`, `onnxruntime`, `onnxsim` (failure #9 + #10).
- Setup-time probe-and-install pulls all of `tpl/DEIMv2/requirements.txt` (failure #11).

## Step 5 — Communication style

The user expects:

- **Direct, short responses.** Use markdown formatting (headers, tables, code blocks).
- **Commit regularly.** Don't batch many changes into one mega-commit; one commit per coherent change with a clear `git log` story. Use `git commit -m "$(cat <<'EOF' ... EOF)"` for multi-paragraph messages with explanatory bodies.
- **End every response that touched code with copy-paste-able rerun commands.** The user has a separate machine with the actual GPU; assume they will run your suggestions from their host shell.
- **Maintain a journal.** Add lessons-learned entries to `docs/lessons.md` as you discover new failure modes. Add benchmark candidates to `tests/candidates/` for invariants worth pinning. This is engineering memory for future contributors.
- **Ask when ambiguous.** Use the `AskUserQuestion` tool (offers structured multiple-choice) for decisions that have multiple defensible answers. Do not invent answers to ambiguous design questions.
- **EMFILE / virtiofs storms**: see the ⚠️ block at the very top of this prompt. Stop and ask for a VM reboot; do not work around.
- **Trust but verify subagent output.** If you spawn subagents, read their actual diffs / tool outputs rather than trusting their summary.
- **Do not amend commits.** Always create new commits (a pre-commit hook failure means the commit didn't happen; --amend would modify the previous commit and lose work).

## Step 6 — Things the prototype got right; preserve them

These are non-obvious but load-bearing patterns from the shitspotter prototype. Don't relitigate them.

1. **Two-axis sweep matrix**: (variant × export_input_hw × train_policy) per cell. Resolution is a first-class axis alongside architecture. The Pareto frontier has 3 dimensions, not 2.

2. **Eligibility manifest is a state machine, not a score function.** A model is `HOST_PROMISING` only after passing host-side gates; `DEPLOY_ELIGIBLE` only after on-device validation. Missing data → `NOT_READY`, not "best guess". `--allow_missing_desktop_bench` is the explicit override.

3. **`candidate_kind=smoke` vs `real`** keeps CI-fixture detectors from accidentally winning real sweeps. The smoke filter is exclusion-by-default.

4. **Round-based hard-negative mining beats online mining.** Each round writes its own workdir, policy.json, mined-negatives kwcoco, score histogram sidecar. You can diff between rounds and reason about whether mining is helping. Don't try to push hard-neg sampling into the dataloader.

5. **Generated config + resolved-effective-config side-by-side.** The trainer writes both the generated YAML *and* the fully-resolved post-`__include__`-expansion view. This catches the "I set eval_spatial_size but did the training pipeline actually use it" bug class.

6. **Per-variant `supports_dynamic_input()` flag.** HGNetv2 encoder pre-bakes positional embeddings at `eval_spatial_size`; multi-scale collate will produce shape-mismatches mid-encoder. DINOv3 encoders interpolate per batch. The kit must respect this: trainer plugin exposes the flag, round-loop coerces `multiscale → fixed` when False.

7. **Auto-shrink batch by input-area ratio.** Per-`(variant, input_hw)` memory table is more maintainable than a flat per-variant default; it auto-handles the OOM cascade at larger input sizes.

8. **Setup-time `--check-env` probe.** Front-loads the install of every transitive dep before any GPU minute is spent. The prototype's `00_setup.sh` does this for ~12 packages across `gdown`, ONNX trio, DEIMv2 deps.

## Step 7 — Acceptance + handoff

When Phase 1 is done:

1. Verify all the §11 acceptance criteria with explicit commands.
2. Write a Phase 1 retrospective: what worked, what was harder than expected, what new failure modes you discovered, what's in `docs/lessons.md`.
3. Tell the user "Phase 1 complete. Ready to confirm Phase 2 (big-DINO + multi-GPU + sealion)?" and wait.

When you have a working Phase 1, the user will give you authorization for Phase 2. Same pattern for Phase 3.

---

End of first-prompt. Begin by reading the seven files in Step 1 in full, then summarise back in ≤ 300 words.
