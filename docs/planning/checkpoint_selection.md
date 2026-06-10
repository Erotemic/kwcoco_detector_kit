# Checkpoint selection & retention — design

Status: **planning** (not yet implemented). This is a spec for an
implementing model to work from. It describes a kit-wide, project-
configurable system for choosing which trained checkpoint(s) to keep
and which one to deploy, with full provenance on every score.

Revision note (2026-06-10, second pass): factored scores into
`(protocol, dataset, subject)`; the probe is now the *same* protocol on
a smaller dataset; selection moved out of the training process into a
detached journal-consuming worker; EMA-always replaces the earlier
raw-before/EMA-after default; retention payloads split slim/full with
leaderboard-anchored resume states.

## Motivation

Today DEIMv2 (`tpl/DEIMv2/engine/solver/det_solver.py`) hand-rolls
checkpoint selection: it tracks a single `best_stat` keyed on COCO
`coco_eval_bbox[0]` (mAP@[0.5:0.95]) over the **validation** split and
writes `best_stg1.pth` (raw model, before `stop_epoch`) / `best_stg2.pth`
(EMA model, after `stop_epoch`).

That metric is computed **whole-image, resized to the train input size**
(`_val_transforms_block` in `kwcoco_detector_kit/trainers/deimv2.py` is
just `Resize([H,W])` — no tiling). For the sea-lion mission this is the
wrong lens: validation images are whole aerial frames (median 5760×3840),
so a ~46 px pup is squashed to ~5 px and disappears. Measured impact:
whole-image pup AP 0.12 vs tiled 0.84 (6.8×). **We are selecting the
deployed checkpoint with the metric that is blind to the binding class.**

Naively tiling the val set is a non-starter: 1060 whole val images expand
to ~59k tiles (full-only, no overlap) or ~289k (train-like overlap +
multiscale) — 56×–272×, adding 6–27 h of eval to a 30-epoch run, every
epoch. So we do **not** tile the whole val set for in-loop eval.

The reframe: **selection-eval and reporting-eval are different jobs.**
Reporting needs full coverage + the exact metric, runs once. Selection
only needs to *rank epochs correctly* in the mission regime — that can be
small, cheap, and constant-cost.

A second motivation is operational. In the last two weeks alone, three
failures traced to eval being tangled into the training/orchestration
critical path: a torch-dynamo ONNX export crash took down post-train
stages; a `1EB all_gather_object` crash at epoch 17 killed a training run
*from inside the in-loop eval*; and the full_8cls stale-eval incident
(2026-06-09) happened because a skip gate trusted file existence over
freshness, scoring an epoch-0 leftover as if it were the trained model.
The architecture below removes eval from the training process entirely
and makes "is this score current?" a keyed lookup, not an mtime guess.

## Core model: every score is (protocol, dataset, subject) → measures

The atomic unit is a **ScoreRecord**, never a bare number. It factors
into three orthogonal parts plus unhashed circumstances:

```
ScoreRecord:
  protocol:      WHICH procedure measured        (the lens)
  dataset:       WHAT it was measured against    (the target)
  subject:       WHO was measured                (the checkpoint)
  measures:      {metric_key: value, ...}        (one pass, many metrics)
  circumstances: {kit_sha, deimv2_sha, host, timestamp, ...}

protocol  = versioned frozen-dataclass registry constant (resolved params included)
dataset   = {role: vali|probe|test, kwcoco_hash | probe_id, n_images}
subject   = {weights_kind: ema|raw, epoch: N, ckpt_hash}
```

### Fingerprint = hash(protocol_id ⊕ dataset_id)

The **fingerprint** is the hash of the resolved protocol and the dataset
binding — *composition of two identities*, not "hash of a context bag
minus excluded fields". It answers: "do these scores live in the same
comparison space?" The subject is the row key within a fingerprint; the
circumstances are stored verbatim for forensics but never hashed.

One eval pass under one fingerprint emits the **whole measures dict**
(`AP@0.5`, `mAP@[.5:.95]`, `ap/<class>`, …). Metrics are addressed as
`(fingerprint, metric_key)` — a protocol does NOT embed a single metric.

### Comparability invariant (kills a class of bugs)

**Leaderboards and rankings are keyed by fingerprint; the system refuses
to compare two ScoreRecords with different fingerprints.** This
structurally prevents the mistakes that have already bitten this project:
ranking a whole-image 0.12 against a tiled 0.84, or placing a
gen004-corpus number beside a gen005-corpus number in one column. The
types don't line up, so it can't happen silently.

### The probe is the same protocol on a smaller dataset

The in-loop tiled signal is **not a separate procedure**. It is the
`true_tiled` protocol — real sliding-window inference with cross-tile
NMS, scored by the kit's real eval — bound to a **small frozen subset of
whole validation images** (~40–60 frames, stratified and pup/rare-positive
enriched, fixed seed, frozen per scheme; `probe_id` = manifest hash).

Cost: ~50 frames × ~56 windows ≈ 3k forward passes per epoch — the same
budget as a pre-cut-tile probe, but with **zero procedural divergence**
from the reported metric (no per-tile/no-NMS caveat to document) and zero
new scoring code (reuses `eval/tiled_predictor.py` verbatim).

Proxy error is then *pure dataset-subsampling error*, visible in the
fingerprint: probe scores and full-val scores differ only in
`dataset_id`, and the invariant already forbids comparing them.

### Content-addressed context store (cheap drill-down)

A score row stores only the short fingerprint hash. The full un-hashed
`(protocol, dataset)` definition lives **once** in a
`fingerprint → definition` registry, deduplicated — thousands of scores
share a handful of fingerprints, so each definition is stored exactly
once (a few KB). Full drill-down is a join from hash back to registry;
it is never reconstructed (reconstruction is where provenance rots).
Cost: negligible.

### `eval_protocol_version` and `code.*_sha`

Full per-commit `kit_sha`/`deimv2_sha` in the comparison key would
invalidate cross-run comparability on every commit. Instead:

- A curated **`eval_protocol_version`** is part of the protocol (hence
  the fingerprint) and bumps **only when something eval-affecting
  changes** (the DEIMv2 patch queue, a regime/metric/probe-construction
  change). Significant-difference judgment is human and deliberate.
- Raw `kit_sha`/`deimv2_sha` live in `circumstances` (un-hashed) for
  forensic drill-down but are **not** part of the comparison key.

## Architecture: the trainer emits a journal; a detached worker decides

The training process has exactly one selection-related responsibility:
**append to a run journal**. Everything else — scoring, leaderboards,
retention, the final re-rank — is a *fold over the journal* performed by
a separate selection worker.

```
trainer (DDP, GPU-heavy, fragile)          selection worker (detached)
  │ per epoch:                               │ tails the journal:
  │  stage last.pth → staging/epoch_N.pth    │  score epoch_N under each in-loop
  │  append journal row (epoch, ckpt_hash,   │    fingerprint (probe true_tiled,
  │    train losses, paths)                  │    whole_640) → append ScoreRecords
  │                                          │  fold leaderboards (derived state,
  │                                          │    recomputable from journal)
  │                                          │  GC staging to the retained union
  └ at end: appends train_complete           └ at end: final re-rank fold
```

Why detached (each justified by a failure we actually had):

- **Crash isolation.** Eval bugs cannot kill a training run (the 1EB
  `all_gather_object` crash was *inside* in-loop eval). The training loop
  becomes pure train-and-emit.
- **No training stall.** DEIMv2's existing in-loop COCO eval pass is
  removed/disabled; the worker computes `whole_640` from the staged
  checkpoint — identical numbers, off the critical path.
- **Staleness is structurally impossible.** Scores are journal rows keyed
  by `ckpt_hash × fingerprint`. "Has this checkpoint been scored under
  this lens?" is a lookup. No mtime guessing — the bug class behind the
  full_8cls incident cannot be expressed.
- **Lag tolerance.** The worker may trail (shared GPU, separate host, or
  catch up after a crash). GC is **fail-retentive**: an epoch may be
  deleted only when it has been scored under *every* configured in-loop
  fingerprint *and* sits on no leaderboard. Worker dies → nothing is
  deleted.
- **Resume for free.** Leaderboards are derived state; any restart
  recomputes them by re-folding the journal. The journal is append-only
  and is itself the provenance record.

The worker needs a GPU for scoring but its load is bursty and small
(batch 1–4, ~1–2 min/epoch vs ~10 min epochs); it can share a training
GPU, use a spare, or run elsewhere and trail.

## The mechanism: leaderboards over union retention

### Leaderboards (top-K per metric)

A **bucket** is `(fingerprint, metric_key, K)` maintaining the top-K
epochs by that metric. A project declares which buckets it tracks — the
only project-facing selection config.

### Union retention + GC

The on-disk keep-set is the **union of all buckets' current members**,
plus `last.pth` (resume, always kept), plus not-yet-scored staged epochs
(fail-retentive rule above).

`best_stg1`/`best_stg2` are deleted as concepts. **Today's behavior is a
strict special case**: one bucket `(whole_640·vali, mAP, K=1)` with
`primary` pointing at it reproduces the current selection exactly.

### Retention payloads: slim by default, full at the anchors

Retained checkpoints serve two distinct purposes with different payload
needs:

- **Selection/deploy** needs `model + EMA` only (**slim**). AdamW moments
  are ≈ 2× model params, so stripping optimizer state roughly halves the
  per-checkpoint cost (~819 MB → ~440 MB for dinov3_x).
- **Recovery** — restarting training from a known-good state when the
  tail of the run has collapsed — needs `model + EMA + optimizer`
  (**full**).

Policy:

- `last.pth` is always full (it is the resume point for normal
  continuation).
- **Resume anchors:** the top-M epochs of the **primary in-loop
  leaderboard** (default M=2) keep the full payload. Rationale: a
  collapse manifests as falling scores, so the leaderboard leader is
  pre-collapse *by construction* — unlike a trailing ring of recent
  epochs, which is post-collapse exactly when you need it. The anchors
  are "the best known-good states", derived from machinery that already
  exists, with no new mechanism.
- **Anchors on the primary board only — decided (2026-06-10).**
  Secondary boards (e.g. whole-image) and per-class boards get slim
  copies; their recovery path is a **warm start** (re-init optimizer from
  the slim weights) — workable, just not an exact resume. Two reasons
  this is enough: (1) the primary board is the mission criterion, so it
  guards collapse in exactly the case we care about; (2) the primary
  metric is class-agnostic AP on the **rare-positive-enriched probe**, so
  a binding-class collapse (e.g. pup) necessarily drags the primary score
  and freezes the anchors pre-collapse — the "per-class collapse
  invisible to the primary board" scenario is largely closed by probe
  construction, not by extra anchors. One board decides recovery and the
  same board decides deployment. A future extension may allow
  `anchors.board` to take a list of buckets; deliberately **not built
  now** — no observed failure motivates it, and warm start covers the
  hypothetical.
- Everything else in the union is slim. The worker strips lazily: when an
  epoch loses anchor status (displaced from top-M by a better epoch), its
  optimizer state is dropped. Scores are immutable, so displacement is
  monotone — a stripped epoch can never need its optimizer state back.

Worst case (fresh collapse-restart): resume from anchor epoch `A` costs
re-training epochs `A+1..N` — acceptable, and strictly better than
resuming from a collapsed `last.pth`.

## The K knob: disk ↔ selection-correctness

Retained size is `|⋃ buckets|`, bounded by `Σ K` (all buckets disjoint)
and in practice far smaller, because a generally-strong epoch wins many
buckets at once. Where buckets **diverge** — a pup-specialist epoch vs an
overall-best epoch — those are exactly the checkpoints worth keeping; the
divergence is the feature.

K is a single dial:

- small K → tiny keep-set, small risk the true-best epoch was mis-ranked
  out of the union by the proxy.
- K = ∞ → save-all → **zero** miss risk.

Multi-bucket union is itself the hedge against proxy error: the true-best
is very likely caught by *some* bucket (whole, probe, or a per-class one)
even if the probe mis-ranks it in another. This is why several buckets
beat one bucket with a bigger K. With slim payloads (above), even large
Σ K stays cheap.

## Provenance threaded end to end

- **Per epoch:** the worker emits ScoreRecords with `(protocol, dataset,
  subject)` attached at emit time; journal rows are
  `fingerprint → {definition_ref, subject, measures}`, never flat dicts.
- **Leaderboards:** top-K per `(fingerprint, metric_key)`; every retained
  checkpoint knows which buckets kept it alive.
- **Final re-rank:** see below; same ScoreRecords, full-val datasets.
- **Deployed artifact** bakes a `selection_provenance` record, e.g.:
  *"epoch 23 (ema), selected as argmax of `true_tiled@640 · vali_full ·
  AP@0.5` = 0.892, re-ranked from a union of 6 candidates retained by
  buckets {true_tiled·probe·AP@0.5·k3, true_tiled·probe·ap/pup·k3,
  whole_640·vali·mAP·k2}; probe_id=…; eval_protocol_version=N."*
- **`projects/*/docs/training_runs.yaml`** records the fingerprint beside
  every metric, so the comparison table is apples-to-apples by
  construction. The registry stops being a place regime mismatches hide.

## Final re-rank is multi-objective (a matrix, not a ranking)

After `train_complete`, the worker scores the retained union under the
**full-validation** bindings of the declared axes — e.g.
`true_tiled · vali_full` (real sliding-window over all 1060 frames) and
`whole_640 · vali_full` — a handful of checkpoints, never 30× per-epoch.

**The re-rank dataset is the full validation split, never test.** The
re-rank *selects*; test is reserved for reporting the already-selected
model. (The existing rescore harness scores test — that is the reporting
step, downstream of selection.)

Output is a **`candidate × (fingerprint, metric)` matrix**. Selection
over the matrix is multi-objective:

- **`argmax`** on a declared primary axis — the default automation.
- **`aggregate`** — a composite score. A derived metric (e.g.
  `combined_v1 = hmean(true_tiled·AP@0.5, whole_640·AP@0.5)`) is a
  **named, versioned registry entry** whose provenance embeds the
  fingerprints of its inputs — as auditable as a primitive.
- **`pareto`** — the non-dominated frontier across the chosen axes;
  auto-pick by tiebreak (closest-to-ideal / max min-normalized score) but
  **surface the whole frontier**.

Whatever is auto-picked, **the full matrix and the Pareto frontier are
persisted** as durable artifacts, so a jack-of-all-trades checkpoint can
be chosen — by tiebreak or manually — with **zero recompute**, later.

## Protocol registry: versioned frozen dataclasses in kit code (decided)

```python
# kwcoco_detector_kit/eval/protocols.py
TRUE_TILED_V1 = EvalProtocol(
    name="true_tiled", version=1,
    regime=SlidingWindow(win=Param("train_input_hw"), overlap=0.5, nms_iou=0.5),
    class_filter=ClassFilter(class_agnostic=True, exclude_distractors=True),
    score_thresh=0.001,
)
WHOLE_RESIZE_V1 = EvalProtocol(
    name="whole_resize", version=1,
    regime=Resize(Param("train_input_hw")),
    class_filter=ClassFilter(class_agnostic=False),
)
```

- Protocols are **parameterized families**; the *resolved* params (e.g.
  `win=640` vs `win=1280`) are hashed into the fingerprint, so tiled@640
  and tiled@1280 are correctly incomparable.
- The class-filter holds the **rule** (`exclude_distractors`), never the
  **list** — the distractor list (`northern_fur_seal`) belongs to the
  dataset/scheme side (`class_schemes.yaml` `distractor_classes`). Kit
  code never names a project's species.
- Protocols do **not** embed a metric; one pass emits the full measures
  dict, addressed by `(fingerprint, metric_key)`.
- Type-checked, importable, git-versioned. Projects reference protocols
  **by name**; adding/changing one is a reviewed code change —
  appropriate, because it changes what "comparable" means. (Validated
  project-declared protocols: considered, deferred.)

## Config schema (per project)

```yaml
checkpoint_selection:
  inloop:                      # what the worker scores every epoch
    - { protocol: true_tiled, dataset: probe }
    - { protocol: whole_resize, dataset: vali }
  probe:
    source: vali
    frames: 50                 # whole images, sliding-window scored
    stratify: auto             # class-stratified, rare-positive enriched
    seed: 0                    # frozen per scheme -> probe_id
  buckets:
    - { protocol: true_tiled,   dataset: probe, metric: AP@0.5, k: 3 }
    - { protocol: true_tiled,   dataset: probe, metric: ap/pup, k: 3 }
    - { protocol: whole_resize, dataset: vali,  metric: mAP,    k: 2 }
  retention:
    anchors: { board: primary_inloop, top_m: 2 }   # full payload (+optimizer)
    payload: slim                                   # default for the rest
  rerank:
    axes:
      - { protocol: true_tiled,   dataset: vali_full, metric: AP@0.5 }
      - { protocol: whole_resize, dataset: vali_full, metric: mAP }
      - { derived: combined_v1 }
    policy: argmax             # argmax | aggregate | pareto
    primary: { protocol: true_tiled, dataset: vali_full, metric: AP@0.5 }
  min_epoch_frac: 0.1          # leaderboards inert for the first 10% of epochs
```

A whole-image project sets `inloop: [{whole_resize, vali}]`, one bucket
`(whole_resize·vali, mAP, k:1)`, primary on the same axis → exactly
today's behavior.

## Decided defaults (2026-06-10)

All overridable per project; chosen to be correct out of the box for the
common case and *visible* (materialized into the run's resolved config and
logged), never magic.

### Val-regime default — project-type-derived knob

The val-regime is a per-project knob whose **default value is derived from
project type**, always written into the resolved config and logged:

- trains on a **tile cache** → in-loop `{true_tiled·probe, whole_resize·vali}`,
  primary `true_tiled·vali_full·AP@0.5`
- trains on **whole images** → in-loop `{whole_resize·vali}`,
  primary `whole_resize·vali_full·mAP`

Rationale: "off by default" silently mis-serves every tiled project until
someone flips it (we already got burned selecting on the wrong regime);
pure auto-magic is non-inspectable. Explicit-knob-with-project-type-default
gives correct out-of-box behavior *and* a value you can see and override.
"Am I tiling my training data?" is a near-perfect proxy for "do I care
about tiled performance?"

### Final re-rank policy default — `argmax` on `primary`, frontier persisted

The auto-deployed pick defaults to `policy: argmax` on the declared
`primary` axis — deterministic, reproducible, no hidden multi-objective
weighting baked into the deploy choice. The full matrix **and** the
Pareto frontier are always computed and persisted regardless of policy,
so the jack-of-all-trades pick is a deliberate, **zero-recompute**
follow-up (tiebreak or manual) at any later time. `pareto`/`aggregate`
remain opt-in for runs that want the generalist auto-selected.

### Weights kind — EMA, always (revised 2026-06-10)

Selection scores **EMA weights at every epoch** (the deployable artifact
is always EMA). This replaces the earlier raw-before/EMA-after default,
which mirrored DEIMv2's stg1/stg2 quirk and broke the comparability
invariant at the `stop_epoch` boundary (raw-epoch-3 and EMA-epoch-20
either blur one fingerprint or silently split every bucket in two).
`min_epoch_frac` already guards EMA warm-up noise. Raw weights remain in
`last.pth`/anchors for resume; a project that wants raw-weight selection
declares it as a distinct subject kind = distinct comparison space,
explicitly. The staged checkpoint carries the full state dict; *the
worker* extracts EMA — the trainer never decides.

### Mechanics defaults

- **Rare-class buckets:** auto-disable any per-class bucket whose class
  support (in its bound dataset) is below **50 instances**; log the
  disable. Keeps noisy rare-class leaderboards (e.g. `ap/dead_nonpup`, 49
  corpus-wide) from thrashing the retained union.
- **Probe:** **frozen per scheme** (a rebuild = a new `probe_id` = a new
  fingerprint), default **50 whole vali frames**, class-stratified with
  rare-positive enrichment, **fixed seed**.
- **Retention payloads:** slim (`model+EMA`) by default; **top-2 of the
  primary in-loop leaderboard keep full payload** (+optimizer) as resume
  anchors; `last.pth` always full.
- **Combined metric:** `combined_v1 = harmonic_mean(true_tiled·AP@0.5,
  whole_resize·AP@0.5)` — harmonic mean punishes weakness on *either*
  axis, which is the generalist signal. Registry-defined, **opt-in** via
  `rerank.axes`.
- **Worker placement:** same node, sharing a training GPU (eval is bursty
  and small) — may trail; GC is fail-retentive under lag.

## Phase 2: the journal as the run's single source of truth

Once the journal exists, the sweep's ad-hoc state probes become views
over it: the `.train_complete` marker, the eval-skip gates (the
file-existence check behind the full_8cls stale-eval incident),
`manifest.tsv`, and the numbers pushed to `training_runs.yaml` are all
derivable by folding journal rows. That convergence — one append-only
record of what happened, everything else derived — is the long-term
payoff, but it is **explicitly out of scope for the first
implementation**; checkpoint selection must not block on it.

## Open points for the implementer

1. **Journal format & locking.** Append-only JSONL per run is the obvious
   choice; define the row schema (event kinds: `epoch_staged`,
   `score_record`, `gc`, `train_complete`, `rerank_result`) and the
   single-writer rule (trainer appends train events; worker appends
   score/GC events; separate files or a partitioned single file).
2. **Rare-class noise — beyond the floor.** Default is auto-disable below
   50 support. Decide whether a displacement margin / metric smoothing is
   *also* wanted for classes just above the floor that still wobble.
3. **Probe builder.** Offline builder producing the frozen 50-frame
   manifest (`probe_id`); decide stratification targets (rare-positive
   enrichment ratios, empty-frame ratio) and emit a human-readable
   manifest beside the hash.
4. **Worker lifecycle.** How the sweep launches/supervises the worker
   (same slurm job, sibling step, or trailing process), and how
   `train_complete` hands off to the re-rank fold.
5. **Derived-metric provenance.** Implement `combined_v1`; its definition
   embeds the input fingerprints so an aggregate score is auditable.
6. **DEIMv2 emit patch.** Disable in-loop COCO eval (or make optional),
   add the stage-copy + journal-append at epoch end; keep the patch
   minimal and upstreamable (see `[[reference-deimv2-upstream-patches]]`).

## Related project context

- `[[detection-ap-is-selection-criterion]]` — class-agnostic, NFS-excluded
  AP@0.5 is the mission metric; per-class AP is diagnostic.
- `[[project-gen005-small-object-floor]]` — the whole-vs-tiled pup gap and
  why selection regime matters.
- `[[project-lastpth-is-epoch-zero]]` — DEIMv2 staging / `stop_epoch`
  quirks that interact with checkpoint writing.
- `[[feedback-kwcoco-bakes-absolute-paths]]` — dataset hashing must handle
  the absolute-path baking issue when computing `dataset.kwcoco_hash`.
- `[[reference-deimv2-upstream-patches]]` — the emit patch joins the
  existing DEIMv2 patch queue.
