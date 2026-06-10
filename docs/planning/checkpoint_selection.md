# Checkpoint selection & retention — design

Status: **planning** (not yet implemented). This is a spec for an
implementing model to work from. It describes a kit-wide, project-
configurable system for choosing which trained checkpoint(s) to keep
and which one to deploy, with full provenance on every score.

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

## The one mechanism

Three parts, nothing more.

### 1. Evaluators (per epoch, constant-cost)

Each evaluator emits a set of `ScoreRecord`s. Two of them:

- **`whole_image`** — the resized-whole-val COCO eval DEIMv2 already runs.
  Nearly free; reuse its numbers.
- **`tile_probe`** — a *fixed, small, stratified* tiled probe (~1.5k
  tiles), scored with the kit's real eval (`kwcoco_detector_kit/eval/`)
  so it can emit the actual mission metric (class-agnostic, NFS-excluded
  AP@0.5) in the tiled regime. Constant cost regardless of corpus size.

The probe is a *proxy*: per-tile, no cross-tile NMS, subsampled. It is
good enough to rank epochs; it is not the reported number.

### 2. Leaderboards (top-K per metric)

A **bucket** is `(fingerprint, metric_key, K)` maintaining the top-K
epochs by that metric. Metric keys are whatever the evaluators emit, e.g.
`AP@0.5`, `mAP@[.5:.95]`, `ap/pup`. A project declares which buckets it
tracks — the only project-facing selection config.

### 3. Union retention + GC

The on-disk keep-set is the **union of all buckets' current members**.
Each epoch: stage `last.pth → epoch_N.pth`, update every leaderboard,
delete any staged epoch that is in **no** leaderboard. `last.pth` (for
resume) is always kept, independent of leaderboards.

`best_stg1`/`best_stg2` become two EMA-axis variants of a metric, not two
special files. **Today's behavior is a strict special case**: one bucket
`(whole_640, mAP, K=1)` with `primary = whole_640·mAP` reproduces it
exactly.

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
beat one bucket with a bigger K.

## Provenance: the atomic unit is a ScoreRecord, not a number

Every score the system touches is `(value, context)`. `context` is a
structured, hashable descriptor:

```
context = {
  eval_protocol_version: int          # bumps only on eval-AFFECTING change
  evaluator:    whole_image | tile_probe | true_tiled
  dataset:      { split: vali|test, kwcoco_hash, n_images }
  probe_id:     <frozen probe manifest hash>   # null unless tile_probe
  regime:       resize@640 | sliding_window(win=640, ov=0.5)+nms(iou=0.5)
  metric:       AP@0.5 | mAP@[.5:.95] | ap/<class>
  class_filter: { class_agnostic: bool, nfs_excluded: bool, distractors: [...] }
  weights:      { kind: raw|ema, epoch: N, ckpt_hash }
  code:         { kit_sha, deimv2_sha }          # stored, NOT in the key
  score_thresh, nms_thresh, ...
}
```

### Fingerprint = hash of the comparison-relevant context

The **fingerprint** is the hash of `context` *excluding* `weights.epoch`,
`weights.ckpt_hash` (these vary per checkpoint) and *excluding* raw
`code.*_sha` (see below). The fingerprint answers: "do these scores live
in the same comparison space?"

### Comparability invariant (kills a class of bugs)

**A leaderboard is keyed by fingerprint; the system refuses to compare two
ScoreRecords with different fingerprints.** This structurally prevents the
mistakes that have already bitten this project: ranking a whole-image 0.12
against a tiled 0.84, or placing a gen004-corpus number beside a
gen005-corpus number in one column. The types don't line up, so it can't
happen silently.

### Content-addressed context store (cheap drill-down)

A score row stores only the short fingerprint hash. The full (un-hashed)
context lives **once** in a `fingerprint → context` registry,
deduplicated — thousands of scores share a handful of fingerprints, so
each context object is stored exactly once (a few KB). Full drill-down is
a join from hash back to registry; it is never reconstructed (reconstruction
is where provenance rots). Cost: negligible.

### `eval_protocol_version` and `code.*_sha`

Full per-commit `kit_sha`/`deimv2_sha` in the comparison key would
invalidate cross-run comparability on every commit. Instead:

- A curated **`eval_protocol_version`** is part of the fingerprint and
  bumps **only when something eval-affecting changes** (the DEIMv2 patch
  queue, a regime/metric/probe-construction change). Significant-difference
  judgment is human and deliberate.
- Raw `kit_sha`/`deimv2_sha` are **stored in the context (un-hashed)** for
  forensic drill-down but are **not** part of the comparison key.

## Provenance threaded end to end

- **Per epoch:** evaluators emit ScoreRecords *with context attached at
  emit time*. The metrics row is `fingerprint → {context, {metric: value}}`,
  not a flat dict.
- **Leaderboards:** top-K per `(fingerprint, metric)`; every retained
  checkpoint knows which fingerprints kept it alive.
- **Final re-rank (always runs):** see below.
- **Deployed artifact** bakes a `selection_provenance` record, e.g.:
  *"epoch 23, selected as argmax of `true_tiled · AP@0.5 · class_agnostic
  · nfs_excluded` = 0.892, re-ranked from a union of 6 candidates retained
  by buckets {probe/AP@0.5·k3, probe/ap/pup·k3, whole/mAP·k2}; in-loop
  proxy = probe `b9540ace…`; eval_protocol_version=N."*
- **`projects/*/docs/training_runs.yaml`** records the fingerprint beside
  every metric, so the comparison table is apples-to-apples by
  construction. The registry stops being a place regime mismatches hide.

## Final re-rank is multi-objective (a matrix, not a ranking)

The retained union is the input to the authoritative, post-training step.
It runs the **real** evals (full sliding-window + NMS for `true_tiled`,
plus `whole_640`, plus any derived "combined" axes) on the handful of
retained checkpoints — never 30× per-epoch. Output is a
**`candidate × fingerprint-metric` matrix**.

A run may legitimately care about several axes at once (exceptional-on-
tiled vs exceptional-on-whole-image vs a generalist). So selection over
the matrix is multi-objective:

- **`argmax`** on a declared primary `(fingerprint, metric)` — the default
  automation.
- **`aggregate`** — a composite score (e.g. weighted mean, or a derived
  `combined` metric). "Combined" is itself a **named, versioned derived
  metric** carrying its own provenance (e.g. `combined_v1 =
  hmean(true_tiled·AP@0.5, whole_640·AP@0.5)`), so an aggregate score is
  as auditable as a primitive one.
- **`pareto`** — compute the non-dominated frontier across the chosen
  axes; auto-pick by a tiebreak (e.g. closest-to-ideal-point / max
  min-normalized score) but **surface the whole frontier**.

Whatever is auto-picked for deployment, **the full matrix and the Pareto
frontier are persisted** as a durable artifact, so a jack-of-all-trades
checkpoint can be chosen — by tiebreak or manually — with **zero
recompute**, at any later time.

## Where canonical protocols live: code registry (decided)

Canonical eval protocols / fingerprints are **versioned frozen-dataclass
constants in a kit module** (e.g. `kwcoco_detector_kit/eval/protocols.py`):

```python
TRUE_TILED_V1 = EvalProtocol(
    name="true_tiled", version=1,
    regime=SlidingWindow(win=640, ov=0.5, nms_iou=0.5),
    metric="AP@0.5", class_agnostic=True, nfs_excluded=True,
)
WHOLE_640_V1 = EvalProtocol(
    name="whole_640", version=1,
    regime=Resize(640), metric="mAP@[.5:.95]",
)
```

Type-checked, importable, single source of truth, git-versioned. Projects
reference a protocol **by name**, never by re-spelling a context dict
(which would drift). Adding/changing a protocol is a reviewed code change
— appropriate, because it changes what "comparable" means. (A future
hybrid allowing validated project-declared protocols was considered and
deferred.)

## Config schema (per project)

```yaml
checkpoint_selection:
  evaluators: [whole_image, tile_probe]
  tile_probe:
    source: vali
    budget: 1500
    stratify: auto            # class-stratified, pup-enriched, realistic empty ratio
  buckets:
    - { protocol: true_tiled, metric: AP@0.5, k: 3 }   # protocol -> fingerprint
    - { protocol: true_tiled, metric: ap/pup, k: 3 }
    - { protocol: whole_640,  metric: mAP,    k: 2 }
  rerank:
    protocols: [true_tiled, whole_640, combined_v1]    # axes of the final matrix
    policy: pareto                                      # argmax | aggregate | pareto
    primary: { protocol: true_tiled, metric: AP@0.5 }  # default deploy pick / tiebreak
  min_epoch: 3                # leaderboards inert before this (kills early noise)
```

A whole-image project sets `evaluators: [whole_image]`, one bucket
`(whole_640, mAP, k:1)`, `rerank.primary: whole_640·mAP` → exactly
today's behavior.

## Decided defaults (2026-06-10)

All overridable per project; chosen to be correct out of the box for the
common case and *visible* (materialized into the run's resolved config and
logged), never magic.

### Val-regime default — project-type-derived knob

The val-regime is a per-project knob whose **default value is derived from
project type**, always written into the resolved config and logged:

- trains on a **tile cache** → `evaluators: [whole_image, tile_probe]`,
  `primary: true_tiled·AP@0.5`
- trains on **whole images** → `evaluators: [whole_image]`,
  `primary: whole_640·mAP`

Rationale: "off by default" silently mis-serves every tiled project until
someone flips it (we already got burned selecting on the wrong regime);
pure auto-magic is non-inspectable. Explicit-knob-with-project-type-default
gives correct out-of-box behavior *and* a value you can see and override.
"Am I tiling my training data?" is a near-perfect proxy for "do I care
about tiled performance?"

### Final re-rank policy default — `argmax` on `primary`, frontier persisted

The auto-deployed pick defaults to `policy: argmax` on the declared
`primary` `(protocol, metric)` — deterministic, reproducible, no hidden
multi-objective weighting baked into the deploy choice. The full
`candidate × fingerprint-metric` matrix **and** the Pareto frontier are
always computed and persisted regardless of policy, so the
jack-of-all-trades pick is a deliberate, **zero-recompute** follow-up
(tiebreak or manual) at any later time. `pareto`/`aggregate` remain opt-in
for runs that want the generalist auto-selected.

### Mechanics defaults

- **Rare-class buckets:** auto-disable any per-class bucket whose class
  support (in the probe/val) is below **50 instances**; `log()` the
  disable. Keeps noisy rare-class leaderboards (e.g. `ap/dead_nonpup`, 49
  corpus-wide) from thrashing the retained union.
- **Probe:** **frozen per scheme** (a rebuild = a new `probe_id` = a new
  fingerprint), budget **1500** tiles, class-stratified with pup/rare-
  positive enrichment + a realistic empty ratio, **fixed seed**.
- **EMA axis:** evaluate **EMA weights past `stop_epoch`, raw before**
  (mirrors today's stg1/stg2); deployable = EMA.
- **Combined metric:** `combined_v1 = harmonic_mean(true_tiled·AP@0.5,
  whole_640·AP@0.5)` — harmonic mean punishes being weak on *either* axis,
  which is precisely the generalist signal. Defined in the registry but
  **opt-in** via `rerank.protocols`.
- **Checkpoint I/O:** stage-copy `last.pth → epoch_N.pth` and GC run
  **off the training critical path** (background), never blocking a step.

## Where it lives: DEIMv2 emits, kit decides

DEIMv2 **emits** (write `epoch_N.pth` to staging + a metrics row); the
**kit** owns the retention manager (consumes metrics, runs leaderboards +
GC, owns the probe and the protocol registry). This keeps policy in the
kit where projects configure it and leaves DEIMv2 a dumb, replaceable
emitter — consistent with "image is the repro unit, kit owns
orchestration." It replaces DEIMv2's hand-rolled `best_stat`/stg1/stg2.

Mechanism sketch: DEIMv2 already writes `last.pth` each epoch. A kit hook
at epoch end copies `last.pth → staging/epoch_N.pth` (~819 MB/epoch I/O,
fine on the data drive), runs the probe eval + records whole-image
numbers, updates leaderboards, GCs staging down to the retained union.

## Open points for the implementer

The policy defaults are decided (see "Decided defaults" above); these are
the *implementation* questions those defaults leave open:

1. **Rare-class noise — beyond the floor.** Default is auto-disable below
   50 support. Decide whether a displacement margin / metric smoothing is
   *also* wanted for classes just above the floor that still wobble.
2. **Probe builder.** Implement the offline builder producing a frozen,
   hashed (`probe_id`) probe at budget 1500 with the stratify/enrichment
   default. Decide the exact stratify target ratios and how "empty" is
   defined per scheme.
3. **EMA axis plumbing.** Wire EMA-past-`stop_epoch` / raw-before through
   the evaluator so each ScoreRecord's `weights.kind` is correct.
4. **`combined` metric composition.** Implement `combined_v1` (harmonic
   mean) in the registry and define how its provenance composes from its
   input fingerprints (it references two protocols, so its context must
   embed both).
5. **Async checkpoint I/O.** Implement the off-critical-path stage-copy +
   GC; confirm it doesn't stall the training step under the data-drive's
   I/O.

## Related project context

- `[[detection-ap-is-selection-criterion]]` — class-agnostic, NFS-excluded
  AP@0.5 is the mission metric; per-class AP is diagnostic.
- `[[project-gen005-small-object-floor]]` — the whole-vs-tiled pup gap and
  why selection regime matters.
- `[[project-lastpth-is-epoch-zero]]` — DEIMv2 staging / `stop_epoch`
  quirks that interact with checkpoint writing.
- `[[feedback-kwcoco-bakes-absolute-paths]]` — dataset hashing must handle
  the absolute-path baking issue when computing `dataset.kwcoco_hash`.
