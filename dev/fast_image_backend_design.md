# Fast Image Backend — design

Author: agent
Date: 2026-05-22
Status: draft — for discussion, not yet implemented

## Problem

The current DEIMv2 training pipeline reads tile JPEGs from a kwcoco
bundle via the standard CocoDetection → PIL/cv2 path. Concretely:

- `data/tile.py` produces O(100k) JPEG files per scheme in a single
  directory (e.g. `tiles.kwcoco.zip`'s associated asset dir).
- Each iter, DataLoader workers `open()` N random tiles, `read()` them
  from the filesystem, JPEG-decode, augment, and stack into a batch.
- Symptoms on the current pup_vs_nonpup baselines:
  - Iter time varies 0.84s → 3.38s as the multi-scale cycle changes
    tile size — most of that delta is bytes/tile, i.e. I/O.
  - GPU utilization sits around ~10% for the 1-GPU hgnetv2_n run.
  - `delayed_image` was warning "may not be efficient without gdal"
    (now fixed via `kwcoco finish_install --with_gdal` in the docker
    image — but the underlying many-small-files pattern remains the
    deeper bottleneck).

**Storage reality: arisia's training filesystem is spinning rust
(HDD), not SSD.** This dominates the design space:

- Random seek ~10 ms vs ~0.1 ms on SSD → 100× per-seek penalty.
- Sequential throughput ~100–200 MB/s is still respectable when the
  workload actually reads contiguous bytes.
- Page cache is load-bearing — 250k tiles × ~150 KB ≈ 37 GB; with
  RAM ≥ that, epoch-2+ get cheaper, but cold-start (epoch 1, post-
  reboot, cross-job) is brutal.
- Random-access-per-sample formats (LMDB, packed memmap) inherit
  the same HDD seek penalty and lose much of their theoretical
  advantage.

We want to move from many-small-files-random-read to a
sequential-IO-friendly format without rewriting the trainer plugins.

## Goals

1. **Throughput**: at least 3× iter-rate improvement for the
   pup_vs_nonpup hgnetv2_n baseline. Practically: get the GPU above
   50% utilization.
2. **Drop-in**: trainer plugins (DEIMv2, future OpenGroundingDINO,
   future YOLOX) consume the new backend through a stable interface;
   no plugin-specific glue per backend.
3. **Reversibility**: training against the old kwcoco-JPEG path stays
   supported. The fast backend is an opt-in optimization, not a fork.
4. **Reproducibility**: the conversion step is deterministic. Two
   builds from the same kwcoco bundle + same params produce
   byte-identical (or at least sample-identical) shards.
5. **Upstreamable**: the reader implementation should be portable into
   `kwcoco_dataloader` later. Live in the kit during incubation;
   migrate once the interface is settled and tested.

## Additional capabilities required

### 1. On-the-fly class coarsening (scheme collapse at load time)

The kit's universal-tile pipeline (commit 5d99545) already tiles
against the upstream 'sealion' bundle with `source_category`
preserved on each annotation. The scheme's class-collapse currently
happens via a separate post-step (`apply_scheme_to_kwcoco.py`) that
writes a per-scheme kwcoco bundle.

For the fast backend, the natural place to apply the collapse is
**at load time** — the Sample carries `source_category` from the
shard; the reader maps it to the run's target_classes per the scheme
yaml. Benefits:

- One tar shard set serves *every* scheme. No re-pack per scheme.
- Changing schemes is a runtime swap of the mapping table.
- The on-disk format never lies about the source data — collapse is
  purely a view.

Implication for `Sample`: `category_id` is the *raw* class (matching
the tile bundle's `source_category`); the scheme mapping happens in
the trainer-side glue, not in the backend.

### 2. Dynamic class-balanced sampling (HDD-aware)

We want to vary class-mix ratios at runtime (up-weight rare classes,
50/50 binary, etc.) — analogous to kwcoco_dataloader's
`BalancedSampleTree` but adapted for tar-shard sequential reads.
On HDD this can't be per-sample random; the seek penalty would
dominate.

Three candidate strategies, in tension between flexibility and HDD-
friendliness:

**A. Class-mixed shards + intra-shard shuffle buffer (max-sequential):**
Pack shards so each contains the global class mix. Read sequentially
through a shard; maintain an in-RAM shuffle buffer (~thousands of
samples) for intra-batch variety. Ratios are fixed at pack time. Best
HDD throughput; no runtime ratio control.

**B. Class-segregated shards + weighted shard scheduler (max-flexible):**
Pack one shard set per class (or per coarse group). A scheduler picks
which shard to advance next based on the target ratio, reading a
chunk of K samples before re-deciding. Cross-shard seeks happen at
chunk boundaries; K amortizes the HDD penalty. K is the responsiveness
knob — too small → seek storm; too large → ratio drifts within a
batch. K ≈ 256–1024 likely sweet spot.

Risk: "dovetailing reads from N shards" sounds like an HDD anti-
pattern. In practice modern HDDs have prefetchers + the kernel page
cache helps, so reading a contiguous K samples from one tar before
moving to another is fine. Reading 1 sample then switching is fatal.

**C. Class-mixed shards + reject-sample (lazy filter):**
Read sequentially from class-mixed shards, reject samples that
exceed the target ratio. Zero seek penalty, but wastes IO on
rejected samples — bad for rare-class up-weighting (most reads get
dropped). Works for moderate ratio shifts (±2× from natural).

**Recommendation: B for the main path, with A as the fallback for
"don't care about runtime balance" runs.** A and B can share the same
shard format if shards are packed class-segregated by default;
strategy A is then just "scheduler uses uniform random shard pick."

Open: kwcoco_dataloader's BalancedSampleTree is a tree-weighted
index sampler. The shard scheduler is its natural shard-granularity
analog. Lift the tree-weighting math; rewrite the leaf semantics
from "pick sample index i" → "advance shard s by K samples."

### 3. Resolution jitter via oversized tiles + crop-on-load

`data/tile.py` already supports `oversize_factor>1.0` — cuts larger
tiles than the model input and records `tile_model_input_size` in
the kwcoco image record. Today this only matters when the trainer's
own augment pipeline does the final crop (we currently set
`oversize_factor=1.2`).

For the fast backend, the contract is:

- Shards store the oversized tile bytes (matching the on-disk kwcoco).
- The reader returns a Sample where `image` is the oversized array.
- The trainer-side glue does a `RandomCrop(model_input_hw)` during
  train and `CenterCrop` during eval, BEFORE DEIMv2's transform
  pipeline sees the sample. (Or threads through DEIMv2's transform
  ops as another step.)
- Annotation bboxes get clipped against the crop window — the same
  `_clip_bbox_xywh` logic tile.py already uses.

Tradeoff: ~44% more bytes/sample on disk and per-iter (1.2² = 1.44).
Worth it for the augmentation diversity + eliminates the tile-time
hard commitment to one input resolution. The fast backend should
make the byte penalty manageable.

## Non-goals (for v1)

- Cloud-mount / streaming-from-S3. Local disk only.
- Multispectral / multi-band reads. RGB jpeg first.
- Dynamic sample weighting / balanced sampling. Static shuffle within
  shards.
- GPU decode (DALI). Orthogonal; revisit if CPU decode becomes the
  next bottleneck.

## Survey

| Format | Throughput | Random | Compr | Notes |
|---|---|---|---|---|
| WebDataset (tar shards) | high | shard-level | jpeg-in-tar | Iter-first model. DDP works but needs shard-per-rank. The kwcoco_dataloader writer (1036 lines, real impl) already does this. |
| MosaicML Streaming | high | per-sample | yes | WebDataset++ with random access. Adds a dep. |
| LMDB | high | per-sample | per-sample (we'd jpeg-encode in) | Simple, single-file, no shard math. Good for eval where iter order matters. |
| Packed memmap | very high | per-sample | none (3× disk) | Disk-blows tile bundle. Disqualified for our scale. |
| HDF5 | moderate-high | per-sample | yes | Heavier than LMDB for not much gain. |
| Arrow/Parquet | high columnar | per-row | yes | Great for tabular metadata, awkward for image bytes. |
| NVIDIA DALI | very high | depends | n/a | Replaces the whole augment pipeline. Invasive. |

For v1 (HDD-aware): **WebDataset tar shards for both train AND eval**.
LMDB's per-sample random access is poisoned by HDD seek latency; eval
just needs a no-shuffle iteration order, which WebDataset can do via
single-shard, no-shuffle reads.

If we later move to SSD-backed storage, LMDB / MosaicML Streaming
re-enter consideration. The reader interface is designed so the
backend is swappable without touching trainer plugins.

## Interface

A single `FastImageBackend` protocol the kit's trainer plugins consume.
Lives in `kwcoco_detector_kit/data/fast_backend.py` during incubation;
migrates to `kwcoco_dataloader/readers/` once stable.

```python
# kwcoco_detector_kit/data/fast_backend.py

from dataclasses import dataclass
from typing import Protocol, Iterator, Iterable
import numpy as np

@dataclass
class Sample:
    """One training sample. Mirrors what DEIMv2's CocoDetection
    constructs in __getitem__ minus the model-specific transforms.
    """
    image_id: int                # stable across epochs; matches kwcoco img id
    image: np.ndarray            # H×W×C uint8 RGB
    annotations: list[dict]      # kwcoco-style: bbox, category_id, source_category, ...
    meta: dict                   # tile_source_gid, tile_role, tile_resize_scale, ...


class FastImageBackend(Protocol):
    """A read-only handle on a converted tile bundle.

    Implementations decide whether iteration is sequential (WebDataset
    tar) or random-access (LMDB, memmap). Both produce the same Sample
    shape so trainer plugins don't care which is underneath.
    """

    # --- catalog ---
    def __len__(self) -> int: ...
    @property
    def categories(self) -> list[dict]: ...  # [{id, name, supercategory}, ...]
    @property
    def supports_random_access(self) -> bool: ...

    # --- access ---
    def __iter__(self) -> Iterator[Sample]:
        """Sequential pass. Iter-order may be implementation-defined
        (e.g. tar order). Trainer-side shuffle happens via shard
        shuffling + intra-shard shuffle buffer."""
        ...
    def __getitem__(self, idx: int) -> Sample:
        """Random access by index. Raises NotImplementedError on
        iter-only backends (caller should check supports_random_access)."""
        ...

    # --- DDP plumbing ---
    def shard_for(self, rank: int, world_size: int,
                  worker_id: int = 0, num_workers: int = 1) -> "FastImageBackend":
        """Return a backend handle restricted to one (rank, worker)
        slice. For WebDataset this picks a subset of tar shards; for
        LMDB it's a strided iteration."""
        ...
```

Sample → DEIMv2 glue lives in the trainer plugin as a thin Dataset
wrapper. Same shape regardless of backend.

## Writer

Lives in `kwcoco_dataloader` (already does). We may need to:

1. **Verify it preserves `source_category`** and our other passthrough
   fields — required for the universal-tile pipeline (see kit commit
   5d99545). If it doesn't, we patch upstream.
2. **Verify shard size targets are sensible** (~1 GB shards, 1k–10k
   samples per shard).
3. **Add an LMDB writer** alongside the WebDataset one, sharing the
   kwcoco → sample-iter front-end.
4. **Standardize the manifest sidecar** (count, schema, sha of the
   source kwcoco) so the reader can fail-loud on schema drift.

The user has authorized "change the contracts when they are already
broken" — so we don't need to be polite if the existing writer's API
is awkward.

## Reader (kit-side, v1)

`kwcoco_detector_kit/data/fast_backend.py` exposes:

- `class WebDatasetBackend(FastImageBackend)` — wraps
  `webdataset.WebDataset(shard_glob).decode("rgb").to_tuple(...)`.
  Handles DDP sharding via `shard_for()`.
- `class LMDBBackend(FastImageBackend)` — wraps `lmdb.open(...)` with a
  cursor; supports random access.

Both expose the same `Sample` shape.

## Trainer plugin glue

`kwcoco_detector_kit/trainers/deimv2.py` gains a small Dataset wrapper
that adapts a `FastImageBackend` to whatever the DEIMv2 internals
expect. Critically:

- DEIMv2's pipeline starts from `CocoDetection` which loads images via
  PIL. We override `_load_image(image_id)` to call into the backend.
- The annotation list lives on the Sample, not on the
  filesystem-side kwcoco; we synthesize the per-iter target dict from
  it directly.
- DDP: PyTorch's DataLoader's `worker_init_fn` calls
  `backend.shard_for(rank, world_size, worker_id, num_workers)`.

## Integration plan

Phase 1 — **WebDataset writer round-trip** (no trainer changes yet):

1. Run `kwcoco_dataloader build_webdataset` on the existing universal
   tile bundle.
2. Verify the round-trip: convert kwcoco → WebDataset → read back via
   `WebDatasetBackend.__iter__`. Confirm `source_category` survives,
   counts match, image bytes byte-identical.
3. Benchmark: time `1000 * next(iter(backend))` vs `1000 *
   kwcoco.CocoDataset.delayed_load(...).finalize()`. Decide if the
   throughput delta is meaningful enough to proceed.

Phase 2 — **DEIMv2 trainer plug-in**:

4. Add `kwcoco_detector_kit/data/fast_backend.py` with the interface +
   WebDataset impl.
5. Add `KCD_DATA_BACKEND=webdataset|kwcoco` to the kit's sweep config.
   When `webdataset`, the trainer glue swaps in the FastImageBackend
   adapter.
6. Run the pup_vs_nonpup hgnetv2_n v2 wrapper with
   `KCD_DATA_BACKEND=webdataset`. Compare iter time + GPU util to v2
   without the flag.

Phase 3 — **LMDB for eval / random access**:

7. Add `class LMDBBackend`.
8. Add `KCD_EVAL_DATA_BACKEND=lmdb|kwcoco` so eval can use random
   access while train uses sequential.

Phase 4 — **Upstream**:

9. Move `FastImageBackend` + WebDatasetBackend + LMDBBackend into
   `kwcoco_dataloader/readers/`. Kit re-exports for backwards
   compatibility, then drops the re-export in a follow-up.

## Open questions

- **Augmentation point**: DEIMv2's transforms (Mosaic, IoUCrop,
  CopyBlend, ...) run inside `CocoDetection.__getitem__`. The fast
  backend returns a Sample; do augments run kit-side (cleaner) or do
  we still hand the Sample to DEIMv2's transforms (less change but
  ties us to DEIMv2's transform API forever)?
- **Annotation deduplication**: With Mosaic, four source samples are
  composed into one model input. Each Sample carries its own
  annotation list. The composition step is in DEIMv2's collate_fn.
  Do we need a per-sample primary key for cross-Sample deduplication?
- **WebDataset shuffle semantics**: The WebDataset shuffle is
  shard-level + intra-shard buffer. Is that random enough to match
  the current full-shuffle? Empirical question — small impact on
  final mAP per webdataset community wisdom, but worth measuring.
- **Eval-time determinism**: The kit's eval step expects sample
  iteration in a deterministic order to align preds with GT. LMDBBackend
  is fine; WebDataset would need a "no-shuffle" mode for eval. Easy
  but needs to be wired.
- **Cache invalidation**: Tile params hash already prevents stale
  reuse for the kwcoco bundle. Should we also key the WebDataset
  shards by the same hash? (Yes, IMO — shards live at
  `$KCD_TILE_CACHE_DPATH/_universal/<tile_hash>/shards/`.)
- **DEIMv2 vs CocoDetection**: How much of DEIMv2's
  `_DataLoader/Dataset` plumbing can we replace vs how much do we
  need to preserve to keep their config-driven transform pipeline
  working? Worst case: we re-implement Mosaic+IoUCrop+... kit-side
  (~500 lines).

## Decision asks

Before writing any code:

1. **Sign off on `Sample` shape + `FastImageBackend` protocol**, or
   counter-propose.
2. **Confirm the kit-incubation → kwcoco_dataloader-port path** is
   the right destination, or call out a different home (e.g., the
   reader stays kit-native long-term).
3. **Phase 1 first?** I.e., benchmark + writer round-trip before
   touching trainer plugins. Or jump straight to phase 2 if you're
   already confident the throughput delta will be worth it.

## Reference implementations to study

- `/home/joncrall/code/kwcoco_dataloader/kwcoco_dataloader/cli/build_webdataset.py` (1036 lines, existing writer)
- WebDataset docs: https://webdataset.github.io/webdataset/
- MosaicML Streaming: https://github.com/mosaicml/streaming (for the random-access-with-tar trick if we change our minds later)
- DEIMv2 train.py CocoDetection path (tpl/DEIMv2/engine/data/coco_dataset.py)

---

## Decisions locked in after research pass (2026-05-22)

External-research review of WebDataset + kwcoco_dataloader, focused
on HDD constraints + dynamic balancing + on-the-fly relabel.
Concrete decisions:

### Backend

- **WebDataset for train + eval, NOT WIDS.** WIDS = map-style indexed
  access over WebDataset shards. With `shuffle=True` it becomes a
  per-sample random-seek storm on HDD — exactly the pattern we're
  trying to escape. WIDS docs themselves recommend classic
  `webdataset.WebDataset` for training and reserve WIDS for "random
  access, sparse sampling, indexed samplers, or legacy code." Use it
  for debug / inspection / one-off eval only.
- **Drop `LocalWebdatasetBuckets` from the training path.** It's a
  map-style Dataset that defaults to WIDS, exactly the HDD anti-
  pattern. Repurpose for debug + eval-with-shuffle=False; build a
  new iterable streaming reader for train.
- **DDP via WebDataset's own sharding** (`nodesplitter=split_by_node`,
  `workersplitter=split_by_worker`), NOT `DistributedSampler`.
  Lightning/map-style assumptions don't apply to IterableDatasets.

### Writer (in `kwcoco_dataloader`)

- **Keep `BucketShardWriter`'s core idea** (bucketed sub-dirs +
  per-shard `__header__.json`/`__footer__.json`/`.index`). Useful as
  the data-side index for the runtime scheduler.
- **Change the bucket key from class-tuples → semantic class-presence
  flags.** Detection tiles can contain multiple classes; bucketing by
  the full tuple fragments shards and makes per-class streams hard.
  Bucket by `contains_<class>` booleans + `contains_any_box` +
  `n_empty` instead.
- **Enrich the footer manifest** with raw class histograms so the
  reader's scheduler can compute stream weights:

  ```json
  {"url":"...tar","nsamples":1000,
   "raw_class_hist":{"P":83,"B":410,"S":102},
   "contains_pup":75,"contains_nonpup_sealion":430,
   "contains_any_box":505,"n_empty":495,"byte_size":7e8}
  ```

- **Sample format**: simple `<key>.jpg` + `<key>.json` pairs (NOT the
  `collated.npz + non_collatable.pyd` packing the existing fusion
  datamodule uses). JSON carries: image_id, width, height, list of
  annotations (with `source_category` preserved per the kit's
  universal-tile invariant), and a tile-provenance block
  (`{kwcoco_gid, space_slice}`).
- **No tile-time class collapse.** The scheme mapping is a runtime
  label transform; the on-disk shards encode the *raw* class labels
  (one shard set serves every scheme).

### Reader (kit-side initially; migrates to `kwcoco_dataloader`)

Three layers:

1. **`WebDatasetStream`** — thin wrapper around `wds.WebDataset(urls,
   shardshuffle=N, nodesplitter=split_by_node, workersplitter=
   split_by_worker, ...)` + `.shuffle(buf)` + `.decode("pil")`.
   Yields raw `{__key__, jpg, json}` dicts.
2. **`.map(relabel_detection_sample)`** — runtime scheme collapse.
   Reads `source_category` from each annotation, applies the scheme
   yaml's mapping, drops unmapped (or marks ignore — TBD policy),
   re-densifies `category_id` to `0..N-1` for DETR. Documents the
   mapping in the run's checkpoint metadata.
3. **`WeightedChunkMix`** — custom `IterableDataset` that opens one
   `WebDatasetStream` per bucket group (e.g. `pup-positive`,
   `nonpup-positive`, `empty-negative`), picks a source by weighted
   random, and drains `K=64` samples before re-picking. K is the
   HDD seek-amortization knob; default 64, drop to 32 if responsiveness
   matters, push to 128 if seek thrash is observed.

```python
class WeightedChunkMix(torch.utils.data.IterableDataset):
    def __init__(self, streams, weights, chunk_size=64, epoch_size=None,
                 seed=0):
        ...
    def __iter__(self):
        rng = random.Random(self.seed + 1000003 * self.epoch)
        iters = [iter(s) for s in self.streams]
        n = 0
        while self.epoch_size is None or n < self.epoch_size:
            i = rng.choices(range(len(iters)), weights=self.weights, k=1)[0]
            for _ in range(self.chunk_size):
                if self.epoch_size is not None and n >= self.epoch_size:
                    return
                try:
                    yield next(iters[i]); n += 1
                except StopIteration:
                    iters[i] = iter(self.streams[i])
                    break
```

### Class-balanced sampling strategy

Locked: **strategy B (segregated bucket streams + chunked weighted
mixer)** as the default. Strategy A (mixed-shard intra-shard buffer)
as a fallback for "don't care about runtime balance" runs — same
shard format, different scheduler. Reject-sampling (C) considered
only for moderate-ratio nudges and explicitly NOT for extreme rare-
class up-weighting.

For object detection specifically, also consider **repeat-factor
packing at pack time**: physically duplicate rare-positive tiles 2–5×
in the bucket they land in. ~25 GB dataset can absorb the
duplication; gives smoother stream weights at runtime.

### Resolution jitter (oversize + crop-on-load)

Locked: **defer to phase 3, prove it pays first.** The 44% HDD byte
penalty is real; DEIMv2's existing Mosaic / RandomIoUCrop /
CopyBlend already do scale/context jitter. Three A/B/C variants to
benchmark before locking in:

- A. normal tile → DEIMv2 aug pipeline
- B. oversized tile → kit-side RandomCrop → DEIMv2 aug
- C. oversized tile → DEIMv2 aug (its IoUCrop handles the jitter)

Expect A or C to win on HDD; B costs the 44% without obvious gain.

If we do build a crop-on-load step, it MUST: clip bboxes, drop boxes
below a visibility threshold, update `area`/`iscrowd`, preserve
`source_category` + `kwcoco_ann_id`, record crop offset, and use a
per-worker/per-sample RNG (not a dataset-construction-time one).
`TimeSpaceAugmenter` in kwcoco_dataloader has the right concept but
has known bugs: off-by-one crop high-bound (`H - h` should be `H - h + 1`),
no bbox update, single-RNG correlation across multiprocessing workers.

### Phase plan (updated)

**Phase 1 — Writer + format lockdown (in kwcoco_dataloader)**
1. Add a detection-tile writer mode to `build_webdataset` that
   produces `<key>.jpg` + `<key>.json` pairs (not the fusion-style
   .npz/.pyd packing).
2. Update `BucketShardWriter` bucket keys to use semantic class-
   presence flags instead of class tuples.
3. Enrich footer manifests with raw class histograms +
   contains-class counts.
4. Add a single-pass round-trip test: kwcoco → shards → read first
   N samples → verify counts, bboxes, source_category match.

**Phase 2 — Streaming reader (kit-side, kit/data/fast_backend.py)**
5. `WebDatasetStream` thin wrapper.
6. `relabel_detection_sample` map function (driven by scheme yaml).
7. `WeightedChunkMix` IterableDataset.
8. Trainer-side glue: adapter that exposes Sample → DEIMv2's
   `CocoDetection` interface.

**Phase 3 — Wire into DEIMv2 + benchmark**
9. Add `KCD_DATA_BACKEND=webdataset|kwcoco` to the sweep config.
10. Run pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_v2 against both
    backends; compare iter time + GPU util.
11. If WebDataset wins clearly, set as default. Otherwise diagnose.

**Phase 4 — Oversize + crop (only if A/B/C benchmark says it helps)**

**Phase 5 — Upstream the kit reader → kwcoco_dataloader**, replacing
the WIDS-based `LocalWebdatasetBuckets` train path.
