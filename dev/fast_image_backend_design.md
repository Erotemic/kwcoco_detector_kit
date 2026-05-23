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

For v1: **WebDataset for train (huge throughput, iter-first matches
our shuffle-each-epoch usage) + LMDB for eval (random access matches
COCO-eval's deterministic order)**. Both can live behind one read
interface.

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
