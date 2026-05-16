# Data pipeline — approaches, trade-offs, and design notes

This document explains every data-serving strategy the kit uses or could use,
what trade-offs each carries for object-detection training specifically, and
where the design is still open.  Update it as decisions get made.

---

## 1. The landscape

Five fundamentally different ways to get images + annotations into a detector
training loop:

| approach | how data lives on disk | random access | sequential | balanced sampling | pre-processing | notes |
|---|---|---|---|---|---|---|
| **A. Direct kwcoco** | original images, kwcoco JSON | yes (coco_image) | poor | no (manual) | none | simplest; unbounded workers hit JSON memory issue |
| **B. TileStore / TileLoader** | one JPEG per tile + kwcoco.zip manifest | yes | poor | no (manual) | tile step | kit default; load-time aug |
| **C. KwcocoDetectionDataset** | original images + kwcoco JSON | via ndsampler | poor | yes (BalancedSampleTree) | none | new; richest sampling |
| **D. Webdataset shards** | `.tar` archives (sequential) | via wids index | excellent | via bucket shards | shard-build step | fits cloud/NFS; hard to update |
| **E. ndsampler COG/NPY backend** | one COG/NPY per image + ndsampler cache | yes (mmap) | ok | via ndsampler | backend-prepare | middle ground; underexplored |

---

## 2. Approach details

### A. Direct kwcoco (`_coco_to_batches` in mock_tiny, DEIMv2's own loader)

The simplest path: load from the kwcoco JSON directly.  Every trainer plugin
does this for its own pre-training data prep step; mock_tiny and the kit's
MSCOCO converter (`data/coco_export.py`) use this path.

**Works well when:**
- Dataset fits comfortably in memory (< 10 k images on a workstation)
- Local SSD with random-access headroom
- `num_workers=0` or low worker count

**Problems at scale:**
- Python's copy-on-write fork semantics: the large kwcoco JSON dict gets
  copied to every DataLoader worker's address space on first access, causing
  2–4× memory explosion.  `num_workers > 4` with a million-annotation kwcoco
  is effectively broken without the SQL backend or ndsampler.
- No balanced sampling.  All images equally likely regardless of annotation
  density.

### B. TileStore / TileLoader (kit default)

`data/tile.py` extracts fixed-size (optionally oversized) JPEG tiles from the
source images; a kwcoco.zip manifest records bbox annotations per tile.
`tile_store.KwcocoJpegStore` iterates them; `tile_loader.TileLoader` applies
load-time crop + flip augmentation.

**Works well when:**
- Local SSD or fast NAS
- Tiles are homogeneous size (constant model input)
- Round-based mining: add new negative tiles each round, re-tile is fast
- You want the tile step to be separate from training (reproducible shards)

**Problems:**
- One file per tile → random I/O.  On spinning disks or S3 with per-file
  overhead this collapses to essentially serial throughput.
- No balanced positive/negative sampling at load time.  You get whatever
  the tile grid produced.
- Tiles must be re-extracted if you want a different window size or scale.

### C. KwcocoDetectionDataset (via kwcoco_dataloader)

`data/kwcoco_sampler.KwcocoDetectionDataset` wraps `KWCocoVideoDataset` from
`kwcoco_dataloader`.  Samples directly from the source kwcoco + image files
via ndsampler.  No pre-tiling required.

**Key capabilities (all from KWCocoVideoDataset config):**
- `balance_options` — YAML list of sampling attributes.  Examples:
  ```yaml
  - attribute: contains_annotation          # 50/50 pos/neg
  - attribute: class                        # equal weight per category
    weights: {widget: 1.0, background: 0.2}
  ```
- `use_centered_positives=True` — window centers on annotation centroids,
  guaranteeing the object appears in the frame even for tiny annotations
- `chip_dims` — sample window; no on-disk pre-tiling needed
- `select_images` — JQ filter to restrict which images are even considered
- `modality_dropout`, `channel_dropout` — augmentation knobs for
  multi-sensor training
- `augment_space_shift_rate` — spatial jitter at sample time (NOT possible
  with pre-tiled approaches unless tile is oversized)
- `sampler_backend='npy'` — ndsampler can pre-cache a memory-mapped NPY
  representation; eliminates per-worker kwcoco JSON copies

**Problems:**
- Still random I/O.  On spinning disks this is slow for large datasets.
  mitigated by the `npy` sampler backend (one mmap'd array per image).
- kwcoco JSON memory pressure with `num_workers > 4`.  Use
  `sampler_backend='npy', sampler_workdir='/fast/local/cache'` to cache
  image data out of the JSON.
- Images must have explicit `channels='r|g|b'` metadata for channel-spec
  matching.  Datasets created by `data/tile.py` don't set this.
  **TODO:** propagate channel metadata from the tile step.
- `KWCocoVideoDataset` was designed for multi-frame satellite fusion; it
  carries setup overhead (spacetime grid construction, heuristics HACK
  that overwrites category colors) that is unnecessary for single-frame
  detection.  Construction takes ~5 s for a 10 k-image dataset; the grid
  is cached.

### D. Webdataset shards

Pre-process a kwcoco dataset into `.tar` archives via
`kwcoco_dataloader.cli.build_webdataset.BuildWebdatasetCLI`.  Read during
training via `LocalWebdatasetBuckets` (wids-indexed random access) or a
pure streaming `wds.WebDataset` pipeline.

**Works well when:**
- Cloud / NFS mount where per-file open cost is dominant (S3 especially)
- Very large datasets (> 1 M tiles) where file-count matters
- DDP across multiple nodes — shards split cleanly per node
- You want all data transforms baked in (normalization, augmentation) before
  training — pay pre-processing cost once, iterate many epochs cheaply

**Problems for detection:**
- `BuildWebdatasetCLI.custom_subset` is hardcoded for fusion tasks
  (`imdata_tchw`, `saliency`, `saliency_weights`, `time_index`,
  `timestamp`).  **None of these are useful for bbox detection.**
  A detection-specific subset writer is needed (see §4 below).
- Baked-in augmentation loses adaptivity.  Spatial jitter now requires
  writing oversized shards (e.g. 640×640 tiles for a 512×512 model) and
  doing crop at load time — same trade-off as TileStore.
- Hard-negative mining requires rebuilding shards each round, or
  maintaining parallel shard buckets.  The bucket-balanced approach in
  `build_webdataset.py` is a sketch; not battle-tested for detection.
- Updating labels (e.g. after re-labeling a batch) requires a full shard
  rebuild.

### E. ndsampler COG/NPY backend (underexplored)

ndsampler can pre-build a per-image COG or NPY cache (`sampler_backend='npy'`
in `KWCocoVideoDataset`).  This gives mmap'd random access with zero fork
overhead, without the shard-build step or any format lock-in.

This is probably the best middle ground for local-disk training at moderate
scale (up to a few hundred thousand images) that hasn't been fully evaluated.

---

## 3. When to use which

| scenario | recommendation |
|---|---|
| Smoke test / CI | Direct kwcoco (`_coco_to_batches` or `KwcocoDetectionDataset`); no pre-step |
| Local SSD, < 50 k images | `KwcocoDetectionDataset` with `sampler_backend=None` |
| Local SSD, > 50 k images | `KwcocoDetectionDataset` with `sampler_backend='npy'` |
| Spinning disk, large dataset | Webdataset shards — sequential reads win |
| Cloud / NFS mount (S3, GCS) | Webdataset shards — per-file overhead is catastrophic |
| DDP across multiple nodes | Webdataset shards — clean shard-per-node splitting |
| Multi-round hard-neg mining | TileStore / TileLoader — easy to add neg tiles per round |
| Rare-class oversampling | `KwcocoDetectionDataset` with `balance_options` |
| Multispectral (non-RGB) | `KwcocoDetectionDataset` — handles heterogeneous channels natively |

---

## 4. The webdataset gap: detection-specific subset

`BuildWebdatasetCLI.custom_subset` currently writes fusion task fields:

```python
collated_subkeys = {'imdata_tchw', 'saliency', 'saliency_weights',
                    'time_index', 'timestamp'}
```

For single-frame bbox detection, the right subset is:

```python
collated_subkeys = {'imdata_chw', 'box_ltrb', 'box_cidxs'}
# plus metadata: gid, ann_aids for provenance
```

**Status:** implemented.  `BuildWebdatasetCLI` now has a `task_type` parameter
(`'fusion'` | `'detection'`).  When `task_type='detection'`:

- `data_config` is forced to `time_steps=1`, `output_type='rgb'`,
  `requested_tasks.boxes=True`
- `custom_subset_detection()` writes `{imdata_chw, box_ltrb, box_cidxs}` to
  `det.npz` per sample (instead of the fusion `collated.npz` fields)
- `LocalWebdatasetBuckets` reads the `'.det.npz'` key automatically via
  `our_decode_basic` (`.npz` extension decoded to a numpy dict)
- `TimeSpaceAugmenter` detects the `det.npz` / `.det.npz` key and calls
  `_augment_detection()`, which crops `imdata_chw` spatially and
  clips/drops `box_ltrb` to the new window

---

## 5. Collation: the ragged-box problem

Standard `torch.utils.data.dataloader.default_collate` requires all tensors
to have the same shape.  Detection batches have variable numbers of boxes per
image (M varies) — you can't stack them.

Three strategies:
1. **Ragged list** — return `boxes_ltrb` as a `list[Tensor]`, one per image.
   Simple; incompatible with `default_collate`; requires custom `collate_fn`.
   `detection_collate_fn` in `data/kwcoco_sampler.py` does this.
2. **Zero-padded fixed size** — pad each image's boxes to `max_gt` with zeros.
   mock_tiny uses this (`max_gt=8`).  Simple to collate; wastes memory;
   requires a "valid" mask to ignore padding.
3. **Flat concat with image index** — concatenate all boxes across the batch,
   plus a `batch_idx` tensor.  Used by YOLOv5/v8 and many modern detectors.
   Requires custom loss/head integration.

DEIMv2 and OGDino use their own collation (padded tensors via `PadCollate` in
DEIMv2's `engine/data/dataset.py`).  The kit's Python-level trainers
(mock_tiny, future pure-Python plugins) should use ragged lists and custom
collation; this is the most flexible and matches PyTorch's convention for
detection.

---

## 6. DataLoader workers and kwcoco JSON memory

Python DataLoader workers fork the main process.  If the main process holds a
large kwcoco JSON dict in memory, every worker gets a copy the moment it
touches any dict object (copy-on-write page fault).  With 8 workers × 500 MB
kwcoco dict = 4 GB wasted.

**Mitigations in order of preference:**
1. `sampler_backend='npy'` — ndsampler caches a mmap'd array per image.
   Workers read from the mmap; the JSON is only touched to build the cache
   (one-time cost).
2. `sampler_backend=None, num_workers=0` — no workers, no fork.  Fine for
   fast local storage; only one core for data loading.
3. kwcoco SQL backend (`kwcoco.CocoDataset(dset, backend='sqlite')`) — avoids
   the in-memory dict; random access is ~10× slower but zero fork overhead.
4. `spawn` start method instead of `fork` — avoids the copy-on-write problem
   but re-imports everything per worker; high startup cost.

---

## 7. DDP (multi-GPU, multi-node) considerations

| approach | DDP strategy |
|---|---|
| Direct kwcoco / KwcocoDetectionDataset | `DistributedSampler` over sample grid targets |
| TileStore / TileLoader | `DistributedSampler` over tile indices |
| Webdataset shards | `wds.split_by_node` + `wds.split_by_worker` — each rank reads disjoint shards sequentially |

Webdataset's shard splitting is the cleanest for multi-node training because
there is no shared index — each node's worker just opens the next shard in its
slice of the URL list.  This is particularly valuable when data lives on a
shared network mount (NFS, Lustre) where concurrent random seeks from multiple
nodes would saturate the storage backend.

For single-node DDP (the common kit use case: 4× GPUs on one workstation),
`DistributedSampler` over a `KwcocoDetectionDataset` sample grid works well
and preserves balanced sampling.

---

## 8. Hard-negative mining compatibility

The kit's round-loop (`orchestration/round_loop.py`) runs N iterations of:
train → predict → mine negatives → merge → retrain.

Each approach's mining compatibility:

| approach | how to add new negatives |
|---|---|
| TileStore / kwcoco direct | re-run `data/tile.py` on negative images; `data/merge.py` concatenates kwcoco manifests |
| KwcocoDetectionDataset | same — just point to the merged kwcoco; no shard rebuild |
| Webdataset shards | rebuild shards from the merged kwcoco; or maintain per-round shard dirs |

Conclusion: webdataset shards are **not round-loop friendly** unless you
accept rebuilding shards each round (expensive) or split shards by provenance
(complex).  For hard-negative mining workflows, kwcoco-direct approaches are
better.

---

## 9. Current state and TODO

### Done
- [x] `data/tile.py` — tile extraction
- [x] `data/tile_store.py` — `KwcocoJpegStore` + `WebdatasetStore` (write path)
- [x] `data/tile_loader.py` — `TileLoader` (read path, load-time aug)
- [x] `data/kwcoco_sampler.py` — `KwcocoDetectionDataset` (ndsampler-backed)
- [x] Two bug fixes in `kwcoco_dataloader.utils.kwcoco_extensions.coco_channel_stats`

### TODO (in rough priority order)
- [ ] **Propagate `channels='r|g|b'` in tile output** — `data/tile.py` should
  set channel metadata on the tile kwcoco images so `KwcocoDetectionDataset`
  works on kit-generated tiles without manual annotation.
- [x] **Detection subset for build_webdataset** — `task_type='detection'` added to
  `BuildWebdatasetCLI`; writes `{imdata_chw, box_ltrb, box_cidxs}` via
  `custom_subset_detection()`; `LocalWebdatasetBuckets` reads `.det.npz`
  automatically; `TimeSpaceAugmenter._augment_detection()` crops image and
  clips/drops boxes.
- [ ] **`mock_tiny` trainer: switch to `KwcocoDetectionDataset`** — replace
  the hand-rolled `_coco_to_batches` with the proper balanced sampler.
- [ ] **ndsampler NPY backend integration** — wire `sampler_backend='npy'` as
  the default for `KwcocoDetectionDataset` when a `sampler_workdir` is
  available.
- [ ] **Webdataset kit CLI** — a `kwcoco-detector-kit build-wds` subcommand
  that calls `BuildWebdatasetCLI` with detection-appropriate defaults
  (single frame, RGB, boxes).
- [x] **`TimeSpaceAugmenter` detection support** — `_augment_detection()` added;
  correctly crops `imdata_chw` and shifts/clips/drops `box_ltrb` to the crop
  window.  Paired `box_cidxs` filtered to match surviving boxes.
- [ ] **Evaluate ndsampler COG/NPY vs webdataset** — benchmark on a real
  spinning-disk dataset to determine if the mmap backend eliminates the need
  for webdataset shards in the common case.

---

## 10. Open questions

1. **Is webdataset worth the pre-processing cost for typical detection runs?**
   For local SSD + moderate dataset (< 200 k tiles), the ndsampler NPY backend
   may give the same sequential-read throughput as webdataset shards without
   the shard-build step.  Need a benchmark.

2. **Detection vs fusion collation in a shared pipeline.**  The kit's trainers
   produce MSCOCO/ODVG json for DEIMv2/OGDino (their own data pipeline).
   The Python-level sampler only matters for mock_tiny and future pure-Python
   trainer plugins.  Don't over-engineer data serving for trainers that never
   see it.

3. **Parquet as a middle path.**  A parquet file with `image_bytes` (JPEG
   blob), `box_ltrb` (list column), `cat_id` (list column), and
   `provenance` metadata would be random-access friendly, columnar for
   balanced-sampling queries, and writeable incrementally.  Worth prototyping
   in Phase 3.2 if webdataset proves too rigid.
