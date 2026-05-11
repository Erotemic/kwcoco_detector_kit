# Storage backends — TileStore abstraction

Phase 3 introduces a backend-agnostic `TileStore` Protocol so the kit can read/write tile bundles in whatever on-disk format makes sense for the host. Phase 2's one-JPEG-per-tile + kwcoco-manifest layout stays the **default**; new backends are additive and opt-in.

## Why this exists

The PLAN's webdataset extension assumed shards were the right answer. They might be — but the user's real constraints are:

1. **Tiles oversized for crop augmentation.** Model inputs of 320×320 with `oversize_factor=1.4` written as 448×448 on disk → trainer-side load-time random crop picks position + scale without bleeding into zero-padded borders.
2. **Spinning-disk + slow-network streamable.** SSD must NOT be a requirement. Random per-sample seeks across a million tiny files are a non-starter; sequential reads of contiguous shards or mmap'd binaries are the constraint.
3. **Format-TBD.** webdataset, tar, parquet, lmdb, mmap'd numpy — the right answer depends on the bundle size, IO budget, and class-balanced sampling needs. The kit shouldn't lock in.

`TileStore` is the abstraction; each backend solves (1)+(2) differently.

## Public API

```python
@runtime_checkable
class TileStore(Protocol):
    @property
    def num_tiles(self) -> int: ...
    @property
    def metadata(self) -> dict: ...
    def __iter__(self) -> Iterator[TileRecord]: ...
```

```python
@dataclass
class TileRecord:
    image_np: np.ndarray         # HxWx3 (or HxWxC for multispectral) uint8
    bboxes_xywh: np.ndarray      # Mx4 float
    category_ids: np.ndarray     # M int
    metadata: dict               # tile_role, tile_scale_name, tile_model_input_size, ...
```

Open any backend by path:

```python
from kwcoco_detector_kit.data.tile_store import open_store
store = open_store("/path/to/tiles.kwcoco.zip")   # -> KwcocoJpegStore
store = open_store("/path/to/wds_bundle_dir/")    # -> WebdatasetStore (auto-detect via _bundle_meta.json)
```

## Backends in v1.0

| backend | file layout | use case |
|---|---|---|
| `KwcocoJpegStore` (default) | one `.jpg` per tile + `<name>.kwcoco.zip` manifest | Phase-1 default. Random-access friendly; not great on spinning disks at million-tile scale. |
| `WebdatasetStore` | `shard-NNNNNN.tar` files + `_bundle_meta.json` | sequential reads on rotational disks; streamable from NFS / S3 via fsspec; built-in DDP-aware shard splitting. |

Future backends slot in by implementing the Protocol. Likely candidates:
- `ParquetStore` — row-wise binary payloads with column-store metadata. Good for class-balanced bucket sampling without duplicating bytes on disk.
- `LmdbStore` — single-file key/value, mmap reads. Good for big bundles on local disk.
- `Mmap NumpyStore` — fixed-stride shards of `np.memmap`'d uint8. Maximum throughput when tiles are homogeneous.

## TileLoader — load-time crop augmentation

```python
from kwcoco_detector_kit.data.tile_loader import TileLoader

loader = TileLoader(
    store,
    augment=True,                           # random crop + flip; eval uses augment=False
    normalize={"mean": [...], "std": [...]} # optional; from data.stats
)
for batch in loader:
    # batch.image: 3xHxW float tensor in model-input coords
    # batch.bboxes_xywh, batch.category_ids: in model-input coords
    ...
```

Load-time crop reads `tile_model_input_size` from the tile metadata (written by `data.tile` when `oversize_factor > 1`) and either random-crops (augment=True) or center-crops (augment=False) the on-disk tile to fit.

The bbox warping is implicit: bboxes get clipped to the crop window, then small ones (< 1 pixel side) get dropped.

DDP/sampler decisions live one layer up. Wrap a `TileLoader` in a `torch.utils.data.DataLoader` and let `DistributedSampler` (or the equivalent for webdataset) handle rank-aware sharding.

## CLIs

```bash
# Convert a kwcoco tile bundle to a WebdatasetStore directory
python -m kwcoco_detector_kit convert-store \
    /scratch/kcd/tiles.kwcoco.zip \
    /scratch/kcd/tiles_wds/ \
    --backend webdataset \
    --maxcount 1024 \
    --maxsize_mb 256

# Probe per-channel mean/std (any backend)
python -m kwcoco_detector_kit stats \
    /scratch/kcd/tiles_wds/ \
    /scratch/kcd/tiles.stats.json \
    --sample_size 256
```

## Trainer integration

DEIMv2 + OpenGroundingDINO consume MSCOCO/ODVG json paths, not TileStores — the kit's `data_format` knob has no effect on those trainers yet. They keep reading from the kwcoco bundle via the existing MSCOCO conversion path inside `generate_config()`. Phase 2 path is unchanged.

A trainer plugin opts into TileStore-aware inputs by:

```python
class MyTrainer:
    def supports_webdataset_input(self) -> bool: return True
    # generate_config inspects data_format and writes a config that
    # points the upstream trainer at the shards directory.
```

mock_tiny is the natural first plugin to demonstrate this (Phase 3.1 — not in v1.0).

## What's NOT in v1.0

- **Class-balanced bucket sampling** — `kwcoco_dataloader.cli.build_webdataset` has a sketch of this; the kit doesn't yet. Future expansion in Phase 3.1.
- **DDP-aware shard splitting for WebdatasetStore** — the webdataset library has `wds.split_by_node` / `wds.split_by_worker`. The kit's `TileLoader` doesn't wrap them yet. Add when a real DDP run needs it.
- **Native webdataset input to DEIMv2/OpenGroundingDINO** — would require upstream changes. Out of scope.
- **Upstream contributions back to `kwcoco_dataloader`** — the PLAN's §6 lists several TODO/FIXME locations in `build_webdataset.py`. Deferred until the kit's WebdatasetStore is exercised against a real dataset.

## Failure modes the abstraction defends against

- **Tile-bundle format coupling** — trainers don't reach inside the bundle's layout; they get `TileRecord` instances. Swapping the backend doesn't touch trainer code.
- **`oversize_factor` carrying through** — every backend's metadata round-trips the kit's tile-augmentation parameters, so the load-time crop knows what to do.
- **Multispectral channel handling** — `TileRecord.image_np` can be HxWxC for any C; `compute_per_channel_stats` derives the normalization for that bundle's specific channel set.
