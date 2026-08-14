# Phase 3 status — webdataset / multispectral / cloud

One place to see what Phase 3 surface is **shipped** vs **deferred**, so agents stop
rediscovering the edges. Phases 1–2 are complete; Phase 3 is partial. Update this
table as pieces land. (Created 2026-06-13 for KCD-DOC-02.)

## Shipped

| Capability | Owner file(s) | Notes |
|---|---|---|
| `TileStore` abstraction | `kwcoco_detector_kit/data/tile_store.py` | `KwcocoJpegStore` (default, Phase-1 path unchanged) + `WebdatasetStore` |
| `TileLoader` IterableDataset | `kwcoco_detector_kit/data/tile_loader.py` | load-time crop aug + normalize |
| Per-channel stats | `kwcoco_detector_kit/data/stats.py` + CLI `stats` | multispectral prep |
| Store conversion CLI | CLI `convert-store` (`data/tile_store.py`) | kwcoco_jpeg ↔ webdataset |
| Checkpoint selection scaffolding | `kwcoco_detector_kit/selection/` | boards/probe/rerank/scoring/worker/journal |

## Deferred (NOT yet implemented)

| Capability | Why it matters | Tracking |
|---|---|---|
| Native webdataset input to DEIMv2/OGDino trainers | trainers still consume kwcoco_jpeg; wds is data-prep only | Phase 3.1 |
| Class-balanced bucket sampling over wds shards | balanced sampling currently duplicates data on disk | Phase 3.1 |
| DDP-aware shard splitting | multi-GPU + wds needs a canonical split recipe | Phase 3.1 |
| Multispectral example end-to-end | stats exists; no worked multi-band example | Phase 3.2 |
| Upstream contributions to `kwcoco_dataloader` | shard builder lives upstream | ongoing |

## Related action items (from the cross-repo audit)

The shitspotter↔kit audit (`shitspotter/dev/design/kit_refactor_and_action_items.md`)
adds these kit items; status here for convenience:

| ID | Status |
|---|---|
| KCD-EVAL-01 (first-class tiled eval recipe mode) | ✅ done |
| KCD-EVAL-02 (eval_mode + tile params in metrics sidecar) | ✅ done |
| KCD-CFG-01 (`${VAR}` interpolation in recipes/specs) | ✅ done |
| KCD-DATA-01 (`data-manifest` op + `data.expect` guard) | ✅ done |
| KCD-DOC-01 (examples vs projects disambiguation) | ✅ done |
| KCD-DOC-02 (this page) | ✅ done |
| KCD-CFG-02 (`project-init` generator) | ⬜ open |
| KCD-EVAL-03 (unified metrics spec: distractors + gates) | ⬜ open |
| KCD-EVAL-04 (batched-NMS in mining/tiled eval throughput) | ⬜ open |
| KCD-BUILD-01 (single source for version pins) | ⬜ open |
| KCD-TEST-01 (consumer-edge guard tests) | ⬜ open |
