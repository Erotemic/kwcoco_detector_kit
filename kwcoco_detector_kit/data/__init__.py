"""
Data plumbing — kwcoco tile augmentation, merge, mine, MSCOCO export, +
the Phase-3 TileStore abstraction for alternative on-disk layouts.

Public surface:

  tile.TileConfig          scriptconfig CLI / DataConfig for tiling.
  tile.run                 programmatic entry point.
  merge.MergeConfig        positive + negative tile merger CLI.
  mine.MineConfig          offline hard-negative miner CLI.
  coco_export.export_mscoco  kwcoco -> MSCOCO json (DEIMv2/OGDino input).
  tile_store.TileStore     Protocol — backend-agnostic tile bundle API.
  tile_store.KwcocoJpegStore  Phase 1 default (one-JPEG-per-tile + kwcoco).
  tile_store.WebdatasetStore  Phase 3 tar-shard backend.
  tile_store.open_store    Auto-detect backend by path.
  tile_loader.TileLoader   Iterable dataset w/ load-time crop aug.
  stats.compute_per_channel_stats  Welford mean/std probe (multispectral).
"""
from kwcoco_detector_kit.data import (
    tile, merge, mine, coco_export, tile_store, tile_loader, stats,
)

__all__ = ["tile", "merge", "mine", "coco_export",
           "tile_store", "tile_loader", "stats"]
