"""
Data plumbing — kwcoco tile augmentation, merge, mine, MSCOCO export.

Public surface:

  tile.TileConfig          scriptconfig CLI / DataConfig for tiling.
  tile.run                 programmatic entry point.
  merge.MergeConfig        positive + negative tile merger CLI.
  merge.run                programmatic entry point.
  mine.MineConfig          offline hard-negative miner CLI.
  mine.run                 programmatic entry point.
  coco_export.export_mscoco  kwcoco -> MSCOCO json (used by DEIMv2/OGDino).
"""
from kwcoco_detector_kit.data import tile, merge, mine, coco_export

__all__ = ["tile", "merge", "mine", "coco_export"]
