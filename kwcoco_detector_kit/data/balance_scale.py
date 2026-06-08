"""
Scale-balanced resampling of a tiled corpus kwcoco (materialize path).

A multi-scale tile corpus is heavily skewed: full-resolution crops vastly
outnumber downscaled / whole-frame tiles, and the corpus is mostly empty
(e.g. shitspotter v13: 1.27M tiles, 70x scale imbalance, ~7% positive, the
dominant near-scale ~97% empty). Uniform sampling lets the dominant scale and
the empties drown the rare/coarse views, which is exactly wrong when the goal
is uniform competence across apparent object scale.

This balances over ``(apparent scale x has-annotation)`` with
:class:`BalancedSampleForest` (the same tree balancer the kwcoco_dataloader
fusion datamodule uses) and *materializes* the draw as a new kwcoco whose image
entries reference the **same on-disk tiles** at balanced rates (oversample the
rare/coarse scales, subsample the dominant). Assets are never duplicated or
recomputed.

This is the JPEG-path analogue of a runtime balanced sampler, chosen because
stock DEIMv2 ``CocoDetection`` is map-style with no sampler-injection hook.

TODO (proper integration, approach B): a runtime ``BalancedCocoDetection`` in
the DEIMv2 fork (sibling to ``engine/data/dataset/wds_coco_dataset.py``,
registered the same way) that holds a ``BalancedSampleForest`` over columnar
attributes and draws a *fresh* balanced tile each step -- no materialized
duplicate kwcoco, full data utilization, per-epoch variety of the dominant
scale. Track training-length via a nominal epoch size. This module is the
stopgap until that lands.
"""
from __future__ import annotations

from pathlib import Path

import scriptconfig as scfg


class BalanceScaleConfig(scfg.DataConfig):
    """Materialize a scale + positive/negative balanced kwcoco from a tiled corpus."""

    src = scfg.Value(None, position=1, required=True,
                     help="input tiled corpus kwcoco (output of tile-corpus)")
    dst = scfg.Value(None, position=2, required=True,
                     help="output balanced kwcoco")
    target_size = scfg.Value(None,
                             help="number of image entries to draw; default len(src)//5")
    pos_fraction = scfg.Value(0.4,
                              help="target fraction of drawn tiles (per scale) that "
                                   "contain >=1 annotation")
    scale_weights = scfg.Value(None,
                               help='optional JSON {scale_bucket: weight}; default '
                                    'uniform over the detected buckets')
    n_trees = scfg.Value(16, help="BalancedSampleForest tree count")
    rng = scfg.Value(0, help="sampling seed (reproducible)")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        return run(config)


def _scale_bucket(im: dict) -> str:
    """Apparent-scale bucket from the tile metadata tile-corpus already stamps."""
    if im.get("tile_scale_name"):
        return "ms_" + str(im["tile_scale_name"])
    if im.get("tile_grid") is not None and im.get("tile_role") == "tile":
        return "quad%s" % im["tile_grid"]
    if im.get("tile_role") == "full":
        return "full"
    return "other"


def run(config):
    import json
    import kwcoco
    import ubelt as ub
    from kwcoco_dataloader.tasks.fusion.datamodules.balanced_sampling import (
        BalancedSampleForest,
    )

    src = kwcoco.CocoDataset.coerce(str(config.src))
    # Absolute file_names so the materialized bundle resolves its tiles from
    # anywhere (same lesson as tile-corpus).
    src.reroot(absolute=True)

    images = list(src.dataset.get("images", []))
    anns = list(src.dataset.get("annotations", []))
    pos_gids = {a["image_id"] for a in anns}
    anns_by_gid: dict = {}
    for a in anns:
        anns_by_gid.setdefault(a["image_id"], []).append(a)

    grid = [{"scale": _scale_bucket(im), "has_poop": (im["id"] in pos_gids)}
            for im in images]

    target = int(config.target_size) if config.target_size else max(1, len(images) // 5)
    scale_w = json.loads(config.scale_weights) if config.scale_weights else None
    pf = float(config.pos_fraction)

    forest = BalancedSampleForest(grid, rng=int(config.rng), n_trees=int(config.n_trees))
    forest.subdivide("scale", weights=scale_w)          # None -> uniform
    forest.subdivide("has_poop", weights={True: pf, False: 1.0 - pf})
    drawn = forest.sample_many(int(target))

    # Materialize: one image entry per draw (unique ids), annotations copied.
    new_images, new_annots = [], []
    next_gid, next_aid = 1, 1
    realized = ub.ddict(lambda: [0, 0])  # scale -> [neg, pos]
    for i in drawn:
        im = images[i]
        nim = dict(im)
        nim["id"] = next_gid
        new_images.append(nim)
        for a in anns_by_gid.get(im["id"], []):
            na = dict(a)
            na["id"] = next_aid
            na["image_id"] = next_gid
            new_annots.append(na)
            next_aid += 1
        realized[grid[i]["scale"]][int(grid[i]["has_poop"])] += 1
        next_gid += 1

    out = kwcoco.CocoDataset(
        {
            "categories": [dict(c) for c in src.dataset.get("categories", [])],
            "images": new_images,
            "annotations": new_annots,
            "info": src.dataset.get("info", []),
        },
    )
    out.fpath = str(Path(config.dst).expanduser())
    out.dump()

    print(f"[balance-scale] {len(images)} -> {len(new_images)} image entries "
          f"({len(new_annots)} annots), target_size={target}, pos_fraction={pf}")
    print("[balance-scale] realized per-scale (pos/total):")
    for k in sorted(realized):
        neg, pos = realized[k]
        tot = neg + pos
        print(f"  {k:8} {tot:>8}  pos%={100 * pos / max(tot, 1):4.0f}")
    return out.fpath


__cli__ = BalanceScaleConfig

if __name__ == "__main__":
    __cli__.main()
