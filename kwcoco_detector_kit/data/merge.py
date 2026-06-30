"""
Merge positive tiles + (round-0 random or round-N hard) negatives into a
single training kwcoco for the next round.

Round-loop semantics (see orchestration/round_loop.py):

  Round 0 training kwcoco =
      all positive tiles
    + a random subsample of negative tiles, ratio ``neg_over_pos``

  Round N training kwcoco =
      all positive tiles
    + hard negatives mined from round N-1

The positive pool is constant across rounds. Only the negative half
changes — round 0 samples uniformly, later rounds use the previous
round's false-positive predictions (from ``data.mine``).
"""
from __future__ import annotations

from pathlib import Path

import kwconf


class MergeConfig(kwconf.Config):
    """Merge a positive-tile kwcoco with a negative-tile kwcoco for one training round."""

    pos_kwcoco = kwconf.Value(None, help="kwcoco bundle of positive tiles (constant across rounds)", required=True)
    neg_kwcoco = kwconf.Value(
        None,
        help="kwcoco of negatives — random subsample for round 0, hard negatives for round N>0",
        required=True,
    )
    dst = kwconf.Value(None, help="output kwcoco for this round's training", required=True)

    category_names = kwconf.Value(
        "widget",
        help=(
            "comma-separated category names to copy from the positives "
            "bundle. Order is preserved and assigned to output "
            "category_id 1, 2, ... — matches data.tile output."
        ),
    )
    neg_over_pos = kwconf.Value(
        3.0,
        help="target ratio of negatives to positives in output. Capped by the actual neg pool. <=0 keeps ALL negatives.",
    )
    seed = kwconf.Value(0, help="RNG seed for negative subsampling")
    round_index = kwconf.Value(0, help="informational — which mining round this merge is for")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        run(config)


def run(config):
    import kwcoco
    import numpy as np

    pos_fpath = Path(str(config.pos_kwcoco)).expanduser().resolve()
    neg_fpath = Path(str(config.neg_kwcoco)).expanduser().resolve()
    dst_fpath = Path(str(config.dst)).expanduser().resolve()

    raw_names = config.category_names
    if isinstance(raw_names, (list, tuple)):
        target_cats = [str(n).strip() for n in raw_names if str(n).strip()]
    else:
        target_cats = [s.strip() for s in str(raw_names).split(",") if s.strip()]
    if not target_cats:
        raise RuntimeError("--category_names must contain at least one name")

    pos_dset = kwcoco.CocoDataset.coerce(str(pos_fpath))
    neg_dset = kwcoco.CocoDataset.coerce(str(neg_fpath))

    pos_gids = [
        img["id"] for img in pos_dset.images().objs
        if img.get("tile_role", "positive") == "positive"
    ]
    neg_gids = [
        img["id"] for img in neg_dset.images().objs
        if img.get("tile_role", "negative") == "negative"
    ]
    if not pos_gids:
        raise RuntimeError(f"no positive tiles in {pos_fpath}")

    rng = np.random.RandomState(int(config.seed))
    if float(config.neg_over_pos) > 0:
        target_n_neg = int(round(float(config.neg_over_pos) * len(pos_gids)))
        target_n_neg = min(target_n_neg, len(neg_gids))
    else:
        target_n_neg = len(neg_gids)

    if len(neg_gids) > target_n_neg:
        neg_gids_picked = list(rng.choice(neg_gids, size=target_n_neg, replace=False))
    else:
        neg_gids_picked = list(neg_gids)

    print(
        f"merge: round={int(config.round_index)} pos={len(pos_gids)} "
        f"neg_pool={len(neg_gids)} neg_picked={len(neg_gids_picked)} "
        f"(target ratio neg/pos={float(config.neg_over_pos)})"
    )

    out_dset = kwcoco.CocoDataset()
    out_dset.fpath = str(dst_fpath)
    # Output category IDs are 1, 2, ... in the order names were given —
    # matches the convention used by data.tile and downstream MSCOCO export.
    new_cat_ids = [out_dset.add_category(name=name) for name in target_cats]
    target_name_to_new_cid = dict(zip(target_cats, new_cat_ids))

    # positives — copy images + their annotations.
    # Rewrite file_name to the SOURCE bundle's absolute path before adding
    # to the merged bundle. The merged bundle gets dumped in a different
    # directory (e.g. rounds/round0/) and a copied-as-is relative
    # file_name would resolve to a nonexistent path under that dir.
    # Same treatment for the negatives block below.
    pos_set = set(pos_gids)
    src_gid_to_new_gid: dict = {}
    for img in pos_dset.images().objs:
        if img["id"] not in pos_set:
            continue
        new_img = {k: v for k, v in img.items() if k != "id"}
        try:
            new_img["file_name"] = str(pos_dset.get_image_fpath(img["id"]))
        except Exception:
            pass
        new_gid = out_dset.add_image(**new_img)
        src_gid_to_new_gid[("pos", img["id"])] = new_gid

    pos_cats_by_id = {c["id"]: c for c in pos_dset.dataset.get("categories", [])}
    for ann in pos_dset.dataset.get("annotations", []):
        src_gid = ann.get("image_id")
        if src_gid not in pos_set:
            continue
        src_cat = pos_cats_by_id.get(ann.get("category_id"))
        if not src_cat or src_cat["name"] not in target_name_to_new_cid:
            continue
        new_gid = src_gid_to_new_gid[("pos", src_gid)]
        bbox = ann.get("bbox")
        if not bbox:
            continue
        out_dset.add_annotation(
            image_id=new_gid, category_id=target_name_to_new_cid[src_cat["name"]],
            bbox=list(bbox), area=float(ann.get("area", bbox[2] * bbox[3])),
            iscrowd=int(ann.get("iscrowd", 0)),
        )

    # negatives — images only, no annotations
    neg_set = set(neg_gids_picked)
    for img in neg_dset.images().objs:
        if img["id"] not in neg_set:
            continue
        new_img = {k: v for k, v in img.items() if k != "id"}
        try:
            new_img["file_name"] = str(neg_dset.get_image_fpath(img["id"]))
        except Exception:
            pass
        out_dset.add_image(**new_img)

    out_dset.dump()
    pos_after = sum(1 for img in out_dset.images().objs if img.get("tile_role") == "positive")
    neg_after = sum(1 for img in out_dset.images().objs if img.get("tile_role") == "negative")
    print(
        f"  wrote {pos_after} pos + {neg_after} neg images, "
        f"{out_dset.n_annots} annots to {dst_fpath}"
    )


__cli__ = MergeConfig


if __name__ == "__main__":
    __cli__.main()
