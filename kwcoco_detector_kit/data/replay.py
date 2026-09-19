"""Deterministic cumulative hard-negative replay selection."""
from __future__ import annotations

from pathlib import Path

import kwconf


class ReplayConfig(kwconf.Config):
    hard_kwcocos = kwconf.Value([], help="ordered list of completed hard-negative manifests")
    broad_kwcoco = kwconf.Value(None, required=True, help="safe broad negative pool")
    dst = kwconf.Value(None, required=True)
    hard_quota = kwconf.Value(5000)
    random_quota = kwconf.Value(5000)
    per_source_cap = kwconf.Value(0, help="0 disables")
    per_scale_cap = kwconf.Value(0, help="0 disables")
    seed = kwconf.Value(0)


def _identity(img):
    value = (
        img.get("tile_identity") or img.get("tile_id")
        or img.get("materialization_identity") or img.get("tile_materialization_id")
    )
    if value:
        return str(value)
    raise KeyError("replay item lacks explicit tile identity metadata")


def run(config):
    import kwcoco
    import numpy as np

    broad = kwcoco.CocoDataset.coerce(str(config.broad_kwcoco))
    raw_hards = config.hard_kwcocos
    if isinstance(raw_hards, str):
        raw_hards = [part for part in raw_hards.split(",") if part]

    # Later manifests can rescore an old tile. Retain one record at its best
    # observed score, independent of manifest order.
    hard_by_id = {}
    for fpath in raw_hards:
        dset = kwcoco.CocoDataset.coerce(str(fpath))
        for img in dset.images().objs:
            ident = _identity(img)
            score = float(img.get("max_pred_score", 0))
            old = hard_by_id.get(ident)
            if old is None or score > old[0]:
                hard_by_id[ident] = (score, dset, img)

    selected = []
    source_counts, scale_counts = {}, {}

    def _admit(dset, img, origin):
        if img.get("tile_role", "negative") != "negative":
            return False
        if "tile_source_gid" not in img:
            raise KeyError("replay item lacks tile_source_gid")
        source = img["tile_source_gid"]
        scale = img.get("tile_scale_name", tuple(img.get("tile_actual_scale_xy", [])))
        if scale in (None, ()):
            raise KeyError("replay item lacks explicit scale metadata")
        if int(config.per_source_cap) > 0 and source_counts.get(source, 0) >= int(config.per_source_cap):
            return False
        if int(config.per_scale_cap) > 0 and scale_counts.get(scale, 0) >= int(config.per_scale_cap):
            return False
        source_counts[source] = source_counts.get(source, 0) + 1
        scale_counts[scale] = scale_counts.get(scale, 0) + 1
        selected.append((dset, img, origin))
        return True

    ranked = sorted(hard_by_id.values(), key=lambda row: (-row[0], _identity(row[2])))
    for _score, dset, img in ranked:
        if sum(origin == "hard" for _, _, origin in selected) >= int(config.hard_quota):
            break
        _admit(dset, img, "hard")

    used = {_identity(img) for _, img, _ in selected}
    candidates = [img for img in broad.images().objs if _identity(img) not in used]
    rng = np.random.RandomState(int(config.seed))
    order = rng.permutation(len(candidates))
    n_random = 0
    for idx in order:
        if n_random >= int(config.random_quota):
            break
        img = candidates[int(idx)]
        if _admit(broad, img, "random"):
            n_random += 1

    out = kwcoco.CocoDataset()
    out.fpath = str(Path(config.dst).expanduser().resolve())
    out.add_category(name="background_candidate")
    for dset, img, origin in selected:
        new = {k: v for k, v in img.items() if k != "id"}
        new["file_name"] = str(dset.get_image_fpath(img["id"]))
        new["replay_origin"] = origin
        out.add_image(**new)
    out.dataset["info"] = [{
        "name": "kwcoco_detector_kit.data.replay",
        "hard_inputs": list(map(str, raw_hards)),
        "broad_input": str(config.broad_kwcoco),
        "hard_selected": sum(origin == "hard" for _, _, origin in selected),
        "random_selected": sum(origin == "random" for _, _, origin in selected),
        "seed": int(config.seed),
    }]
    Path(out.fpath).parent.mkdir(parents=True, exist_ok=True)
    out.dump()
    return Path(out.fpath)


__cli__ = ReplayConfig
