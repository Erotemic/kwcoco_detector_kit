"""
Offline hard-negative miner.

Given a trained detector + a kwcoco bundle of NEGATIVE tiles (``tile_role
== 'negative'`` as produced by ``data.tile``), score each tile with the
predictor and emit a kwcoco subset of "hard" negatives — tiles where the
model produces a high-confidence false detection.

The output kwcoco can then be unioned with the positive-tile bundle by
``data.merge`` to form the next training round.

Predictor adapter
-----------------
This module is **predictor-agnostic**. The trainer's predictor plugin
returns an object satisfying ``predictors._interface.DetectorPredictor``::

    class DetectorPredictor(Protocol):
        def predict_image(self, image_np, orig_size) -> list[dict]:
            '''[{'label': int, 'bbox_xyxy': [...], 'score': float}, ...]'''
        @property
        def eval_spatial_size(self) -> tuple[int, int]: ...

The miner asks the predictor for ``predict_image()`` on each negative
tile and keeps the ``max(score)`` over the returned detections.

The ``--trainer NAME`` knob picks which predictor plugin to use; the
default (``mock_tiny``) is the kit's CPU smoke predictor.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import scriptconfig as scfg


class MineConfig(scfg.DataConfig):
    """Score every negative tile with a trained detector; emit a kwcoco of the hardest."""

    neg_kwcoco = scfg.Value(None, help="input kwcoco of negative tiles", required=True)
    workdir = scfg.Value(None, help="trainer workdir (contains the checkpoint + config)", required=True)
    dst = scfg.Value(None, help="output kwcoco of hard negatives", required=True)

    trainer = scfg.Value(
        "mock_tiny",
        help='trainer plugin name; resolved via trainers._registry',
    )
    score_thresh = scfg.Value(0.30, help='tile is "hard" iff max pred score >= this')
    max_hard_per_round = scfg.Value(5000, help="cap total hard negatives; keep highest-scoring")
    # Mining budget — how many negative tiles to actually SCORE this
    # round. Without this, a full sweep over a million-tile negative pool
    # on CPU can take 12+ hours per round and dominate the experiment
    # wall-clock. Default (0) means "score them all" (legacy behavior).
    max_candidates = scfg.Value(
        0,
        help=(
            "cap on the number of negative tiles to score this round. "
            "0 = no cap (score every tile). Recommended: 30000-100000 for "
            "the shitspotter multi-scale tile pool. Cuts mining wall-clock "
            "by ~30x with only modest hard-neg-recall loss."
        ),
    )
    candidate_strategy = scfg.Value(
        "stratified_by_image",
        choices=["first", "random", "stratified_by_image"],
        help=(
            "How to sub-sample negatives when max_candidates > 0: "
            "'first' = first N gids (deterministic, biased toward earlier "
            "images); 'random' = uniform sample; 'stratified_by_image' = "
            "pick a balanced count per source image so the round 0 pool "
            "doesn't oversample one scene (recommended)."
        ),
    )
    candidate_seed = scfg.Value(0, help="rng seed for random/stratified strategies")
    device = scfg.Value("cpu", help="torch device (cpu / cuda:N)")
    progress = scfg.Value(True, help="show ProgIter")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        run(config)


def _load_predictor(trainer_name: str, workdir: Path, device: str):
    """Instantiate the predictor plugin for ``trainer_name`` from ``workdir``."""
    from kwcoco_detector_kit.trainers._registry import get_trainer

    trainer = get_trainer(trainer_name)
    return trainer.build_predictor(workdir, device=device)


def run(config):
    import kwcoco
    import numpy as np
    import ubelt as ub

    workdir = Path(str(config.workdir)).expanduser().resolve()
    neg_fpath = Path(str(config.neg_kwcoco)).expanduser().resolve()
    dst_fpath = Path(str(config.dst)).expanduser().resolve()

    print(f"mine: trainer={config.trainer} workdir={workdir}")
    print(f"      neg_kwcoco={neg_fpath}")
    print(f"      dst={dst_fpath}")
    print(f"      score_thresh={config.score_thresh}")
    print(f"      max_hard_per_round={config.max_hard_per_round}")

    predictor = _load_predictor(str(config.trainer), workdir, str(config.device))

    neg_dset = kwcoco.CocoDataset.coerce(str(neg_fpath))
    candidate_gids = [
        img["id"] for img in neg_dset.images().objs
        if img.get("tile_role") in (None, "negative")
    ]
    n_pool = len(candidate_gids)
    print(f"      pool: {n_pool} negative tiles")

    # Apply mining budget. Without this, scoring an N-tile pool runs in
    # O(N) and dominates the round-loop wall-clock for the shitspotter
    # multi-scale tile bundles (~1.8M tiles ≈ 16 h per round on a 3090).
    max_candidates = int(config.max_candidates or 0)
    if 0 < max_candidates < n_pool:
        strategy = str(config.candidate_strategy)
        rng = np.random.RandomState(int(config.candidate_seed))
        if strategy == "first":
            candidate_gids = candidate_gids[:max_candidates]
        elif strategy == "random":
            candidate_gids = list(
                rng.choice(candidate_gids, size=max_candidates, replace=False)
            )
        elif strategy == "stratified_by_image":
            # The "source image" for a tile is encoded in the
            # multiscale-tile filename; fall back to the kwcoco image's
            # `source_gid` field, else its own gid.
            def _src_key(img):
                if "source_gid" in img:
                    return img["source_gid"]
                fn = img.get("file_name", "")
                # filenames look like gid00005049_s10_x00320_y00960_negative.jpg
                # use the gid prefix as the source key
                base = Path(fn).name
                prefix = base.split("_")[0] if "_" in base else base
                return prefix
            groups: dict = {}
            for img in neg_dset.images().objs:
                if img["id"] not in set(candidate_gids):
                    continue
                groups.setdefault(_src_key(img), []).append(img["id"])
            # round-robin pick per-source until budget hit
            picked: list = []
            keys = list(groups.keys())
            rng.shuffle(keys)
            cycles = max(1, max_candidates // max(1, len(keys)) + 1)
            for _ in range(cycles):
                for k in keys:
                    bucket = groups[k]
                    if bucket:
                        picked.append(bucket.pop(
                            int(rng.randint(0, len(bucket)))
                        ))
                        if len(picked) >= max_candidates:
                            break
                if len(picked) >= max_candidates:
                    break
            candidate_gids = picked
        else:
            raise ValueError(f"unknown candidate_strategy: {strategy!r}")
        print(
            f"      budget: {max_candidates} of {n_pool} via "
            f"strategy={strategy} -> scoring {len(candidate_gids)} tiles"
        )
    else:
        print(f"      budget: unlimited (scoring all {n_pool} candidates)")

    scored: List[Tuple[float, int]] = []
    iterator = ub.ProgIter(candidate_gids, desc="mine score neg tiles", enabled=bool(config.progress))
    for gid in iterator:
        try:
            arr = neg_dset.coco_image(gid).imdelay().finalize()
        except Exception as ex:
            print(f"  warn: failed to read gid {gid}: {ex}")
            continue
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        orig_h, orig_w = arr.shape[:2]
        detections = predictor.predict_image(arr, (orig_w, orig_h))
        s = max((float(d.get("score", 0.0)) for d in detections), default=0.0)
        scored.append((s, gid))

    thresh = float(config.score_thresh)
    max_keep = int(config.max_hard_per_round)
    hard = [(s, g) for (s, g) in scored if s >= thresh]
    hard.sort(reverse=True)
    if len(hard) > max_keep:
        hard = hard[:max_keep]
    hard_gids = {g for _s, g in hard}
    score_by_gid = {g: s for s, g in hard}

    print(
        f"      {len(hard)} hard negatives kept "
        f"(of {len(scored)} scored; threshold {thresh})"
    )

    out_dset = kwcoco.CocoDataset()
    out_dset.fpath = str(dst_fpath)
    out_dset.add_category(name="widget")  # placeholder — negatives carry no anns
    n_kept = 0
    for img in neg_dset.images().objs:
        gid = img["id"]
        if gid not in hard_gids:
            continue
        new = {k: v for k, v in img.items() if k != "id"}
        new["max_pred_score"] = float(score_by_gid[gid])
        new["mined_for_round"] = int(os.environ.get("KCD_ROUND", "0"))
        out_dset.add_image(id=gid, **new)
        n_kept += 1

    out_dset.dump()
    print(f"  wrote {n_kept} hard-neg tile images to {dst_fpath}")

    # Sidecar score histogram — useful for picking next-round threshold.
    if scored:
        bins = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.01]
        hist = [0] * (len(bins) - 1)
        for s, _ in scored:
            for i in range(len(bins) - 1):
                if bins[i] <= s < bins[i + 1]:
                    hist[i] += 1
                    break
        sidecar = dst_fpath.with_suffix(".mine_stats.json")
        sidecar.write_text(json.dumps({
            "n_scored": len(scored),
            "n_hard": len(hard),
            "score_thresh": thresh,
            "max_hard_per_round": max_keep,
            "score_bins": bins,
            "score_hist": hist,
        }, indent=2))
        print(f"  wrote score histogram to {sidecar}")


__cli__ = MineConfig


if __name__ == "__main__":
    __cli__.main()
