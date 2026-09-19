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
import hashlib
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import kwconf


def stable_shard_for_key(key: str, num_shards: int) -> int:
    """Deterministic shard assignment independent of Python hash randomization."""
    if int(num_shards) < 1:
        raise ValueError("num_shards must be positive")
    return int(hashlib.sha256(str(key).encode()).hexdigest(), 16) % int(num_shards)


def _semantic_source_scale(item):
    """Semantic stratification keys; filenames are deliberately irrelevant."""
    if "tile_source_gid" not in item:
        raise KeyError("candidate is missing required tile_source_gid metadata")
    scale = item.get("tile_scale_name")
    if scale is None:
        scale = tuple(item.get("tile_actual_scale_xy", []))
    if scale in (None, ()):
        raise KeyError("candidate is missing explicit scale metadata")
    return item["tile_source_gid"], scale


def stratified_candidate_ids(items, budget, seed=0):
    """Round-robin over source and scale groups for broad finite coverage."""
    import numpy as np

    items = list(items)
    if not budget or budget >= len(items):
        return [item["id"] if "id" in item else item["tile_id"] for item in items]
    groups = {}
    for item in items:
        groups.setdefault(_semantic_source_scale(item), []).append(item)
    rng = np.random.RandomState(int(seed))
    keys = sorted(groups, key=repr)
    rng.shuffle(keys)
    for rows in groups.values():
        rng.shuffle(rows)
    chosen = []
    while len(chosen) < budget and keys:
        next_keys = []
        for key in keys:
            rows = groups[key]
            if rows:
                item = rows.pop()
                chosen.append(item["id"] if "id" in item else item["tile_id"])
                if len(chosen) >= budget:
                    break
            if rows:
                next_keys.append(key)
        keys = next_keys
    return chosen


def merge_shard_ledgers(ledger_paths, expected_ids):
    """Merge complete shards while proving disjoint, exact global coverage."""
    records = []
    seen = set()
    for path in ledger_paths:
        doc = json.loads(Path(path).read_text())
        if not doc.get("scan_complete"):
            raise RuntimeError(f"incomplete mining shard: {path}")
        ids = {row["tile_id"] for row in doc["records"]}
        overlap = seen & ids
        if overlap:
            raise RuntimeError(f"shard ledgers overlap: {sorted(overlap)[:3]}")
        seen |= ids
        records.extend(doc["records"])
    expected = set(expected_ids)
    if seen != expected:
        raise RuntimeError(
            f"merged shard coverage mismatch: missing={len(expected - seen)} "
            f"extra={len(seen - expected)}"
        )
    records.sort(key=lambda row: row["tile_id"])
    return records


class MineConfig(kwconf.Config):
    """Score every negative tile with a trained detector; emit a kwcoco of the hardest."""

    neg_kwcoco = kwconf.Value(None, help="input materialized kwcoco negative tiles")
    candidate_index = kwconf.Value(None, help="virtual negative candidate JSON index")
    cache_dpath = kwconf.Value(None, help="cache for admitted virtual candidates")
    jpeg_quality = kwconf.Value(90)
    workdir = kwconf.Value(None, help="trainer workdir (contains the checkpoint + config)", required=True)
    dst = kwconf.Value(None, help="output kwcoco of hard negatives", required=True)

    trainer = kwconf.Value(
        "mock_tiny",
        help='trainer plugin name; resolved via trainers._registry',
    )
    score_thresh = kwconf.Value(0.30, help='tile is "hard" iff max pred score >= this')
    max_hard_per_round = kwconf.Value(5000, help="cap total hard negatives; keep highest-scoring")
    # Mining budget — how many negative tiles to actually SCORE this
    # round. Without this, a full sweep over a million-tile negative pool
    # on CPU can take 12+ hours per round and dominate the experiment
    # wall-clock. Default (0) means "score them all" (legacy behavior).
    max_candidates = kwconf.Value(
        0,
        help=(
            "cap on the number of negative tiles to score this round. "
            "0 = no cap (score every tile). Recommended: 30000-100000 for "
            "the shitspotter multi-scale tile pool. Cuts mining wall-clock "
            "by ~30x with only modest hard-neg-recall loss."
        ),
    )
    candidate_strategy = kwconf.Value(
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
    candidate_seed = kwconf.Value(0, help="rng seed for random/stratified strategies")
    device = kwconf.Value("cpu", help="torch device (cpu / cuda:N)")
    progress = kwconf.Value(True, help="show ProgIter")
    batch_size = kwconf.Value(16, help="predictor batch size; scalar backends fall back safely")
    shard_index = kwconf.Value(0, help="deterministic mining shard index")
    num_shards = kwconf.Value(1, help="number of disjoint deterministic shards")
    ledger = kwconf.Value(None, help="optional atomic JSON score ledger (defaults beside dst)")
    allow_failures = kwconf.Value(False, help="permit scan_successful=false output")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        run(config)


def _load_predictor(trainer_name: str, workdir: Path, device: str):
    """Instantiate the predictor plugin for ``trainer_name`` from ``workdir``."""
    from kwcoco_detector_kit.trainers._registry import get_trainer

    trainer = get_trainer(trainer_name)
    return trainer.build_predictor(workdir, device=device)


def _atomic_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")
        tmp = Path(file.name)
    os.replace(tmp, path)


def _run_virtual(config, predictor, dst_fpath):
    """Mine a virtual index with durable per-batch progress and exact resume."""
    from kwcoco_detector_kit.data.candidates import (
        load_candidate_index, materialize_candidates, realize_candidate_arrays,
    )
    from kwcoco_detector_kit.predictors._interface import predict_batch

    index = load_candidate_index(config.candidate_index)
    rows = index["candidates"]
    max_candidates = int(config.max_candidates or 0)
    if max_candidates and max_candidates < len(rows):
        wanted = set(stratified_candidate_ids(rows, max_candidates, config.candidate_seed))
        rows = [row for row in rows if row["tile_id"] in wanted]
    shard_index, num_shards = int(config.shard_index), int(config.num_shards)
    rows = [
        row for row in rows
        if stable_shard_for_key(row["tile_id"], num_shards) == shard_index
    ]
    rows.sort(key=lambda row: row["tile_id"])
    expected_ids = [row["tile_id"] for row in rows]
    expected_digest = hashlib.sha256("\n".join(expected_ids).encode()).hexdigest()
    ledger_path = Path(config.ledger) if config.ledger else dst_fpath.with_suffix(".mine_ledger.json")
    progress_path = ledger_path.with_suffix(ledger_path.suffix + ".progress.jsonl")
    records_by_id = {}
    expected_set = set(expected_ids)
    if progress_path.is_file():
        for line in progress_path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break  # killed during the final append; safely redo that bounded row
            if record.get("tile_id") in expected_set:
                records_by_id[record["tile_id"]] = record
    pending = [row for row in rows if row["tile_id"] not in records_by_id]
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = max(1, int(config.batch_size))
    with open(progress_path, "a", encoding="utf8") as progress:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            batch_records = []
            try:
                arrays_by_id = realize_candidate_arrays(index, batch)
                arrays = [arrays_by_id[row["tile_id"]] for row in batch]
                sizes = [(arr.shape[1], arr.shape[0]) for arr in arrays]
            except Exception as ex:
                batch_records = [{
                    "tile_id": row["tile_id"], "status": "read_error",
                    "error": f"{type(ex).__name__}: {ex}",
                } for row in batch]
            if not batch_records:
                try:
                    results = predict_batch(predictor, arrays, sizes)
                    if len(results) != len(batch):
                        raise RuntimeError(
                            f"predict_batch cardinality mismatch: {len(results)} != {len(batch)}"
                        )
                    for row, detections in zip(batch, results):
                        top = max(detections, key=lambda d: float(d.get("score", 0)), default=None)
                        batch_records.append({
                            "tile_id": row["tile_id"], "status": "ok",
                            "max_score": float(top.get("score", 0)) if top else 0.0,
                            "top_label": None if top is None else int(top.get("label", 0)),
                            "top_bbox_xyxy": None if top is None else top.get("bbox_xyxy"),
                        })
                except Exception as ex:
                    batch_records = [{
                        "tile_id": row["tile_id"], "status": "predict_error",
                        "error": f"{type(ex).__name__}: {ex}",
                    } for row in batch]
            for record in batch_records:
                progress.write(json.dumps(record, sort_keys=True) + "\n")
                records_by_id[record["tile_id"]] = record
            progress.flush()
            os.fsync(progress.fileno())
            delay = float(os.environ.get("KCD_MINE_TEST_BATCH_DELAY", "0"))
            if delay:
                time.sleep(delay)

    records = [records_by_id[tile_id] for tile_id in expected_ids]
    failures = [row for row in records if row["status"] != "ok"]
    ledger = {
        "schema_version": 2, "shard_index": shard_index, "num_shards": num_shards,
        "expected_identity_digest": expected_digest,
        "expected_tile_ids": expected_ids,
        "num_expected": len(expected_ids), "num_records": len(records),
        "num_failures": len(failures), "scan_complete": len(records) == len(expected_ids),
        "scan_successful": not failures, "records": records,
    }
    _atomic_json(ledger, ledger_path)
    if failures and not bool(config.allow_failures):
        raise RuntimeError(f"mining scan completed with {len(failures)} failures; see {ledger_path}")

    good = [row for row in records if row["status"] == "ok"]
    good.sort(key=lambda row: (-row["max_score"], row["tile_id"]))
    selected_scores = {
        row["tile_id"]: row["max_score"]
        for row in good if row["max_score"] >= float(config.score_thresh)
    }
    selected_scores = dict(list(selected_scores.items())[:int(config.max_hard_per_round)])
    selected = [row for row in rows if row["tile_id"] in selected_scores]
    if selected and not config.cache_dpath:
        raise ValueError("cache_dpath is required to admit virtual candidates")
    out = materialize_candidates(
        index, selected, cache_dpath=config.cache_dpath,
        jpeg_quality=int(config.jpeg_quality),
    ) if selected else __import__("kwcoco").CocoDataset()
    out.fpath = str(dst_fpath)
    for image in out.images().objs:
        image["max_pred_score"] = selected_scores[image["tile_id"]]
        image["mined_for_round"] = int(os.environ.get("KCD_ROUND", "0"))
    dst_fpath.parent.mkdir(parents=True, exist_ok=True)
    out.dump()
    return dst_fpath


def run(config):
    import kwcoco
    import numpy as np
    import ubelt as ub

    workdir = Path(str(config.workdir)).expanduser().resolve()
    dst_fpath = Path(str(config.dst)).expanduser().resolve()

    if bool(config.neg_kwcoco) == bool(config.candidate_index):
        raise ValueError("specify exactly one of neg_kwcoco or candidate_index")

    neg_fpath = (
        Path(str(config.neg_kwcoco)).expanduser().resolve()
        if config.neg_kwcoco else None
    )

    print(f"mine: trainer={config.trainer} workdir={workdir}")
    print(f"      neg_kwcoco={neg_fpath}")
    print(f"      dst={dst_fpath}")
    print(f"      score_thresh={config.score_thresh}")
    print(f"      max_hard_per_round={config.max_hard_per_round}")

    predictor = _load_predictor(str(config.trainer), workdir, str(config.device))
    if config.candidate_index:
        return _run_virtual(config, predictor, dst_fpath)

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
            # Build the membership set ONCE -- with a 1.8M-tile pool,
            # rebuilding the set per-iteration is O(N^2) and hangs for
            # hours before the first ProgIter line prints.
            candidate_id_set = set(candidate_gids)
            items = [
                img for img in neg_dset.images().objs
                if img["id"] in candidate_id_set
            ]
            candidate_gids = stratified_candidate_ids(
                items, max_candidates, config.candidate_seed,
            )
        else:
            raise ValueError(f"unknown candidate_strategy: {strategy!r}")
        print(
            f"      budget: {max_candidates} of {n_pool} via "
            f"strategy={strategy} -> scoring {len(candidate_gids)} tiles"
        )
    else:
        print(f"      budget: unlimited (scoring all {n_pool} candidates)")

    num_shards = int(config.num_shards)
    shard_index = int(config.shard_index)
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"invalid shard {shard_index}/{num_shards}")

    def _stable_key(gid):
        img = neg_dset.imgs[gid]
        key = img.get("tile_identity") or img.get("tile_id")
        if key is None:
            source, scale = _semantic_source_scale(img)
            extent = img.get("tile_extent_xyxy_in_source")
            if extent is None:
                raise KeyError("candidate lacks explicit source crop metadata")
            key = json.dumps({
                "source": source, "scale": scale, "extent": extent,
            }, sort_keys=True, separators=(",", ":"))
        return str(key)

    candidate_gids = [
        gid for gid in candidate_gids
        if stable_shard_for_key(_stable_key(gid), num_shards) == shard_index
    ]
    print(f"      shard: {shard_index}/{num_shards} -> {len(candidate_gids)} candidates")

    scored: List[Tuple[float, int]] = []
    ledger_records = []
    batch_size = max(1, int(config.batch_size))
    from kwcoco_detector_kit.predictors._interface import predict_batch
    iterator = ub.ProgIter(
        range(0, len(candidate_gids), batch_size),
        total=(len(candidate_gids) + batch_size - 1) // batch_size,
        desc="mine score neg tile batches", enabled=bool(config.progress),
    )
    for start in iterator:
        gids = candidate_gids[start:start + batch_size]
        arrays, sizes, valid_gids = [], [], []
        for gid in gids:
            try:
                arr = neg_dset.coco_image(gid).imdelay().finalize()
                if arr.ndim == 2:
                    arr = np.repeat(arr[..., None], 3, axis=-1)
                if arr.shape[2] == 4:
                    arr = arr[..., :3]
                arrays.append(arr)
                sizes.append((arr.shape[1], arr.shape[0]))
                valid_gids.append(gid)
            except Exception as ex:
                ledger_records.append({
                    "gid": gid, "tile_id": _stable_key(gid),
                    "status": "read_error", "error": f"{type(ex).__name__}: {ex}",
                })
        if not arrays:
            continue
        try:
            result_batch = predict_batch(predictor, arrays, sizes)
            if len(result_batch) != len(valid_gids):
                raise RuntimeError(
                    f"predict_batch cardinality mismatch: "
                    f"{len(result_batch)} != {len(valid_gids)}"
                )
        except Exception as ex:
            for gid in valid_gids:
                ledger_records.append({
                    "gid": gid, "tile_id": _stable_key(gid),
                    "status": "predict_error", "error": f"{type(ex).__name__}: {ex}",
                })
            continue
        for gid, detections in zip(valid_gids, result_batch):
            top = max(detections, key=lambda d: float(d.get("score", 0.0)), default=None)
            score = float(top.get("score", 0.0)) if top else 0.0
            scored.append((score, gid))
            ledger_records.append({
                "gid": gid, "tile_id": _stable_key(gid), "status": "ok",
                "max_score": score,
                "top_label": None if top is None else int(top.get("label", 0)),
                "top_bbox_xyxy": None if top is None else top.get("bbox_xyxy"),
            })

    ledger_fpath = Path(config.ledger) if config.ledger else dst_fpath.with_suffix(".mine_ledger.json")
    ledger_fpath.parent.mkdir(parents=True, exist_ok=True)
    ledger_doc = {
        "schema_version": 2, "complete": True, "scan_complete": True,
        "shard_index": shard_index, "num_shards": num_shards,
        "num_expected": len(candidate_gids), "num_records": len(ledger_records),
        "num_failures": sum(r["status"] != "ok" for r in ledger_records),
        "scan_successful": all(r["status"] == "ok" for r in ledger_records),
        "records": ledger_records,
    }
    with tempfile.NamedTemporaryFile("w", dir=ledger_fpath.parent, delete=False) as file:
        json.dump(ledger_doc, file, indent=2)
        file.write("\n")
        tmp_ledger = Path(file.name)
    os.replace(tmp_ledger, ledger_fpath)
    failures = [record for record in ledger_records if record["status"] != "ok"]
    if failures and not bool(config.allow_failures):
        raise RuntimeError(
            f"mining scan completed with {len(failures)} failures; see {ledger_fpath}"
        )

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
        # Rewrite file_name to absolute via the SOURCE bundle's resolver.
        # The output bundle is dumped to a different directory than the
        # input pool (rounds/roundN/hard_negs.kwcoco.zip vs the original
        # data/train_tiles_neg.kwcoco.zip), so a copied-as-is relative
        # file_name resolves to a nonexistent path downstream. Same fix
        # pattern as data/merge.py (commit b0db63c).
        try:
            new["file_name"] = str(neg_dset.get_image_fpath(gid))
        except Exception:
            pass
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
