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


def iter_candidate_shard_assignments(rows, num_shards, locality_chunk_size=256):
    """Assign bounded source/scale chunks while preserving traversal locality."""
    chunk_size = max(1, int(locality_chunk_size))
    prior_group = None
    group_offset = 0
    for row in rows:
        group = _semantic_source_scale(row)
        if group != prior_group:
            prior_group = group
            group_offset = 0
        chunk_index = group_offset // chunk_size
        group_offset += 1
        key = json.dumps(
            [group[0], group[1], chunk_index],
            sort_keys=True, separators=(",", ":"),
        )
        yield row, stable_shard_for_key(key, num_shards)


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
        if "records_path" in doc:
            shard_records = [
                json.loads(line)
                for line in Path(doc["records_path"]).read_text().splitlines()
            ]
        else:
            shard_records = doc["records"]
        ids = {row["tile_id"] for row in shard_records}
        overlap = seen & ids
        if overlap:
            raise RuntimeError(f"shard ledgers overlap: {sorted(overlap)[:3]}")
        seen |= ids
        records.extend(shard_records)
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
    locality_chunk_size = kwconf.Value(
        256,
        help="maximum source/scale-local candidate chunk assigned as one shard unit",
    )
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


class _ScoreDistribution:
    """Bounded deterministic score summary suitable for million-row scans."""

    bins = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.01]

    def __init__(self, sample_size=4096):
        self.sample_size = int(sample_size)
        self.count = 0
        self.total = 0.0
        self.minimum = None
        self.maximum = None
        self.hist = [0] * (len(self.bins) - 1)
        self._sample_heap = []

    def add(self, identity, score):
        import heapq

        score = float(score)
        self.count += 1
        self.total += score
        self.minimum = score if self.minimum is None else min(self.minimum, score)
        self.maximum = score if self.maximum is None else max(self.maximum, score)
        for idx, (low, high) in enumerate(zip(self.bins, self.bins[1:])):
            if low <= score < high:
                self.hist[idx] += 1
                break
        priority = int(hashlib.sha256(str(identity).encode()).hexdigest(), 16)
        item = (-priority, str(identity), score)
        if len(self._sample_heap) < self.sample_size:
            heapq.heappush(self._sample_heap, item)
        elif item > self._sample_heap[0]:
            heapq.heapreplace(self._sample_heap, item)

    def as_dict(self):
        values = sorted(item[2] for item in self._sample_heap)

        def quantile(q):
            if not values:
                return None
            position = q * (len(values) - 1)
            lower = int(position)
            upper = min(lower + 1, len(values) - 1)
            frac = position - lower
            return values[lower] * (1 - frac) + values[upper] * frac

        return {
            "n_scored": self.count,
            "score_min": self.minimum,
            "score_max": self.maximum,
            "score_mean": None if not self.count else self.total / self.count,
            "score_bins": self.bins,
            "score_hist": self.hist,
            "score_quantiles": {
                key: quantile(q) for key, q in [
                    ("p00", 0.0), ("p25", 0.25), ("p50", 0.5),
                    ("p75", 0.75), ("p90", 0.9), ("p95", 0.95),
                    ("p99", 0.99), ("p100", 1.0),
                ]
            },
            "quantile_sample_size": len(values),
            "quantiles_exact": self.count <= self.sample_size,
        }


def _write_mine_stats(dst, distribution, *, n_hard, score_thresh,
                      max_hard_per_round, num_failures=0):
    stats = {
        **distribution.as_dict(),
        "n_hard": int(n_hard),
        "num_failures": int(num_failures),
        "score_thresh": float(score_thresh),
        "max_hard_per_round": int(max_hard_per_round),
    }
    sidecar = Path(dst).with_suffix(".mine_stats.json")
    _atomic_json(stats, sidecar)
    return sidecar


def _workdir_model_identity(workdir):
    """Digest model/config inputs while excluding transient mining outputs."""
    from kwcoco_detector_kit.data.tile_cache import canonical_digest, sha256_file

    root = Path(workdir)
    suffixes = {".pt", ".pth", ".ckpt", ".json", ".yaml", ".yml"}
    files = []
    for path in sorted(root.rglob("*")) if root.is_dir() else []:
        if path.is_file() and path.suffix.lower() in suffixes:
            files.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path)})
    return canonical_digest(files)


def _budgeted_candidate_rows(index, max_candidates, seed):
    """Return a factory for deterministic streaming/bounded candidate traversal."""
    from kwcoco_detector_kit.data.candidates import iter_candidate_records

    if max_candidates and max_candidates < index["num_candidates"]:
        import heapq
        heap = []
        for row in iter_candidate_records(index):
            priority = int(hashlib.sha256(
                f"{int(seed)}:{row['tile_id']}".encode()
            ).hexdigest(), 16)
            item = (-priority, row["tile_id"], row)
            if len(heap) < max_candidates:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        selected = [item[2] for item in heap]
        selected.sort(key=lambda row: (
            row["tile_source_gid"], tuple(row["tile_actual_scale_xy"]),
            row["tile_scaled_extent_xyxy"], row["tile_id"],
        ))
        return lambda: iter(selected)
    return lambda: iter_candidate_records(index)


def finalize_virtual_mining(candidate_index, ledger_paths, dst, *, cache_dpath,
                            jpeg_quality=90, score_thresh=0.3,
                            max_hard_per_round=5000, allow_failures=False):
    """Validate all score shards, choose global top-K, then materialize."""
    import heapq
    import kwcoco

    from kwcoco_detector_kit.data.candidates import (
        iter_candidate_records, load_candidate_index, materialize_candidates,
    )

    index = load_candidate_index(candidate_index)
    docs = [json.loads(Path(path).read_text()) for path in ledger_paths]
    if not docs:
        raise ValueError("no mining shard ledgers supplied")
    fingerprints = {doc.get("mining_run_fingerprint") for doc in docs}
    if len(fingerprints) != 1 or None in fingerprints:
        raise RuntimeError("mining shard fingerprint mismatch")
    run_spec = docs[0]["run_spec"]
    from kwcoco_detector_kit.data.tile_cache import canonical_digest
    fingerprint = next(iter(fingerprints))
    if any(doc.get("run_spec") != run_spec for doc in docs):
        raise RuntimeError("mining shard run-spec mismatch")
    if canonical_digest(run_spec) != fingerprint:
        raise RuntimeError("mining shard fingerprint does not match run spec")
    if float(score_thresh) != float(run_spec["score_thresh"]):
        raise RuntimeError("finalizer score threshold differs from fingerprinted run")
    if int(max_hard_per_round) != int(run_spec["max_hard_per_round"]):
        raise RuntimeError("finalizer top-K differs from fingerprinted run")
    num_shards = int(run_spec["num_shards"])
    locality_chunk_size = int(run_spec["locality_chunk_size"])
    if len(docs) != num_shards or {doc["shard_index"] for doc in docs} != set(range(num_shards)):
        raise RuntimeError("missing or duplicate mining shard ledger")
    if any(not doc.get("scan_complete") for doc in docs):
        raise RuntimeError("incomplete mining shard")

    row_factory = _budgeted_candidate_rows(
        index, int(run_spec["max_candidates"]), int(run_spec["candidate_seed"]),
    )
    expected_hashers = [hashlib.sha256() for _ in range(num_shards)]
    expected_counts = [0] * num_shards
    for row, rank in iter_candidate_shard_assignments(
        row_factory(), num_shards, locality_chunk_size,
    ):
        expected_hashers[rank].update((row["tile_id"] + "\n").encode())
        expected_counts[rank] += 1
    for doc in docs:
        rank = doc["shard_index"]
        if doc["num_expected"] != expected_counts[rank] or doc["expected_identity_digest"] != expected_hashers[rank].hexdigest():
            raise RuntimeError(f"shard {rank} expected-set mismatch")

    class _ReverseLex(str):
        """Make the lexicographically largest ID the worst equal-score item."""

        def __lt__(self, other):
            return str.__gt__(self, other)

    heap = []
    max_keep = max(0, int(max_hard_per_round))
    seen = set()
    failures = 0
    distribution = _ScoreDistribution()
    for doc in docs:
        rank = doc["shard_index"]
        progress_path = Path(doc["records_path"])
        count = 0
        observed_hasher = hashlib.sha256()
        with open(progress_path, encoding="utf8") as progress:
            records = (json.loads(line) for line in progress)
            for record in records:
                tile_id = record["tile_id"]
                if tile_id in seen:
                    raise RuntimeError(f"duplicate terminal candidate identity: {tile_id}")
                seen.add(tile_id)
                count += 1
                observed_hasher.update((tile_id + "\n").encode())
                if record["status"] != "ok":
                    failures += 1
                    continue
                score = float(record["max_score"])
                distribution.add(tile_id, score)
                if score < float(score_thresh):
                    continue
                item = (score, _ReverseLex(tile_id), tile_id)
                if not max_keep:
                    continue
                if len(heap) < max_keep:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
        if count != doc["num_expected"]:
            raise RuntimeError(f"shard {rank} terminal-result count mismatch")
        if observed_hasher.hexdigest() != doc["expected_identity_digest"]:
            raise RuntimeError(f"shard {rank} terminal identity mismatch")
    if len(seen) != sum(expected_counts):
        raise RuntimeError("global terminal coverage mismatch")
    if failures and not allow_failures:
        raise RuntimeError(f"global mining scan contains {failures} failures")

    ranked = sorted(
        [(score, tile_id) for score, _reverse_id, tile_id in heap],
        key=lambda item: (-item[0], item[1]),
    )
    score_by_id = {tile_id: score for score, tile_id in ranked}
    selected = [row for row in iter_candidate_records(index) if row["tile_id"] in score_by_id]
    out = materialize_candidates(
        index, selected, cache_dpath=cache_dpath, jpeg_quality=jpeg_quality,
    ) if selected else kwcoco.CocoDataset()
    out.fpath = str(Path(dst).resolve())
    for image in out.images().objs:
        image["max_pred_score"] = score_by_id[image["tile_id"]]
        image["mined_for_round"] = int(os.environ.get("KCD_ROUND", "0"))
    Path(out.fpath).parent.mkdir(parents=True, exist_ok=True)
    out.dump()
    _atomic_json({
        "schema_version": 1,
        "mining_run_fingerprint": fingerprint,
        "num_scored": len(seen), "num_failures": failures,
        "score_thresh": float(score_thresh),
        "max_hard_per_round": int(max_hard_per_round),
        "selected": [{"tile_id": tile_id, "max_score": score} for score, tile_id in ranked],
    }, Path(dst).with_suffix(".selected_candidates.json"))
    _write_mine_stats(
        dst, distribution, n_hard=len(ranked), score_thresh=score_thresh,
        max_hard_per_round=max_hard_per_round, num_failures=failures,
    )
    return Path(out.fpath)


def _run_virtual(config, predictor, dst_fpath):
    """Mine a virtual index with durable per-batch progress and exact resume."""
    from kwcoco_detector_kit.data.candidates import (
        iter_candidate_records, iter_realized_candidate_batches,
        load_candidate_index, materialize_candidates,
    )
    from kwcoco_detector_kit.predictors._interface import predict_batch

    index = load_candidate_index(config.candidate_index)
    max_candidates = int(config.max_candidates or 0)
    row_factory = _budgeted_candidate_rows(index, max_candidates, config.candidate_seed)
    shard_index, num_shards = int(config.shard_index), int(config.num_shards)
    locality_chunk_size = int(config.locality_chunk_size)
    ledger_path = Path(config.ledger) if config.ledger else dst_fpath.with_suffix(".mine_ledger.json")
    progress_path = ledger_path.with_suffix(ledger_path.suffix + ".progress.jsonl")
    expected_hasher = hashlib.sha256()
    global_hasher = hashlib.sha256()
    global_count = 0
    expected_count = 0
    for row, rank in iter_candidate_shard_assignments(
        row_factory(), num_shards, locality_chunk_size,
    ):
        global_hasher.update((row["tile_id"] + "\n").encode())
        global_count += 1
        if rank == shard_index:
            expected_hasher.update((row["tile_id"] + "\n").encode())
            expected_count += 1
    expected_digest = expected_hasher.hexdigest()
    from kwcoco_detector_kit.data.tile_cache import canonical_digest
    run_spec = {
        "candidate_index_digest": index["candidate_content_digest"],
        "selected_universe_digest": global_hasher.hexdigest(),
        "selected_universe_count": global_count,
        "num_shards": num_shards,
        "shard_assignment_schema": "sha256-source-scale-spatial-chunk-mod-v1",
        "locality_chunk_size": locality_chunk_size,
        "max_candidates": max_candidates,
        "candidate_seed": int(config.candidate_seed),
        "candidate_strategy": str(config.candidate_strategy),
        "trainer": str(config.trainer),
        "model_workdir_identity": _workdir_model_identity(config.workdir),
        "device": str(config.device),
        "batch_size": int(config.batch_size),
        "score_thresh": float(config.score_thresh),
        "max_hard_per_round": int(config.max_hard_per_round),
        "candidate_policy_fingerprint": index["policy_fingerprint"],
    }
    mining_run_fingerprint = canonical_digest(run_spec)
    progress_meta_path = progress_path.with_suffix(progress_path.suffix + ".meta.json")
    if progress_path.exists():
        if not progress_meta_path.is_file():
            raise RuntimeError("existing mining progress lacks run fingerprint metadata")
        prior_meta = json.loads(progress_meta_path.read_text())
        if prior_meta.get("mining_run_fingerprint") != mining_run_fingerprint:
            raise RuntimeError("existing mining progress fingerprint mismatch")
    else:
        _atomic_json({
            "schema_version": 1,
            "mining_run_fingerprint": mining_run_fingerprint,
            "run_spec": run_spec,
        }, progress_meta_path)

    completed_ids = set()
    num_failures = 0
    if progress_path.is_file():
        with open(progress_path, "r+", encoding="utf8") as progress:
            valid_end = progress.tell()
            while line := progress.readline():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A killed append can leave only the final line torn.
                    # Discard it in place without retaining prior JSONL rows.
                    progress.seek(valid_end)
                    progress.truncate()
                    break
                valid_end = progress.tell()
                if record.get("tile_id"):
                    completed_ids.add(record["tile_id"])
                    num_failures += record.get("status") != "ok"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = max(1, int(config.batch_size))

    def _score_batch(batch, arrays, read_error, progress):
        nonlocal num_failures
        if not batch:
            return
        batch_records = []
        if read_error is not None:
            batch_records = [{
                "tile_id": row["tile_id"], "status": "read_error",
                "error": f"{type(read_error).__name__}: {read_error}",
            } for row in batch]
        if not batch_records:
            sizes = [(arr.shape[1], arr.shape[0]) for arr in arrays]
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
            completed_ids.add(record["tile_id"])
            num_failures += record["status"] != "ok"
        progress.flush()
        os.fsync(progress.fileno())
        delay = float(os.environ.get("KCD_MINE_TEST_BATCH_DELAY", "0"))
        if delay:
            time.sleep(delay)

    with open(progress_path, "a", encoding="utf8") as progress:
        assigned_rows = (
            row
            for row, rank in iter_candidate_shard_assignments(
                row_factory(), num_shards, locality_chunk_size,
            )
            if rank == shard_index and row["tile_id"] not in completed_ids
        )
        for batch, arrays, read_error in iter_realized_candidate_batches(
            index, assigned_rows, batch_size,
        ):
            _score_batch(batch, arrays, read_error, progress)

    ledger = {
        "schema_version": 2, "shard_index": shard_index, "num_shards": num_shards,
        "mining_run_fingerprint": mining_run_fingerprint, "run_spec": run_spec,
        "expected_identity_digest": expected_digest,
        "num_expected": expected_count, "num_records": len(completed_ids),
        "num_failures": num_failures,
        "scan_complete": len(completed_ids) == expected_count,
        "scan_successful": not num_failures,
        "records_path": str(progress_path.resolve()),
    }
    _atomic_json(ledger, ledger_path)
    if num_failures and not bool(config.allow_failures):
        raise RuntimeError(
            f"mining scan completed with {num_failures} failures; see {ledger_path}"
        )

    if num_shards == 1:
        return finalize_virtual_mining(
            config.candidate_index, [ledger_path], dst_fpath,
            cache_dpath=config.cache_dpath, jpeg_quality=int(config.jpeg_quality),
            score_thresh=float(config.score_thresh),
            max_hard_per_round=int(config.max_hard_per_round),
            allow_failures=bool(config.allow_failures),
        )
    return ledger_path


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

    # Shared bounded score summary for threshold/round decisions.
    if scored:
        distribution = _ScoreDistribution()
        for score, gid in scored:
            distribution.add(_stable_key(gid), score)
        sidecar = _write_mine_stats(
            dst_fpath, distribution, n_hard=len(hard), score_thresh=thresh,
            max_hard_per_round=max_keep, num_failures=len(failures),
        )
        print(f"  wrote score histogram to {sidecar}")


__cli__ = MineConfig


if __name__ == "__main__":
    __cli__.main()
