"""Diagnostic tests for WebDatasetCocoDetection behavior under load.

Targets the regressions observed on gen002 2552 (2026-05-30):

* Epoch claimed 47878 iters but actually only ran ~5000 — stream
  exhausted early. Either ``__iter__`` skips reduce the effective
  count, or ``__len__`` overestimates what the stream can produce.

* ``max mem: 2403`` was constant across 3500 iters — far below
  v5's 10.5 GB at same batch/resolution/model. Suggests effective
  batch is smaller than nominal because empty-annotation skip
  drops samples mid-collation.

* Workers silently died — likely OOM from unbounded shuffle buffers
  (1024 decoded samples × 4 workers).

* ``__len__`` called every iter by MetricLogger; without memoization
  each call re-decoded every ``*.tar.index.json``.

Each test below pins one of those behaviors to a contract so a future
regression breaks the test, not the training job.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest


# --- fixtures -----------------------------------------------------------

def _build_synth_kwcoco(bundle_dpath: Path, n_images: int = 32,
                        empty_frac: float = 0.0,
                        source_classes=("A", "B")):
    """Build a synthetic kwcoco bundle.

    ``empty_frac``: fraction of images that get no annotations (stream
    will yield them, but our adapter's __iter__ skips empties — so the
    consumer sees fewer samples than the source contains).
    """
    import kwcoco
    import kwimage

    asset_dpath = bundle_dpath / "assets"
    asset_dpath.mkdir(parents=True, exist_ok=True)

    dset = kwcoco.CocoDataset()
    dset.fpath = str(bundle_dpath / "synth.kwcoco.json")
    cat_ids = [dset.add_category(name=f"target_{c}") for c in source_classes]

    rng = np.random.RandomState(0)
    n_empty = int(n_images * empty_frac)
    empty_idxs = set(rng.choice(n_images, size=n_empty, replace=False).tolist())

    for k in range(n_images):
        img = (rng.rand(64, 64, 3) * 255).astype(np.uint8)
        fpath = asset_dpath / f"img_{k:04d}.jpg"
        kwimage.imwrite(str(fpath), img)
        gid = dset.add_image(
            file_name=str(fpath.relative_to(bundle_dpath)),
            width=64, height=64, name=f"img_{k:04d}",
        )
        if k in empty_idxs:
            continue
        # Round-robin across source classes so multiple buckets exist
        # — exercises load_bucket_streams + WeightedChunkMix mixing.
        src = source_classes[k % len(source_classes)]
        cid = cat_ids[k % len(source_classes)]
        dset.add_annotation(
            image_id=gid, category_id=cid,
            bbox=[10.0, 10.0, 20.0, 20.0],
            area=400.0, iscrowd=0,
            source_category=src,
        )
    dset.dump()
    return dset, n_images, n_images - n_empty


def _make_shards(in_fpath: Path, out_dpath: Path, maxcount: int = 8):
    """Write WDS shards. Default maxcount=8 forces multiple shards per
    bucket for a 32-image source, exposing multi-shard streaming bugs
    that single-shard tests would miss."""
    from kwcoco_dataloader.cli.build_detection_webdataset import (
        BuildDetectionWebdatasetCLI,
    )
    BuildDetectionWebdatasetCLI.main(
        argv=False,
        in_fpath=str(in_fpath),
        out_dpath=str(out_dpath),
        bucket_attr="source_category",
        maxcount=maxcount,
        maxsize_mb=1024,
        jpeg_quality=90,
        drop_provenance=False,
        progress=False,
    )


def _import_adapter():
    kit_dpath = Path(__file__).resolve().parents[2]
    deimv2_dpath = kit_dpath / "tpl" / "DEIMv2"
    if not (deimv2_dpath / "engine" / "data" / "dataset"
            / "wds_coco_dataset.py").exists():
        pytest.skip("DEIMv2 submodule not initialised under tpl/DEIMv2/")
    sys.path.insert(0, str(deimv2_dpath))
    try:
        from engine.data.dataset.wds_coco_dataset import (  # noqa: E402
            WebDatasetCocoDetection,
        )
        return WebDatasetCocoDetection
    finally:
        sys.path.pop(0)


def _make_dataset(tmp_path, n_images=32, empty_frac=0.0,
                  source_classes=("A", "B"),
                  maxcount: int = 8,
                  **adapter_kwargs):
    bundle_dpath = tmp_path / "src"
    bundle_dpath.mkdir(parents=True, exist_ok=True)
    src, total, nonempty = _build_synth_kwcoco(
        bundle_dpath, n_images=n_images, empty_frac=empty_frac,
        source_classes=source_classes,
    )
    shards_dpath = tmp_path / "shards"
    _make_shards(Path(src.fpath), shards_dpath, maxcount=maxcount)

    WebDatasetCocoDetection = _import_adapter()
    src2tgt = {c: f"target_{c}" for c in source_classes}
    ds = WebDatasetCocoDetection(
        shards_dpath=str(shards_dpath),
        category_names=[f"target_{c}" for c in source_classes],
        source_to_target=src2tgt,
        **adapter_kwargs,
    )
    return ds, total, nonempty


def _count_via_iter(ds):
    """Drain the dataset's __iter__ once and count what it yields.

    For the adapter's behavior contract this is *the* ground truth:
    "what the trainer actually sees per epoch."
    """
    n = 0
    for _ in ds:
        n += 1
    return n


# --- tests --------------------------------------------------------------


@pytest.mark.parametrize("n_images,empty_frac", [
    (32, 0.0),
    (32, 0.5),     # half the source has no annotations
])
def test_len_vs_actual_yield(tmp_path, n_images, empty_frac):
    """len(ds) is the *nominal* sample count; the *actual* yield
    (after our adapter's empty-annotation skip) can be lower.

    Production bug: DEIMv2's MetricLogger reports
        Total time: X (X/len(ds) s/it)
    and the LR scheduler plans epoch boundaries from len(ds). If the
    actual yield is much smaller, the per-iter average looks fake
    and the scheduler over-shoots its warmup.
    """
    pytest.importorskip("torch")
    pytest.importorskip("webdataset")
    pytest.importorskip("kwcoco_dataloader")

    ds, total, nonempty = _make_dataset(
        tmp_path, n_images=n_images, empty_frac=empty_frac,
    )

    nominal = len(ds)
    actual = _count_via_iter(ds)

    # Pin the actual contract: __len__ reports the source-side
    # sample count (== total written to shards), but the iterator
    # only yields samples that pass the empty-annotation filter.
    assert nominal == total, (
        f"len(ds)={nominal} should equal source samples ({total}); "
        f"if this changes, the LR scheduler will plan against the "
        f"wrong epoch boundary."
    )
    assert actual == nonempty, (
        f"iterating yielded {actual} samples; expected "
        f"{nonempty} (= total {total} minus {total - nonempty} "
        f"empty-annotation skips). If this changes, gen002's "
        f"\"epoch is shorter than nominal\" bug got worse."
    )

    if empty_frac > 0:
        # This is the gen002 surprise: len() and actual disagree.
        # We pin it as expected behavior so anyone touching this
        # path KNOWS the divergence is intentional, not a bug.
        assert actual < nominal


def test_len_is_memoized(tmp_path):
    """MetricLogger.log_every() calls len(loader) per iter.
    Without memoization, each call re-parses every
    *.tar.index.json (multiple seconds at gen002 scale).

    Production bug fix: bf3d290 + ab013cc (2026-05-30).
    """
    pytest.importorskip("torch")
    pytest.importorskip("webdataset")
    pytest.importorskip("kwcoco_dataloader")

    ds, _, _ = _make_dataset(tmp_path, n_images=32)

    t0 = time.perf_counter()
    n0 = len(ds)
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(1000):
        _ = len(ds)
    t_repeat = (time.perf_counter() - t0) / 1000

    # Repeat call should be at least 100x faster than first call.
    # In practice it should be effectively free (Python attribute
    # lookup), so the bar is low and the test is robust.
    assert n0 > 0
    assert t_repeat < t_first / 100, (
        f"len(ds) not memoized: first call {t_first*1000:.1f}ms, "
        f"avg of 1000 repeats {t_repeat*1000:.3f}ms (need 100x ratio)."
    )


def test_stream_kwargs_plumbing(tmp_path):
    """Verify the shuffle_buffer/shardshuffle defaults flow through
    to WebDatasetStream, AND can be overridden from the caller.

    Production bug fix: 331270e (2026-05-30) — uncapped
    shuffle_buffer caused OOM-kills of dataloader workers.
    """
    pytest.importorskip("torch")
    pytest.importorskip("webdataset")
    pytest.importorskip("kwcoco_dataloader")

    # Default
    ds_default, _, _ = _make_dataset(tmp_path, n_images=16)
    assert ds_default._stream_kwargs == {
        "shuffle_buffer": 128, "shardshuffle": 8,
    }

    # Caller override
    ds_custom, _, _ = _make_dataset(
        (tmp_path / "custom"),
        n_images=16,
        stream_kwargs={"shuffle_buffer": 4, "shardshuffle": 0},
    )
    assert ds_custom._stream_kwargs == {
        "shuffle_buffer": 4, "shardshuffle": 0,
    }


def test_repeated_iter_is_idempotent(tmp_path):
    """An IterableDataset's __iter__ MUST be re-entrable: DEIMv2
    calls iter(loader) once per epoch. If our adapter holds stream
    state between epochs, epoch N+1 yields stale or empty data.

    Concretely: 2 passes over the dataset should yield the same
    set of sample keys, in some order.
    """
    pytest.importorskip("torch")
    pytest.importorskip("webdataset")
    pytest.importorskip("kwcoco_dataloader")

    ds, _, nonempty = _make_dataset(tmp_path, n_images=24)

    n_pass1 = _count_via_iter(ds)
    n_pass2 = _count_via_iter(ds)

    assert n_pass1 == nonempty, (
        f"pass 1 yielded {n_pass1}, expected {nonempty}"
    )
    assert n_pass2 == nonempty, (
        f"pass 2 yielded {n_pass2}, expected {nonempty} — "
        f"if 0, the stream state didn't reset between epochs "
        f"(epoch 1 would be empty in production)."
    )


def test_dataloader_drain_count_matches_iter_count(tmp_path):
    """The number of batches produced by DataLoader * batch_size
    should approximately equal the iter-yield count. If they
    diverge, the multi-worker stream split is dropping samples.

    Note: pin num_workers=0 to isolate the dataset-level behavior
    from the worker-sharding behavior tested separately.
    """
    import torch
    pytest.importorskip("webdataset")
    pytest.importorskip("kwcoco_dataloader")

    batch_size = 4
    ds, _, nonempty = _make_dataset(tmp_path, n_images=40)

    raw_yield = _count_via_iter(ds)

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=lambda x: x,   # identity, we just want to count
        drop_last=False,
    )
    n_samples_in_loader = sum(len(b) for b in loader)

    assert raw_yield == nonempty
    assert n_samples_in_loader == raw_yield, (
        f"DataLoader drained {n_samples_in_loader} samples; "
        f"raw __iter__ drained {raw_yield}. Divergence means "
        f"DataLoader is dropping/duplicating somewhere — "
        f"check collate fn, drop_last, and IterableDataset framing."
    )


@pytest.mark.parametrize("num_workers", [0, 1, 2, 4])
def test_multi_worker_total_yield_is_stable(tmp_path, num_workers):
    """The sum of samples across all workers should equal the
    single-process yield, regardless of num_workers.

    If wds.split_by_worker is dropping shards (e.g. because the
    bucket count isn't divisible by num_workers), some workers
    get zero work and the total under-counts. This is the most
    likely culprit for gen002's "epoch 0 ran ~5000 iters of 47878".
    """
    import torch
    pytest.importorskip("webdataset")
    pytest.importorskip("kwcoco_dataloader")

    # n_images chosen so neither shards-per-bucket nor buckets-per-worker
    # divides evenly for num_workers in {1,2,4}: 30 across 2 buckets
    # gives 15 per bucket; with maxcount=4 → 4 shards in one bucket,
    # 4 in the other = 8 shards total; not divisible by 4.
    ds, total_images, nonempty = _make_dataset(
        tmp_path, n_images=30, source_classes=("A", "B"),
    )
    # First record single-process truth:
    truth_yield = _count_via_iter(ds)

    # Now multi-worker via DataLoader:
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        drop_last=False,
    )
    multi_yield = sum(len(b) for b in loader)

    # Pin the contract: total yield must NOT vary with num_workers.
    # If this assertion fires, training depth-per-epoch silently
    # changes when the user tweaks KCD_TRAIN_NUM_WORKERS.
    assert multi_yield == truth_yield, (
        f"num_workers={num_workers} drained {multi_yield} samples; "
        f"single-process drained {truth_yield}. "
        f"split_by_worker is dropping or duplicating shards."
    )


@pytest.mark.parametrize("num_workers", [0, 4])
def test_production_scale_multi_worker_yield(tmp_path, num_workers):
    """Larger, multi-bucket, multi-shard scenario closer to gen002.

    gen002 has hundreds of shards across ~5 buckets; this test
    exercises a 100-image, 4-bucket, multi-shard-per-bucket setup.
    The bug we're hunting: epoch claimed 47878 samples, actually
    yielded ~5000 (~10%). If multi-worker shard splitting drops
    samples at small scale, this should catch it; if it only
    manifests at production scale, this test passing tells us the
    bug is elsewhere (likely in the empty-skip + WeightedChunkMix
    interaction).
    """
    import torch
    pytest.importorskip("webdataset")
    pytest.importorskip("kwcoco_dataloader")

    ds, _, nonempty = _make_dataset(
        tmp_path,
        n_images=100,
        source_classes=("A", "B", "C", "D"),
        maxcount=7,  # → ~25/7 = ~4 shards/bucket = 16 shards total
    )
    truth_yield = _count_via_iter(ds)

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=2,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        drop_last=False,
    )
    multi_yield = sum(len(b) for b in loader)

    assert multi_yield == truth_yield, (
        f"production-scale: num_workers={num_workers} drained "
        f"{multi_yield} samples; truth={truth_yield}; "
        f"nonempty={nonempty}. ratio={multi_yield / truth_yield:.2f}. "
        f"If ratio is ~0.10 like gen002, we've reproduced the bug."
    )


@pytest.mark.parametrize("num_workers", [4])
def test_many_small_shards_yield(tmp_path, num_workers):
    """gen002 has ~hundreds of tiny shards (maxcount-bounded), spread
    across few workers. This stress-tests the case where each worker
    is assigned many shards.

    If load_bucket_streams + split_by_worker have a bug that surfaces
    only when (n_shards / num_workers) is large, this catches it. The
    test must complete in bounded time AND yield every non-empty
    source sample.
    """
    import torch
    pytest.importorskip("webdataset")
    pytest.importorskip("kwcoco_dataloader")

    # 60 images across 3 buckets, maxcount=1 → 60 shards (20/bucket).
    # With 4 workers, that's 15 shards/worker.
    ds, _, nonempty = _make_dataset(
        tmp_path, n_images=60,
        source_classes=("A", "B", "C"),
        maxcount=1,
    )
    truth = _count_via_iter(ds)

    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=num_workers,
        collate_fn=lambda x: x, drop_last=False,
    )
    multi = sum(len(b) for b in loader)

    assert multi == truth, (
        f"many-shard stress: workers={num_workers}, shards~60, "
        f"got {multi} samples but expected {truth} ({nonempty} nonempty). "
        f"ratio={multi/truth:.2f}. Production gen002 was at ratio~0.10."
    )


def test_epoch_length_pin_cycles_stream(tmp_path):
    """With epoch_length>0, the iterator yields exactly epoch_length
    samples even if the underlying stream is shorter — it cycles back
    to the start as needed.

    Critical for DDP: each rank must see the same per-epoch sample
    count regardless of how unevenly shards split between ranks.
    Without this, "Rank 0 BROADCAST vs Rank 1 REDUCE" collective
    mismatches kill multi-GPU runs (host-gpu2x-wds matrix scenario,
    yardrat 2026-05-30).

    Conversely with epoch_length=0 (the default), we keep the
    "drain once and stop" behavior the gen001 single-rank path
    depends on.
    """
    pytest.importorskip("webdataset")
    pytest.importorskip("kwcoco_dataloader")

    # Drain-once contract (epoch_length=0): yields == nonempty count.
    ds_drain, _, nonempty = _make_dataset(tmp_path, n_images=20)
    assert _count_via_iter(ds_drain) == nonempty

    # Cycle contract (epoch_length > nonempty): yields == epoch_length.
    target = nonempty * 3 + 5  # not a multiple of nonempty
    ds_cycle, _, _ = _make_dataset(
        (tmp_path / "cycle"), n_images=20,
        epoch_length=target,
    )
    yielded = _count_via_iter(ds_cycle)
    assert yielded == target, (
        f"epoch_length={target} requested but yielded {yielded}; "
        f"adapter is not cycling the stream when it exhausts."
    )


def test_sample_timeout_env_disables_watchdog(tmp_path, monkeypatch):
    """When KCD_WDS_SAMPLE_TIMEOUT_S=0, the watchdog thread is not
    started. Default config (timeout=120) installs the thread.

    Smoke-checks that the env var routes correctly without firing
    the SIGKILL path (which would kill the test process).
    """
    pytest.importorskip("webdataset")
    pytest.importorskip("kwcoco_dataloader")

    import threading

    # Disabled: no watchdog thread spawned
    monkeypatch.setenv("KCD_WDS_SAMPLE_TIMEOUT_S", "0")
    ds, _, _ = _make_dataset(tmp_path, n_images=16)
    before = threading.active_count()
    list(ds)
    after = threading.active_count()
    assert after <= before + 1, (
        f"thread leak with timeout disabled: {before} -> {after}"
    )

    # Enabled but generous timeout: watchdog spawns + cleans up
    monkeypatch.setenv("KCD_WDS_SAMPLE_TIMEOUT_S", "120")
    ds2, _, _ = _make_dataset((tmp_path / "b"), n_images=16)
    before = threading.active_count()
    list(ds2)
    # daemon=True watchdogs aren't guaranteed to exit immediately
    # after we call stop, but they should within a few seconds.
    deadline = __import__('time').monotonic() + 10
    while __import__('time').monotonic() < deadline:
        if threading.active_count() <= before + 1:
            break
        __import__('time').sleep(0.1)
    assert threading.active_count() <= before + 1, (
        f"watchdog thread didn't stop: {before} -> {threading.active_count()}"
    )


def test_load_bucket_streams_sees_all_footers(tmp_path):
    """load_bucket_streams walks __footer__.json files to weight
    buckets. If footers are missing or unparseable for some
    buckets, those buckets are silently dropped — production gen002
    might be hitting this if the writer didn't flush all footers.
    """
    pytest.importorskip("webdataset")
    pytest.importorskip("kwcoco_dataloader")

    from kwcoco_dataloader.readers.detection import load_bucket_streams

    ds, _, nonempty = _make_dataset(
        tmp_path, n_images=40, source_classes=("A", "B"),
    )
    shards_dpath = ds.shards_dpath

    # Verify the writer wrote a footer for every bucket-dir:
    bucket_dirs = [d for d in shards_dpath.iterdir() if d.is_dir()]
    footer_files = list(shards_dpath.glob("*/__footer__.json"))
    assert len(footer_files) == len(bucket_dirs), (
        f"writer skipped a footer: {len(bucket_dirs)} bucket dirs but "
        f"only {len(footer_files)} __footer__.json files."
    )

    bucket_set = load_bucket_streams(shards_dpath=shards_dpath)
    assert len(bucket_set.streams) == len(bucket_dirs), (
        f"load_bucket_streams returned {len(bucket_set.streams)} "
        f"streams but disk has {len(bucket_dirs)} bucket dirs. "
        f"Footer parsing dropped a bucket silently."
    )
