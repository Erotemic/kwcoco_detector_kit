from __future__ import annotations

import threading
import time


def test_source_prefetch_runs_future_prepare_while_consumer_works():
    from kwcoco_detector_kit.predictors.pipeline import PredictionPipeline

    second_started = threading.Event()

    def prepare(item):
        if item == 1:
            second_started.set()
        return item * 10

    with PredictionPipeline(
        source_workers=1,
        source_prefetch=1,
        window_prefetch=0,
        postprocess_workers=0,
    ) as pipeline:
        iterator = pipeline.iter_prepared_sources([0, 1, 2], prepare)
        item, result = next(iterator)
        assert (item, result) == (0, 0)
        assert second_started.wait(timeout=1.0)
        assert list(iterator) == [(1, 10), (2, 20)]
        stats = pipeline.profile_dict()["source_prepare"]
        assert stats["submitted"] == 3
        assert stats["completed"] == 3
        assert stats["max_pending"] <= 2


def test_postprocess_pool_is_ordered_and_bounded():
    from kwcoco_detector_kit.predictors.pipeline import PredictionPipeline

    release_first = threading.Event()
    second_done = threading.Event()

    def first():
        assert release_first.wait(timeout=2.0)
        return "first"

    def second():
        second_done.set()
        return "second"

    with PredictionPipeline(
        source_workers=0,
        source_prefetch=0,
        window_prefetch=0,
        postprocess_workers=2,
        postprocess_inflight=2,
    ) as pipeline:
        pipeline.submit_postprocess(0, first)
        pipeline.submit_postprocess(1, second)
        assert second_done.wait(timeout=1.0)
        # Source order is the commit contract: a fast later result cannot jump
        # ahead of an unfinished earlier source.
        assert pipeline.drain_postprocess_ready() == []
        release_first.set()
        assert pipeline.finish_postprocess() == [
            (0, "first"),
            (1, "second"),
        ]
        stats = pipeline.profile_dict()["postprocess"]
        assert stats["submitted"] == 2
        assert stats["completed"] == 2
        assert stats["max_pending"] == 2


def test_wait_for_postprocess_capacity_applies_backpressure():
    from kwcoco_detector_kit.predictors.pipeline import PredictionPipeline

    release = threading.Event()

    def blocked(value):
        assert release.wait(timeout=2.0)
        return value

    with PredictionPipeline(
        source_workers=0,
        source_prefetch=0,
        window_prefetch=0,
        postprocess_workers=1,
        postprocess_inflight=1,
    ) as pipeline:
        pipeline.submit_postprocess("a", blocked, 1)

        def release_soon():
            time.sleep(0.05)
            release.set()

        thread = threading.Thread(target=release_soon)
        thread.start()
        ready = pipeline.wait_for_postprocess_capacity(reserve=1)
        thread.join()
        assert ready == [("a", 1)]
        assert pipeline.profile_dict()["postprocess"]["wait_seconds"] > 0
