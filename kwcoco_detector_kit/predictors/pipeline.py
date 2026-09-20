"""Bounded concurrency primitives for source-space prediction.

The prediction pipeline deliberately keeps CUDA ownership on the calling
thread.  Background threads are used only for source/window realization and
CPU postprocessing.  This avoids duplicating models or shuttling large NumPy
arrays through multiprocessing IPC while still overlapping the three stages
that dominate source-space detector inference::

    source/window I/O  ->  GPU inference  ->  CPU merge/postprocess

All queues are bounded and results are committed in input order.  The latter
is important for deterministic KWCoco output and for surfacing worker errors
at a predictable source-image boundary.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Any, Callable, Deque, Iterable, Iterator, Optional, Tuple


@dataclass
class StageStats:
    """Timing and backpressure counters for one asynchronous pipeline stage."""

    submitted: int = 0
    completed: int = 0
    work_seconds: float = 0.0
    wait_seconds: float = 0.0
    max_pending: int = 0

    def note_submit(self, pending: int) -> None:
        self.submitted += 1
        self.max_pending = max(self.max_pending, int(pending))

    def note_complete(self, work_seconds: float, wait_seconds: float) -> None:
        self.completed += 1
        self.work_seconds += float(work_seconds)
        self.wait_seconds += float(wait_seconds)

    def as_dict(self) -> dict[str, Any]:
        return {
            "submitted": int(self.submitted),
            "completed": int(self.completed),
            "work_seconds": float(self.work_seconds),
            "wait_seconds": float(self.wait_seconds),
            "max_pending": int(self.max_pending),
        }


def _timed_call(func: Callable, *args, **kwargs):
    started = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - started


class PredictionPipeline:
    """Own the bounded worker pools used by one prediction pass.

    Args:
        source_workers:
            Threads allowed to prepare upcoming source images.  Source
            preparation may decode JPEG-like imagery once, but never touches
            CUDA.
        source_prefetch:
            Number of *future* source images to keep queued beyond the source
            currently consumed by the main thread.
        window_prefetch:
            Number of future window batches to realize while the main thread
            runs detector inference on the current batch.  A single worker is
            intentionally used here so one source reader is never accessed by
            multiple window-read threads concurrently.
        postprocess_workers:
            CPU workers for merge/NMS/mask polygonization.  CUDA-backed
            segmenters should not be submitted to this stage.
        postprocess_inflight:
            Maximum number of completed-GPU source results retained for CPU
            postprocessing at once.  This is a memory/backpressure bound, not
            an unbounded writer queue.
    """

    def __init__(
        self,
        *,
        source_workers: int = 2,
        source_prefetch: int = 2,
        window_prefetch: int = 2,
        postprocess_workers: int = 1,
        postprocess_inflight: int = 2,
    ):
        self.source_workers = max(0, int(source_workers))
        self.source_prefetch = max(0, int(source_prefetch))
        self.window_prefetch = max(0, int(window_prefetch))
        self.postprocess_workers = max(0, int(postprocess_workers))
        self.postprocess_inflight = max(1, int(postprocess_inflight))

        self.source_stats = StageStats()
        self.window_stats = StageStats()
        self.postprocess_stats = StageStats()

        self._source_pool: Optional[ThreadPoolExecutor] = None
        self._window_pool: Optional[ThreadPoolExecutor] = None
        self._post_pool: Optional[ThreadPoolExecutor] = None
        self._post_pending: Deque[Tuple[Any, Future]] = deque()
        self._entered = False

    def __enter__(self):
        if self._entered:
            raise RuntimeError("PredictionPipeline cannot be entered twice")
        self._entered = True
        if self.source_workers > 0 and self.source_prefetch > 0:
            self._source_pool = ThreadPoolExecutor(
                max_workers=self.source_workers,
                thread_name_prefix="kdk-source",
            )
        if self.window_prefetch > 0:
            # Deliberately one producer for one SourceWindowReader.  The
            # expensive work generally releases the GIL (GDAL/OpenCV/NumPy),
            # and a single producer avoids backend-specific thread-safety
            # assumptions while still overlapping reads with CUDA.
            self._window_pool = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="kdk-window",
            )
        if self.postprocess_workers > 0:
            self._post_pool = ThreadPoolExecutor(
                max_workers=self.postprocess_workers,
                thread_name_prefix="kdk-post",
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        # Do not silently discard failures from already-submitted work.  The
        # caller normally drains postprocessing explicitly; on an exception we
        # cancel work that has not started and wait for running calls to leave
        # shared resources in a clean state.
        for pool in [self._post_pool, self._window_pool, self._source_pool]:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=exc_type is not None)
        self._post_pool = None
        self._window_pool = None
        self._source_pool = None
        self._entered = False
        return False

    @property
    def async_postprocess(self) -> bool:
        return self._post_pool is not None

    def _ordered_prefetch(
        self,
        items: Iterable,
        func: Callable,
        *,
        pool: Optional[ThreadPoolExecutor],
        ahead: int,
        stats: StageStats,
    ) -> Iterator[tuple[Any, Any]]:
        """Map ``func`` ahead of the consumer while preserving item order."""
        if pool is None or ahead <= 0:
            for item in items:
                stats.note_submit(1)
                started = time.perf_counter()
                result = func(item)
                work = time.perf_counter() - started
                stats.note_complete(work, 0.0)
                yield item, result
            return

        # ``ahead`` means future items beyond the one about to be consumed.
        capacity = max(1, int(ahead) + 1)
        source = iter(items)
        pending: Deque[Tuple[Any, Future]] = deque()

        def submit_one(item) -> None:
            future = pool.submit(_timed_call, func, item)
            pending.append((item, future))
            stats.note_submit(len(pending))

        for _ in range(capacity):
            try:
                submit_one(next(source))
            except StopIteration:
                break

        while pending:
            item, future = pending.popleft()
            wait_started = time.perf_counter()
            result, work = future.result()
            wait = time.perf_counter() - wait_started
            stats.note_complete(work, wait)
            yield item, result
            # Refill only after the consumer asks for the next result. While
            # it works, exactly ``ahead`` future items remain queued beyond
            # the item it currently owns, keeping the memory bound literal.
            try:
                submit_one(next(source))
            except StopIteration:
                pass

    def iter_prepared_sources(self, items: Iterable, prepare_fn: Callable):
        """Yield source objects in order while preparing later sources ahead."""
        yield from self._ordered_prefetch(
            items,
            prepare_fn,
            pool=self._source_pool,
            ahead=self.source_prefetch,
            stats=self.source_stats,
        )

    def iter_window_batches(self, items: Iterable, read_fn: Callable):
        """Yield realized window batches in order while reading future batches."""
        yield from self._ordered_prefetch(
            items,
            read_fn,
            pool=self._window_pool,
            ahead=self.window_prefetch,
            stats=self.window_stats,
        )

    def _resolve_post_future(self, key, future: Future, *, block: bool):
        if not block and not future.done():
            return None
        wait_started = time.perf_counter()
        result, work = future.result()
        wait = time.perf_counter() - wait_started
        self.postprocess_stats.note_complete(work, wait)
        return key, result

    def drain_postprocess_ready(self) -> list[tuple[Any, Any]]:
        """Return the completed in-order prefix without blocking."""
        ready = []
        while self._post_pending:
            key, future = self._post_pending[0]
            resolved = self._resolve_post_future(key, future, block=False)
            if resolved is None:
                break
            self._post_pending.popleft()
            ready.append(resolved)
        return ready

    def wait_for_postprocess_capacity(self, reserve: int = 1) -> list[tuple[Any, Any]]:
        """Apply backpressure before another GPU result is produced.

        ``reserve=1`` ensures there is room to submit the source about to be
        inferred without temporarily exceeding the configured memory bound.
        Results are returned in deterministic source order.
        """
        if self._post_pool is None:
            return []
        ready = self.drain_postprocess_ready()
        while len(self._post_pending) + int(reserve) > self.postprocess_inflight:
            key, future = self._post_pending.popleft()
            resolved = self._resolve_post_future(key, future, block=True)
            assert resolved is not None
            ready.append(resolved)
            ready.extend(self.drain_postprocess_ready())
        return ready

    def submit_postprocess(self, key, func: Callable, *args, **kwargs) -> None:
        """Submit CPU finalization without allowing an unbounded queue."""
        if self._post_pool is None:
            raise RuntimeError("postprocess pool is disabled")
        if len(self._post_pending) >= self.postprocess_inflight:
            raise RuntimeError(
                "postprocess queue is full; call wait_for_postprocess_capacity() "
                "before producing another GPU result"
            )
        future = self._post_pool.submit(_timed_call, func, *args, **kwargs)
        self._post_pending.append((key, future))
        self.postprocess_stats.note_submit(len(self._post_pending))

    def finish_postprocess(self) -> list[tuple[Any, Any]]:
        """Block until all CPU finalizers finish, preserving submit order."""
        finished = []
        while self._post_pending:
            key, future = self._post_pending.popleft()
            resolved = self._resolve_post_future(key, future, block=True)
            assert resolved is not None
            finished.append(resolved)
        return finished

    def profile_dict(self) -> dict[str, Any]:
        """Serializable pipeline configuration and stage statistics."""
        return {
            "config": {
                "source_workers": self.source_workers,
                "source_prefetch": self.source_prefetch,
                "window_prefetch": self.window_prefetch,
                "postprocess_workers": self.postprocess_workers,
                "postprocess_inflight": self.postprocess_inflight,
            },
            "source_prepare": self.source_stats.as_dict(),
            "window_read": self.window_stats.as_dict(),
            "postprocess": self.postprocess_stats.as_dict(),
        }
