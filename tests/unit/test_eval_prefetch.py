"""Unit tests for the threaded image prefetcher used by the eval loop."""
from kwcoco_detector_kit.eval.kwcoco_eval import _iter_prefetched


def test_order_preserved_with_workers():
    items = list(range(50))
    out = list(_iter_prefetched(items, read_fn=lambda x: x * 10, workers=4))
    assert [i for i, _ in out] == items
    assert [v for _, v in out] == [i * 10 for i in items]


def test_sequential_fallback():
    items = list(range(5))
    out = list(_iter_prefetched(items, read_fn=lambda x: x + 1, workers=0))
    assert out == [(i, i + 1) for i in items]


def test_read_failure_is_returned_not_raised():
    # read_fn returns the exception object (mirrors kwcoco_eval._read), so the
    # consumer can skip a bad image instead of the whole eval aborting.
    def read(x):
        if x == 3:
            return ValueError("boom")
        return x
    out = dict(_iter_prefetched(range(5), read_fn=read, workers=3))
    assert isinstance(out[3], ValueError)
    assert out[0] == 0 and out[4] == 4


def test_empty():
    assert list(_iter_prefetched([], read_fn=lambda x: x, workers=4)) == []
