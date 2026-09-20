from __future__ import annotations

import numpy as np


class _FakeDelayed:
    def __init__(self, arr, counter, space_slice=None):
        self.arr = arr
        self.counter = counter
        self.space_slice = space_slice

    def finalize(self):
        if self.space_slice is None:
            self.counter["full"] += 1
            return self.arr.copy()
        self.counter["region"] += 1
        return self.arr[self.space_slice].copy()

    def crop(self, space_slice):
        return _FakeDelayed(self.arr, self.counter, space_slice)


class _FakeCocoImage:
    def __init__(self, arr, suffix):
        self.arr = arr
        self.counter = {"full": 0, "region": 0}
        self.img = {
            "file_name": f"fake{suffix}",
            "height": arr.shape[0],
            "width": arr.shape[1],
        }

    def imdelay(self):
        return _FakeDelayed(self.arr, self.counter)


def test_source_window_reader_decode_once_and_region_equivalence():
    from kwcoco_detector_kit.predictors.source_window import SourceWindowReader

    arr = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    jpg = _FakeCocoImage(arr, ".jpg")
    r1 = SourceWindowReader(jpg, strategy="auto")
    a = r1.read_window(0, 0, 32, 32)
    b = r1.read_window(16, 16, 48, 48)
    assert r1.strategy == "decode_once"
    assert jpg.counter == {"full": 1, "region": 0}

    tif = _FakeCocoImage(arr, ".tif")
    r2 = SourceWindowReader(tif, strategy="auto")
    a2 = r2.read_window(0, 0, 32, 32)
    b2 = r2.read_window(16, 16, 48, 48)
    assert r2.strategy == "delayed_region"
    assert tif.counter == {"full": 0, "region": 2}
    assert np.array_equal(a, a2)
    assert np.array_equal(b, b2)
