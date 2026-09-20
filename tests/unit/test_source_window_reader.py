from __future__ import annotations

import numpy as np


class _FakeDelayed:
    def __init__(self, arr, counter, space_slice=None, scale_xy=(1.0, 1.0)):
        self.arr = arr
        self.counter = counter
        self.space_slice = space_slice
        self.scale_xy = scale_xy

    def _scaled(self):
        sx, sy = self.scale_xy
        if sx == 1.0 and sy == 1.0:
            return self.arr
        src_h, src_w = self.arr.shape[:2]
        out_w = max(1, int(round(src_w * sx)))
        out_h = max(1, int(round(src_h * sy)))
        ys = np.minimum((np.arange(out_h) / sy).astype(int), src_h - 1)
        xs = np.minimum((np.arange(out_w) / sx).astype(int), src_w - 1)
        return self.arr[ys[:, None], xs[None, :]]

    def finalize(self):
        scaled = self._scaled()
        if self.space_slice is None:
            self.counter["full"] += 1
            return scaled.copy()
        self.counter["region"] += 1
        return scaled[self.space_slice].copy()

    def crop(self, space_slice):
        return _FakeDelayed(
            self.arr, self.counter, space_slice, scale_xy=self.scale_xy
        )

    def scale(self, factor):
        self.counter["scale"] += 1
        if isinstance(factor, tuple):
            sx, sy = map(float, factor)
        else:
            sx = sy = float(factor)
        return _FakeDelayed(
            self.arr, self.counter, self.space_slice, scale_xy=(sx, sy)
        )


class _FakeCocoImage:
    def __init__(self, arr, suffix):
        self.arr = arr
        self.counter = {"imdelay": 0, "scale": 0, "full": 0, "region": 0}
        self.img = {
            "file_name": f"fake{suffix}",
            "height": arr.shape[0],
            "width": arr.shape[1],
        }

    def imdelay(self):
        self.counter["imdelay"] += 1
        return _FakeDelayed(self.arr, self.counter)


def test_prediction_space_roundtrip_geometry():
    from kwcoco_detector_kit.predictors.space import PredictionSpace

    space = PredictionSpace.from_scale((101, 203), 0.4)
    assert space.native_hw == (101, 203)
    assert space.prediction_hw == (40, 81)
    box = [3.5, 4.5, 20.0, 25.0]
    native = space.box_to_native_xyxy(box)
    roundtrip = space.box_to_prediction_xyxy(native)
    assert np.allclose(roundtrip, box)


def test_source_window_reader_decode_once_and_region_equivalence():
    from kwcoco_detector_kit.predictors.source_window import SourceWindowReader

    arr = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    jpg = _FakeCocoImage(arr, ".jpg")
    r1 = SourceWindowReader(jpg, strategy="auto")
    a = r1.read_window(0, 0, 32, 32)
    b = r1.read_window(16, 16, 48, 48)
    assert r1.strategy == "decode_once"
    assert jpg.counter == {"imdelay": 1, "scale": 0, "full": 1, "region": 0}

    tif = _FakeCocoImage(arr, ".tif")
    r2 = SourceWindowReader(tif, strategy="auto")
    a2 = r2.read_window(0, 0, 32, 32)
    b2 = r2.read_window(16, 16, 48, 48)
    assert r2.strategy == "delayed_region"
    assert tif.counter == {"imdelay": 1, "scale": 0, "full": 0, "region": 2}
    assert np.array_equal(a, a2)
    assert np.array_equal(b, b2)


def test_prepare_decodes_decode_once_but_keeps_region_source_lazy():
    from kwcoco_detector_kit.predictors.source_window import SourceWindowReader

    arr = np.zeros((32, 48, 3), dtype=np.uint8)

    jpg = _FakeCocoImage(arr, ".jpg")
    jpg_reader = SourceWindowReader(jpg, strategy="auto")
    jpg_reader.prepare()
    assert jpg.counter == {"imdelay": 1, "scale": 0, "full": 1, "region": 0}
    jpg_reader.read_window(0, 0, 16, 16)
    assert jpg.counter == {"imdelay": 1, "scale": 0, "full": 1, "region": 0}

    tif = _FakeCocoImage(arr, ".tif")
    tif_reader = SourceWindowReader(tif, strategy="auto")
    tif_reader.prepare()
    assert tif.counter == {"imdelay": 0, "scale": 0, "full": 0, "region": 0}
    tif_reader.read_window(0, 0, 16, 16)
    assert tif.counter == {"imdelay": 1, "scale": 0, "full": 0, "region": 1}


def test_prediction_scale_is_applied_before_delayed_region_crop():
    from kwcoco_detector_kit.predictors.source_window import SourceWindowReader

    arr = np.arange(64 * 80 * 3, dtype=np.uint8).reshape(64, 80, 3)
    tif = _FakeCocoImage(arr, ".tif")
    reader = SourceWindowReader(tif, strategy="auto", prediction_scale=0.5)

    assert reader.source_hw == (64, 80)
    assert reader.prediction_hw == (32, 40)
    crop = reader.read_window(4, 3, 20, 19)

    expected_scaled = arr[::2, ::2]
    assert np.array_equal(crop, expected_scaled[3:19, 4:20])
    # The scale node is part of the delayed graph and the crop is finalized
    # without forcing a full prediction-space image.
    assert tif.counter == {"imdelay": 1, "scale": 1, "full": 0, "region": 1}
    assert reader.full_decode_count == 0
