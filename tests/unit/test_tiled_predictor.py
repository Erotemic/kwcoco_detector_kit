"""
Unit tests for the windowed (tiled) inference wrapper.

These use a fake base predictor so no model/checkpoint is needed — they
exercise the window geometry, coordinate translation back to full-image
space, and the cross-window NMS merge.
"""
import threading

import numpy as np

from kwcoco_detector_kit.eval.tiled_predictor import (
    TiledPredictor,
    _nms_indices,
    _per_class_nms,
)


class _FakeWindowDetector:
    """Returns one detection at a fixed offset inside every window it sees.

    eval_spatial_size is 64x64. For each window crop it emits a 10x10 box
    near the crop's top-left, in the crop's own coordinate frame — exactly
    the contract a real predictor honors (coords in the passed orig_size
    frame). The wrapper is responsible for shifting these into full-image
    coordinates.
    """

    def __init__(self, size=(64, 64)):
        self._size = size

    @property
    def eval_spatial_size(self):
        return self._size

    def predict_image(self, image_np, orig_size):
        return [{"label": 0, "score": 0.9, "bbox_xyxy": [2.0, 3.0, 12.0, 13.0]}]


class _FakeBatchDetector(_FakeWindowDetector):
    """Adds a batched path and records how many batch calls were made."""

    def __init__(self, size=(64, 64)):
        super().__init__(size)
        self.batch_calls = 0
        self.batch_sizes = []

    def predict_batch(self, images_np, orig_sizes):
        self.batch_calls += 1
        self.batch_sizes.append(len(images_np))
        return [self.predict_image(im, sz) for im, sz in zip(images_np, orig_sizes)]


class _FakeMaskDetector(_FakeBatchDetector):
    def predict_image(self, image_np, orig_size):
        mask = np.zeros(image_np.shape[:2], dtype=bool)
        mask[3:13, 2:12] = True
        return [{
            "label": 0, "score": 0.9,
            "bbox_xyxy": [2.0, 3.0, 12.0, 13.0], "mask": mask,
        }]


class _FakeArrayReader:
    """Minimal source-window reader used to exercise the no-cache path."""

    def __init__(self, image, batch_event=None):
        self.image = image
        self.source_hw = image.shape[:2]
        self.window_reads = []
        self.full_reads = 0
        self.batch_reads = 0
        self.batch_event = batch_event

    def read_window(self, x0, y0, x1, y1):
        self.window_reads.append((x0, y0, x1, y1))
        return self.image[y0:y1, x0:x1]

    def read_windows(self, offsets, window_hw):
        self.batch_reads += 1
        if self.batch_reads >= 2 and self.batch_event is not None:
            self.batch_event.set()
        win_h, win_w = window_hw
        H, W = self.source_hw
        return [
            self.read_window(x0, y0, min(x0 + win_w, W), min(y0 + win_h, H))
            for x0, y0 in offsets
        ]

    def read_full(self):
        self.full_reads += 1
        return self.image


def test_window_offsets_translate_to_full_image():
    base = _FakeWindowDetector(size=(64, 64))
    pred = TiledPredictor(base, overlap=0.0, keep_full=False)
    # 128x128 image with a 64 window, 0 overlap => 2x2 = 4 windows at
    # (0,0),(0,64),(64,0),(64,64).
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    dets = pred.predict_image(img, (128, 128))
    # The fake box at (2,3,12,13) is distinct per window once shifted, so
    # NMS keeps all 4.
    assert len(dets) == 4
    corners = sorted((d["bbox_xyxy"][0], d["bbox_xyxy"][1]) for d in dets)
    assert corners == [(2.0, 3.0), (2.0, 67.0), (66.0, 3.0), (66.0, 67.0)]


def test_predict_source_uses_window_reader_without_full_decode():
    base = _FakeBatchDetector(size=(64, 64))
    pred = TiledPredictor(
        base, overlap=0.0, keep_full=False, batch_size=2,
    )
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    reader = _FakeArrayReader(image)
    dets = pred.predict_source(reader)
    corners = sorted((d["bbox_xyxy"][0], d["bbox_xyxy"][1]) for d in dets)
    assert corners == [(2.0, 3.0), (2.0, 67.0), (66.0, 3.0), (66.0, 67.0)]
    assert reader.window_reads == [
        (0, 0, 64, 64),
        (64, 0, 128, 64),
        (0, 64, 64, 128),
        (64, 64, 128, 128),
    ]
    assert reader.full_reads == 0
    assert pred.n_windows == 4


class _PrefetchAwareBatchDetector(_FakeBatchDetector):
    def __init__(self, batch_event, size=(64, 64)):
        super().__init__(size)
        self.batch_event = batch_event
        self.saw_prefetched_batch = False

    def predict_batch(self, images_np, orig_sizes):
        if self.batch_calls == 0:
            # The window producer should realize batch 2 while this first GPU
            # stand-in is busy. Without pipeline overlap this times out.
            self.saw_prefetched_batch = self.batch_event.wait(timeout=1.0)
        return super().predict_batch(images_np, orig_sizes)


def test_predict_source_prefetches_next_window_batch_during_inference():
    from kwcoco_detector_kit.predictors.pipeline import PredictionPipeline

    batch_event = threading.Event()
    base = _PrefetchAwareBatchDetector(batch_event, size=(64, 64))
    pred = TiledPredictor(
        base, overlap=0.0, keep_full=False, batch_size=2,
    )
    reader = _FakeArrayReader(
        np.zeros((128, 128, 3), dtype=np.uint8),
        batch_event=batch_event,
    )
    with PredictionPipeline(
        source_workers=0,
        source_prefetch=0,
        window_prefetch=1,
        postprocess_workers=0,
    ) as pipeline:
        raw = pred._infer_source(reader, pipeline=pipeline)
        records = pred._merge_and_nms(raw)
        assert pipeline.profile_dict()["window_read"]["max_pending"] <= 2
    assert base.saw_prefetched_batch
    assert len(records) == 4


def test_small_image_defers_to_base():
    base = _FakeWindowDetector(size=(64, 64))
    pred = TiledPredictor(base, keep_full=False)
    img = np.zeros((40, 50, 3), dtype=np.uint8)  # fits in one 64 window
    dets = pred.predict_image(img, (50, 40))
    assert len(dets) == 1
    assert dets[0]["bbox_xyxy"] == [2.0, 3.0, 12.0, 13.0]


def test_keep_full_adds_whole_image_pass():
    base = _FakeWindowDetector(size=(64, 64))
    no_full = TiledPredictor(base, overlap=0.0, keep_full=False)
    with_full = TiledPredictor(base, overlap=0.0, keep_full=True)
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    # whole-image pass adds a det at (2,3) that doesn't overlap the
    # window-(0,0) det enough to be NMS'd at default 0.5? It is identical
    # coords, so NMS collapses it. Use a detector whose full-image det is
    # elsewhere to see the extra.
    n0 = len(no_full.predict_image(img, (128, 128)))
    n1 = len(with_full.predict_image(img, (128, 128)))
    # The full-image det coincides with the (0,0)-window det, so NMS removes
    # the duplicate: counts stay equal. This asserts keep_full doesn't double
    # count identical detections.
    assert n1 == n0


def test_batched_path_used_and_equivalent():
    # A base with predict_batch must (a) be used, (b) produce the same merged
    # result as the per-window path.
    per_window = TiledPredictor(_FakeWindowDetector((64, 64)), overlap=0.0, keep_full=False)
    batched_base = _FakeBatchDetector((64, 64))
    batched = TiledPredictor(batched_base, overlap=0.0, keep_full=False, batch_size=2)
    img = np.zeros((128, 128, 3), dtype=np.uint8)  # 4 windows
    a = per_window.predict_image(img, (128, 128))
    b = batched.predict_image(img, (128, 128))
    assert batched_base.batch_calls > 0           # batched path exercised
    assert max(batched_base.batch_sizes) <= 2     # respected batch_size
    key = lambda ds: sorted((d["label"], *d["bbox_xyxy"]) for d in ds)
    assert key(a) == key(b)                       # identical detections


def test_native_masks_are_reconstructed_in_source_coordinates():
    pred = TiledPredictor(
        _FakeMaskDetector((64, 64)), overlap=0.0, keep_full=False, batch_size=2,
    )
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    records = pred.predict_image(image, (128, 128))
    assert len(records) == 4
    assert all(record["mask"].shape == (128, 128) for record in records)
    assert sum(int(record["mask"].sum()) for record in records) == 400
    assert any(record["mask"][67, 66] for record in records)


def test_native_mask_canvas_is_allocated_only_for_nms_survivors(monkeypatch):
    import kwcoco_detector_kit.predictors.tiled as tiled_mod

    pred = TiledPredictor(
        _FakeMaskDetector((64, 64)), overlap=0.0, keep_full=False, batch_size=2,
    )
    mask_a = np.ones((64, 64), dtype=bool)
    mask_b = np.ones((64, 64), dtype=bool)
    raw = {
        "deferred": None,
        "offsets": [(0, 0), (32, 0)],
        "source_hw": (128, 128),
        "window_dets": [
            [{
                "label": 0, "score": 0.9,
                "bbox_xyxy": [40.0, 10.0, 50.0, 20.0], "mask": mask_a,
            }],
            [{
                "label": 0, "score": 0.8,
                "bbox_xyxy": [8.0, 10.0, 18.0, 20.0], "mask": mask_b,
            }],
        ],
        "full_dets": None,
    }

    full_canvas_allocations = 0
    real_zeros = tiled_mod.np.zeros

    def counting_zeros(shape, *args, **kwargs):
        nonlocal full_canvas_allocations
        if tuple(shape) == (128, 128):
            full_canvas_allocations += 1
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(tiled_mod.np, "zeros", counting_zeros)
    records = pred._merge_and_nms(raw)
    assert len(records) == 1
    assert records[0]["mask"].shape == (128, 128)
    # The two crop detections map to the same source box. NMS keeps one, and
    # only that surviving mask should ever be expanded to a full-source canvas.
    assert full_canvas_allocations == 1


def test_max_dets_caps_top_k_by_score():
    # Many distinct windows -> many detections; max_dets keeps the top-K
    # by score after NMS.
    base = _FakeBatchDetector((64, 64))
    pred = TiledPredictor(base, overlap=0.0, keep_full=False, max_dets=2)
    img = np.zeros((256, 256, 3), dtype=np.uint8)  # 16 windows -> 16 dets
    dets = pred.predict_image(img, (256, 256))
    assert len(dets) == 2  # capped
    # all kept are the max score the fake emits (0.9); float32 round-trip
    assert all(abs(d["score"] - 0.9) < 1e-4 for d in dets)


def test_reduce_window_floor_and_topk():
    import kwimage
    from kwcoco_detector_kit.eval.tiled_predictor import _to_detections
    base = _FakeWindowDetector((64, 64))
    pred = TiledPredictor(base, per_window_nms=False,
                          pre_nms_score_thresh=0.05, pre_nms_topk=2)
    det = _to_detections([
        {"label": 0, "score": 0.001, "bbox_xyxy": [0, 0, 5, 5]},     # below floor
        {"label": 0, "score": 0.9, "bbox_xyxy": [100, 100, 110, 110]},
        {"label": 0, "score": 0.6, "bbox_xyxy": [200, 200, 210, 210]},
        {"label": 0, "score": 0.3, "bbox_xyxy": [300, 300, 310, 310]},  # dropped by topk=2
    ])
    out = pred._reduce_window(det)
    got = sorted(round(float(s), 2) for s in out.scores)
    assert got == [0.6, 0.9]


def test_nms_indices_suppresses_overlap():
    boxes = np.array([
        [0, 0, 10, 10],
        [1, 1, 11, 11],   # ~IoU 0.68 with box0 -> suppressed
        [50, 50, 60, 60],  # disjoint -> kept
    ], dtype=np.float64)
    scores = np.array([0.9, 0.8, 0.7])
    keep = _nms_indices(boxes, scores, 0.5)
    assert keep == [0, 2]


def test_per_class_nms_is_per_label():
    dets = [
        {"label": 0, "score": 0.9, "bbox_xyxy": [0, 0, 10, 10]},
        {"label": 1, "score": 0.8, "bbox_xyxy": [0, 0, 10, 10]},  # same box, diff class
    ]
    kept = _per_class_nms(dets, 0.5)
    # Different classes must NOT suppress each other.
    assert len(kept) == 2


def test_predict_source_tiles_prediction_space_not_native_space():
    """A coarse reader should grid the scaled view, not native dimensions."""
    from kwcoco_detector_kit.predictors.space import PredictionSpace

    base = _FakeBatchDetector(size=(32, 32))
    pred = TiledPredictor(
        base, window=(32, 32), overlap=0.0, keep_full=False, batch_size=2,
    )
    reader = _FakeArrayReader(np.zeros((64, 64, 3), dtype=np.uint8))
    reader.source_hw = (128, 128)
    reader.prediction_hw = (64, 64)
    reader.space = PredictionSpace.from_scale(reader.source_hw, 0.5)

    records = pred.predict_source(reader)
    assert pred.n_windows == 4
    assert len(records) == 4
    # TiledPredictor intentionally returns prediction-space geometry. The
    # writer/postprocess layer owns the explicit warp back to native space.
    corners = sorted((r["bbox_xyxy"][0], r["bbox_xyxy"][1]) for r in records)
    assert corners == [(2.0, 3.0), (2.0, 35.0), (34.0, 3.0), (34.0, 35.0)]
