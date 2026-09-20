from __future__ import annotations


class _FakeDetections:
    def __init__(self):
        import numpy as np
        self.xyxy = np.array([[0, 0, 4, 4], [1, 1, 5, 5]], dtype=float)
        self.confidence = np.array([0.9, 0.8], dtype=float)
        self.class_id = np.array([0, 1], dtype=int)
        self.mask = None


def test_rfdetr_records_drop_background_sentinel():
    from kwcoco_detector_kit.trainers.rfdetr import RFDETRSegPredictor

    records = RFDETRSegPredictor._records(
        _FakeDetections(), orig_size=(8, 8), actual_hw=(8, 8), num_classes=1
    )
    assert len(records) == 1
    assert records[0]["label"] == 0
