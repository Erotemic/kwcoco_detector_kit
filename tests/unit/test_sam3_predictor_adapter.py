from __future__ import annotations


def test_sam3_state_to_records_scales_boxes_and_masks():
    import numpy as np

    from kwcoco_detector_kit.predictors.sam3 import SAM3TextPredictor

    state = {
        "boxes": np.array([[1, 2, 5, 6]], dtype=np.float32),
        "scores": np.array([0.75], dtype=np.float32),
        "masks": np.array([[[[0, 1], [1, 1]]]], dtype=bool),
    }
    records = SAM3TextPredictor._state_to_records(
        state,
        actual_hw=(8, 8),
        orig_size=(16, 12),
    )
    assert len(records) == 1
    rec = records[0]
    assert rec["label"] == 0
    assert abs(rec["score"] - 0.75) < 1e-6
    assert rec["bbox_xyxy"] == [2.0, 3.0, 10.0, 9.0]
    assert rec["mask"].shape == (12, 16)
    assert rec["mask"].dtype == bool


def test_sam3_empty_state_to_records():
    from kwcoco_detector_kit.predictors.sam3 import SAM3TextPredictor

    records = SAM3TextPredictor._state_to_records(
        {}, actual_hw=(8, 8), orig_size=(8, 8)
    )
    assert records == []


def test_sam3_admission_review_defaults_preserve_uncertain_truth():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2]
        / "kwcoco_detector_kit/eval/sam3_admission.py"
    ).read_text()
    assert 'review = kwconf.Value(' in source
    assert '"ignore,unknown,unkown"' in source
    assert 'default_non_target_policy = kwconf.Value(' in source
    assert '"background", choices=["background", "ignore", "error"]' in source
    assert '"true": str(src)' in source


def test_coerce_numpy_accepts_torch_bfloat16():
    import numpy as np
    import pytest

    torch = pytest.importorskip("torch")
    from kwcoco_detector_kit.predictors.sam3 import _coerce_numpy

    value = torch.tensor([0.25, 0.75], dtype=torch.bfloat16)
    got = _coerce_numpy(value)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, [0.25, 0.75], rtol=0, atol=1e-3)


def test_sam3_admission_uses_ubelt_compatible_postfix_api():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2]
        / "kwcoco_detector_kit/eval/sam3_admission.py"
    ).read_text()
    assert "prog.set_postfix(progress_postfix)" in source
    assert "prog.set_postfix(\n            dets=" not in source
