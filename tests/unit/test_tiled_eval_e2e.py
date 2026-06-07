"""
End-to-end tiled-eval tests on kwcoco demo data with a mock predictor.

Exercises the real run_kwcoco_eval code paths (windowing, columnar
kwimage.Detections merge, per-window reduction, NMS, cap, the
serial vs producer/consumer-pipelined loops, and the dump + kwcoco-eval
metric pass) without needing a GPU or a trained checkpoint. The mock
predictor returns kwimage.Detections from predict_batch (the columnar
contract) and dicts from predict_image (whole-image / keep_full).
"""
import numpy as np
import kwimage
import kwcoco
import pytest


class _MockPredictor:
    """Deterministic windowed predictor. Emits one box per window at a fixed
    spot in the window's frame, so results are reproducible and the offset /
    merge / NMS logic is checkable."""

    def __init__(self, size=(32, 32), n_classes=2):
        self._size = size
        self._n = n_classes

    @property
    def eval_spatial_size(self):
        return self._size

    def predict_image(self, image_np, orig_size):
        # whole-image / keep_full path returns dicts (the protocol shape)
        return [{"label": 0, "score": 0.5, "bbox_xyxy": [1.0, 1.0, 6.0, 6.0]}]

    def predict_batch(self, images_np, orig_sizes):
        # columnar contract: one kwimage.Detections per crop
        out = []
        for k, _ in enumerate(images_np):
            boxes = np.array([[2.0, 2.0, 8.0, 8.0],
                              [3.0, 3.0, 9.0, 9.0]], dtype=np.float32)
            scores = np.array([0.9, 0.2], dtype=np.float32)
            cidx = np.array([k % self._n, (k + 1) % self._n], dtype=np.int64)
            out.append(kwimage.Detections(
                boxes=kwimage.Boxes(boxes, "ltrb"),
                scores=scores, class_idxs=cidx))
        return out


class _MockTrainer:
    name = "mock"

    def __init__(self, predictor):
        self._predictor = predictor

    def build_predictor(self, workdir, *, device="cpu"):
        return self._predictor


def _demo_kwcoco(tmp_path, n_imgs=2, dim=80):
    """A tiny on-disk kwcoco bundle whose images are bigger than the mock
    window (so the tiled path actually tiles)."""
    dset = kwcoco.CocoDataset.demo("shapes{}".format(n_imgs),
                                   gsize=(dim, dim), rng=0)
    return dset


def _run(tmp_path, *, tiled, nms_workers=0, max_dets=None, pre_floor=0.0,
         per_window_nms=True):
    from kwcoco_detector_kit.eval.kwcoco_eval import run_kwcoco_eval
    dset = _demo_kwcoco(tmp_path)
    cats = [c["name"] for c in dset.dataset["categories"]][:2]
    while len(cats) < 2:
        cats.append(f"cat{len(cats)}")
    trainer = _MockTrainer(_MockPredictor(size=(32, 32), n_classes=len(cats)))
    metrics = run_kwcoco_eval(
        trainer=trainer,
        workdir=tmp_path / "wd",
        test_kwcoco=dset.fpath,
        kcd_root=tmp_path / "kcd",
        candidate_id="mock_32x32_fixed",
        category_names=cats,
        force=True,
        tiled_eval=tiled,
        tiled_max_dets=max_dets,
        tiled_pre_nms_score_thresh=pre_floor,
        tiled_per_window_nms=per_window_nms,
        eval_nms_workers=nms_workers,
        read_workers=0,
        device="cpu",
    )
    return metrics


def _pred_boxes(metrics_fpath):
    # pred bundle sits next to the eval/ dir: <candidate>/pred_boxes.kwcoco.zip
    from pathlib import Path
    pred = Path(metrics_fpath).parent.parent / "pred_boxes.kwcoco.zip"
    dset = kwcoco.CocoDataset.coerce(str(pred))
    return sorted(
        (a["image_id"], a["category_id"], round(a["score"], 4),
         tuple(round(v, 2) for v in a["bbox"]))
        for a in dset.dataset["annotations"]
    )


def test_tiled_eval_runs_and_writes_metrics(tmp_path):
    m = _run(tmp_path, tiled=True)
    assert m.exists()


def test_wholeimage_eval_runs(tmp_path):
    m = _run(tmp_path, tiled=False)
    assert m.exists()


def test_pipelined_equals_serial(tmp_path):
    # The producer/consumer pipeline must not change predictions.
    serial = _pred_boxes(_run(tmp_path / "a", tiled=True, nms_workers=0))
    piped = _pred_boxes(_run(tmp_path / "b", tiled=True, nms_workers=3))
    assert serial == piped
    assert len(serial) > 0


def test_max_dets_cap_reduces_predictions(tmp_path):
    uncapped = _pred_boxes(_run(tmp_path / "u", tiled=True, max_dets=None))
    capped = _pred_boxes(_run(tmp_path / "c", tiled=True, max_dets=1))
    # cap of 1/image can only reduce (or equal) the prediction count
    assert len(capped) <= len(uncapped)
    assert len(capped) > 0


def test_pre_nms_floor_drops_low_scores(tmp_path):
    # floor above the 0.2 secondary box should drop it everywhere
    high = _pred_boxes(_run(tmp_path / "h", tiled=True, pre_floor=0.5))
    assert all(s >= 0.5 for (_g, _c, s, _b) in high)


def test_per_window_nms_off_matches_on(tmp_path):
    # The global NMS subsumes per-window NMS, so toggling per_window_nms must
    # not change the final predictions (it's purely a speed knob).
    on = _pred_boxes(_run(tmp_path / "on", tiled=True, per_window_nms=True))
    off = _pred_boxes(_run(tmp_path / "off", tiled=True, per_window_nms=False))
    assert on == off
