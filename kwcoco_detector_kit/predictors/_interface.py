"""
Predictor plugin protocol.

A predictor wraps a trained checkpoint and exposes a tiny interface
for the eval + hard-negative-mining paths to drive inference. The
trainer plugin's ``build_predictor()`` factory returns an instance.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


def predict_batch(predictor, images_np, orig_sizes):
    """Use an optional backend batch method, with a correct serial fallback."""
    method = getattr(predictor, "predict_batch", None)
    if callable(method):
        return method(images_np, orig_sizes)
    return [
        predictor.predict_image(image, size)
        for image, size in zip(images_np, orig_sizes)
    ]


@runtime_checkable
class DetectorPredictor(Protocol):
    """The trainer-plugin-supplied inference adapter.

    Implementations are free to load whatever they like in ``__init__``
    (torch checkpoint, ONNX session, etc.). The hard-negative miner
    only calls ``predict_image`` and reads ``eval_spatial_size``.
    """

    @property
    def eval_spatial_size(self) -> tuple[int, int]:
        """(H, W) the model evaluates at — used to validate tile inputs."""

    def predict_image(self, image_np, orig_size) -> list[dict]:
        """Score one image; return a list of detections.

        Args:
            image_np: HxWx3 uint8 numpy array.
            orig_size: (W, H) of the image — coords in the returned
                ``bbox_xyxy`` are in this pixel frame.

        Returns:
            list of ``{'label': int, 'bbox_xyxy': [x0, y0, x1, y1],
            'score': float}`` dicts. A native instance-segmentation backend
            adds ``mask`` as a 2-D bool/uint8 array in the same ``orig_size``
            coordinate frame. May be empty.
        """


@runtime_checkable
class BatchDetectorPredictor(DetectorPredictor, Protocol):
    """Optional acceleration protocol; callers must support serial fallback."""

    def predict_batch(self, images_np, orig_sizes) -> list[list[dict]]:
        """Score a batch, preserving input order and the per-image contract."""
