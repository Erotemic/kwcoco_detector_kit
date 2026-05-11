"""
Predictor plugin protocol.

A predictor wraps a trained checkpoint and exposes a tiny interface
for the eval + hard-negative-mining paths to drive inference. The
trainer plugin's ``build_predictor()`` factory returns an instance.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


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
            'score': float}`` dicts. May be empty.
        """
