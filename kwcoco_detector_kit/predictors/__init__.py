"""
Trained-checkpoint inference adapters.

  _interface.DetectorPredictor   Protocol every predictor satisfies.
  mock_tiny                       Predict using a mock_tiny checkpoint.
  deimv2                          Predict using a DEIMv2 checkpoint.
  onnx.OnnxPredictor              Predict using a kit-exported ONNX package
                                  (onnxruntime only - no PyTorch).
  source_window.SourceWindowReader
                                  No-cache source/window realization.
  pipeline.PredictionPipeline     Bounded I/O/GPU/CPU pipeline orchestration.
  tiled.TiledPredictor            Source-coordinate tiled inference/merge.
"""
from kwcoco_detector_kit.predictors import _interface
from kwcoco_detector_kit.predictors.onnx import OnnxPredictor
from kwcoco_detector_kit.predictors.pipeline import PredictionPipeline
from kwcoco_detector_kit.predictors.source_window import SourceWindowReader

__all__ = [
    "_interface",
    "OnnxPredictor",
    "PredictionPipeline",
    "SourceWindowReader",
    "TiledPredictor",
]


def __getattr__(name):
    """Keep the heavier tiled/data CLI dependency lazy at package import time."""
    if name == "TiledPredictor":
        from kwcoco_detector_kit.predictors.tiled import TiledPredictor
        return TiledPredictor
    raise AttributeError(name)
