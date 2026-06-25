"""
Trained-checkpoint inference adapters.

  _interface.DetectorPredictor   Protocol every predictor satisfies.
  mock_tiny                       Predict using a mock_tiny checkpoint.
  deimv2                          Predict using a DEIMv2 checkpoint.
  onnx.OnnxPredictor              Predict using a kit-exported ONNX package
                                  (onnxruntime only — no PyTorch).
"""
from kwcoco_detector_kit.predictors import _interface
from kwcoco_detector_kit.predictors.onnx import OnnxPredictor

__all__ = ["_interface", "OnnxPredictor"]
