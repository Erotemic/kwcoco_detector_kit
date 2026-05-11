"""
Trained-checkpoint inference adapters.

  _interface.DetectorPredictor   Protocol every predictor satisfies.
  mock_tiny                       Predict using a mock_tiny checkpoint.
  deimv2                          Predict using a DEIMv2 checkpoint.
"""
from kwcoco_detector_kit.predictors import _interface

__all__ = ["_interface"]
