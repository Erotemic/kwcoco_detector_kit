"""
kwcoco-detector-kit
===================

Domain-agnostic object-detector training pipeline on kwcoco datasets.

Top-level subpackages:

  data           kwcoco tile augmentation, merge, mine, MSCOCO export
  trainers       trainer plugin interface + DEIMv2 / OpenGroundingDINO / mock_tiny
  predictors     trained-checkpoint inference adapters
  export         ONNX export + modelspec sidecar + parity guard + package YAML
  eval           kwcoco eval driver + checkpoint shortlist + ONNX bench
  orchestration  pareto sweep + round loop + eligibility manifest + setup audit
  cli            scriptconfig-based command-line entry points
"""
from kwcoco_detector_kit._version import __version__

__all__ = ["__version__"]
