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
  cli            kwconf-based command-line entry points
"""
from kwcoco_detector_kit._version import __version__

# Run the geowatch-style binary-module pre-import dance BEFORE the kit's
# CLI subcommands import torch / kwcoco / etc. The default profile imports
# GDAL via osgeo first (and calls UseExceptions()), which sidesteps a
# class of "DelayedLoad may not be efficient without gdal" warnings + the
# slow-mining symptom they correlate with. Override with KCD_PREIMPORT=0
# to disable, or KCD_PREIMPORT=pyproj,gdal etc. to customise.
from kwcoco_detector_kit._preimport import execute_ordered_preimports as _pre
_pre()
del _pre

__all__ = ["__version__"]
