"""
Trainer plugins.

  _interface.DetectorTrainer    Protocol every trainer satisfies.
  _registry.register_trainer    decorator: registers by name.
  _registry.get_trainer         lookup by name.
  _tier                         GPU-tier auto-detection + memory table.
  mock_tiny                     CPU smoke detector.
  deimv2                        DEIMv2 (12 variants: HGNetv2 + DINOv3).
"""
from kwcoco_detector_kit.trainers import _interface, _registry, _tier

# Import-side-effect: register the built-in trainer plugins.
from kwcoco_detector_kit.trainers import mock_tiny  # noqa: F401
from kwcoco_detector_kit.trainers import deimv2  # noqa: F401

__all__ = ["_interface", "_registry", "_tier", "mock_tiny", "deimv2"]
