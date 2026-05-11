"""
Deployment-package YAML — declares which ONNX + which postprocess
parameters constitute a deployable model. Ported in shape from the
prior project's ``cli_package``, dropped the SAM2 segmenter coupling
(Phase 2 will reintroduce it as an optional segmenter slot).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml


def build_package_yaml(
    *,
    out_fpath: Path,
    backend: str,
    detector_onnx_fpath: Path,
    detector_modelspec_fpath: Path,
    metadata_name: str,
    category_name: str = "widget",
    score_thresh: float = 0.30,
    nms_iou_thresh: float = 0.50,
    extra: Optional[dict] = None,
) -> Path:
    spec = {
        "backend": str(backend),
        "metadata_name": str(metadata_name),
        "category_name": str(category_name),
        "detector": {
            "onnx_fpath": str(detector_onnx_fpath),
            "modelspec_fpath": str(detector_modelspec_fpath),
        },
        "postprocess": {
            "score_thresh": float(score_thresh),
            "nms_iou_thresh": float(nms_iou_thresh),
        },
    }
    if extra:
        spec.update(extra)
    out_fpath = Path(out_fpath)
    out_fpath.parent.mkdir(parents=True, exist_ok=True)
    out_fpath.write_text(yaml.safe_dump(spec, sort_keys=False))
    return out_fpath
